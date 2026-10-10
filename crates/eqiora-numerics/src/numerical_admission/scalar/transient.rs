use super::*;

impl CommonLinearPlan {
    pub(crate) fn scalar_state(
        &self,
        time_s: f64,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        let (mapping, input) = self.scalar_assembly_at(time_s)?;
        let history = self.history_fields(&mapping, &values)?;
        let keys = mapping.keys().collect::<Vec<_>>();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "scalar State requires a transient Plan and complete finite nodal coefficients",
            ));
        }
        let prescribed = mapping.lift(&vec![0.0; mapping.free_count()], false)?;
        for (key, expected) in &input.prescribed_states {
            if history
                .get(&key.field)
                .and_then(|field| field.coefficients.get(key))
                != Some(expected)
            {
                return Err(invalid(
                    "linear State contradicts prescribed displacement boundary values",
                ));
            }
        }
        for key in &keys {
            let value = &history[&key.field].coefficients[key];
            if mapping.free_dof(*key).is_none() {
                let (_, layout) = mapping.field_layout(key.field).expect("exact Field");
                let expected =
                    prescribed[mapping.global_dof(*key).expect("exact coordinate")] * layout.scale;
                if *value != expected {
                    return Err(invalid(
                        "scalar State contradicts prescribed boundary values",
                    ));
                }
            }
        }
        CommonState::new(
            self.identity().to_owned(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            CommonStateKind::Linear(values.into_boxed_slice()),
        )
    }

    pub(super) fn scalar_assembly(
        &self,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        self.scalar_assembly_at(0.0)
    }

    pub(super) fn scalar_assembly_at(
        &self,
        time_s: f64,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        self.scalar_assembly_with_history(time_s, None, None)
    }

    fn scalar_assembly_with_history(
        &self,
        time_s: f64,
        previous_time_s: Option<f64>,
        previous: Option<
            &BTreeMap<
                eqiora_core::RawId,
                crate::region_assembly::mapping::RecoveredRegionField<f64>,
            >,
        >,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        let temporal = self
            .admission
            .temporal
            .ok_or_else(|| invalid("scalar storage requires BackwardEuler"))?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        if equations
            .regions
            .iter()
            .flat_map(|region| region.form.boundary_laws().values())
            .flat_map(|laws| laws.values())
            .any(|law| law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Trace)
        {
            return Err(invalid(
                "scalar storage requires complete essential boundary data",
            ));
        }
        let mut bound = equations.clone();
        for region in &mut bound.regions {
            if region.form.motion().is_some() {
                region.form = if let Some(previous) = previous_time_s {
                    crate::form_compiler::linear::CompiledLinearBlockForm::derive_over_step(
                        self.admission.program(),
                        region.form.domain(),
                        previous,
                        time_s,
                        temporal.step().value(),
                    )?
                } else {
                    crate::form_compiler::linear::CompiledLinearBlockForm::derive_at_time(
                        self.admission.program(),
                        region.form.domain(),
                        2,
                        &BTreeSet::new(),
                        Some(time_s),
                    )?
                };
            }
            if region.form.motion().is_none() && !region.form.kinematics().is_empty() {
                region
                    .form
                    .bind_boundary_time(self.admission.program(), time_s)?;
            }
            region.form = region.form.bind_backward_euler(temporal.step())?;
        }
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                let mut prescribed_states = BTreeMap::new();
                let (mapping, mut input) = bound.cartesian_assembly_with_boundary(mesh.mesh(), |key, law, value| {
                    let Some(state) = law.trace_field.filter(|field| *field != law.tested) else { return Ok(Some(value)); };
                    let state_key = crate::region_assembly::mapping::FieldDof { field: state, ..key };
                    let physical = crate::cartesian_elliptic::support::require_compatible_boundary_value(
                        prescribed_states.get(&state_key).copied(), value,
                    )?.expect("finite boundary datum");
                    prescribed_states.insert(state_key, physical);
                    match previous {
                        None => Ok(None), // State validation uses physical displacement, without imposing a surrogate rate.
                        Some(fields) => {
                            let old = fields.get(&state).and_then(|field| field.coefficients.get(&state_key))
                                .ok_or_else(|| invalid("displacement boundary lacks its exact previous coefficient"))?;
                            Ok(Some((physical - *old) / temporal.step().value()))
                        }
                    }
                })?;
                input.prescribed_states = prescribed_states;
                Ok((mapping, input))
            }
            NativeMeshResources::GmshSimplicial { mesh, .. } => {
                if bound
                    .regions
                    .iter()
                    .flat_map(|region| region.form.boundary_laws().values())
                    .flat_map(|laws| laws.values())
                    .any(|law| law.trace_field.is_some_and(|field| field != law.tested))
                {
                    return Err(invalid(
                        "displacement boundary history currently requires a Cartesian nodal Mesh",
                    ));
                }
                let state = equations
                    .regions
                    .iter()
                    .find_map(|region| region.form.motion())
                    .map(|motion| {
                        motion
                            .bind(self.admission.program(), time_s)?
                            .geometry_state(mesh.mesh())
                    })
                    .transpose()?;
                let (mapping, forms, natural) = bound.simplicial_assembly_at(
                    mesh,
                    Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                    state.as_ref(),
                )?;
                Ok((
                    mapping,
                    crate::region_assembly::mapping::RegionSolveInput {
                        operator_properties: eqiora_solver::LinearOperatorProperties::General,
                        geometry_action: None,
                        forms,
                        natural,
                        previous: None,
                        prescribed_states: BTreeMap::new(),
                    },
                ))
            }
            _ => Err(invalid("scalar storage requires its exact nodal Mesh")),
        }
    }

    fn scalar_step_assembly(
        &self,
        state: &CommonState,
        next_time_s: f64,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        if state.state_space_identity() != self.identity() {
            return Err(invalid("scalar State belongs to a foreign exact Plan"));
        }
        let CommonStateKind::Linear(values) = &state.kind else {
            return Err(invalid("scalar Run requires scalar State"));
        };
        let (old_mapping, _) = self.scalar_assembly_at(state.time_s())?;
        let previous = self.history_fields(&old_mapping, values)?;
        let (mapping, mut input) =
            self.scalar_assembly_with_history(next_time_s, Some(state.time_s()), Some(&previous))?;
        let RecognizedNativeModel::Linear(bound) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        if let Some(motion) = bound.regions.iter().find_map(|region| region.form.motion()) {
            let NativeMeshResources::GmshSimplicial { mesh, .. } = self.admission.resources()
            else {
                return Err(invalid(
                    "moving scalar history requires its authenticated simplicial Mesh",
                ));
            };
            let step = self
                .admission
                .temporal
                .expect("scalar storage")
                .step()
                .value();
            // The Run owns start + accepted_count * step. Recomputing that
            // timestamp by repeated addition can differ by an ulp. The sealed
            // geometry action and both forms still bind the exact Plan step.
            if !next_time_s.is_finite() || next_time_s <= state.time_s() {
                return Err(invalid(
                    "moving scalar action requires a later finite accepted Time",
                ));
            }
            let previous = motion
                .bind(self.admission.program(), state.time_s())?
                .geometry_state(mesh.mesh())?;
            let current = motion
                .bind(self.admission.program(), next_time_s)?
                .geometry_state(mesh.mesh())?;
            input.geometry_action = Some(eqiora_meshing::FixedTopologyGeometryAction::new(
                mesh.mesh(),
                &previous,
                &current,
                step,
            )?);
        }
        input.previous = Some(previous);
        Ok((mapping, input))
    }

    pub(in crate::numerical_admission) fn advance_scalar(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        next_time_s: f64,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar inventory"));
        };
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.scalar_step_assembly(state, next_time_s)?;
        let request = LinearSolveRequest::new(&checked, self.admission.linear.solver);
        let current_mesh = input
            .geometry_action
            .as_ref()
            .map(|action| action.current_mesh().clone());
        let output = match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => mapping.solve(
                mesh.mesh(),
                input,
                self.admission.linear.workers,
                request,
                |reactions, values| reactions.recover(values),
            )?,
            NativeMeshResources::GmshSimplicial { mesh, .. } => mapping.solve(
                current_mesh.as_ref().unwrap_or_else(|| mesh.mesh()),
                input,
                self.admission.linear.workers,
                request,
                |reactions, values| reactions.recover(values),
            )?,
            _ => return Err(invalid("scalar storage requires its exact nodal Mesh")),
        };
        let values = output
            .fields
            .into_values()
            .flat_map(|field| field.coefficients.into_values())
            .collect();
        self.scalar_state(next_time_s, values)
    }
}
impl ResolvedCommonPlan {
    pub(crate) fn spatial_state_space_identity(&self) -> Result<String, Diagnostic> {
        match self {
            Self::Linear(plan) if plan.admission.temporal.is_some() => {
                Ok(plan.identity().to_owned())
            }
            Self::TransientFlow(plan) => Ok(plan.state_space_identity()),
            _ => Err(invalid(
                "Plan does not own a scalar/flow transient state space",
            )),
        }
    }
}

impl CommonState {
    /// Field-major, node-major, component-minor coefficients of this linear spatial State.
    pub fn linear_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Linear(values) => Some(values),
            _ => None,
        }
    }
}
