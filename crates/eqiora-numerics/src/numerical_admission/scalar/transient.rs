use super::*;
use eqiora_core::RawId;

impl CommonLinearPlan {
    pub(crate) fn scalar_state(
        &self,
        time_s: f64,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        let (mapping, _) = self.scalar_assembly_at(time_s)?;
        let keys = mapping.keys().collect::<Vec<_>>();
        if values.len() != keys.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "scalar State requires a transient Plan and complete finite nodal coefficients",
            ));
        }
        let prescribed = mapping.lift(&vec![0.0; mapping.free_count()], false)?;
        for (key, value) in keys.iter().zip(&values) {
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
        history_fields(&mapping, &values)?;
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
        self.scalar_assembly_with_history(time_s, None)
    }

    fn scalar_assembly_with_history(
        &self,
        time_s: f64,
        previous_time_s: Option<f64>,
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
            region.form = region.form.bind_backward_euler(temporal.step())?;
        }
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => bound.cartesian_assembly(mesh.mesh()),
            NativeMeshResources::GmshSimplicial { mesh, .. } => {
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
        let (mapping, mut input) =
            self.scalar_assembly_with_history(next_time_s, Some(state.time_s()))?;
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
        input.previous = Some(history_fields(&mapping, values)?);
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

fn history_fields(
    mapping: &crate::region_assembly::mapping::RegionDofMap<f64>,
    values: &[f64],
) -> Result<BTreeMap<RawId, crate::region_assembly::mapping::RecoveredRegionField<f64>>, Diagnostic>
{
    let keys = mapping.keys().collect::<Vec<_>>();
    if keys.len() != values.len() {
        return Err(invalid(
            "scalar history differs from the exact mapped coefficient inventory",
        ));
    }
    let mut fields = BTreeMap::new();
    for (key, &value) in keys.into_iter().zip(values) {
        let (domain, layout) = mapping.field_layout(key.field).expect("mapped Field");
        fields
            .entry(key.field)
            .or_insert_with(|| crate::region_assembly::mapping::RecoveredRegionField {
                domain,
                value_type: layout.value_type.clone(),
                space: layout.space,
                coefficients: BTreeMap::new(),
            })
            .coefficients
            .insert(key, value);
    }
    mapping.validate_physical(&fields)?;
    Ok(fields)
}
