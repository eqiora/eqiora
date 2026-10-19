use super::*;

impl CommonSpatialPlan {
    pub(super) fn prescribed_geometry_at(
        &self,
        time_s: f64,
    ) -> Result<Option<eqiora_meshing::FixedTopologyGeometryState<2>>, Diagnostic> {
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("geometry projection requires spatial equations"));
        };
        let Some(motion) = equations
            .regions
            .iter()
            .find_map(|region| region.form.motion())
        else {
            return Ok(None);
        };
        let (NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }) = self.admission.resources()
        else {
            return Err(invalid(
                "moving geometry requires its authenticated simplicial Mesh",
            ));
        };
        motion
            .bind(self.admission.program(), time_s)?
            .geometry_state(mesh.mesh())
            .map(Some)
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
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let mut bound = equations.clone();
        for region in &mut bound.regions {
            if region.form.motion().is_some() {
                region.form = if let Some(previous) = previous_time_s {
                    crate::form_compiler::spatial::CompiledSpatialForm::derive_over_step(
                        self.admission.program(),
                        region.form.domain(),
                        previous,
                        time_s,
                        temporal.step().value(),
                    )?
                } else {
                    crate::form_compiler::spatial::CompiledSpatialForm::derive_at_time(
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
        let mut prescribed_states = BTreeMap::new();
        let boundary_value = |key: crate::region_assembly::mapping::FieldDof,
                              law: &crate::form_compiler::region::RegionBoundaryLaw<f64>,
                              value: f64| {
            let Some(state) = law.trace_field.filter(|field| *field != law.tested) else {
                return Ok(Some(value));
            };
            let state_key = crate::region_assembly::mapping::FieldDof {
                field: state,
                ..key
            };
            let physical = crate::cartesian_elliptic::support::require_compatible_boundary_value(
                prescribed_states.get(&state_key).copied(),
                value,
            )?
            .expect("finite boundary datum");
            prescribed_states.insert(state_key, physical);
            match previous {
                None => Ok(None), // State validation uses physical displacement, without imposing a surrogate rate.
                Some(fields) => {
                    let old = fields
                        .get(&state)
                        .and_then(|field| field.coefficients.get(&state_key))
                        .ok_or_else(|| {
                            invalid("displacement boundary lacks its exact previous coefficient")
                        })?;
                    Ok(Some((physical - *old) / temporal.step().value()))
                }
            }
        };
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                let (mapping, mut input) = bound.cartesian_assembly_with_boundary(
                    mesh.mesh(),
                    &self.admission.discretizations,
                    &self.admission.system_scaling,
                    boundary_value,
                )?;
                input.prescribed_states = prescribed_states;
                Ok((mapping, input))
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                let state = self.prescribed_geometry_at(time_s)?;
                let (mapping, forms, natural) = bound.simplicial_assembly_with_boundary(
                    mesh,
                    &self.admission.discretizations,
                    &self.admission.system_scaling,
                    state.as_ref(),
                    boundary_value,
                )?;
                Ok((
                    mapping,
                    crate::region_assembly::mapping::RegionSolveInput {
                        operator_properties: bound.finite_element_operator_properties(),
                        geometry_action: None,
                        forms,
                        natural,
                        previous: None,
                        prescribed_states,
                    },
                ))
            }
            _ => Err(invalid("scalar storage requires its exact nodal Mesh")),
        }
    }

    pub(super) fn scalar_step_assembly(
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
        let CommonStateKind::Spatial(values) = &state.kind else {
            return Err(invalid("scalar Run requires scalar State"));
        };
        let (old_mapping, _) = self.scalar_assembly_at(state.require_time_s()?)?;
        let previous = self.physical_fields(&old_mapping, values)?;
        let (mapping, mut input) = self.scalar_assembly_with_history(
            next_time_s,
            Some(state.require_time_s()?),
            Some(&previous),
        )?;
        let RecognizedNativeModel::Spatial(bound) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        if bound
            .regions
            .iter()
            .any(|region| region.form.motion().is_some())
        {
            let (NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }) =
                self.admission.resources()
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
            if !next_time_s.is_finite() || next_time_s <= state.require_time_s()? {
                return Err(invalid(
                    "moving scalar action requires a later finite accepted Time",
                ));
            }
            let previous = self
                .prescribed_geometry_at(state.require_time_s()?)?
                .expect("retained motion");
            let current = self
                .prescribed_geometry_at(next_time_s)?
                .expect("retained motion");
            input.geometry_action =
                Some(Box::new(eqiora_meshing::FixedTopologyGeometryAction::new(
                    mesh.mesh(),
                    &previous,
                    &current,
                    step,
                )?));
        }
        input.previous = Some(previous);
        Ok((mapping, input))
    }

    pub(in crate::numerical_admission) fn advance_scalar(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        next_time_s: f64,
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar inventory"));
        };
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.scalar_step_assembly(state, next_time_s)?;
        let request = LinearSolveRequest::new(&checked, self.admission.linear.solver);
        if self.admission.nonlinear.is_some() {
            let initial = input.previous.clone().expect("transient history");
            let output = self.solve_region_newton(&mapping, input, &initial, request)?;
            let values = output
                .fields
                .into_values()
                .flat_map(|field| field.coefficients.into_values())
                .collect();
            let mut accepted = CommonAcceptedSpatialStep::new(
                state,
                self.scalar_state(Some(next_time_s), values)?,
            )?;
            accepted.residual = Some((
                output.solve.residual_norm,
                output.solve.evidence.residual_target,
            ));
            accepted.execution = Some(super::nonlinear::newton_execution(&output.solve));
            return Ok(accepted);
        }
        let current_mesh = input
            .geometry_action
            .as_ref()
            .map(|action| action.current_mesh().clone());
        let output = match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.solve(mesh.mesh(), input, self.admission.linear.workers, request)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping.solve(
                current_mesh.as_ref().unwrap_or_else(|| mesh.mesh()),
                input,
                self.admission.linear.workers,
                request,
            )?,
            _ => return Err(invalid("scalar storage requires its exact nodal Mesh")),
        };
        let values = output
            .fields
            .into_values()
            .flat_map(|field| field.coefficients.into_values())
            .collect();
        let mut accepted =
            CommonAcceptedSpatialStep::new(state, self.scalar_state(Some(next_time_s), values)?)?;
        accepted.execution = Some(CommonSpatialExecution {
            solve: crate::common_result::evidence::CommonSolveHistory::Linear(Box::new(
                crate::common_result::evidence::CommonSolveEvidence::from_report(
                    &output.solve_report,
                ),
            )),
            assembly: crate::common_result::evidence::CommonAssemblyEvidence::from_report(
                &output.assembly_report,
            ),
        });
        Ok(accepted)
    }
}
impl ResolvedCommonPlan {
    pub(crate) fn spatial_state_space_identity(&self) -> Result<String, Diagnostic> {
        match self {
            Self::Spatial(plan) if plan.admission.temporal.is_some() => {
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
    /// Field-major, node-major, component-minor coefficients of this spatial State.
    pub(crate) fn spatial_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Spatial(values) => Some(values),
            _ => None,
        }
    }
}
