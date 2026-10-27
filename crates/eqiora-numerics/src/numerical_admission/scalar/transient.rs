use super::*;
use crate::form_compiler::region::coordinates::{self, Real};
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap};
use crate::spatial_expression::Coefficient;

impl CommonSpatialPlan {
    pub(super) fn solve_region_linear<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
        input: &crate::region_assembly::mapping::RegionSolveInput<S>,
        request: LinearSolveRequest<'_>,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver>,
    ) -> Result<crate::region_assembly::mapping::RegionSolveOutput<S>, Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let current_mesh = input
            .geometry_action
            .as_ref()
            .map(|action| action.current_mesh());
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => mapping.solve(
                mesh.mesh(),
                input,
                self.admission.linear.workers,
                request,
                prepared,
            ),
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping.solve(
                current_mesh.unwrap_or_else(|| mesh.mesh()),
                input,
                self.admission.linear.workers,
                request,
                prepared,
            ),
            _ => Err(invalid(
                "Region linear solve requires its exact affine Mesh",
            )),
        }
    }

    pub(super) fn prescribed_geometry_at(
        &self,
        time_s: f64,
    ) -> Result<Option<eqiora_meshing::FixedTopologyGeometryState<2>>, Diagnostic> {
        let motion = match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => equations
                .regions
                .iter()
                .find_map(|region| region.form.motion()),
            RecognizedNativeModel::ComplexSpatial(equations) => equations
                .regions
                .iter()
                .find_map(|region| region.form.motion()),
            _ => return Err(invalid("geometry projection requires spatial equations")),
        };
        let Some(motion) = motion else {
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

    pub(in crate::numerical_admission) fn scalar_assembly_at<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        time_s: f64,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        self.scalar_assembly_with_history(equations, time_s, None, None, &[])
    }

    pub(super) fn scalar_time_partial<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        time_s: f64,
        scopes: &[crate::form_compiler::spatial::InputPartialScope],
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        let coordinate = eqiora_ir::ScalarSymbolCoordinate::for_value(
            eqiora_schema::kernel::SymbolRef::Time,
            &eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, TIME)
                .expect("real scalar Time"),
        )?
        .remove(0);
        self.scalar_assembly_with_history(
            equations,
            time_s,
            None,
            None,
            &scopes
                .iter()
                .map(|scope| (&coordinate, *scope))
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn scalar_assembly_with_history<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        time_s: f64,
        previous_time_s: Option<f64>,
        previous: Option<
            &BTreeMap<eqiora_core::RawId, crate::region_assembly::mapping::RecoveredRegionField<S>>,
        >,
        partials: &[(
            &eqiora_ir::ScalarSymbolCoordinate,
            crate::form_compiler::spatial::InputPartialScope,
        )],
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        let step = || {
            self.admission
                .temporal
                .as_ref()
                .ok_or_else(|| invalid("discrete spatial history requires its time policy"))?
                .require_backward_euler()
        };
        // A method-dependent transform is shared by the whole coupled system;
        // its fluid and solid Regions must agree on the exact step.
        let bind_step = previous_time_s.is_some()
            || equations.regions.iter().any(|region| {
                region.form.motion().is_some() || !region.form.kinematics().is_empty()
            });
        let mut bound = equations.clone();
        for region in &mut bound.regions {
            if region.form.motion().is_some() {
                region.form = if let Some(previous) = previous_time_s {
                    crate::form_compiler::spatial::CompiledSpatialForm::derive_over_step(
                        self.admission.program(),
                        region.form.domain(),
                        previous,
                        time_s,
                        step()?.value(),
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
            if region.form.motion().is_none() {
                region.form = region
                    .form
                    .bind_inputs(crate::form_compiler::spatial::InputPoint::Time(time_s))?;
            }
            // Continuous, untransformed State/rate evaluation has no step.
            // Existing kinematic elimination and moving-inventory transforms
            // retain their explicit method binding until evaluated in original coordinates.
            if bind_step {
                region.form = region.form.bind_backward_euler(step()?)?;
            }
            for &(coordinate, scope) in partials {
                region.form = region.form.input_partial(coordinate, scope)?;
            }
        }
        let mut prescribed_states = BTreeMap::new();
        let boundary_value = |key: crate::region_assembly::mapping::FieldDof,
                              law: &crate::form_compiler::region::RegionBoundaryLaw<S>,
                              value: S| {
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
                    Ok(Some((physical - *old) / step()?.value()))
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
                        conservative: None,
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

    pub(super) fn scalar_step_assembly<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        state: &CommonState,
        grid: TimeGrid,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        if state.state_space_identity() != self.identity() {
            return Err(invalid("scalar State belongs to a foreign exact Plan"));
        }
        if grid.previous()? != state.time_grid()? {
            return Err(invalid(
                "scalar step differs from its predecessor's Time grid",
            ));
        }
        let step_s = self
            .admission
            .temporal
            .as_ref()
            .ok_or_else(|| invalid("scalar step requires BackwardEuler"))?
            .require_backward_euler()?
            .value();
        if grid.previous()?.time(step_s)?.to_bits() != state.require_time_s()?.to_bits() {
            return Err(invalid("scalar State differs from its retained Time grid"));
        }
        let CommonStateKind::Spatial { values, .. } = &state.kind else {
            return Err(invalid("scalar Run requires scalar State"));
        };
        self.scalar_history_assembly(
            equations,
            &SpatialHistory {
                grid,
                values: values.clone(),
            },
            grid.time(step_s)?,
        )
    }

    pub(super) fn scalar_history_assembly<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        history: &SpatialHistory,
        next_time_s: f64,
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        let step_s = self
            .admission
            .temporal
            .as_ref()
            .ok_or_else(|| invalid("spatial step requires BackwardEuler"))?
            .require_backward_euler()?
            .value();
        let previous_time_s = history.grid.previous()?.time(step_s)?;
        if history.grid.time(step_s)?.to_bits() != next_time_s.to_bits() {
            return Err(invalid(
                "spatial State Time differs from its exact Plan time grid",
            ));
        }
        let (old_mapping, old_input) = self.scalar_assembly_at(equations, previous_time_s)?;
        let previous = self.validate_state_fields(&old_mapping, &old_input, &history.values)?;
        let (mapping, mut input) = self.scalar_assembly_with_history(
            equations,
            next_time_s,
            Some(previous_time_s),
            Some(&previous),
            &[],
        )?;
        if equations
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
                .as_ref()
                .expect("scalar storage")
                .require_backward_euler()?
                .value();
            let previous = self
                .prescribed_geometry_at(previous_time_s)?
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

    #[cfg(test)] // Internal cold single-step reference; Runs own their provider session.
    pub(in crate::numerical_admission) fn advance_scalar(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic> {
        self.advance_scalar_prepared(state, backend, None)
    }

    pub(in crate::numerical_admission) fn advance_scalar_prepared(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver>,
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let grid = state.time_grid()?.advance(1)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                self.advance_typed(equations, state, backend, grid, prepared)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                self.advance_typed(equations, state, backend, grid, prepared)
            }
            _ => Err(invalid("missing spatial inventory")),
        }
    }

    fn advance_typed<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        grid: TimeGrid,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver>,
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let next_time_s = grid.time(
            self.admission
                .temporal
                .as_ref()
                .expect("temporal Plan")
                .require_backward_euler()?
                .value(),
        )?;
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.scalar_step_assembly(equations, state, grid)?;
        let request = LinearSolveRequest::new(&checked, self.admission.linear.solver);
        let (fields, residual, execution) = if self.admission.nonlinear.is_some() {
            let initial = input.previous.clone().expect("transient history");
            let output = self.solve_region_newton(&mapping, &input, &initial, request)?;
            let residual = Some((
                output.solve.residual_norm,
                output.solve.evidence.residual_target()?,
            ));
            let execution = super::nonlinear::newton_execution(&output.solve);
            (output.fields, residual, execution)
        } else {
            let output = self.solve_region_linear(&mapping, &input, request, prepared)?;
            let execution = CommonSpatialExecution {
                solve: crate::common_result::evidence::CommonSolveHistory::Linear(Box::new(
                    crate::common_result::evidence::CommonSolveEvidence::from_report(
                        &output.solve_report,
                    ),
                )),
                assembly: crate::common_result::evidence::CommonAssemblyEvidence::from_report(
                    &output.assembly_report,
                ),
            };
            // Validate operational provenance before publishing a State or
            // allowing another action to reuse the provider session. Targets
            // and counts come from the finalized ordinary assembly/core.
            execution.validate(
                self,
                output.residual_target,
                output.assembly_report.packet_count(),
                output.assembly_report.target_count(),
                None,
            )?;
            (output.fields, None, execution)
        };
        let mut values = Vec::new();
        for field in fields.into_values() {
            values.extend(coordinates::values::<S, Real>(
                &field.value_type,
                &field.coefficients.into_values().collect::<Vec<_>>(),
            )?);
        }
        let previous = SpatialHistory {
            grid,
            values: state
                .spatial_values()
                .expect("validated spatial State")
                .into(),
        };
        let current = self.spatial_state(Some(next_time_s), values, Some(previous))?;
        let mut accepted = CommonAcceptedSpatialStep::new(state, current)?;
        accepted.residual = residual;
        accepted.execution = Some(execution);
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
    /// Coordinate payload of the current State and its immediate restart history.
    pub(crate) fn spatial_retained_values_bytes(&self) -> Option<usize> {
        match &self.kind {
            CommonStateKind::Spatial { values, previous } => size_of_val(values.as_ref())
                .checked_add(
                    previous
                        .as_ref()
                        .map_or(0, |history| size_of_val(history.values.as_ref())),
                ),
            _ => None,
        }
    }

    /// Field-major, node-major, component-minor coefficients of this spatial State.
    pub(crate) fn spatial_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Spatial { values, .. } => Some(values),
            _ => None,
        }
    }
}
