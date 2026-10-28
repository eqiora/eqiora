//! Newton execution and acceptance share the spatial map without inventing Time.
use super::*;
use crate::common_result::evidence::{CommonAssemblyEvidence, CommonSolveHistory};
use crate::form_compiler::region::coordinates::{self, Real};
use crate::region_assembly::mapping::{
    RecoveredRegionField, RegionAlgebraicCoordinates, RegionDofMap, RegionSolveInput,
};
use crate::spatial_expression::Coefficient;

impl CommonSpatialPlan {
    pub(super) fn solve_region_newton<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        initial: &BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>,
        request: LinearSolveRequest<'_>,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver>,
    ) -> Result<crate::region_assembly::mapping::RegionNonlinearOutput<S>, Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let nonlinear = self
            .admission
            .nonlinear
            .ok_or_else(|| invalid("Newton execution requires a Newton Plan"))?;
        let workers = self.admission.linear.workers;
        let current_mesh = input
            .geometry_action
            .as_ref()
            .map(|action| action.current_mesh().clone());
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => mapping.solve_nonlinear(
                mesh.mesh(),
                input,
                initial,
                nonlinear,
                workers,
                request,
                prepared,
            ),
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping
                .solve_nonlinear(
                    current_mesh.as_ref().unwrap_or_else(|| mesh.mesh()),
                    input,
                    initial,
                    nonlinear,
                    workers,
                    request,
                    prepared,
                ),
            _ => Err(invalid("Region Newton requires its exact affine Mesh")),
        }
    }

    pub(super) fn run_newton_result(
        &self,
        initial: Option<&CommonState>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let initial = initial
            .ok_or_else(|| invalid("stationary Newton requires an explicit initial State"))?;
        self.static_initial_fields(initial)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                self.run_typed_newton(equations, initial, backend)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                self.run_typed_newton(equations, initial, backend)
            }
            _ => Err(invalid("Newton requires admitted spatial equations")),
        }
    }

    fn run_typed_newton<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        initial: &CommonState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.stationary_assembly(equations)?;
        let seed = self.physical_fields(
            &mapping,
            initial
                .spatial_values()
                .ok_or_else(|| invalid("Newton seed requires physical spatial Fields"))?,
        )?;
        let mut prepared = checked.prepare_linear(self.admission.linear.solver)?;
        if let Some(linear) = &mut prepared {
            linear.begin_action()?;
        }
        let result = (|| {
            let output = self.solve_region_newton(
                &mapping,
                &input,
                &seed,
                LinearSolveRequest::new(&checked, self.admission.linear.solver),
                prepared
                    .as_mut()
                    .map(|linear| linear.as_mut() as &mut dyn eqiora_solver::PreparedLinearSolver),
            )?;
            let execution = newton_execution(&output.solve);
            let fields = output
                .fields
                .into_iter()
                .map(|(id, field)| {
                    let values = coordinates::values::<S, Real>(
                        &field.value_type,
                        &field.coefficients.into_values().collect::<Vec<_>>(),
                    )?;
                    Ok((
                        id.downcast().expect("compiled Field identity"),
                        field.value_type,
                        values,
                        field.space,
                    ))
                })
                .collect::<Result<_, Diagnostic>>()?;
            crate::CommonResult::accept_spatial(
                self.clone(),
                Some(initial.clone()),
                0.,
                fields,
                execution,
                None,
            )
        })();
        match result {
            Ok(accepted) => {
                if let Some(linear) = &mut prepared {
                    if let Err(error) = linear.commit_action() {
                        let _ = linear.discard_action();
                        return Err(error);
                    }
                }
                Ok(accepted)
            }
            Err(error) => {
                if let Some(linear) = &mut prepared {
                    let _ = linear.discard_action();
                }
                Err(error)
            }
        }
    }

    pub(crate) fn reaccept_static_execution(
        &self,
        values: &BTreeMap<eqiora_core::RawId, Vec<f64>>,
        nullspace: Option<&crate::nullspace::NullspaceEvidence>,
        initial: Option<&CommonState>,
        execution: &CommonSpatialExecution,
    ) -> Result<Option<(f64, f64)>, Diagnostic> {
        crate::common_result::artifact::require_reference_assembly(&execution.assembly)?;
        match (&execution.solve, self.admission.nonlinear) {
            (CommonSolveHistory::Linear(solve), None) => {
                solve.validate_plan(&ResolvedCommonPlan::Spatial(Box::new(self.clone())))?;
                self.reaccept_static_coefficients(
                    values,
                    nullspace,
                    initial,
                    solve.initial_residual_norm(),
                )?;
                Ok(None)
            }
            (CommonSolveHistory::Newton { iterations, .. }, Some(nonlinear)) => {
                if nullspace.is_some() {
                    return Err(invalid(
                        "Newton spatial Result cannot carry a linear nullspace receipt",
                    ));
                }
                let initial = initial
                    .ok_or_else(|| invalid("Newton Result omits its exact initial State"))?;
                self.static_initial_fields(initial)?;
                match self.admission.recognized_model() {
                    RecognizedNativeModel::Spatial(equations) => self.reaccept_typed_newton(
                        equations,
                        values,
                        initial,
                        nonlinear,
                        *iterations,
                        execution,
                    ),
                    RecognizedNativeModel::ComplexSpatial(equations) => self.reaccept_typed_newton(
                        equations,
                        values,
                        initial,
                        nonlinear,
                        *iterations,
                        execution,
                    ),
                    _ => Err(invalid("Newton replay requires admitted spatial equations")),
                }
            }
            _ => Err(invalid(
                "static Result solve history differs from its Plan strategy",
            )),
        }
    }
    fn reaccept_typed_newton<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        values: &BTreeMap<eqiora_core::RawId, Vec<f64>>,
        initial: &CommonState,
        nonlinear: NonlinearSolvePlan,
        iterations: usize,
        execution: &CommonSpatialExecution,
    ) -> Result<Option<(f64, f64)>, Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let (mapping, input) = self.stationary_assembly(equations)?;
        let values = values
            .iter()
            .map(|(id, words)| {
                let (_, layout) = mapping
                    .field_layout(*id)
                    .ok_or_else(|| invalid("Newton Result contains an unowned Field"))?;
                Ok((*id, Real::decode::<S>(&layout.value_type, words)?))
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        let fields = mapping.physical_fields_from_coefficients(&values)?;
        let seed = self.physical_fields(
            &mapping,
            initial
                .spatial_values()
                .ok_or_else(|| invalid("Newton seed requires physical spatial Fields"))?,
        )?;
        let validate = |mesh: &eqiora_meshing::SimplicialMesh| {
            mapping.validate_nonlinear_coefficients(
                mesh, &input, &fields, &seed, nonlinear, iterations,
            )
        };
        let (norm, target, packets, targets, derived_initial) = match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => mapping
                .validate_nonlinear_coefficients(
                    mesh.mesh(),
                    &input,
                    &fields,
                    &seed,
                    nonlinear,
                    iterations,
                )?,
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => validate(mesh.mesh())?,
            _ => return Err(invalid("Newton replay requires its exact affine Mesh")),
        };
        execution.validate(self, target, packets, targets, derived_initial)?;
        Ok(Some((norm, target)))
    }
}

pub(super) fn newton_execution(
    solve: &crate::region_assembly::RegionNewtonSolution,
) -> CommonSpatialExecution {
    CommonSpatialExecution {
        solve: CommonSolveHistory::from_newton(&solve.evidence),
        assembly: CommonAssemblyEvidence::from_report(&solve.assembly.report),
    }
}
