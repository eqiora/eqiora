//! Newton execution and acceptance share the spatial map without inventing Time.
use super::*;
use crate::common_result::evidence::{
    CommonAssemblyEvidence, CommonSolveEvidence, CommonSolveHistory,
};
use crate::region_assembly::mapping::{RecoveredRegionField, RegionDofMap, RegionSolveInput};

impl CommonSpatialPlan {
    pub(super) fn solve_region_newton(
        &self,
        mapping: &RegionDofMap<f64>,
        input: &RegionSolveInput<f64>,
        initial: &BTreeMap<eqiora_core::RawId, RecoveredRegionField<f64>>,
        request: LinearSolveRequest<'_>,
    ) -> Result<crate::region_assembly::mapping::RegionNonlinearOutput, Diagnostic> {
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
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.solve_nonlinear(mesh.mesh(), input, initial, nonlinear, workers, request)
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping
                .solve_nonlinear(
                    current_mesh.as_ref().unwrap_or_else(|| mesh.mesh()),
                    input,
                    initial,
                    nonlinear,
                    workers,
                    request,
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
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("Newton requires admitted real spatial equations"));
        };
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.state_assembly(None)?;
        let seed = self.physical_fields(
            &mapping,
            initial
                .spatial_values()
                .ok_or_else(|| invalid("Newton seed requires physical spatial Fields"))?,
        )?;
        let output = self.solve_region_newton(
            &mapping,
            &input,
            &seed,
            LinearSolveRequest::new(&checked, self.admission.linear.solver),
        )?;
        let execution = newton_execution(&output.solve);
        let fields = output
            .fields
            .into_iter()
            .map(|(id, field)| {
                (
                    id.downcast().expect("compiled Field identity"),
                    field.value_type,
                    field.coefficients.into_values().collect(),
                    field.space,
                )
            })
            .collect();
        crate::CommonResult::accept_spatial(
            self.clone(),
            Some(initial.clone()),
            0.,
            fields,
            execution,
            None,
        )
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
                let (mapping, input) = self.state_assembly(None)?;
                let fields = mapping.physical_fields_from_coefficients(values)?;
                let seed = self.physical_fields(
                    &mapping,
                    initial
                        .spatial_values()
                        .ok_or_else(|| invalid("Newton seed requires physical spatial Fields"))?,
                )?;
                let validate = |mesh: &eqiora_meshing::SimplicialMesh| {
                    mapping.validate_nonlinear_coefficients(
                        mesh,
                        &input,
                        &fields,
                        &seed,
                        nonlinear,
                        *iterations,
                    )
                };
                let (norm, target, packets, targets, derived_initial) =
                    match self.admission.resources() {
                        NativeMeshResources::Cartesian { mesh, .. } => mapping
                            .validate_nonlinear_coefficients(
                                mesh.mesh(),
                                &input,
                                &fields,
                                &seed,
                                nonlinear,
                                *iterations,
                            )?,
                        NativeMeshResources::GmshSimplicial { mesh, .. }
                        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                            validate(mesh.mesh())?
                        }
                        _ => return Err(invalid("Newton replay requires its exact affine Mesh")),
                    };
                execution.validate(self, target, packets, targets, derived_initial)?;
                Ok(Some((norm, target)))
            }
            _ => Err(invalid(
                "static Result solve history differs from its Plan strategy",
            )),
        }
    }
}

pub(super) fn newton_execution(
    solve: &crate::region_assembly::RegionNewtonSolution,
) -> CommonSpatialExecution {
    CommonSpatialExecution {
        solve: CommonSolveHistory::Newton {
            initial_residual_norm: solve.evidence.initial_residual_norm,
            iterations: solve.evidence.iterations,
            linear_solves: solve
                .evidence
                .linear_solves
                .iter()
                .map(CommonSolveEvidence::from_report)
                .collect(),
        },
        assembly: CommonAssemblyEvidence::from_report(&solve.assembly.report),
    }
}
