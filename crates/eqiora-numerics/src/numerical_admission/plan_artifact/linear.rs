//! One canonical solver-intent projection shared by Plan bytes and lineage.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireSolverObjective {
    Robust,
    Fast,
    LowMemory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireLinearControls {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: usize,
    intent: WireLinearIntent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireLinearIntent {
    ProgramControlled {
        objective: WireSolverObjective,
    },
    Exact {
        algorithm: String,
        preconditioner: String,
        reduction: String,
        provider: WireLinearProvider,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLinearProvider {
    id: String,
    implementation_version: String,
    libraries: Vec<(String, String)>,
}

impl From<SolverProvider> for WireLinearProvider {
    fn from(provider: SolverProvider) -> Self {
        Self {
            id: provider.id().as_str().into(),
            implementation_version: provider.implementation_version().into(),
            libraries: provider
                .libraries()
                .iter()
                .map(|library| (library.name().into(), library.version().into()))
                .collect(),
        }
    }
}

impl WireLinearControls {
    pub(super) fn to_native(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonLinearRequest, Diagnostic> {
        let maximum = nonzero(self.maximum_iterations, "linear maximum_iterations")?;
        match &self.intent {
            WireLinearIntent::ProgramControlled { objective } => {
                CommonLinearRequest::program_controlled(
                    self.relative_tolerance,
                    self.absolute_tolerance,
                    maximum,
                    (*objective).into(),
                )
            }
            WireLinearIntent::Exact {
                algorithm,
                preconditioner,
                reduction,
                provider,
            } => {
                let algorithm = match algorithm.as_str() {
                    "conjugate-gradient" => LinearSolver::ConjugateGradient,
                    "minimum-residual" => LinearSolver::MinimumResidual,
                    "bicgstab" => LinearSolver::BiConjugateGradientStabilized,
                    "sparse-lu" => LinearSolver::SparseLu,
                    _ => return Err(invalid("unknown exact linear algorithm")),
                };
                let preconditioner = match preconditioner.as_str() {
                    "identity" => PreconditionerPolicy::Identity,
                    "jacobi" => PreconditionerPolicy::Jacobi,
                    _ => return Err(invalid("unknown exact linear preconditioner")),
                };
                let reduction = match reduction.as_str() {
                    "reproducible" => ReductionPolicy::Reproducible,
                    "fast" => ReductionPolicy::Fast,
                    _ => return Err(invalid("unknown exact linear reduction")),
                };
                let reference = REFERENCE_LINEAR_SOLVER.provider();
                let provider = if *provider == WireLinearProvider::from(reference) {
                    reference
                } else if *provider == WireLinearProvider::from(backend.provider()) {
                    backend.provider()
                } else {
                    return Err(invalid(
                        "persisted exact solver provider release or library inventory is unavailable",
                    ));
                };
                let plan = SolverPlan::new(
                    algorithm,
                    self.relative_tolerance,
                    self.absolute_tolerance,
                    maximum,
                )?
                .with_preconditioner(preconditioner)
                .with_reduction(reduction);
                CommonLinearRequest::exact(plan, provider)
            }
        }
    }
}

impl From<SolverPlanningObjective> for WireSolverObjective {
    fn from(value: SolverPlanningObjective) -> Self {
        match value {
            SolverPlanningObjective::Robust => Self::Robust,
            SolverPlanningObjective::Fast => Self::Fast,
            SolverPlanningObjective::LowMemory => Self::LowMemory,
        }
    }
}

impl From<WireSolverObjective> for SolverPlanningObjective {
    fn from(value: WireSolverObjective) -> Self {
        match value {
            WireSolverObjective::Robust => Self::Robust,
            WireSolverObjective::Fast => Self::Fast,
            WireSolverObjective::LowMemory => Self::LowMemory,
        }
    }
}

pub(in crate::numerical_admission) fn linear_intent_bytes(
    request: CommonLinearRequest,
) -> Result<Vec<u8>, Diagnostic> {
    serde_json::to_vec(&WireLinearControls::from(request))
        .map_err(|error| invalid(format!("cannot encode exact linear intent: {error}")))
}

impl From<CommonLinearRequest> for WireLinearControls {
    fn from(request: CommonLinearRequest) -> Self {
        let intent = match request.exact_request() {
            Some((solver, provider)) => WireLinearIntent::Exact {
                algorithm: match solver.algorithm() {
                    LinearSolver::ConjugateGradient => "conjugate-gradient",
                    LinearSolver::MinimumResidual => "minimum-residual",
                    LinearSolver::BiConjugateGradientStabilized => "bicgstab",
                    LinearSolver::SparseLu => "sparse-lu",
                }
                .into(),
                preconditioner: match solver.preconditioner() {
                    PreconditionerPolicy::Identity => "identity",
                    PreconditionerPolicy::Jacobi => "jacobi",
                }
                .into(),
                reduction: match solver.reduction() {
                    ReductionPolicy::Reproducible => "reproducible",
                    ReductionPolicy::Fast => "fast",
                }
                .into(),
                provider: provider.into(),
            },
            None => WireLinearIntent::ProgramControlled {
                objective: request
                    .objective()
                    .expect("non-exact intent owns an objective")
                    .into(),
            },
        };
        Self {
            relative_tolerance: request.relative_tolerance(),
            absolute_tolerance: request.absolute_tolerance(),
            maximum_iterations: request.maximum_iterations().get(),
            intent,
        }
    }
}
