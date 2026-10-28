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
    cost: Option<WireCostRequest>,
    execution: WireLinearProvider,
    workers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCostRequest {
    solves: usize,
    structures: usize,
    numeric_preparations: usize,
    identity_iterations: Option<usize>,
    jacobi_iterations: Option<usize>,
    numeric_payload_limit: Option<usize>,
    declared_horizon: bool,
}

impl From<eqiora_solver::SolverCostRequest> for WireCostRequest {
    fn from(value: eqiora_solver::SolverCostRequest) -> Self {
        Self {
            solves: value.solves().get(),
            structures: value.structures().get(),
            numeric_preparations: value.numeric_preparations().get(),
            identity_iterations: value.identity_iterations().map(NonZeroUsize::get),
            jacobi_iterations: value.jacobi_iterations().map(NonZeroUsize::get),
            numeric_payload_limit: value.numeric_payload_limit().map(NonZeroUsize::get),
            declared_horizon: value.declared_horizon(),
        }
    }
}

impl WireCostRequest {
    fn to_native(&self) -> Result<eqiora_solver::SolverCostRequest, Diagnostic> {
        let mut request = if self.declared_horizon {
            eqiora_solver::SolverCostRequest::new(
                nonzero(self.solves, "cost solves")?,
                nonzero(self.structures, "cost structures")?,
                nonzero(self.numeric_preparations, "cost numeric preparations")?,
            )?
        } else {
            if (self.solves, self.structures, self.numeric_preparations) != (1, 1, 1) {
                return Err(invalid(
                    "undeclared cost horizon must be the one-shot assumption",
                ));
            }
            eqiora_solver::SolverCostRequest::default()
        };
        request = request.with_iteration_forecasts(
            self.identity_iterations
                .map(|value| nonzero(value, "Identity iteration forecast"))
                .transpose()?,
            self.jacobi_iterations
                .map(|value| nonzero(value, "Jacobi iteration forecast"))
                .transpose()?,
        );
        if let Some(limit) = self.numeric_payload_limit {
            request = request.with_numeric_payload_limit(nonzero(limit, "numeric payload limit")?);
        }
        Ok(request)
    }
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

impl From<ExecutionProvider> for WireLinearProvider {
    fn from(provider: ExecutionProvider) -> Self {
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
        let request = match &self.intent {
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
        }?;
        let execution = if self.execution == WireLinearProvider::from(SERIAL_EXECUTION_PROVIDER) {
            SERIAL_EXECUTION_PROVIDER
        } else if self.execution == WireLinearProvider::from(backend.default_execution_provider()) {
            backend.default_execution_provider()
        } else {
            return Err(invalid(
                "persisted execution provider release or library inventory is unavailable",
            ));
        };
        let request =
            request.with_execution(execution, nonzero(self.workers, "execution workers")?);
        match &self.cost {
            Some(cost) => Ok(request.with_cost(cost.to_native()?)),
            None => Ok(request),
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
            cost: request.cost().map(Into::into),
            execution: request.execution().0.into(),
            workers: request.execution().1.get(),
        }
    }
}

pub(crate) fn solve_intent_bytes(request: CommonSolvePolicy) -> Result<Vec<u8>, Diagnostic> {
    serde_json::to_vec(&wire_solve(request))
        .map_err(|error| invalid(format!("cannot encode solve controls: {error}")))
}

pub(super) fn wire_solve(request: CommonSolvePolicy) -> WireSolve {
    match request {
        CommonSolvePolicy::Linear(request) => WireSolve::Linear {
            linear: request.into(),
        },
        CommonSolvePolicy::Newton { linear, nonlinear } => WireSolve::Newton {
            linear: linear.into(),
            nonlinear: WireNonlinearControls {
                relative_tolerance: nonlinear.relative_tolerance(),
                absolute_tolerance: nonlinear.absolute_tolerance(),
                maximum_iterations: nonlinear.maximum_iterations().get(),
                maximum_line_search_steps: nonlinear.maximum_line_search_steps(),
            },
        },
    }
}
