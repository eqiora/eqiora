//! Owned solver and assembly evidence without process-local static lifetimes.

use super::invalid;
use eqiora_assembly::AssemblyReport;
use eqiora_solver::{
    ConvergenceReason, ExecutionReport, ExecutionTopology, LinearOperatorOrientation, LinearSolver,
    PreconditionerPolicy, ProviderLibrary, ReductionPolicy, SolveReport,
};

/// Owned provider identity and declared library inventory, without static lifetimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonProviderEvidence {
    pub(crate) id: String,
    pub(crate) implementation_version: String,
    pub(crate) libraries: Vec<(String, String)>,
}

impl CommonProviderEvidence {
    fn from_solver(report: &SolveReport) -> Self {
        let provider = report.solver_provider();
        Self::from_parts(
            provider.id().as_str(),
            provider.implementation_version(),
            provider.libraries(),
        )
    }

    fn from_execution_provider(provider: eqiora_solver::ExecutionProvider) -> Self {
        Self::from_parts(
            provider.id().as_str(),
            provider.implementation_version(),
            provider.libraries(),
        )
    }

    fn from_parts(id: &str, version: &str, libraries: &[ProviderLibrary]) -> Self {
        Self {
            id: id.to_owned(),
            implementation_version: version.to_owned(),
            libraries: libraries
                .iter()
                .map(|library| (library.name().to_owned(), library.version().to_owned()))
                .collect(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn implementation_version(&self) -> &str {
        &self.implementation_version
    }
    pub fn libraries(&self) -> &[(String, String)] {
        &self.libraries
    }
}

/// Recorded execution placement; no worker pool or device handle is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonExecutionTopology {
    Host {
        workers: usize,
    },
    Distributed {
        ranks: usize,
        workers_per_partition: usize,
    },
    Cuda {
        device: u16,
    },
}

/// Owned provider and placement metadata for one execution phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonExecutionEvidence {
    pub(crate) provider: CommonProviderEvidence,
    pub(crate) adapter: String,
    pub(crate) topology: CommonExecutionTopology,
}

impl CommonExecutionEvidence {
    fn from_report(provider: eqiora_solver::ExecutionProvider, report: ExecutionReport) -> Self {
        let topology = topology(report);
        Self {
            provider: CommonProviderEvidence::from_execution_provider(provider),
            adapter: report.adapter().as_str().to_owned(),
            topology,
        }
    }

    pub const fn provider(&self) -> &CommonProviderEvidence {
        &self.provider
    }
    pub fn adapter(&self) -> &str {
        &self.adapter
    }
    pub const fn topology(&self) -> CommonExecutionTopology {
        self.topology
    }
}

/// Owned provenance of an accepted linear solve.
///
/// Decoding checks report consistency; the owning Result separately binds the
/// exact Plan, candidate values and original-equation acceptance. This record
/// alone neither authenticates a producer nor establishes numerical acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonSolveEvidence {
    pub(crate) solver: CommonProviderEvidence,
    pub(crate) execution: CommonExecutionEvidence,
    pub(crate) verification: CommonExecutionEvidence,
    pub(crate) orientation: LinearOperatorOrientation,
    pub(crate) algorithm: LinearSolver,
    pub(crate) preconditioner: PreconditionerPolicy,
    pub(crate) reduction: ReductionPolicy,
    pub(crate) relative_tolerance: f64,
    pub(crate) absolute_tolerance: f64,
    pub(crate) maximum_iterations: usize,
    pub(crate) reason: ConvergenceReason,
    pub(crate) completed_iterations: usize,
    pub(crate) initial_residual_norm: f64,
    pub(crate) reported_residual_norm: f64,
    pub(crate) true_residual_norm: f64,
    pub(crate) residual_target: f64,
}

impl CommonSolveEvidence {
    /// Check recorded policies and provider inventories against the exact Plan.
    /// Candidate values and original-equation acceptance are checked separately.
    pub fn validate_plan(
        &self,
        plan: &crate::ResolvedCommonPlan,
    ) -> Result<(), eqiora_core::Diagnostic> {
        let expected = plan
            .effective_solver()
            .ok_or_else(|| invalid("Result solve evidence requires a linear Plan"))?;
        let solver_provider = plan
            .linear_solver_provider()
            .ok_or_else(|| invalid("Result solve evidence requires a solver provider"))?;
        let (execution_provider, workers) = plan
            .linear_execution_provider()
            .ok_or_else(|| invalid("Result solve evidence requires an execution provider"))?;
        super::artifact::require_solver_contract(
            expected,
            solver_provider,
            execution_provider,
            workers,
            self,
        )
    }

    /// Deterministic owned metadata charge, excluding allocator capacity and overhead.
    /// This has no dependence on solution values or completed iteration counts.
    pub fn storage_bytes(&self) -> Option<usize> {
        let mut total = size_of::<Self>();
        for provider in [
            &self.solver,
            &self.execution.provider,
            &self.verification.provider,
        ] {
            total = total.checked_add(provider.id.len())?;
            total = total.checked_add(provider.implementation_version.len())?;
            total = total.checked_add(size_of_val(provider.libraries.as_slice()))?;
            for (name, version) in &provider.libraries {
                total = total.checked_add(name.len())?.checked_add(version.len())?;
            }
        }
        total
            .checked_add(self.execution.adapter.len())?
            .checked_add(self.verification.adapter.len())
    }

    pub fn from_report(report: &SolveReport) -> Self {
        let plan = report.solver_plan();
        Self {
            solver: CommonProviderEvidence::from_solver(report),
            execution: CommonExecutionEvidence::from_report(
                report.execution_provider(),
                report.execution(),
            ),
            verification: CommonExecutionEvidence::from_report(
                report.verification_provider(),
                report.verification(),
            ),
            orientation: report.orientation(),
            algorithm: report.algorithm(),
            preconditioner: report.preconditioner(),
            reduction: report.reduction(),
            relative_tolerance: plan.relative_tolerance(),
            absolute_tolerance: plan.absolute_tolerance(),
            maximum_iterations: plan.maximum_iterations().get(),
            reason: report.reason(),
            completed_iterations: report.completed_iterations(),
            initial_residual_norm: report.initial_residual_norm(),
            reported_residual_norm: report.reported_residual_norm(),
            true_residual_norm: report.true_residual_norm(),
            residual_target: report.residual_target(),
        }
    }

    pub const fn solver(&self) -> &CommonProviderEvidence {
        &self.solver
    }
    pub const fn execution(&self) -> &CommonExecutionEvidence {
        &self.execution
    }
    pub const fn verification(&self) -> &CommonExecutionEvidence {
        &self.verification
    }
    pub const fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }
    pub const fn algorithm(&self) -> LinearSolver {
        self.algorithm
    }
    pub const fn preconditioner(&self) -> PreconditionerPolicy {
        self.preconditioner
    }
    pub const fn reduction(&self) -> ReductionPolicy {
        self.reduction
    }
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    pub const fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }
    pub const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }
    pub const fn reason(&self) -> ConvergenceReason {
        self.reason
    }
    pub const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
    pub const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }
    pub const fn reported_residual_norm(&self) -> f64 {
        self.reported_residual_norm
    }
    pub const fn true_residual_norm(&self) -> f64 {
        self.true_residual_norm
    }
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }
}

impl serde::Serialize for CommonSolveEvidence {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        super::artifact::WireSolve::from_solve(self)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for CommonSolveEvidence {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = super::artifact::WireSolve::deserialize(deserializer)?;
        wire.replay().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommonAssemblyEvidence {
    pub(crate) adapter: String,
    pub(crate) topology: CommonExecutionTopology,
    pub(crate) packet_count: usize,
    pub(crate) target_count: usize,
}

impl CommonAssemblyEvidence {
    pub(crate) fn from_report(report: &AssemblyReport) -> Self {
        let execution = report.execution();
        Self {
            adapter: execution.adapter().as_str().to_owned(),
            topology: topology(execution),
            packet_count: report.packet_count(),
            target_count: report.target_count(),
        }
    }

    pub(crate) fn adapter(&self) -> &str {
        &self.adapter
    }
    pub(crate) const fn topology(&self) -> CommonExecutionTopology {
        self.topology
    }
    pub(crate) const fn packet_count(&self) -> usize {
        self.packet_count
    }
    pub(crate) const fn target_count(&self) -> usize {
        self.target_count
    }
}

fn topology(report: ExecutionReport) -> CommonExecutionTopology {
    match report.topology() {
        ExecutionTopology::Host { workers } => CommonExecutionTopology::Host {
            workers: workers.get(),
        },
        ExecutionTopology::Distributed {
            ranks,
            workers_per_partition,
        } => CommonExecutionTopology::Distributed {
            ranks: ranks.get(),
            workers_per_partition: workers_per_partition.get(),
        },
        ExecutionTopology::Cuda { device } => CommonExecutionTopology::Cuda { device },
    }
}

/// Distinct acceptance records for an affine solve and a nonlinear iteration.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CommonSolveHistory {
    Linear(Box<CommonSolveEvidence>),
    Newton {
        initial_residual_norm: f64,
        iterations: usize,
        linear_solves: Vec<CommonSolveEvidence>,
    },
}

impl CommonSolveHistory {
    pub(crate) fn from_newton(value: &crate::nonlinear_execution::NewtonEvidence) -> Self {
        Self::Newton {
            initial_residual_norm: value.initial_residual_norm,
            iterations: value.iterations,
            linear_solves: value
                .linear_solves
                .iter()
                .map(CommonSolveEvidence::from_report)
                .collect(),
        }
    }
}
