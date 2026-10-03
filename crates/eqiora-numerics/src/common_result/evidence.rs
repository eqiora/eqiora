//! Owned solver and assembly evidence without process-local static lifetimes.

use eqiora_assembly::AssemblyReport;
use eqiora_solver::{
    ConvergenceReason, ExecutionReport, ExecutionTopology, LinearOperatorOrientation, LinearSolver,
    PreconditionerPolicy, ProviderLibrary, ReductionPolicy, SolveReport,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CommonProviderEvidence {
    pub(super) id: String,
    pub(super) implementation_version: String,
    pub(super) libraries: Vec<(String, String)>,
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

    pub(super) fn id(&self) -> &str {
        &self.id
    }
    pub(super) fn implementation_version(&self) -> &str {
        &self.implementation_version
    }
    pub(super) fn libraries(&self) -> &[(String, String)] {
        &self.libraries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommonExecutionTopology {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CommonExecutionEvidence {
    pub(super) provider: CommonProviderEvidence,
    pub(super) adapter: String,
    pub(super) topology: CommonExecutionTopology,
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

    pub(super) const fn provider(&self) -> &CommonProviderEvidence {
        &self.provider
    }
    pub(super) fn adapter(&self) -> &str {
        &self.adapter
    }
    pub(super) const fn topology(&self) -> CommonExecutionTopology {
        self.topology
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CommonSolveEvidence {
    pub(super) solver: CommonProviderEvidence,
    pub(super) execution: CommonExecutionEvidence,
    pub(super) verification: CommonExecutionEvidence,
    pub(super) orientation: LinearOperatorOrientation,
    pub(super) algorithm: LinearSolver,
    pub(super) preconditioner: PreconditionerPolicy,
    pub(super) reduction: ReductionPolicy,
    pub(super) relative_tolerance: f64,
    pub(super) absolute_tolerance: f64,
    pub(super) maximum_iterations: usize,
    pub(super) reason: ConvergenceReason,
    pub(super) completed_iterations: usize,
    pub(super) initial_residual_norm: f64,
    pub(super) reported_residual_norm: f64,
    pub(super) true_residual_norm: f64,
    pub(super) residual_target: f64,
}

impl CommonSolveEvidence {
    pub(super) fn from_report(report: &SolveReport) -> Self {
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

    pub(super) const fn solver(&self) -> &CommonProviderEvidence {
        &self.solver
    }
    pub(super) const fn execution(&self) -> &CommonExecutionEvidence {
        &self.execution
    }
    pub(super) const fn verification(&self) -> &CommonExecutionEvidence {
        &self.verification
    }
    pub(super) const fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }
    pub(super) const fn algorithm(&self) -> LinearSolver {
        self.algorithm
    }
    pub(super) const fn preconditioner(&self) -> PreconditionerPolicy {
        self.preconditioner
    }
    pub(super) const fn reduction(&self) -> ReductionPolicy {
        self.reduction
    }
    pub(super) const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    pub(super) const fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }
    pub(super) const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }
    pub(super) const fn reason(&self) -> ConvergenceReason {
        self.reason
    }
    pub(super) const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
    pub(super) const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }
    pub(super) const fn reported_residual_norm(&self) -> f64 {
        self.reported_residual_norm
    }
    pub(super) const fn true_residual_norm(&self) -> f64 {
        self.true_residual_norm
    }
    pub(super) const fn residual_target(&self) -> f64 {
        self.residual_target
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CommonAssemblyEvidence {
    pub(super) adapter: String,
    pub(super) topology: CommonExecutionTopology,
    pub(super) packet_count: usize,
    pub(super) target_count: usize,
}

impl CommonAssemblyEvidence {
    pub(super) fn from_report(report: &AssemblyReport) -> Self {
        let execution = report.execution();
        Self {
            adapter: execution.adapter().as_str().to_owned(),
            topology: topology(execution),
            packet_count: report.packet_count(),
            target_count: report.target_count(),
        }
    }

    pub(super) fn adapter(&self) -> &str {
        &self.adapter
    }
    pub(super) const fn topology(&self) -> CommonExecutionTopology {
        self.topology
    }
    pub(super) const fn packet_count(&self) -> usize {
        self.packet_count
    }
    pub(super) const fn target_count(&self) -> usize {
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
pub(super) enum AlgebraicSolveEvidence {
    Linear(Box<CommonSolveEvidence>),
    Newton {
        initial_residual_norm: f64,
        iterations: usize,
        linear_solves: Vec<CommonSolveEvidence>,
    },
}
