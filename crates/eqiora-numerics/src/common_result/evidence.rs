//! Owned solver and assembly evidence without process-local static lifetimes.

use eqiora_assembly::AssemblyReport;
use eqiora_solver::{
    ConvergenceReason, ExecutionReport, ExecutionTopology, LinearOperatorOrientation, LinearSolver,
    PreconditionerPolicy, ProviderLibrary, ReductionPolicy, SolveReport,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommonProviderEvidence {
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

    pub(crate) fn id(&self) -> &str {
        &self.id
    }
    pub(crate) fn implementation_version(&self) -> &str {
        &self.implementation_version
    }
    pub(crate) fn libraries(&self) -> &[(String, String)] {
        &self.libraries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommonExecutionTopology {
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
pub(crate) struct CommonExecutionEvidence {
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

    pub(crate) const fn provider(&self) -> &CommonProviderEvidence {
        &self.provider
    }
    pub(crate) fn adapter(&self) -> &str {
        &self.adapter
    }
    pub(crate) const fn topology(&self) -> CommonExecutionTopology {
        self.topology
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonSolveEvidence {
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
    pub(crate) fn from_report(report: &SolveReport) -> Self {
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

    pub(crate) const fn solver(&self) -> &CommonProviderEvidence {
        &self.solver
    }
    pub(crate) const fn execution(&self) -> &CommonExecutionEvidence {
        &self.execution
    }
    pub(crate) const fn verification(&self) -> &CommonExecutionEvidence {
        &self.verification
    }
    pub(crate) const fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }
    pub(crate) const fn algorithm(&self) -> LinearSolver {
        self.algorithm
    }
    pub(crate) const fn preconditioner(&self) -> PreconditionerPolicy {
        self.preconditioner
    }
    pub(crate) const fn reduction(&self) -> ReductionPolicy {
        self.reduction
    }
    pub(crate) const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    pub(crate) const fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }
    pub(crate) const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }
    pub(crate) const fn reason(&self) -> ConvergenceReason {
        self.reason
    }
    pub(crate) const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
    pub(crate) const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }
    pub(crate) const fn reported_residual_norm(&self) -> f64 {
        self.reported_residual_norm
    }
    pub(crate) const fn true_residual_norm(&self) -> f64 {
        self.true_residual_norm
    }
    pub(crate) const fn residual_target(&self) -> f64 {
        self.residual_target
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
