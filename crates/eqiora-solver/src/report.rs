use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    BackendId, ExecutionProvider, FixedOrderInnerProduct, LinearOperatorOrientation, LinearProblem,
    LinearSolver, PreconditionerPolicy, ReductionPolicy, ReplicatedLinearExecution,
    SERIAL_LINEAR_EXECUTION, SolverPlan, SolverProvider,
};

/// Successful termination condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// The supplied initial guess already met the declared tolerance.
    InitialResidualSatisfied,
    /// Solver work produced an accepted independently verified residual.
    ResidualToleranceSatisfied,
}

/// Stable Eqiora-owned identity for an operator execution adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionId(&'static str);

impl ExecutionId {
    /// Construct a namespaced compile-time execution identity.
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    /// Namespaced execution identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// Parallel topology used by one accepted execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTopology {
    /// One host process with a bounded worker pool.
    Host { workers: NonZeroUsize },
    /// One distributed execution group with fixed rank and per-rank worker
    /// counts.
    Distributed {
        /// Participating ranks/partitions.
        ranks: NonZeroUsize,
        /// Workers admitted inside every partition.
        workers_per_partition: NonZeroUsize,
    },
    /// One CUDA device selected by its runtime-visible ordinal.
    Cuda { device: u16 },
}

/// Backend-neutral placement evidence for one accepted execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionReport {
    adapter: ExecutionId,
    topology: ExecutionTopology,
}

impl ExecutionReport {
    /// The direct one-worker host path used in the absence of an adapter.
    #[must_use]
    pub const fn host_serial() -> Self {
        Self {
            adapter: ExecutionId::new("eqiora.host.serial"),
            topology: ExecutionTopology::Host {
                workers: NonZeroUsize::MIN,
            },
        }
    }

    /// Record a host execution adapter and its bounded worker count.
    #[must_use]
    pub const fn host(adapter: ExecutionId, workers: NonZeroUsize) -> Self {
        Self {
            adapter,
            topology: ExecutionTopology::Host { workers },
        }
    }

    /// Record a distributed adapter and execution-group rank count.
    #[must_use]
    pub const fn distributed(adapter: ExecutionId, ranks: NonZeroUsize) -> Self {
        Self {
            adapter,
            topology: ExecutionTopology::Distributed {
                ranks,
                workers_per_partition: NonZeroUsize::MIN,
            },
        }
    }

    /// Record a CUDA execution adapter and selected runtime-visible device.
    #[must_use]
    pub const fn cuda(adapter: ExecutionId, device: u16) -> Self {
        Self {
            adapter,
            topology: ExecutionTopology::Cuda { device },
        }
    }

    /// Execution adapter identity, separate from the solver backend.
    #[must_use]
    pub const fn adapter(self) -> ExecutionId {
        self.adapter
    }

    /// Exact parallel topology admitted to this execution phase.
    #[must_use]
    pub const fn topology(self) -> ExecutionTopology {
        self.topology
    }
}

/// Auditable evidence for one accepted solve.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveReport {
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    execution: ExecutionReport,
    verification_provider: ExecutionProvider,
    verification: ExecutionReport,
    orientation: LinearOperatorOrientation,
    plan: SolverPlan,
    reason: ConvergenceReason,
    completed_iterations: usize,
    initial_residual_norm: f64,
    reported_residual_norm: f64,
    true_residual_norm: f64,
    residual_target: f64,
}

impl SolveReport {
    /// Construct evidence after independent true-residual acceptance.
    ///
    /// # Errors
    /// Returns `EQ0802` for non-finite/negative values, inconsistent
    /// termination evidence, an exceeded iteration limit, or an unaccepted
    /// true residual.
    #[allow(clippy::too_many_arguments)]
    pub fn accepted(
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        execution: ExecutionReport,
        orientation: LinearOperatorOrientation,
        plan: SolverPlan,
        reason: ConvergenceReason,
        completed_iterations: usize,
        initial_residual_norm: f64,
        reported_residual_norm: f64,
        true_residual_norm: f64,
        residual_target: f64,
    ) -> Result<Self, Diagnostic> {
        Self::accepted_with_verification(
            solver_provider,
            execution_provider,
            execution,
            execution_provider,
            execution,
            orientation,
            plan,
            reason,
            completed_iterations,
            initial_residual_norm,
            reported_residual_norm,
            true_residual_norm,
            residual_target,
        )
    }

    /// Construct evidence when production and independent verification used
    /// different execution placements.
    ///
    /// # Errors
    /// Returns `EQ0802` for non-finite/negative values, inconsistent
    /// termination evidence, an exceeded iteration limit, or an unaccepted
    /// true residual.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accepted_with_verification(
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        execution: ExecutionReport,
        verification_provider: ExecutionProvider,
        verification: ExecutionReport,
        orientation: LinearOperatorOrientation,
        plan: SolverPlan,
        reason: ConvergenceReason,
        completed_iterations: usize,
        initial_residual_norm: f64,
        reported_residual_norm: f64,
        true_residual_norm: f64,
        residual_target: f64,
    ) -> Result<Self, Diagnostic> {
        solver_provider.validate()?;
        execution_provider.validate()?;
        verification_provider.validate()?;
        if execution_provider.id() != execution.adapter() {
            return Err(invalid_provider(
                "solve execution provider ID contradicts its execution report adapter",
            ));
        }
        if verification_provider.id() != verification.adapter() {
            return Err(invalid_provider(
                "solve verification provider ID contradicts its execution report adapter",
            ));
        }
        if [
            initial_residual_norm,
            reported_residual_norm,
            true_residual_norm,
            residual_target,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(solve_failed(
                "solve report residual values must be finite and non-negative",
            ));
        }
        if completed_iterations > plan.maximum_iterations().get() {
            return Err(solve_failed(format!(
                "solve report completed {completed_iterations} iterations but the SolverPlan permits at most {}",
                plan.maximum_iterations()
            )));
        }
        match (reason, completed_iterations) {
            (ConvergenceReason::InitialResidualSatisfied, 0)
            | (ConvergenceReason::ResidualToleranceSatisfied, 1..) => {}
            (ConvergenceReason::InitialResidualSatisfied, _) => {
                return Err(solve_failed(
                    "initial-residual convergence requires zero completed iterations",
                ));
            }
            (ConvergenceReason::ResidualToleranceSatisfied, 0) => {
                return Err(solve_failed(
                    "post-work residual convergence requires at least one completed iteration",
                ));
            }
        }
        if reason == ConvergenceReason::InitialResidualSatisfied
            && initial_residual_norm > residual_target
        {
            return Err(solve_failed(
                "initial-residual convergence requires the initial residual to satisfy the target",
            ));
        }
        // `reason` and `completed_iterations` are producer evidence, while the
        // residual values may be recomputed by an independent verifier. Their
        // reductions may legitimately place the initial norm on opposite sides
        // of the tolerance threshold, so post-work producer termination does
        // not imply a verifier-side unsatisfied initial residual.
        if true_residual_norm > residual_target {
            return Err(solve_failed(format!(
                "backend reported convergence but true residual {true_residual_norm:e} exceeds target {residual_target:e}"
            )));
        }
        Ok(Self {
            solver_provider,
            execution_provider,
            execution,
            verification_provider,
            verification,
            orientation,
            plan,
            reason,
            completed_iterations,
            initial_residual_norm,
            reported_residual_norm,
            true_residual_norm,
            residual_target,
        })
    }

    /// Adapter identity.
    #[must_use]
    pub const fn backend(&self) -> BackendId {
        self.solver_provider.id()
    }

    /// Stable solver identity and declared release/dependency inventory.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Stable production-execution identity and declared release/dependency inventory.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Operator execution placement, separate from the solver algorithm.
    #[must_use]
    pub const fn execution(&self) -> ExecutionReport {
        self.execution
    }

    /// Stable verifier identity and declared release/dependency inventory.
    #[must_use]
    pub const fn verification_provider(&self) -> ExecutionProvider {
        self.verification_provider
    }

    /// Placement used for the independent true-residual acceptance check.
    #[must_use]
    pub const fn verification(&self) -> ExecutionReport {
        self.verification
    }

    /// Orientation of the independently verified linear action.
    #[must_use]
    pub const fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }

    /// Solver algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> LinearSolver {
        self.plan.algorithm()
    }

    /// Preconditioner policy.
    #[must_use]
    pub const fn preconditioner(&self) -> PreconditionerPolicy {
        self.plan.preconditioner()
    }

    /// Reduction policy.
    #[must_use]
    pub const fn reduction(&self) -> ReductionPolicy {
        self.plan.reduction()
    }

    /// Exact solver policy used to produce and accept the solution.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.plan
    }

    /// Successful termination condition.
    #[must_use]
    pub const fn reason(&self) -> ConvergenceReason {
        self.reason
    }

    /// Completed iterations.
    #[must_use]
    pub const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }

    /// Residual norm of the initial guess.
    #[must_use]
    pub const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }

    /// Recursive/backend-reported residual norm.
    #[must_use]
    pub const fn reported_residual_norm(&self) -> f64 {
        self.reported_residual_norm
    }

    /// Independently recomputed `||b - A x||_2`.
    #[must_use]
    pub const fn true_residual_norm(&self) -> f64 {
        self.true_residual_norm
    }

    /// Accepted absolute residual threshold.
    #[must_use]
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }
}

/// Accepted solution values and their evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolution {
    values: Vec<f64>,
    report: SolveReport,
}

impl LinearSolution {
    /// Pair finite solution values with an accepted report.
    ///
    /// # Errors
    /// Returns `EQ0802` when a solution value is non-finite.
    pub(crate) fn new(values: Vec<f64>, report: SolveReport) -> Result<Self, Diagnostic> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed("linear solution contains a non-finite value"));
        }
        Ok(Self { values, report })
    }

    /// Solution vector.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Auditable convergence evidence.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }

    /// Consume the accepted solution without copying its values or evidence.
    ///
    /// This ownership boundary lets a method-native result retain the exact
    /// accepted vector and the paired [`SolveReport`] that admitted it.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, SolveReport) {
        (self.values, self.report)
    }
}

/// Independently verify and accept a backend-produced solution.
///
/// The backend supplies only operational evidence. Eqiora recomputes the
/// initial and final residuals through the admitted operator, derives the
/// target from the sole [`SolverPlan`], and constructs the public report only
/// after the true residual passes.
///
/// # Errors
/// Returns `EQ0802` for shape/non-finite operator behavior, inconsistent
/// termination/iteration evidence, or when the true residual exceeds the
/// declared target.
#[allow(clippy::too_many_arguments)]
pub fn accept_linear_solution(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    solver_provider: SolverProvider,
    reason: ConvergenceReason,
    completed_iterations: usize,
    reported_residual_norm: f64,
    values: Vec<f64>,
) -> Result<LinearSolution, Diagnostic> {
    accept_linear_solution_with_execution(
        problem,
        plan,
        solver_provider,
        reason,
        completed_iterations,
        reported_residual_norm,
        values,
        &SERIAL_LINEAR_EXECUTION,
    )
}

/// Independently verify and accept a backend-produced solution through the
/// exact execution that produced it.
///
/// # Errors
/// Returns `EQ0802` for shape/non-finite execution behavior, inconsistent
/// termination/iteration evidence, or when the true residual exceeds the
/// declared target, and `EQ0807` for an incompatible reduction policy.
#[allow(clippy::too_many_arguments)]
pub fn accept_linear_solution_with_execution(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    solver_provider: SolverProvider,
    reason: ConvergenceReason,
    completed_iterations: usize,
    reported_residual_norm: f64,
    values: Vec<f64>,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    execution.require_reduction(plan.reduction())?;
    accept_linear_solution_with_verifier(
        problem,
        plan,
        solver_provider,
        execution.provider(),
        execution.report(),
        reason,
        completed_iterations,
        reported_residual_norm,
        values,
        execution,
    )
}

/// Independently verify a backend-produced solution through a distinct
/// verifier while preserving the execution that produced the values.
///
/// Verification always uses Eqiora's fixed-order inner-product action. This
/// keeps a backend-native fast reduction out of the acceptance oracle and
/// makes heterogeneous production/verification evidence explicit.
///
/// # Errors
/// Returns `EQ0802` for shape/non-finite verification behavior, inconsistent
/// termination/iteration evidence, or when the true residual exceeds the
/// declared target, and `EQ0807` when the verifier cannot execute the
/// reproducible acceptance reduction.
#[allow(clippy::too_many_arguments)]
pub fn accept_linear_solution_with_verifier(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    execution: ExecutionReport,
    reason: ConvergenceReason,
    completed_iterations: usize,
    reported_residual_norm: f64,
    values: Vec<f64>,
    verifier: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    let mut workspace = LinearAcceptanceWorkspace::new(problem)?;
    workspace.accept_with_verifier(
        problem,
        plan,
        solver_provider,
        execution_provider,
        execution,
        reason,
        completed_iterations,
        reported_residual_norm,
        values,
        verifier,
    )
}

/// Reusable buffers for independent true-residual acceptance.
///
/// Backends with collective liveness constraints construct this workspace
/// during admission, before entering any execution communication. The
/// workspace is shape-bound to one [`LinearProblem`] dimension but contains
/// no backend or solution state.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAcceptanceWorkspace {
    applied: Vec<f64>,
    zero_initial: Vec<f64>,
}

impl LinearAcceptanceWorkspace {
    /// Allocate the complete verifier workspace fallibly.
    ///
    /// # Errors
    /// Returns `EQ0802` if either bounded vector cannot be reserved.
    pub fn new(problem: &LinearProblem<'_>) -> Result<Self, Diagnostic> {
        Ok(Self {
            applied: zeroed_vector(problem.operator().rows(), "acceptance action")?,
            zero_initial: zeroed_vector(problem.operator().columns(), "acceptance initial guess")?,
        })
    }

    /// Independently verify a backend-produced solution using admitted buffers.
    ///
    /// This is the allocation-free execution counterpart of
    /// [`accept_linear_solution_with_verifier`]. It is intended for transports
    /// that must prove all dynamic workspace exists before communication begins.
    ///
    /// # Errors
    /// Returns the same diagnostics as [`accept_linear_solution_with_verifier`],
    /// plus `EQ0802` when this workspace has the wrong shape.
    #[allow(clippy::too_many_arguments)]
    pub fn accept_with_verifier(
        &mut self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        execution: ExecutionReport,
        reason: ConvergenceReason,
        completed_iterations: usize,
        reported_residual_norm: f64,
        values: Vec<f64>,
        verifier: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        verifier.require_reduction(ReductionPolicy::Reproducible)?;
        if values.len() != problem.operator().columns() {
            return Err(solve_failed(format!(
                "backend returned {} values for operator dimension {}",
                values.len(),
                problem.operator().columns()
            )));
        }
        if self.applied.len() != problem.operator().rows()
            || self.zero_initial.len() != problem.operator().columns()
        {
            return Err(solve_failed(
                "linear acceptance workspace does not match the admitted problem",
            ));
        }
        let initial = problem.initial_guess().unwrap_or(&self.zero_initial);
        let initial_residual_norm = residual_norm(verifier, problem, initial, &mut self.applied)?;
        let true_residual_norm = residual_norm(verifier, problem, &values, &mut self.applied)?;
        let right_hand_side_norm = euclidean_norm(verifier, problem.right_hand_side())?;
        let residual_target = plan.residual_target(right_hand_side_norm)?;
        let report = SolveReport::accepted_with_verification(
            solver_provider,
            execution_provider,
            execution,
            verifier.provider(),
            verifier.report(),
            problem.operator().orientation(),
            plan,
            reason,
            completed_iterations,
            initial_residual_norm,
            reported_residual_norm,
            true_residual_norm,
            residual_target,
        )?;
        LinearSolution::new(values, report)
    }
}

fn zeroed_vector(length: usize, purpose: &'static str) -> Result<Vec<f64>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| solve_failed(format!("could not reserve {purpose} workspace")))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn residual_norm(
    execution: &dyn ReplicatedLinearExecution,
    problem: &LinearProblem<'_>,
    values: &[f64],
    applied: &mut [f64],
) -> Result<f64, Diagnostic> {
    execution.apply(problem.operator(), values, applied)?;
    for (value, right) in applied.iter_mut().zip(problem.right_hand_side()) {
        *value = right - *value;
    }
    euclidean_norm(execution, applied)
}

fn euclidean_norm(
    execution: &dyn ReplicatedLinearExecution,
    values: &[f64],
) -> Result<f64, Diagnostic> {
    let squared = execution.inner_product(FixedOrderInnerProduct::new(values, values)?)?;
    Ok(squared.sqrt())
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

fn invalid_provider(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{LinearOperator, LinearOperatorProperties};

    fn solver_provider(id: &'static str) -> SolverProvider {
        SolverProvider::new(BackendId::new(id), "test", &[])
    }

    fn execution_provider(id: &'static str) -> ExecutionProvider {
        ExecutionProvider::new(ExecutionId::new(id), "test", &[])
    }

    #[derive(Debug)]
    struct Diagonal;

    impl LinearOperator for Diagonal {
        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            output[0] = 2.0 * input[0];
            output[1] = 4.0 * input[1];
            Ok(())
        }
    }

    #[test]
    fn heterogeneous_acceptance_preserves_producer_and_verifier() {
        let problem = LinearProblem::new(
            &Diagonal,
            &[2.0, 8.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let cuda = ExecutionReport::cuda(ExecutionId::new("eqiora.cuda.test"), 3);

        let solution = accept_linear_solution_with_verifier(
            &problem,
            plan,
            solver_provider("eqiora.cuda.test-solver"),
            execution_provider("eqiora.cuda.test"),
            cuda,
            ConvergenceReason::ResidualToleranceSatisfied,
            2,
            0.0,
            vec![1.0, 2.0],
            &SERIAL_LINEAR_EXECUTION,
        )
        .unwrap();

        assert_eq!(solution.report().execution(), cuda);
        assert_eq!(solution.report().solver_plan(), plan);
        assert_eq!(
            solution.report().verification(),
            ExecutionReport::host_serial()
        );
        assert_eq!(solution.report().true_residual_norm(), 0.0);
    }

    #[test]
    fn accepted_report_rejects_impossible_iteration_evidence() {
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap();
        let accepted = |reason, iterations, initial_residual_norm| {
            SolveReport::accepted(
                solver_provider("eqiora.test.solver"),
                crate::SERIAL_EXECUTION_PROVIDER,
                ExecutionReport::host_serial(),
                LinearOperatorOrientation::Normal,
                plan,
                reason,
                iterations,
                initial_residual_norm,
                0.0,
                0.0,
                1.0e-12,
            )
        };

        let over_limit =
            accepted(ConvergenceReason::ResidualToleranceSatisfied, 9, 1.0).unwrap_err();
        assert!(over_limit.message().contains("permits at most 8"));

        let false_initial =
            accepted(ConvergenceReason::InitialResidualSatisfied, 1, 0.0).unwrap_err();
        assert!(false_initial.message().contains("requires zero"));

        let false_iteration =
            accepted(ConvergenceReason::ResidualToleranceSatisfied, 0, 1.0).unwrap_err();
        assert!(false_iteration.message().contains("at least one"));

        let unsatisfied_initial =
            accepted(ConvergenceReason::InitialResidualSatisfied, 0, 1.0).unwrap_err();
        assert!(
            unsatisfied_initial
                .message()
                .contains("to satisfy the target")
        );
    }

    #[test]
    fn accepted_report_allows_producer_and_verifier_initial_norms_to_straddle_target() {
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap();

        // Termination reason and iteration count describe the producer. The
        // residual norms describe the independent verifier, whose reduction
        // may already place the initial norm exactly at the accepted target.
        let producer = ExecutionReport::cuda(ExecutionId::new("eqiora.cuda.test"), 2);
        let verifier = ExecutionReport::host_serial();
        let report = SolveReport::accepted_with_verification(
            solver_provider("eqiora.test.solver"),
            execution_provider("eqiora.cuda.test"),
            producer,
            crate::SERIAL_EXECUTION_PROVIDER,
            verifier,
            LinearOperatorOrientation::Normal,
            plan,
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            1.0e-12,
            0.0,
            0.0,
            1.0e-12,
        )
        .unwrap();

        assert_eq!(
            report.reason(),
            ConvergenceReason::ResidualToleranceSatisfied
        );
        assert_eq!(report.completed_iterations(), 1);
        assert_eq!(report.initial_residual_norm(), 1.0e-12);
        assert_eq!(report.execution(), producer);
        assert_eq!(report.verification(), verifier);
    }

    #[test]
    fn accepted_report_rejects_provider_and_report_identity_drift() {
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap();
        let report = ExecutionReport::host_serial();
        let error = SolveReport::accepted(
            solver_provider("eqiora.test.solver"),
            execution_provider("eqiora.test.substituted"),
            report,
            LinearOperatorOrientation::Normal,
            plan,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            0.0,
            0.0,
            0.0,
            1.0e-12,
        )
        .unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("contradicts"));
    }
}
