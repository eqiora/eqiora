//! Method-owned row acceptance must survive global norm convergence and lowering.
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_solver::{
    CanonicalCsrSystemView, ConvergenceReason, CsrStorage, ExecutionId, ExecutionReport,
    FixedOrderInnerProduct, LinearOperator, LinearOperatorProperties, LinearProblem, LinearSolver,
    LinearSolverBackend, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, ReductionPolicy,
    ReplicatedLinearExecution, SERIAL_EXECUTION_PROVIDER, SERIAL_LINEAR_EXECUTION, SolverPlan,
};
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Diagonal;
impl LinearOperator for Diagonal {
    type Scalar = f64;
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output.copy_from_slice(&[input[0], 2. * input[1]]);
        Ok(())
    }
}
fn plan(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(algorithm, 2f64.powi(-20), 2f64.powi(-40), NonZeroUsize::MIN).unwrap()
}
fn row_check(x: &[f64]) -> Result<bool, Diagnostic> {
    // Independently: |2*x_1-1| <= max(atol, rtol * (|2*x_1|+|b_1|)).
    let target = 2f64.powi(-40).max(2f64.powi(-20) * (2. * x[1].abs() + 1.));
    Ok((2. * x[1] - 1.).abs() <= target)
}
#[test]
fn bicgstab_completes_omega_in_the_same_original_iteration() {
    // B^2 dominates all dot products: alpha rounds to 1, leaving row two
    // wrong by 1 although the global target is 1024. Omega=1/2 solves it.
    let rhs = [2f64.powi(30), 1.];
    let ordinary = LinearProblem::new(&Diagonal, &rhs, LinearOperatorProperties::General).unwrap();
    let policy = plan(LinearSolver::BiConjugateGradientStabilized);
    let premature = REFERENCE_LINEAR_SOLVER.solve(&ordinary, policy).unwrap();
    assert_eq!(premature.values(), &[rhs[0], 1.]);
    assert!(!row_check(premature.values()).unwrap());
    let checked = ordinary.with_candidate_check(&row_check);
    let solved = REFERENCE_LINEAR_SOLVER.solve(&checked, policy).unwrap();
    assert_eq!(solved.values(), &[rhs[0], 0.5]);
    assert_eq!(solved.report().completed_iterations(), 1);
    assert_eq!(solved.report().solver_plan(), policy);
    assert_eq!(solved.report().solver_provider(), REFERENCE_SOLVER_PROVIDER);
    assert_eq!(
        solved.report().execution_provider(),
        SERIAL_EXECUTION_PROVIDER
    );
    assert_eq!(solved.report().true_residual_norm(), 0.);
}
#[test]
fn globally_accepted_initial_guess_still_requires_physical_acceptance() {
    let rhs = [2f64.powi(30), 1.];
    let initial = [rhs[0], 1.];
    for algorithm in [
        LinearSolver::ConjugateGradient,
        LinearSolver::MinimumResidual,
        LinearSolver::BiConjugateGradientStabilized,
    ] {
        let problem = LinearProblem::new(
            &Diagonal,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_initial_guess(&initial)
        .unwrap()
        .with_candidate_check(&row_check);
        let solved = REFERENCE_LINEAR_SOLVER
            .solve(&problem, plan(algorithm))
            .unwrap();
        assert_eq!(solved.values(), &[rhs[0], 0.5]);
        assert_eq!(solved.report().completed_iterations(), 1);
    }
}
#[test]
fn structural_errors_and_zero_residual_rejection_do_not_become_success() {
    let calls = AtomicUsize::new(0);
    let reject = |_: &[f64]| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    };
    let fail = |_: &[f64]| {
        Err(Diagnostic::error(
            codes::INVALID_REALIZATION,
            "physical domain failure",
        ))
    };
    let rhs = [0., 0.];
    for algorithm in [
        LinearSolver::ConjugateGradient,
        LinearSolver::MinimumResidual,
        LinearSolver::BiConjugateGradientStabilized,
    ] {
        let policy = plan(algorithm);
        let problem = LinearProblem::new(
            &Diagonal,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        calls.store(0, Ordering::SeqCst);
        assert_eq!(
            REFERENCE_LINEAR_SOLVER
                .solve(&problem.with_candidate_check(&reject), policy)
                .unwrap_err()
                .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let problem = LinearProblem::new(
            &Diagonal,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&fail);
        assert_eq!(
            REFERENCE_LINEAR_SOLVER
                .solve(&problem, policy)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
}
#[derive(Debug)]
struct ForbiddenExecution(ExecutionReport);
impl ReplicatedLinearExecution for ForbiddenExecution {
    fn provider(&self) -> eqiora_solver::ExecutionProvider {
        SERIAL_EXECUTION_PROVIDER
    }
    fn report(&self) -> ExecutionReport {
        self.0
    }
    fn require_reduction(&self, _: ReductionPolicy) -> Result<(), Diagnostic> {
        panic!("reduction admission must not run")
    }
    fn apply(
        &self,
        _: &dyn LinearOperator<Scalar = f64>,
        _: &[f64],
        _: &mut [f64],
    ) -> Result<(), Diagnostic> {
        panic!("apply must not run")
    }
    fn inner_product(&self, _: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        panic!("reduction must not run")
    }
}
#[test]
fn admission_rejects_actual_producer_and_verifier_before_execution() {
    for report in [
        ExecutionReport::distributed(ExecutionId::new("test.distributed"), NonZeroUsize::MIN),
        ExecutionReport::cuda(ExecutionId::new("test.cuda"), 0),
    ] {
        check_forbidden_execution(&ForbiddenExecution(report));
    }
}

fn check_forbidden_execution(forbidden: &ForbiddenExecution) {
    let check = |_: &[f64]| -> Result<bool, Diagnostic> { panic!("callback must not run") };
    let problem = LinearProblem::new(&Diagonal, &[0., 0.], LinearOperatorProperties::General)
        .unwrap()
        .with_candidate_check(&check);
    problem
        .require_candidate_check_execution(ExecutionReport::host(
            ExecutionId::new("test.parallel"),
            NonZeroUsize::new(4).unwrap(),
        ))
        .unwrap();
    let policy = plan(LinearSolver::BiConjugateGradientStabilized);
    assert_eq!(
        REFERENCE_LINEAR_SOLVER
            .solve_with_execution(&problem, policy, forbidden)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    assert_eq!(
        eqiora_solver::LinearSolution::accept_with_execution(
            &problem,
            policy,
            REFERENCE_SOLVER_PROVIDER,
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.,
            vec![0., 0.],
            forbidden
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
    for (producer, verifier) in [
        (
            forbidden.report(),
            &SERIAL_LINEAR_EXECUTION as &dyn ReplicatedLinearExecution,
        ),
        (ExecutionReport::host_serial(), forbidden),
    ] {
        assert_eq!(
            eqiora_solver::LinearSolution::accept_with_verifier(
                &problem,
                policy,
                REFERENCE_SOLVER_PROVIDER,
                SERIAL_EXECUTION_PROVIDER,
                producer,
                ConvergenceReason::ResidualToleranceSatisfied,
                1,
                0.,
                vec![0., 0.],
                verifier
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
    }
}
struct ComplexDiagonal;
impl CsrStorage<num_complex::Complex64> for ComplexDiagonal {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 1, 2]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1]
    }
    fn values(&self) -> &[num_complex::Complex64] {
        const VALUES: [num_complex::Complex64; 2] = [
            num_complex::Complex64::new(1., 0.),
            num_complex::Complex64::new(2., 0.),
        ];
        &VALUES
    }
}
#[test]
fn complex_cg_lowering_checks_physical_complex_coordinates_with_one_budget() {
    use num_complex::Complex64 as C;
    let rhs = [C::new(2f64.powi(30), 2f64.powi(29)), C::new(1., -1.)];
    let initial = [rhs[0], C::new(1., -1.)];
    let system = CanonicalCsrSystemView::new(
        &ComplexDiagonal,
        &rhs,
        LinearOperatorProperties::HermitianPositiveDefinite,
    )
    .unwrap();
    let check = |x: &[C]| {
        assert_eq!(x.len(), 2);
        Ok((2. * x[1] - rhs[1]).norm() <= 2f64.powi(-20))
    };
    let problem = system
        .linear_problem()
        .unwrap()
        .with_initial_guess(&initial)
        .unwrap()
        .with_candidate_check(&check);
    let policy = plan(LinearSolver::ConjugateGradient);
    let solved = REFERENCE_LINEAR_SOLVER.solve(&problem, policy).unwrap();
    assert_eq!(solved.values(), &[rhs[0], C::new(0.5, -0.5)]);
    assert_eq!(solved.report().completed_iterations(), 1);
    assert_eq!(solved.report().solver_plan(), policy);
}

#[test]
fn shared_final_acceptance_cannot_bypass_a_method_rejection() {
    let reject = |_: &[f64]| Ok(false);
    let rhs = [2., 1.];
    let policy = plan(LinearSolver::BiConjugateGradientStabilized);
    let problem = LinearProblem::new(&Diagonal, &rhs, LinearOperatorProperties::General)
        .unwrap()
        .with_candidate_check(&reject);
    assert_eq!(
        eqiora_solver::LinearSolution::accept_with_execution(
            &problem,
            policy,
            REFERENCE_SOLVER_PROVIDER,
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.,
            vec![2., 0.5],
            &SERIAL_LINEAR_EXECUTION
        )
        .unwrap_err()
        .code(),
        codes::NUMERICAL_SOLVE_FAILED
    );
}
