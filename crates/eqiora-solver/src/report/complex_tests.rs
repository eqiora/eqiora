use num_complex::Complex64 as C;

use super::*;
use crate::{FixedOrderInnerProduct, LinearOperator, LinearOperatorProperties};

#[derive(Debug)]
struct Diagonal;

impl LinearOperator for Diagonal {
    type Scalar = C;
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn apply(&self, input: &[C], output: &mut [C]) -> Result<(), Diagnostic> {
        output[0] = C::new(2.0, 1.0) * input[0];
        output[1] = C::new(3.0, -2.0) * input[1];
        Ok(())
    }
}

fn plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        0.0,
        NonZeroUsize::MIN,
    )
    .unwrap()
}

fn provider() -> SolverProvider {
    SolverProvider::new(BackendId::new("eqiora.test.typed-acceptance"), "1", &[])
}

#[test]
fn complex_acceptance_checks_the_original_action_and_imaginary_residual() {
    // (2+i)(1-3i)=5-5i and (3-2i)(-2+i)=-4+7i.
    let rhs = [C::new(5.0, -5.0), C::new(-4.0, 7.0)];
    let values = vec![C::new(1.0, -3.0), C::new(-2.0, 1.0)];
    let problem = LinearProblem::new(&Diagonal, &rhs, LinearOperatorProperties::General).unwrap();
    let accepted = accept_linear_solution(
        &problem,
        plan(),
        provider(),
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        0.0,
        values.clone(),
    )
    .unwrap();
    assert_eq!(accepted.values(), values);
    assert_eq!(accepted.report().true_residual_norm(), 0.0);
    assert_eq!(
        accepted.report().residual_target(),
        115.0_f64.sqrt() * 1.0e-12
    );
    assert_eq!(
        accepted.report().verification_provider(),
        SERIAL_LINEAR_EXECUTION.provider()
    );
    let mut wrong = values.clone();
    wrong[0].im += 1.0;
    assert!(
        accept_linear_solution(
            &problem,
            plan(),
            provider(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.0,
            wrong
        )
        .is_err()
    );
    let initial_problem = problem.with_initial_guess(&values).unwrap();
    let initial = accept_linear_solution(
        &initial_problem,
        plan(),
        provider(),
        ConvergenceReason::InitialResidualSatisfied,
        0,
        0.0,
        values.clone(),
    )
    .unwrap();
    assert_eq!(initial.report().completed_iterations(), 0);
}

#[derive(Debug)]
struct NonrealNorm;
impl ReplicatedLinearExecution<C> for NonrealNorm {
    fn provider(&self) -> ExecutionProvider {
        SERIAL_LINEAR_EXECUTION.provider()
    }
    fn report(&self) -> ExecutionReport {
        SERIAL_LINEAR_EXECUTION.report()
    }
    fn require_reduction(&self, _: ReductionPolicy) -> Result<(), Diagnostic> {
        Ok(())
    }
    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = C>,
        input: &[C],
        output: &mut [C],
    ) -> Result<(), Diagnostic> {
        operator.apply(input, output)
    }
    fn inner_product(&self, action: FixedOrderInnerProduct<'_, C>) -> Result<C, Diagnostic> {
        Ok(SERIAL_LINEAR_EXECUTION.inner_product(action)? + C::new(0.0, 1.0))
    }
}

#[test]
fn complex_acceptance_rejects_a_nonreal_norm_from_the_verifier() {
    let rhs = [C::new(5.0, -5.0), C::new(-4.0, 7.0)];
    let values = vec![C::new(1.0, -3.0), C::new(-2.0, 1.0)];
    let problem = LinearProblem::new(&Diagonal, &rhs, LinearOperatorProperties::General).unwrap();
    assert!(
        accept_linear_solution_with_execution(
            &problem,
            plan(),
            provider(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.0,
            values,
            &NonrealNorm
        )
        .is_err()
    );
}
