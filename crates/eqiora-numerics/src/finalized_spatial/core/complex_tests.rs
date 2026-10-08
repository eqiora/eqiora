use super::*;
use eqiora_solver::{
    CompleteCsrStorage, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER,
};
use std::num::NonZeroUsize;

type C = Complex64;

struct ScalarSystem {
    a: [C; 1],
    b: [C; 1],
}
impl CompleteCsrStorage<C> for ScalarSystem {
    fn rows(&self) -> usize {
        1
    }
    fn columns(&self) -> usize {
        1
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 1]
    }
    fn column_indices(&self) -> &[usize] {
        &[0]
    }
    fn values(&self) -> &[C] {
        &self.a
    }
    fn right_hand_side(&self) -> &[C] {
        &self.b
    }
}

fn policy(relative: f64) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        relative,
        1e-12,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
}

fn core(a: C, b: C, policy: SolverPlan) -> FinalizedLinearCore<C> {
    FinalizedLinearCore::new(
        policy,
        VectorLayoutKind::Replicated,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        Arc::new(
            CanonicalCsrSystemView::new(
                &ScalarSystem { a: [a], b: [b] },
                LinearOperatorProperties::General,
            )
            .unwrap(),
        ),
    )
}

fn solve(core: &FinalizedLinearCore<C>) -> LinearSolution<C> {
    REFERENCE_LINEAR_SOLVER
        .solve(&core.linear_problem().unwrap(), core.solver_plan())
        .unwrap()
}

#[test]
fn complex_finalization_recomputes_both_residual_channels_and_rhs_norm() {
    // (2+i)(2+i)=3+4i and |3+4i|=5, independently of solver output.
    let accepted = core(C::new(2., 1.), C::new(3., 4.), policy(1e-3));
    let solution = solve(&accepted);
    assert!((solution.values()[0] - C::new(2., 1.)).norm() < 1e-12);
    assert_eq!(
        solution.report().residual_target().to_bits(),
        (5e-3_f64).to_bits()
    );
    accepted.validate_solution(&solution).unwrap();

    let lost_imaginary_rhs = core(C::new(2., 1.), C::new(3., 0.), policy(1e-3));
    let error = lost_imaginary_rhs.validate_solution(&solution).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(error.message().contains("tolerance evidence"));

    // Identical absolute targets prevent the target check from masking the
    // residual falsifier: both receiving systems have purely imaginary defects.
    for (producer_a, producer_b, receiving_a, receiving_b) in [
        (
            C::new(1., 0.),
            C::new(2., -1.),
            C::new(1., 0.),
            C::new(2., 1.),
        ),
        (
            C::new(2., 1.),
            C::new(4., 2.),
            C::new(2., -1.),
            C::new(4., 2.),
        ),
    ] {
        let producer = core(producer_a, producer_b, policy(0.));
        let candidate = solve(&producer);
        producer.validate_solution(&candidate).unwrap();
        let receiving = core(receiving_a, receiving_b, policy(0.));
        let error = receiving.validate_solution(&candidate).unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert!(error.message().contains("residual"));
    }
}

#[test]
fn complex_norm_keeps_all_components_across_the_fixed_partial_boundary() {
    // 513 complex values cross the 1024-real-component partial boundary.
    // Each contributes 3²+4²=25; imaginary terms must never cancel real terms.
    assert_eq!(
        C::squared_norm(&vec![C::new(3., 4.); 513]).unwrap(),
        513. * 25.
    );
    for value in [
        C::new(0., f64::NAN),
        C::new(0., f64::INFINITY),
        C::new(0., f64::MAX),
    ] {
        assert_eq!(
            C::squared_norm(&[value]).unwrap_err().code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
    }
    assert_eq!(
        fallible_residual::<C>(usize::MAX).unwrap_err().code(),
        codes::NUMERICAL_SOLVE_FAILED
    );
}
