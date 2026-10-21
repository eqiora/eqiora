use super::*;
use eqiora_solver::{CsrStorage, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER};
use std::num::NonZeroUsize;

use num_complex::Complex64 as C;

struct ScalarSystem {
    a: [C; 1],
    b: [C; 1],
}
impl CsrStorage<C> for ScalarSystem {
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
}

impl ScalarSystem {
    pub(crate) fn rhs(&self) -> &[C] {
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
            {
                let storage = &ScalarSystem { a: [a], b: [b] };
                CanonicalCsrSystemView::new(
                    storage,
                    storage.rhs(),
                    LinearOperatorProperties::General,
                )
            }
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
    // 1025 complex values cross multiple 1024-real-coordinate partial boundaries.
    // Each contributes 3²+4²=25; imaginary terms must never cancel real terms.
    assert_eq!(
        FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &vec![C::new(3., 4.); 1025])
            .unwrap(),
        (1025.0_f64 * 25.0).sqrt()
    );
    // The magnitude of one purely imaginary finite value is its absolute
    // value; squaring that coordinate must not introduce overflow.
    assert_eq!(
        FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[C::new(0., f64::MAX)]).unwrap(),
        f64::MAX
    );
    for value in [
        C::new(0., f64::NAN),
        C::new(0., f64::INFINITY),
        // sqrt(2) * MAX is outside binary64's finite range.
        C::new(f64::MAX, f64::MAX),
    ] {
        assert_eq!(
            FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[value])
                .unwrap_err()
                .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
    }
    assert_eq!(
        fallible_residual::<C>(usize::MAX).unwrap_err().code(),
        codes::NUMERICAL_SOLVE_FAILED
    );
}

#[test]
fn empty_complete_equation_rejects_zero_rhs_solve_and_replayed_coordinates() {
    let zero = core(C::new(0., 0.), C::new(0., 0.), policy(1e-3));
    let error = zero.linear_problem().unwrap_err();
    assert!(error.message().contains("complete spatial equation row 0"));
    let error = zero.validate_coefficients(&[C::new(4., 7.)]).unwrap_err();
    assert!(error.message().contains("complete spatial equation row 0"));
}
