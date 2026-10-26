use super::*;
use eqiora_solver::{CsrStorage, LinearOperatorProperties, LinearSolver};
use std::num::NonZeroUsize;

struct Scalar([f64; 1]);
impl CsrStorage for Scalar {
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
    fn values(&self) -> &[f64] {
        &self.0
    }
}

fn system(value: f64) -> CanonicalCsrSystemView {
    CanonicalCsrSystemView::new(
        &Scalar([value]),
        &[-0.25],
        LinearOperatorProperties::General,
    )
    .unwrap()
}

fn criterion(
    system: &CanonicalCsrSystemView,
    direction: &[f64],
    accepted: &[f64],
    absolute: &[f64],
) -> PrimalAcceptance {
    PrimalAcceptance::NewtonCorrection {
        previous: vec![1.5],
        direction: direction.to_vec(),
        accepted: accepted.to_vec(),
        system: Arc::new(system.clone()),
        solve: SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(10).unwrap(),
        )
        .unwrap(),
        relative: 1e-15,
        absolute: absolute.to_vec(),
    }
}

#[test]
fn full_newton_correction_does_not_fabricate_a_residual_tolerance() {
    // f(x)=x²-2, x0=3/2: J0=3, F0=1/4, delta=-1/12.
    // x1=17/12 has a nonzero residual 1/144, despite a correction
    // below the declared physical absolute tolerance 1/10.
    let matrix = system(3.);
    let direction = [-1.0_f64 / 12.];
    let accepted = [1.5 + direction[0]];
    let residual = [accepted[0] * accepted[0] - 2.];
    assert!((residual[0] - 1. / 144.).abs() < 1e-15);
    assert!(
        PrimalAcceptance::Residual(1e-12)
            .validate(&residual)
            .is_err()
    );
    assert_eq!(
        criterion(&matrix, &direction, &accepted, &[0.1])
            .validate(&residual)
            .unwrap(),
        None
    );
    assert!(
        criterion(&matrix, &direction, &accepted, &[1e-4])
            .validate(&residual)
            .is_err()
    );
    assert!(
        criterion(&matrix, &direction, &[accepted[0] + 1e-6], &[0.1])
            .validate(&residual)
            .is_err()
    );
    // A tiny claimed correction passes the coordinate norm but fails the
    // actual old Jacobian equation. No recorded weighted norm is trusted.
    assert!(
        criterion(&matrix, &[-1e-9], &[1.5 - 1e-9], &[0.1])
            .validate(&residual)
            .is_err()
    );
    assert!(
        criterion(&system(4.), &direction, &accepted, &[0.1])
            .validate(&residual)
            .is_err()
    );
}

#[test]
fn composed_acceptance_keeps_each_equation_block_criterion() {
    let residual_blocks = [
        (1, PrimalAcceptance::Residual(0.01)),
        (1, PrimalAcceptance::Residual(0.02)),
    ];
    let bound = PrimalAcceptance::Blocks(residual_blocks.to_vec())
        .validate(&[0.005, 0.01])
        .unwrap()
        .unwrap();
    assert_eq!(bound, 0.01_f64.hypot(0.02));
    // A combined norm alone would admit this failure of the first block.
    assert!(
        PrimalAcceptance::Blocks(residual_blocks.to_vec())
            .validate(&[0.015, 0.])
            .is_err()
    );
    assert!(
        PrimalAcceptance::Blocks(residual_blocks.to_vec())
            .validate(&[0.])
            .is_err()
    );
    assert!(
        PrimalAcceptance::Blocks(residual_blocks.to_vec())
            .validate(&[0.; 3])
            .is_err()
    );
    let matrix = system(3.);
    let direction = [-1.0_f64 / 12.];
    let accepted = [1.5 + direction[0]];
    let mixed = [
        (1, PrimalAcceptance::Residual(0.01)),
        (1, criterion(&matrix, &direction, &accepted, &[0.1])),
    ];
    assert_eq!(
        PrimalAcceptance::Blocks(mixed.to_vec())
            .validate(&[0., accepted[0] * accepted[0] - 2.])
            .unwrap(),
        None
    );
}
