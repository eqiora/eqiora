use super::*;
// Independent two-node network: conductance 2, injection [6,-6], voltage drop 3.
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolveRequest,
    LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use std::num::NonZeroUsize;

struct Network {
    matrix: [f64; 4],
    load: [f64; 2],
}
impl CompleteCsrStorage for Network {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }
    fn values(&self) -> &[f64] {
        &self.matrix
    }
    fn right_hand_side(&self) -> &[f64] {
        &self.load
    }
}
fn system(matrix: [f64; 4], load: [f64; 2]) -> CanonicalCsrSystemView {
    CanonicalCsrSystemView::new(
        &Network { matrix, load },
        LinearOperatorProperties::Symmetric,
    )
    .unwrap()
}
fn request() -> LinearSolveRequest<'static> {
    LinearSolveRequest::new(
        &REFERENCE_LINEAR_SOLVER,
        SolverPlan::new(
            LinearSolver::MinimumResidual,
            1e-12,
            1e-12,
            NonZeroUsize::new(16).unwrap(),
        )
        .unwrap(),
    )
}
fn constraint(basis: [f64; 2], weights: [f64; 2], value: f64) -> NullspaceConstraint {
    NullspaceConstraint::new(basis.into(), weights.into(), value).unwrap()
}

#[test]
fn explicit_reference_changes_only_coordinates_not_network_drop_or_load() {
    let system = system([2., -2., -2., 2.], [6., -6.]);
    for (weights, value, expected) in [
        ([0.5, 0.5], 0., [1.5, -1.5]),
        ([1., 0.], 4., [4., 1.]),
        ([0., 1.], -2., [1., -2.]),
    ] {
        let solved = solve_canonical_with_nullspace(
            request(),
            &system,
            &constraint([1., 1.], weights, value),
        )
        .unwrap();
        for (actual, expected) in solved.values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert!((solved.values[0] - solved.values[1] - 3.).abs() < 1e-10);
        assert_eq!(solved.evidence.compatibility_residual, 0.);
        assert!(solved.evidence.original_residual_norm < 1e-10);
        assert!(solved.evidence.gauge_residual.abs() < 1e-10);
        assert!(solved.evidence.multiplier.abs() < 1e-10);
    }
    assert_eq!(system.values(), &[2., -2., -2., 2.]);
    assert_eq!(system.right_hand_side(), &[6., -6.]);
}

#[test]
fn wrong_null_vector_and_hidden_pin_fail_against_actual_operator() {
    let original = system([2., -2., -2., 2.], [6., -6.]);
    let wrong =
        solve_canonical_with_nullspace(request(), &original, &constraint([1., -1.], [1., 0.], 0.))
            .unwrap_err();
    assert!(wrong.to_string().contains("fails the actual operator"));
    let pinned = system([1., 0., 0., 2.], [0., -6.]);
    let hidden_pin =
        solve_canonical_with_nullspace(request(), &pinned, &constraint([1., 1.], [1., 0.], 0.))
            .unwrap_err();
    assert!(hidden_pin.to_string().contains("fails the actual operator"));
}

#[test]
fn incompatible_load_rejects_even_when_solver_tolerance_would_accept_it() {
    let original = system([2., -2., -2., 2.], [6., -5.]);
    let loose = LinearSolveRequest::new(
        &REFERENCE_LINEAR_SOLVER,
        SolverPlan::new(
            LinearSolver::MinimumResidual,
            1.,
            100.,
            NonZeroUsize::new(16).unwrap(),
        )
        .unwrap(),
    );
    let error =
        solve_canonical_with_nullspace(loose, &original, &constraint([1., 1.], [0.5, 0.5], 0.))
            .unwrap_err();
    assert!(error.to_string().contains("load is incompatible"));
}

#[test]
fn reference_must_remove_the_declared_null_direction() {
    assert!(NullspaceConstraint::new(vec![1., 1.], vec![1., -1.], 0.).is_err());
    assert!(NullspaceConstraint::new(vec![0., 0.], vec![1., 1.], 0.).is_err());
    assert!(NullspaceConstraint::new(vec![1., f64::NAN], vec![1., 1.], 0.).is_err());
    assert!(NullspaceConstraint::new(vec![1., 1.], vec![1.], 0.).is_err());
}

#[test]
fn general_operator_cannot_reuse_right_null_vector_as_left_compatibility_test() {
    // A*[1,1]=0, but the left kernel is [2,-1], not [1,1].
    let original = CanonicalCsrSystemView::new(
        &Network {
            matrix: [1., -1., 2., -2.],
            load: [1., -1.],
        },
        LinearOperatorProperties::General,
    )
    .unwrap();
    let error =
        solve_canonical_with_nullspace(request(), &original, &constraint([1., 1.], [1., 0.], 0.))
            .unwrap_err();
    assert!(error.to_string().contains("explicitly symmetric"));
}

#[test]
fn basis_scaling_cannot_hide_an_invalid_mode_or_change_the_reference() {
    let original = system([2., -2., -2., 2.], [6., -6.]);
    for scale in [1e-200, -1., 1e200] {
        let solved = solve_canonical_with_nullspace(
            request(),
            &original,
            &constraint([scale, scale], [1., 0.], 4.),
        )
        .unwrap();
        assert!((solved.values[0] - 4.).abs() < 1e-10);
        assert!((solved.values[1] - 1.).abs() < 1e-10);
        assert!(
            solve_canonical_with_nullspace(
                request(),
                &original,
                &constraint([scale, 2. * scale], [1., 0.], 4.),
            )
            .is_err()
        );
    }
    assert!(NullspaceConstraint::new(vec![1e300, 1e-300], vec![1., 1.], 0.).is_err());
}
