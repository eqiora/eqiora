use super::*;
use crate::{
    CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties, REFERENCE_LINEAR_SOLVER,
};
use eqiora_core::Scalar;
use num_complex::{Complex64, ComplexFloat};

struct Triangular<S>([S; 5]);
impl<S> CsrStorage<S> for Triangular<S> {
    fn rows(&self) -> usize {
        3
    }
    fn columns(&self) -> usize {
        3
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 1, 3, 5]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 0, 1, 0, 2]
    }
    fn values(&self) -> &[S] {
        &self.0
    }
}

fn check_shadow_restart<S: Scalar + ComplexFloat<Real = f64> + From<f64> + Sync>(phase: S)
where
    super::super::ReferenceLinearSolver: LinearSolverBackend<S>,
{
    // A = [[1,0,0],[1,2,0],[1,0,3]], b = phase * e_1.
    // First stabilization gives r=phase*(0,-3/13,2/13), orthogonal to r_0.
    // Forward substitution independently gives x=phase*(1,-1/2,-1/3).
    let storage = Triangular([1., 1., 2., 1., 3.].map(<S as From<f64>>::from));
    let rhs = [
        phase,
        <S as From<f64>>::from(0.),
        <S as From<f64>>::from(0.),
    ];
    let system =
        CanonicalCsrSystemView::new(&storage, &rhs, LinearOperatorProperties::General).unwrap();
    let problem = system.linear_problem().unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        20.try_into().unwrap(),
    )
    .unwrap();
    let result = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
    for (actual, expected) in result.values().iter().zip([1., -0.5, -1. / 3.]) {
        assert!((*actual - phase * <S as From<f64>>::from(expected)).abs() < 1e-12);
    }
    assert!(result.report().true_residual_norm() <= result.report().residual_target());
    let limited = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        1.try_into().unwrap(),
    )
    .unwrap();
    let error = REFERENCE_LINEAR_SOLVER
        .solve(&problem, limited)
        .unwrap_err();
    assert!(error.message().contains("reported residual"));
    assert!(error.message().contains("true residual"));
    assert!(error.message().contains("target"));
}

#[test]
fn real_shadow_orthogonality_restarts_without_accepting_an_unconverged_solution() {
    check_shadow_restart(1.);
}

#[test]
fn complex_shadow_orthogonality_restarts_without_accepting_an_unconverged_solution() {
    check_shadow_restart(Complex64::new(1., 1.));
}

struct OffDiagonal<S>([S; 2]);
impl<S> CsrStorage<S> for OffDiagonal<S> {
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
        &[1, 0]
    }
    fn values(&self) -> &[S] {
        &self.0
    }
}

fn check_fresh_shadow_recovery<S: Scalar + ComplexFloat<Real = f64> + From<f64> + Sync>(phase: S)
where
    super::super::ReferenceLinearSolver: LinearSolverBackend<S>,
{
    // A = phase * [[0,2],[3,0]]. For either coordinate RHS, r* A r=0,
    // although det(A)=-6*phase² is nonzero. Column values give the exact
    // coordinate solutions without calling an implementation to derive them.
    for scale in [0.0009765625, 1., 1024.] {
        let phase = phase * <S as From<f64>>::from(scale);
        let storage = OffDiagonal([
            <S as From<f64>>::from(2.) * phase,
            <S as From<f64>>::from(3.) * phase,
        ]);
        for (rhs, expected) in [
            (
                [<S as From<f64>>::from(2.) * phase, S::zero()],
                [S::zero(), S::one()],
            ),
            (
                [S::zero(), <S as From<f64>>::from(3.) * phase],
                [S::one(), S::zero()],
            ),
        ] {
            let system =
                CanonicalCsrSystemView::new(&storage, &rhs, LinearOperatorProperties::General)
                    .unwrap();
            let plan = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                0.,
                8.try_into().unwrap(),
            )
            .unwrap();
            let result = REFERENCE_LINEAR_SOLVER
                .solve(&system.linear_problem().unwrap(), plan)
                .unwrap();
            for (actual, expected) in result.values().iter().zip(expected) {
                assert!((*actual - expected).abs() < 1e-12);
            }
            assert!(result.report().true_residual_norm() <= result.report().residual_target());
            assert!(result.report().completed_iterations() <= 8);
            // Independently reapply the two authored rows.
            let residual = [
                <S as From<f64>>::from(2.) * phase * result.values()[1] - rhs[0],
                <S as From<f64>>::from(3.) * phase * result.values()[0] - rhs[1],
            ];
            assert!(
                residual[0].abs().hypot(residual[1].abs()) <= result.report().residual_target()
            );
        }
    }
    let singular = OffDiagonal([S::zero(), S::zero()]);
    let system = CanonicalCsrSystemView::new(
        &singular,
        &[S::one(), S::zero()],
        LinearOperatorProperties::General,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        0.,
        8.try_into().unwrap(),
    )
    .unwrap();
    assert!(
        REFERENCE_LINEAR_SOLVER
            .solve(&system.linear_problem().unwrap(), plan)
            .is_err()
    );
}

#[test]
fn real_fresh_shadow_recovers_orthogonal_operator_images_without_accepting_singular_systems() {
    check_fresh_shadow_recovery(1.);
}

#[test]
fn complex_fresh_shadow_recovers_orthogonal_operator_images_without_accepting_singular_systems() {
    check_fresh_shadow_recovery(Complex64::new(1., -2.));
}
