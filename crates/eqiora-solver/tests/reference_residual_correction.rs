//! Independent polynomial adjoint exposes loss of small residual corrections.
use eqiora_solver::{
    CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties, LinearSolver,
    LinearSolverBackend, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use num_complex::Complex64 as C;

struct Matrix<S> {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<S>,
}
impl<S> CsrStorage<S> for Matrix<S> {
    fn rows(&self) -> usize {
        126
    }
    fn columns(&self) -> usize {
        126
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.columns
    }
    fn values(&self) -> &[S] {
        &self.values
    }
}

fn fixture() -> (Matrix<f64>, Vec<f64>, Vec<f64>) {
    let mut matrix = Matrix {
        offsets: vec![0],
        columns: Vec::new(),
        values: Vec::new(),
    };
    for row in 0..126 {
        if row >= 2 {
            matrix.columns.push(row - 2);
            matrix.values.push(-64.);
        }
        matrix.columns.push(row);
        matrix.values.push(128.);
        if row + 2 < 126 {
            matrix.columns.push(row + 2);
            matrix.values.push(-64.);
        }
        matrix.offsets.push(matrix.values.len());
    }
    let mut rhs = Vec::new();
    let mut expected = Vec::new();
    for j in 1..64i64 {
        // A = 64 tridiag(-1,2,-1), independently for each interleaved channel.
        // The quadratic RHS comes from exact Q1 mass moments on [0,3].
        // Solve its second difference symbolically: x_j is quartic, x_0=x_64=0.
        // Integer division by 3 is exact before conversion; remaining denominators
        // are powers of two. Neither coefficients nor expected values use a solve.
        let t = j as f64;
        rhs.push((409567. + 532544. * t - 99. * t * t) / 1048576.);
        rhs.push((274423. + 263488. * t - 27. * t * t) / 524288.);
        for (coefficients, denominator) in [
            ([4493928256, -2457501, -1065088, 99], 268435456.),
            ([2256795968, -1646565, -526976, 27], 134217728.),
        ] {
            let numerator = j
                * (coefficients[0]
                    + j * (coefficients[1] + j * (coefficients[2] + j * coefficients[3])));
            assert_eq!(numerator % 3, 0);
            expected.push((numerator / 3) as f64 / denominator);
        }
    }
    (matrix, rhs, expected)
}

fn policy(iterations: usize) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-14,
        1e-14,
        iterations.try_into().unwrap(),
    )
    .unwrap()
}

#[test]
fn real_correction_reaches_the_original_true_residual_within_the_original_budget() {
    let (matrix, rhs, expected) = fixture();
    let system =
        CanonicalCsrSystemView::new(&matrix, &rhs, LinearOperatorProperties::General).unwrap();
    for initial in [vec![0.; 126], vec![0.5; 126], expected.clone()] {
        let point = system
            .linear_problem()
            .unwrap()
            .with_initial_guess(&initial)
            .unwrap();
        let result = REFERENCE_LINEAR_SOLVER.solve(&point, policy(256)).unwrap();
        assert!(result.report().true_residual_norm() <= result.report().residual_target());
        assert!(result.report().completed_iterations() <= 256);
        for (actual, expected) in result.values().iter().zip(&expected) {
            assert!((actual - expected).abs() <= 2e-12);
        }
    }
    assert!(
        REFERENCE_LINEAR_SOLVER
            .solve(&system.linear_problem().unwrap(), policy(1))
            .is_err()
    );
}

#[test]
fn complex_and_real_coordinate_solves_retain_the_same_analytic_correction() {
    let (matrix, rhs, expected) = fixture();
    let phase = C::new(0.75, -0.5);
    let matrix = Matrix {
        offsets: matrix.offsets,
        columns: matrix.columns,
        values: matrix.values.into_iter().map(|v| phase * v).collect(),
    };
    let rhs = rhs.into_iter().map(|v| phase * v).collect::<Vec<_>>();
    let system =
        CanonicalCsrSystemView::new(&matrix, &rhs, LinearOperatorProperties::General).unwrap();
    let problem = system.linear_problem().unwrap();
    let result = REFERENCE_LINEAR_SOLVER
        .solve(&problem, policy(4096))
        .unwrap();
    assert!(result.report().true_residual_norm() <= result.report().residual_target());
    for (actual, expected) in result.values().iter().zip(&expected) {
        assert!((*actual - C::new(*expected, 0.)).norm() <= 2e-12);
    }
    let real = system
        .real_coordinate_system(eqiora_solver::LinearOperatorOrientation::Normal, &rhs)
        .unwrap();
    let result = REFERENCE_LINEAR_SOLVER
        .solve(&real.linear_problem().unwrap(), policy(4096))
        .unwrap();
    for (actual, expected) in result.values().as_chunks::<2>().0.iter().zip(expected) {
        assert!((actual[0] - expected).abs() <= 2e-12);
        assert!(actual[1].abs() <= 2e-12);
    }
}
