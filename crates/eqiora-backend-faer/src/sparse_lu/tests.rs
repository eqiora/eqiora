use std::num::NonZeroUsize;

use eqiora_solver::{
    CanonicalCsrSystemView, CsrStorage, LinearOperatorOrientation, LinearOperatorProperties,
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, PreparedLinearSolver,
    PreparedLinearStructureIdentity, ReductionPolicy, SolverPlan,
};

use crate::FaerLinearSolver;
use crate::prepared_sparse_lu::FaerPreparedSparseLu;

struct Storage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl CsrStorage for Storage {
    fn rows(&self) -> usize {
        64
    }
    fn columns(&self) -> usize {
        64
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.columns
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

fn row_scale(row: usize) -> f64 {
    [1.0, 2.0, 4.0, 8.0][row % 4]
}

fn fixture() -> CanonicalCsrSystemView {
    // T = 64 tridiag(-1, 2, -1), with 3 at the two boundary diagonals.
    // Its exact solution to T z = 1 is z_j = (j(63-j)+32)/128:
    // the interior second difference is 2/128; 3z_0-z_1 = 2/128.
    // A = D T is nonsymmetric. A z = D 1 and A^T D^-1 z = 1.
    // All coefficients, loads and expected coordinates are exactly dyadic.
    let mut storage = Storage {
        offsets: vec![0],
        columns: vec![],
        values: vec![],
    };
    for row in 0..64 {
        let scale = row_scale(row);
        if row > 0 {
            storage.columns.push(row - 1);
            storage.values.push(-64.0 * scale);
        }
        storage.columns.push(row);
        storage.values.push(if row == 0 || row == 63 {
            192.0 * scale
        } else {
            128.0 * scale
        });
        if row < 63 {
            storage.columns.push(row + 1);
            storage.values.push(-64.0 * scale);
        }
        storage.offsets.push(storage.columns.len());
    }
    CanonicalCsrSystemView::new(
        &storage,
        &(0..64).map(row_scale).collect::<Vec<_>>(),
        LinearOperatorProperties::General,
    )
    .unwrap()
}

fn plan(backsolves: usize) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::SparseLu,
        1.0e-14,
        1.0e-14,
        NonZeroUsize::new(backsolves).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn check(solution: &eqiora_solver::LinearSolution, orientation: LinearOperatorOrientation) {
    assert!(solution.report().completed_iterations() > 1);
    assert!(solution.report().completed_iterations() <= 4);
    for (row, value) in solution.values().iter().enumerate() {
        let numerator = u32::try_from(row * (63 - row) + 32).unwrap();
        let expected = f64::from(numerator)
            / 128.0
            / if orientation == LinearOperatorOrientation::Normal {
                1.0
            } else {
                row_scale(row)
            };
        assert!(
            (value - expected).abs() <= 1.0e-12,
            "row {row}: {value} != {expected}"
        );
    }
}

#[test]
fn corrections_preserve_the_original_action_and_backsolve_budget() {
    let source = fixture();
    for orientation in [
        LinearOperatorOrientation::Normal,
        LinearOperatorOrientation::Transposed,
        LinearOperatorOrientation::ConjugateTransposed,
    ] {
        let rhs = if orientation == LinearOperatorOrientation::Normal {
            source.right_hand_side().to_vec()
        } else {
            vec![1.0; 64]
        };
        assert!(
            LinearSolveRequest::new(&FaerLinearSolver, plan(1))
                .solve_canonical_oriented(&source, &rhs, orientation)
                .is_err()
        );
        let solution = LinearSolveRequest::new(&FaerLinearSolver, plan(4))
            .solve_canonical_oriented(&source, &rhs, orientation)
            .unwrap();
        check(&solution, orientation);
    }
}

#[test]
fn prepared_and_ordinary_solves_share_residual_correction() {
    let source = fixture();
    let identity = PreparedLinearStructureIdentity::new(&b"scaled-tridiagonal"[..]).unwrap();
    let problem = source.linear_problem().unwrap();
    assert!(
        FaerPreparedSparseLu::new(plan(1))
            .solve(&identity, &problem)
            .is_err()
    );
    let mut prepared = FaerPreparedSparseLu::new(plan(4));
    for _ in 0..2 {
        let solution = prepared.solve(&identity, &problem).unwrap();
        check(&solution, LinearOperatorOrientation::Normal);
    }
}

#[test]
fn method_check_rejects_globally_accepted_initial_and_zero_residual_candidates() {
    use eqiora_solver::LinearSolverBackend;
    let system = fixture();
    let initial = vec![0.; 64];
    let requires_nonzero = |values: &[f64]| Ok(values.iter().any(|value| *value != 0.));
    // The large original absolute target accepts zero globally; the method
    // criterion requires one ordinary factor solve against the same equation.
    let policy = SolverPlan::new(LinearSolver::SparseLu, 0., 1e6, NonZeroUsize::MIN)
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
    let problem = system
        .linear_problem()
        .unwrap()
        .with_initial_guess(&initial)
        .unwrap()
        .with_candidate_check(&requires_nonzero);
    let solved = FaerLinearSolver.solve(&problem, policy).unwrap();
    assert_eq!(solved.report().completed_iterations(), 1);
    assert_eq!(solved.report().solver_plan(), policy);
    assert!(requires_nonzero(solved.values()).unwrap());

    let rejects = |_: &[f64]| Ok(false);
    let problem = system
        .linear_problem()
        .unwrap()
        .with_candidate_check(&rejects);
    assert_eq!(
        FaerLinearSolver
            .solve(&problem, plan(2))
            .unwrap_err()
            .code(),
        eqiora_core::diagnostic::codes::NUMERICAL_SOLVE_FAILED
    );
    let structural = |_: &[f64]| {
        Err(eqiora_core::Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_REALIZATION,
            "physical domain failure",
        ))
    };
    let problem = system
        .linear_problem()
        .unwrap()
        .with_candidate_check(&structural);
    assert_eq!(
        FaerLinearSolver.solve(&problem, policy).unwrap_err().code(),
        eqiora_core::diagnostic::codes::INVALID_REALIZATION
    );
}
