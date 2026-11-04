use std::num::NonZeroUsize;

use eqiora_solver::{
    CanonicalCsrSystemView, CsrStorage, LinearOperatorOrientation, LinearOperatorProperties,
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, PreparedLinearSolver,
    PreparedLinearStructureIdentity, ReductionPolicy, SolverPlan,
};

use crate::FaerLinearSolver;
use crate::prepared_sparse_lu::FaerPreparedSparseLu;

struct Storage<S = f64> {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<S>,
}

impl<S> CsrStorage<S> for Storage<S> {
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
    fn values(&self) -> &[S] {
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
    let mut constrained = FaerPreparedSparseLu::new(plan(1));
    constrained.begin_action().unwrap();
    assert!(constrained.solve(&identity, &problem).is_err());
    constrained.discard_action().unwrap();
    let mut prepared = FaerPreparedSparseLu::new(plan(4));
    for _ in 0..2 {
        prepared.begin_action().unwrap();
        let solution = prepared.solve(&identity, &problem).unwrap();
        check(&solution, LinearOperatorOrientation::Normal);
        prepared.commit_action().unwrap();
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

fn loose_plan(backsolves: usize) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::SparseLu,
        1e-8,
        1e-8,
        NonZeroUsize::new(backsolves).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast)
}

#[test]
fn accepted_nonzero_factor_solves_get_one_safeguarded_correction_within_budget() {
    let source = fixture();
    for orientation in [
        LinearOperatorOrientation::Normal,
        LinearOperatorOrientation::Transposed,
        LinearOperatorOrientation::ConjugateTransposed,
    ] {
        let rhs = if orientation == LinearOperatorOrientation::Normal {
            source.right_hand_side().to_vec()
        } else {
            vec![1.; 64]
        };
        let one = LinearSolveRequest::new(&FaerLinearSolver, loose_plan(1))
            .solve_canonical_oriented(&source, &rhs, orientation)
            .unwrap();
        let two = LinearSolveRequest::new(&FaerLinearSolver, loose_plan(2))
            .solve_canonical_oriented(&source, &rhs, orientation)
            .unwrap();
        assert_eq!(one.report().completed_iterations(), 1);
        assert_eq!(two.report().completed_iterations(), 2);
        assert!(two.report().true_residual_norm() <= one.report().true_residual_norm());
        check(&two, orientation);
    }
    let identity = PreparedLinearStructureIdentity::new(&b"accepted-tridiagonal"[..]).unwrap();
    for budget in [1, 2] {
        let mut prepared = FaerPreparedSparseLu::new(loose_plan(budget));
        for _ in 0..2 {
            prepared.begin_action().unwrap();
            let solution = prepared
                .solve(&identity, &source.linear_problem().unwrap())
                .unwrap();
            assert_eq!(solution.report().completed_iterations(), budget);
            if budget == 2 {
                check(&solution, LinearOperatorOrientation::Normal);
            }
            prepared.commit_action().unwrap();
        }
    }
}

#[test]
fn optional_correction_preserves_an_accepted_candidate_and_propagates_method_errors() {
    use eqiora_solver::LinearSolverBackend;
    let source = fixture();
    let first = FaerLinearSolver
        .solve(&source.linear_problem().unwrap(), loose_plan(1))
        .unwrap();
    // This is a method-selection falsifier, not a numerical oracle: only the
    // already accepted candidate is in the caller's declared admissible set.
    let only_original = |values: &[f64]| Ok(values == first.values());
    let problem = source
        .linear_problem()
        .unwrap()
        .with_candidate_check(&only_original);
    let retained = FaerLinearSolver.solve(&problem, loose_plan(2)).unwrap();
    assert_eq!(retained.values(), first.values());
    assert_eq!(
        retained.report().true_residual_norm(),
        first.report().true_residual_norm()
    );
    assert_eq!(retained.report().completed_iterations(), 2);
    let fails_on_trial = |values: &[f64]| {
        if values == first.values() {
            Ok(true)
        } else {
            Err(eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "trial method failure",
            ))
        }
    };
    let problem = source
        .linear_problem()
        .unwrap()
        .with_candidate_check(&fails_on_trial);
    assert_eq!(
        FaerLinearSolver
            .solve(&problem, loose_plan(2))
            .unwrap_err()
            .message(),
        "trial method failure"
    );
}

#[test]
fn exact_factor_solve_and_accepted_initial_guess_do_not_spend_a_correction() {
    use eqiora_solver::LinearSolverBackend;
    let storage = Storage {
        offsets: (0..=64).collect(),
        columns: (0..64).collect(),
        values: vec![2.; 64],
    };
    let source =
        CanonicalCsrSystemView::new(&storage, &[2.; 64], LinearOperatorProperties::General)
            .unwrap();
    let solution = FaerLinearSolver
        .solve(&source.linear_problem().unwrap(), loose_plan(2))
        .unwrap();
    assert_eq!(solution.values(), &[1.; 64]);
    assert_eq!(solution.report().true_residual_norm(), 0.);
    assert_eq!(solution.report().completed_iterations(), 1);
    let problem = source
        .linear_problem()
        .unwrap()
        .with_initial_guess(&[1.; 64])
        .unwrap();
    let initial = FaerLinearSolver.solve(&problem, loose_plan(2)).unwrap();
    assert_eq!(initial.report().completed_iterations(), 0);
}

#[test]
fn complex_optional_refinement_keeps_normal_transpose_and_adjoint_meaning() {
    use faer::complex::Complex64 as C;
    let real = fixture();
    let storage = Storage {
        offsets: real.row_offsets().to_vec(),
        columns: real.column_indices().to_vec(),
        values: real.values().iter().map(|x| C::new(*x, *x)).collect(),
    };
    // A=(1+i) D T, with exactly dyadic coefficients and the independently
    // derived tridiagonal coordinates z_j from fixture(). Form each oriented
    // RHS by exact dyadic complex multiplication, not a solver output.
    let expected = (0..64)
        .map(|j| {
            C::new(
                f64::from(u32::try_from(j * (63 - j) + 32).unwrap()) / 128.,
                0.,
            )
        })
        .collect::<Vec<_>>();
    for orientation in [
        LinearOperatorOrientation::Normal,
        LinearOperatorOrientation::Transposed,
        LinearOperatorOrientation::ConjugateTransposed,
    ] {
        let mut rhs = vec![C::new(0., 0.); 64];
        for row in 0..64 {
            for k in storage.offsets[row]..storage.offsets[row + 1] {
                let col = storage.columns[k];
                let a = storage.values[k];
                match orientation {
                    LinearOperatorOrientation::Normal => rhs[row] += a * expected[col],
                    LinearOperatorOrientation::Transposed => rhs[col] += a * expected[row],
                    LinearOperatorOrientation::ConjugateTransposed => {
                        rhs[col] += a.conj() * expected[row]
                    }
                }
            }
        }
        let source =
            CanonicalCsrSystemView::new(&storage, &rhs, LinearOperatorProperties::General).unwrap();
        let one = LinearSolveRequest::new(&FaerLinearSolver, loose_plan(1))
            .solve_canonical_oriented(&source, &rhs, orientation)
            .unwrap();
        let two = LinearSolveRequest::new(&FaerLinearSolver, loose_plan(2))
            .solve_canonical_oriented(&source, &rhs, orientation)
            .unwrap();
        assert_eq!(one.report().completed_iterations(), 1);
        assert_eq!(two.report().completed_iterations(), 2);
        assert!(two.report().true_residual_norm() <= one.report().true_residual_norm());
        for (actual, expected) in two.values().iter().zip(&expected) {
            assert!((*actual - *expected).norm() <= 1e-12);
        }
    }
}
