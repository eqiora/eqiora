use super::*;
use eqiora_solver::{
    AlgebraicBlock, AlgebraicStructure, CanonicalCsrSystemView, CsrStorage,
    FixedFactorizationChoice, LinearOperator, LinearOperatorOrientation, LinearOperatorProperties,
    LinearSolver, PreconditionerPolicy, ReductionPolicy, UpperDiagonalSchurConfig,
};

struct Matrix {
    values: Vec<f64>,
    rhs: Vec<f64>,
    offsets: Vec<usize>,
    indices: Vec<usize>,
}
impl CsrStorage for Matrix {
    fn rows(&self) -> usize {
        4
    }
    fn columns(&self) -> usize {
        4
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.indices
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}
fn fixture(a: f64, scale: f64) -> CanonicalCsrSystemView {
    // Manufactured exact dyadic original-coordinate solution; independent K action.
    let local = [
        [a, 1., 1., 0.],
        [0., 4., 2., 1.],
        [3., 0., 5., 1.],
        [0., 2., 1., 3.],
    ];
    let map = [2, 0, 3, 1];
    let x = [1. / 8., -1. / 4., 1. / 4., 1. / 8.];
    let mut values = [0.; 16];
    let mut rhs = vec![0.; 4];
    for i in 0..4 {
        for j in 0..4 {
            values[4 * map[i] + map[j]] = local[i][j];
            rhs[map[i]] += local[i][j] * x[map[j]] * scale;
        }
    }
    let mut sparse_values = Vec::new();
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    for row in 0..4 {
        for col in 0..4 {
            let value = values[4 * row + col];
            if value != 0. {
                indices.push(col);
                sparse_values.push(value);
            }
        }
        offsets.push(sparse_values.len());
    }
    let matrix = Matrix {
        values: sparse_values,
        rhs,
        offsets,
        indices,
    };
    CanonicalCsrSystemView::new(&matrix, &matrix.rhs, LinearOperatorProperties::General).unwrap()
}
fn binding() -> (SolverPlan, AlgebraicStructure) {
    let a = eqiora_core::Id::new();
    let b = eqiora_core::Id::new();
    let groups = [
        vec![AlgebraicBlock::Field(a)],
        vec![AlgebraicBlock::Field(b)],
    ];
    let structure = AlgebraicStructure::new([a, b], [])
        .unwrap()
        .with_coordinates(
            4,
            vec![
                (AlgebraicBlock::Field(a), vec![2, 0]),
                (AlgebraicBlock::Field(b), vec![3, 1]),
            ],
        )
        .unwrap();
    let factor = FixedFactorizationChoice::new(LinearSolver::SparseLu, "eqiora.faer").unwrap();
    let policy = PreconditionerPolicy::upper_diagonal_schur(
        UpperDiagonalSchurConfig::new(groups, factor.clone(), factor, 4, 3).unwrap(),
    );
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        64.try_into().unwrap(),
    )
    .unwrap()
    .with_preconditioner(policy)
    .with_reduction(ReductionPolicy::Fast);
    (plan, structure)
}
fn residual(system: &CanonicalCsrSystemView, solution: &LinearSolution) {
    let mut image = vec![0.; 4];
    system.apply(solution.values(), &mut image).unwrap();
    let norm = image
        .iter()
        .zip(system.right_hand_side())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(norm <= solution.report().residual_target());
}
#[test]
fn aggregate_reuse_refresh_and_rollback_follow_original_acceptance() {
    let (plan, structure) = binding();
    let identity = PreparedLinearStructureIdentity::new(&b"analytic-two-block"[..]).unwrap();
    let system = fixture(2., 1.);
    let problem = system
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    let mut prepared = FaerPreparedSparseLu::new(plan.clone());
    prepared.begin_action().unwrap();
    let solved = prepared.solve(&identity, &problem).unwrap();
    residual(&system, &solved);
    assert_eq!(solved.report().solver_plan(), &plan);
    prepared.commit_action().unwrap();
    let first = Arc::clone(prepared.ready.as_ref().unwrap());
    let changed_rhs = fixture(2., 0.5);
    let rhs_problem = changed_rhs
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    prepared.begin_action().unwrap();
    residual(
        &changed_rhs,
        &prepared.solve(&identity, &rhs_problem).unwrap(),
    );
    assert!(Arc::ptr_eq(prepared.tentative.as_ref().unwrap(), &first));
    prepared.commit_action().unwrap();
    let changed = fixture(3., 1.);
    let changed_problem = changed
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    prepared.begin_action().unwrap();
    residual(
        &changed,
        &prepared.solve(&identity, &changed_problem).unwrap(),
    );
    let tentative = prepared.tentative.as_ref().unwrap();
    assert!(!Arc::ptr_eq(tentative, &first));
    assert!(
        tentative
            .block
            .as_ref()
            .unwrap()
            .shares_symbolic_with(first.block.as_ref().unwrap())
    );
    // Later method rejection prevents committing the whole tentative pair.
    let reject = |_: &[f64]| Ok(false);
    let rejected = changed
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap()
        .with_candidate_check(&reject);
    assert!(prepared.solve(&identity, &rejected).is_err());
    assert!(prepared.commit_action().is_err());
    prepared.discard_action().unwrap();
    assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &first));
    // Cached factors cannot bypass orientation admission, including zero RHS.
    let transposed =
        eqiora_solver::Oriented::new(&system, LinearOperatorOrientation::Transposed).unwrap();
    let wrong = LinearProblem::new(&transposed, &[0.; 4], LinearOperatorProperties::General)
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    prepared.begin_action().unwrap();
    assert!(prepared.solve(&identity, &wrong).is_err());
    prepared.discard_action().unwrap();
    assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &first));
}

#[test]
fn cold_and_prepared_admission_reject_missing_provider_orientation_and_singular_updates() {
    use eqiora_solver::LinearSolverBackend;
    let (plan, structure) = binding();
    let system = fixture(2., 1.);
    let problem = system
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    let solved = crate::FaerLinearSolver
        .solve(&problem, plan.clone())
        .unwrap();
    residual(&system, &solved);
    assert_eq!(solved.report().solver_provider(), FAER_SOLVER_PROVIDER);
    assert!(
        crate::FaerLinearSolver
            .solve(&system.linear_problem().unwrap(), plan.clone())
            .is_err()
    );
    let PreconditionerPolicy::UpperDiagonalSchur(config) = plan.preconditioner() else {
        unreachable!()
    };
    let bad = PreconditionerPolicy::upper_diagonal_schur(
        UpperDiagonalSchurConfig::new(
            config.groups().clone(),
            FixedFactorizationChoice::new(LinearSolver::SparseLu, "missing.provider").unwrap(),
            config.schur_factorization().clone(),
            4,
            3,
        )
        .unwrap(),
    );
    let zero = fixture(2., 0.);
    let zero_problem = zero
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    assert!(
        crate::FaerLinearSolver
            .solve(&zero_problem, plan.clone().with_preconditioner(bad))
            .is_err()
    );
    let identity = PreparedLinearStructureIdentity::new(&b"singular-update-retention"[..]).unwrap();
    let mut prepared = FaerPreparedSparseLu::new(plan);
    prepared.begin_action().unwrap();
    prepared.solve(&identity, &problem).unwrap();
    prepared.commit_action().unwrap();
    let accepted = Arc::clone(prepared.ready.as_ref().unwrap());
    let singular = fixture(0., 1.);
    let singular_problem = singular
        .linear_problem()
        .unwrap()
        .with_algebraic_structure(&structure)
        .unwrap();
    prepared.begin_action().unwrap();
    assert!(prepared.solve(&identity, &singular_problem).is_err());
    assert!(prepared.commit_action().is_err());
    prepared.discard_action().unwrap();
    assert!(Arc::ptr_eq(prepared.ready.as_ref().unwrap(), &accepted));
}
