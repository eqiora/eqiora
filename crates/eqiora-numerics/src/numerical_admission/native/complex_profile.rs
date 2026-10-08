use super::*;
use eqiora_solver::{CanonicalCsrSystemView, CompleteCsrStorage, HostSerialSolverProfile};
use num_complex::Complex64 as C;

struct ScalarSystem;
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
        const VALUES: [C; 1] = [C::new(2., 1.)];
        &VALUES
    }
    fn right_hand_side(&self) -> &[C] {
        const RHS: [C; 1] = [C::new(3., 4.)];
        &RHS
    }
}

#[derive(Debug)]
struct Unexecuted;
impl eqiora_solver::LinearOperator for Unexecuted {
    type Scalar = C;
    fn rows(&self) -> usize {
        1
    }
    fn columns(&self) -> usize {
        1
    }
    fn apply(&self, _: &[C], _: &mut [C]) -> Result<(), Diagnostic> {
        panic!("structural rejection must precede operator work")
    }
}

fn policy() -> NativeLinearPolicy {
    NativeLinearPolicy::exact::<C>(
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(16).unwrap(),
        )
        .unwrap(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap()
}

#[test]
fn complex_exact_execution_uses_shared_structural_profile_and_plan_checks() {
    let mut policy = policy();
    policy.planning_profile = Some(HostSerialSolverProfile::general_canonical_csr());
    let checked = policy
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    let system =
        CanonicalCsrSystemView::new(&ScalarSystem, LinearOperatorProperties::General).unwrap();
    let problem = system.linear_problem().unwrap();
    // Independently: (2+i)(2+i)=3+4i. The exact provider still executes its complex path.
    let solution = checked.solve(&problem, policy.solver).unwrap();
    assert!((solution.values()[0] - C::new(2., 1.)).norm() < 1e-12);

    let changed = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-10,
        1e-14,
        NonZeroUsize::new(16).unwrap(),
    )
    .unwrap();
    assert!(
        checked
            .solve(&problem, changed)
            .unwrap_err()
            .message()
            .contains("changed the admitted exact solver plan")
    );
    assert!(
        checked
            .prepare_linear(changed)
            .unwrap_err()
            .message()
            .contains("changed the admitted exact solver plan")
    );

    let rhs = [C::new(3., 4.)];
    let unexecuted =
        eqiora_solver::LinearProblem::new(&Unexecuted, &rhs, LinearOperatorProperties::General)
            .unwrap();
    assert!(
        checked
            .solve(&unexecuted, policy.solver)
            .unwrap_err()
            .message()
            .contains("profile.canonical-csr-mismatch")
    );
    for (properties, diagonal, expected) in [
        (
            LinearOperatorProperties::Hermitian,
            Some(true),
            "profile.operator-properties-mismatch",
        ),
        (
            LinearOperatorProperties::General,
            Some(false),
            "profile.diagonal-availability-mismatch",
        ),
    ] {
        policy.planning_profile = Some(HostSerialSolverProfile::canonical_csr(
            properties, diagonal, None,
        ));
        let checked = policy
            .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, None)
            .unwrap();
        assert!(
            checked
                .solve(&problem, policy.solver)
                .unwrap_err()
                .message()
                .contains(expected)
        );
    }
}

#[test]
fn complex_profile_retains_exact_structure_and_rejects_automatic_ranking() {
    let structure = eqiora_solver::AlgebraicStructure::new([eqiora_core::Id::new()], []).unwrap();
    let foreign = eqiora_solver::AlgebraicStructure::new([eqiora_core::Id::new()], []).unwrap();
    let mut policy = policy();
    assert!(
        policy
            .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
            .is_err()
    );
    policy.planning_profile = Some(
        HostSerialSolverProfile::general_canonical_csr()
            .with_structure(structure.clone())
            .unwrap(),
    );
    policy
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
        .unwrap();
    for candidate in [None, Some(&foreign)] {
        assert!(
            policy
                .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, candidate)
                .unwrap_err()
                .message()
                .contains("profile.algebraic-structure-mismatch")
        );
    }
    policy.planning_objective = Some(SolverPlanningObjective::Robust);
    assert!(
        policy
            .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
            .unwrap_err()
            .message()
            .contains("exact complex solver policy")
    );
}

#[test]
fn prepared_complex_wrapper_reauthenticates_before_provider_work() {
    #[derive(Debug)]
    struct UnexecutedPrepared;
    impl eqiora_solver::PreparedLinearSolver<C> for UnexecutedPrepared {
        fn solve(
            &mut self,
            _: &eqiora_solver::PreparedLinearStructureIdentity,
            _: &eqiora_solver::LinearProblem<'_, C>,
        ) -> Result<eqiora_solver::LinearSolution<C>, Diagnostic> {
            panic!("profile rejection must precede prepared provider work")
        }
    }
    use eqiora_solver::PreparedLinearSolver;
    let mut checked = ProfileCheckedPreparedLinear {
        prepared: Box::new(UnexecutedPrepared),
        profile: Some(HostSerialSolverProfile::general_canonical_csr()),
    };
    let structure =
        eqiora_solver::PreparedLinearStructureIdentity::new(&b"test-ordering"[..]).unwrap();
    let rhs = [C::new(1., 0.)];
    let problem =
        eqiora_solver::LinearProblem::new(&Unexecuted, &rhs, LinearOperatorProperties::General)
            .unwrap();
    assert!(
        checked
            .solve(&structure, &problem)
            .unwrap_err()
            .message()
            .contains("profile.canonical-csr-mismatch")
    );
}
