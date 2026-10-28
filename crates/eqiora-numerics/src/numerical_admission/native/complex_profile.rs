use super::*;
use crate::numerical_admission::linear_policy::{
    PreparedLinearAction, ProfileCheckedPreparedLinear,
};
use eqiora_solver::{CanonicalCsrSystemView, CsrStorage, SolverPlanningProfile};
use num_complex::Complex64 as C;

struct ScalarSystem;
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
        const VALUES: [C; 1] = [C::new(2., 1.)];
        &VALUES
    }
}

impl ScalarSystem {
    pub(crate) fn rhs(&self) -> &[C] {
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
    assert!(
        policy
            .checked_backend::<f64>(&REFERENCE_LINEAR_SOLVER, None)
            .is_err()
    );
    let real = NativeLinearPolicy::exact::<f64>(policy.solver, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert!(
        real.checked_complex_backend(&REFERENCE_LINEAR_SOLVER, None)
            .is_err()
    );
    policy.planning_profile = Some(SolverPlanningProfile::general_canonical_csr());
    let checked = policy
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    let system = CanonicalCsrSystemView::new(
        &ScalarSystem,
        ScalarSystem.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
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
        policy.planning_profile = Some(SolverPlanningProfile::canonical_csr(
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
fn parameter_policy_rebinds_only_after_actual_system_and_capability_checks() {
    let original = policy();
    let system = CanonicalCsrSystemView::new(
        &ScalarSystem,
        ScalarSystem.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let coordinates = system
        .real_coordinate_system(
            eqiora_solver::LinearOperatorOrientation::Normal,
            system.right_hand_side(),
        )
        .unwrap();
    let real = original.at_parameter_problem(&coordinates, None).unwrap();
    assert_eq!(real.scalar_domain, eqiora_core::ScalarDomain::Real);
    assert_eq!(real.solver, original.solver);
    assert_eq!(real.provider, original.provider);
    assert_eq!(real.capabilities, original.capabilities);
    real.checked_backend::<f64>(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    // Ordinary execution still rejects a mismatched numerical domain.
    assert!(
        original
            .checked_backend::<f64>(&REFERENCE_LINEAR_SOLVER, None)
            .is_err()
    );
    let complex = real.at_parameter_problem(&system, None).unwrap();
    assert_eq!(complex, original);
    let checked = complex
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    let solved = checked
        .solve(&system.linear_problem().unwrap(), complex.solver)
        .unwrap();
    assert!((solved.values()[0] - C::new(2., 1.)).norm() < 1e-12);

    let mut unavailable = real.clone();
    unavailable.capabilities = SolverCapabilities::exact([eqiora_solver::SolverCapability {
        scalar_domain: eqiora_core::ScalarDomain::Real,
        algorithm: real.solver.algorithm(),
        operator_properties: LinearOperatorProperties::General,
        preconditioner: real.solver.preconditioner(),
        reduction: real.solver.reduction(),
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    assert!(unavailable.at_parameter_problem(&system, None).is_err());
    let mut wrong_profile = real;
    wrong_profile.planning_profile = Some(SolverPlanningProfile::canonical_csr(
        LinearOperatorProperties::Hermitian,
        None,
        None,
    ));
    assert!(
        wrong_profile
            .at_parameter_problem(&system, None)
            .unwrap_err()
            .message()
            .contains("profile.operator-properties-mismatch")
    );
}

#[test]
fn complex_profile_retains_exact_structure_and_rejects_incoherent_planning() {
    let structure = eqiora_solver::AlgebraicStructure::new([eqiora_core::Id::new()], []).unwrap();
    let foreign = eqiora_solver::AlgebraicStructure::new([eqiora_core::Id::new()], []).unwrap();
    let mut policy = policy();
    assert!(
        policy
            .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
            .is_err()
    );
    policy.planning_profile = Some(
        SolverPlanningProfile::general_canonical_csr()
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
            .contains("coherent solver planning audit")
    );
}

#[test]
fn prepared_complex_wrapper_reauthenticates_before_provider_work() {
    #[derive(Debug)]
    struct UnexecutedPrepared {
        active: bool,
    }
    impl eqiora_solver::PreparedLinearSolver<C> for UnexecutedPrepared {
        fn begin_action(&mut self) -> Result<(), Diagnostic> {
            assert!(!self.active);
            self.active = true;
            Ok(())
        }
        fn commit_action(&mut self) -> Result<(), Diagnostic> {
            panic!("profile rejection must prevent provider commit")
        }
        fn discard_action(&mut self) -> Result<(), Diagnostic> {
            assert!(self.active);
            self.active = false;
            Ok(())
        }
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
        prepared: Box::new(UnexecutedPrepared { active: false }),
        plan: policy().solver,
        provider: policy().provider,
        profile: Some(SolverPlanningProfile::general_canonical_csr()),
        action: PreparedLinearAction::Idle,
    };
    let structure =
        eqiora_solver::PreparedLinearStructureIdentity::new(&b"test-ordering"[..]).unwrap();
    let rhs = [C::new(1., 0.)];
    let problem =
        eqiora_solver::LinearProblem::new(&Unexecuted, &rhs, LinearOperatorProperties::General)
            .unwrap();
    assert!(checked.solve(&structure, &problem).is_err());
    assert_eq!(checked.action, PreparedLinearAction::Idle);
    checked.begin_action().unwrap();
    assert!(checked.begin_action().is_err());
    assert!(
        checked
            .solve(&structure, &problem)
            .unwrap_err()
            .message()
            .contains("profile.canonical-csr-mismatch")
    );
    assert_eq!(checked.action, PreparedLinearAction::Failed);
    assert!(checked.commit_action().is_err());
    assert!(checked.solve(&structure, &problem).is_err());
    checked.discard_action().unwrap();
    assert_eq!(checked.action, PreparedLinearAction::Idle);
    assert!(checked.discard_action().is_err());
    checked.begin_action().unwrap();
    checked.discard_action().unwrap();
}

#[test]
fn automatic_complex_provenance_survives_parameter_domain_rebinding() {
    let request = CommonLinearRequest::program_controlled(
        1e-12,
        1e-14,
        NonZeroUsize::new(16).unwrap(),
        SolverPlanningObjective::Robust,
    )
    .unwrap();
    let original = super::super::solver_planning::resolve_complex_linear(
        request,
        LinearOperatorProperties::General,
        None,
        None,
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(
        original.selected_candidate_id,
        Some("eqiora.reference.bicgstab-general-identity-reproducible-complex-f64")
    );
    let system = CanonicalCsrSystemView::new(
        &ScalarSystem,
        ScalarSystem.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let coordinates = system
        .real_coordinate_system(
            eqiora_solver::LinearOperatorOrientation::Normal,
            system.right_hand_side(),
        )
        .unwrap();
    let real = original.at_parameter_problem(&coordinates, None).unwrap();
    assert_eq!(real.scalar_domain, eqiora_core::ScalarDomain::Real);
    assert_eq!(real.selected_candidate_id, original.selected_candidate_id);
    assert_eq!(real.planning_reasons, original.planning_reasons);
    assert_eq!(real.solver, original.solver);
    assert_eq!(real.provider, original.provider);
    assert_eq!(real.at_parameter_problem(&system, None).unwrap(), original);
    let decision = SolverSelection::resolve_host::<C>(
        SolverPlanningProfile::canonical_csr(LinearOperatorProperties::General, None, None),
        SolverPlanningObjective::Robust,
        1e-12,
        1e-14,
        NonZeroUsize::new(16).unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let wrong_domain =
        NativeLinearPolicy::exact::<f64>(decision.solver_plan(), &REFERENCE_LINEAR_SOLVER).unwrap();
    assert!(wrong_domain.with_planning(&decision).is_err());
}
