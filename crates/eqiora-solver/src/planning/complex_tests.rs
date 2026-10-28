//! Adversarial typed admission does no numerical work and cannot change catalogs.
use super::*;
use num_complex::Complex64 as C;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct TypedBackend {
    provider: SolverProvider,
    capabilities: crate::SolverCapabilities,
    calls: AtomicUsize,
}
impl crate::LinearSolverBackend<C> for TypedBackend {
    fn provider(&self) -> SolverProvider {
        self.provider
    }
    fn capabilities(&self) -> crate::SolverCapabilities {
        self.capabilities.clone()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_, C>,
        _: SolverPlan,
        _: &dyn crate::ReplicatedLinearExecution<C>,
    ) -> Result<LinearSolution<C>, Diagnostic> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        panic!("catalog rejection must precede numerical work")
    }
}
fn controls(algorithm: LinearSolver, reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(100).unwrap())
        .unwrap()
        .with_reduction(reduction)
}
#[test]
pub(super) fn typed_capabilities_and_requested_inventory_precede_ranking() {
    let backend = TypedBackend {
        provider: REFERENCE_PROVIDER,
        capabilities: crate::SolverCapabilities::exact([SolverCapability {
            scalar_domain: ScalarDomain::Real,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .unwrap(),
        calls: AtomicUsize::new(0),
    };
    let profile = HostSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
    let reference = HostSerialSolverCandidate::planned(
        REFERENCE_COMPLEX_ID,
        PLANNING_EVIDENCE,
        controls(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Reproducible,
        ),
        Some(&backend as &dyn crate::LinearSolverBackend<C>),
    );
    let direct = HostSerialSolverCandidate::planned(
        FAER_COMPLEX_LU_ID,
        PLANNING_EVIDENCE,
        controls(LinearSolver::SparseLu, ReductionPolicy::Fast),
        None,
    );
    let iterative_faer = HostSerialSolverCandidate::planned(
        FAER_COMPLEX_BICGSTAB_ID,
        PLANNING_EVIDENCE,
        controls(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Fast,
        ),
        None,
    );
    let jacobi_faer = HostSerialSolverCandidate::planned(
        FAER_COMPLEX_BICGSTAB_JACOBI_ID,
        PLANNING_EVIDENCE,
        iterative_faer
            .plan()
            .with_preconditioner(PreconditionerPolicy::Jacobi),
        None,
    );
    let reference_jacobi = HostSerialSolverCandidate::planned(
        REFERENCE_COMPLEX_JACOBI_ID,
        PLANNING_EVIDENCE,
        reference
            .plan()
            .with_preconditioner(PreconditionerPolicy::Jacobi),
        reference.backend,
    );
    let error = resolve_candidates(
        profile.clone(),
        SolverPlanningObjective::Fast,
        &[
            reference,
            reference_jacobi,
            direct,
            iterative_faer,
            jacobi_faer,
        ],
    )
    .unwrap_err();
    assert_eq!(
        error.message(),
        format!(
            "{POLICY_ID} no admitted candidate; trace=[{FAER_COMPLEX_BICGSTAB_ID}=catalog.provider-unavailable,{FAER_COMPLEX_BICGSTAB_JACOBI_ID}=catalog.provider-unavailable,{FAER_COMPLEX_LU_ID}=catalog.provider-unavailable,{REFERENCE_COMPLEX_ID}=capability.exact-tuple-required,{REFERENCE_COMPLEX_JACOBI_ID}=profile.complete-diagonal-required]"
        )
    );
    // A complete real General catalog must not be accepted as the complex inventory.
    let foreign = [
        REFERENCE_ID,
        REFERENCE_IDENTITY_ID,
        FAER_BICGSTAB_ID,
        FAER_IDENTITY_ID,
        FAER_SPARSE_LU_ID,
    ]
    .map(|id| {
        HostSerialSolverCandidate::planned(
            id,
            PLANNING_EVIDENCE,
            reference.plan(),
            Some(&backend as &dyn crate::LinearSolverBackend<C>),
        )
    });
    let error = resolve_candidates(profile, SolverPlanningObjective::Robust, &foreign).unwrap_err();
    assert!(error.message().contains("catalog.missing-id"));
    // Nor may a complete complex General catalog reinterpret an asserted HPD class.
    let error = resolve_candidates(
        HostSolverProfile::canonical_csr(
            LinearOperatorProperties::HermitianPositiveDefinite,
            None,
            None,
        ),
        SolverPlanningObjective::Robust,
        &[
            reference,
            reference_jacobi,
            direct,
            iterative_faer,
            jacobi_faer,
        ],
    )
    .unwrap_err();
    assert!(error.message().contains("catalog.missing-id"));
    let changed = HostSerialSolverCandidate::planned(
        REFERENCE_COMPLEX_ID,
        PLANNING_EVIDENCE,
        reference
            .plan()
            .with_preconditioner(PreconditionerPolicy::Jacobi),
        reference.backend,
    );
    let error = resolve_candidates(
        HostSolverProfile::canonical_csr(LinearOperatorProperties::General, Some(true), None),
        SolverPlanningObjective::Robust,
        &[
            changed,
            reference_jacobi,
            direct,
            iterative_faer,
            jacobi_faer,
        ],
    )
    .unwrap_err();
    assert!(error.message().contains("catalog.plan-mismatch"));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[test]
pub(super) fn typed_provider_identity_and_operator_class_are_independent() {
    let field = eqiora_core::Id::new();
    let structure = crate::AlgebraicStructure::new(
        [field],
        [crate::AlgebraicConstraint::ZeroIntegral { field }],
    )
    .unwrap();
    let error = HostSolverProfile::canonical_csr(
        LinearOperatorProperties::HermitianPositiveDefinite,
        None,
        None,
    )
    .with_structure(structure)
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("gauge-multiplier-incompatible-with-positive-definite")
    );

    let backend = TypedBackend {
        provider: SolverProvider::new(BackendId::new("eqiora.foreign-complex"), "1", &[]),
        capabilities: crate::SolverCapabilities::reference(),
        calls: AtomicUsize::new(0),
    };
    let error = ResolvedHostSolverPlan::<C>::resolve(
        HostSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None),
        SolverPlanningObjective::Fast,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
        &backend,
        None,
    )
    .unwrap_err();
    assert!(error.message().contains("catalog.provider-mismatch"));
    assert!(error.message().contains("catalog.provider-unavailable"));
    for properties in [
        LinearOperatorProperties::Symmetric,
        LinearOperatorProperties::SymmetricIndefinite,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    ] {
        let error = ResolvedHostSolverPlan::<C>::resolve(
            HostSolverProfile::canonical_csr(properties, None, None),
            SolverPlanningObjective::Robust,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &backend,
            None,
        )
        .unwrap_err();
        assert!(error.message().contains("catalog.unsupported-profile"));
    }
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}
