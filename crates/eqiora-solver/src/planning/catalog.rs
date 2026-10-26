//! Exact existing candidate tuples and their admission checks.
use super::*;

pub(super) fn rejection_reason<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    profile: &HostSerialSolverProfile,
    candidate: HostSerialSolverCandidate<'_, S>,
) -> Option<&'static str> {
    let expected = expected_candidate(candidate.id());
    if S::DOMAIN != expected.scalar_domain {
        return Some("catalog.scalar-domain-mismatch");
    }
    if candidate.evidence_case() != expected.evidence_case {
        return Some("catalog.evidence-mismatch");
    }
    let Some(backend) = candidate.backend else {
        return Some("catalog.provider-unavailable");
    };
    if backend.provider() != expected.provider {
        return Some("catalog.provider-mismatch");
    }
    if !plan_tuple_matches(candidate.plan(), expected) {
        return Some("catalog.plan-mismatch");
    }
    if profile.facts.properties != expected.properties {
        return Some(match expected.properties {
            LinearOperatorProperties::General => "profile.general-required",
            LinearOperatorProperties::SymmetricPositiveDefinite => "profile.spd-required",
            LinearOperatorProperties::SymmetricIndefinite => {
                "profile.symmetric-indefinite-required"
            }
            LinearOperatorProperties::Symmetric => "profile.symmetric-required",
            LinearOperatorProperties::ComplexSymmetric => "profile.complex-symmetric-required",
            LinearOperatorProperties::Hermitian => "profile.hermitian-required",
            LinearOperatorProperties::HermitianPositiveDefinite => "profile.hpd-required",
        });
    }
    if profile.facts.orientation != LinearOperatorOrientation::Normal {
        return Some("profile.normal-required");
    }
    if !profile.facts.canonical_csr {
        return Some("profile.canonical-csr-required");
    }
    if let Some(reason) = profile.plan_rejection(candidate.plan()) {
        return Some(reason);
    }
    let required = SolverCapability {
        scalar_domain: expected.scalar_domain,
        algorithm: expected.algorithm,
        operator_properties: expected.properties,
        preconditioner: expected.preconditioner,
        reduction: expected.reduction,
        scalar_type: ScalarType::F64,
    };
    if !backend.capabilities().combinations().contains(&required) {
        return Some("capability.exact-tuple-required");
    }
    None
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExpectedCandidate {
    pub(super) scalar_domain: ScalarDomain,
    pub(super) properties: LinearOperatorProperties,
    pub(super) evidence_case: &'static str,
    pub(super) provider: SolverProvider,
    pub(super) algorithm: LinearSolver,
    pub(super) preconditioner: PreconditionerPolicy,
    pub(super) reduction: ReductionPolicy,
}

pub(super) fn expected_candidate(id: &str) -> ExpectedCandidate {
    match id {
        REFERENCE_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::General,
            evidence_case: REFERENCE_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_BICGSTAB_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::General,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_IDENTITY_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::General,
            evidence_case: REFERENCE_IDENTITY_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_IDENTITY_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::General,
            evidence_case: FAER_IDENTITY_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        FAER_SPARSE_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::General,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_CG_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_CG_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
        },
        FAER_SPD_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_MINRES_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::SymmetricIndefinite,
            evidence_case: "numerics.reference-minres-w64",
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::MinimumResidual,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_INDEFINITE_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Real,
            properties: LinearOperatorProperties::SymmetricIndefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_COMPLEX_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::General,
            evidence_case: COMPLEX_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_COMPLEX_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::General,
            evidence_case: COMPLEX_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        FAER_COMPLEX_SYMMETRIC_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::ComplexSymmetric,
            evidence_case: COMPLEX_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        FAER_HERMITIAN_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::Hermitian,
            evidence_case: COMPLEX_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_HPD_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            evidence_case: COMPLEX_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_HPD_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            evidence_case: COMPLEX_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        _ => unreachable!("inventory validation rejects unknown candidate IDs"),
    }
}

fn plan_tuple_matches(plan: SolverPlan, expected: ExpectedCandidate) -> bool {
    plan.algorithm() == expected.algorithm
        && plan.preconditioner() == expected.preconditioner
        && plan.reduction() == expected.reduction
}
