//! Exact existing candidate tuples and their admission checks.
use super::*;

pub(super) fn rejection_reason<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    profile: &SolverPlanningProfile,
    candidate: SolverCandidate<'_, S>,
) -> Option<&'static str> {
    let expected = expected_candidate(candidate.id());
    if S::DOMAIN != expected.scalar_domain {
        return Some("catalog.scalar-domain-mismatch");
    }
    if candidate.evidence_case() != expected.evidence_case {
        return Some("catalog.evidence-mismatch");
    }
    let Some(provider) = candidate.provider() else {
        return Some("catalog.provider-unavailable");
    };
    if provider != expected.provider {
        return Some("catalog.provider-mismatch");
    }
    if candidate.execution_provider() != Some(profile.execution_provider)
        || candidate.execution() != Some(profile.execution)
    {
        return Some("catalog.execution-mismatch");
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
    if !candidate
        .capabilities()
        .expect("available provider")
        .combinations()
        .contains(&required)
    {
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
        selection::CUDA_CG_ID | selection::CUDA_BICGSTAB_ID | selection::CUDA_MINRES_ID => {
            ExpectedCandidate {
                scalar_domain: ScalarDomain::Real,
                properties: match id {
                    selection::CUDA_CG_ID => LinearOperatorProperties::SymmetricPositiveDefinite,
                    selection::CUDA_MINRES_ID => LinearOperatorProperties::SymmetricIndefinite,
                    _ => LinearOperatorProperties::General,
                },
                evidence_case: "numerics.host-serial-solver-planning-private",
                provider: selection::CUDA_PROVIDER,
                algorithm: match id {
                    selection::CUDA_CG_ID => LinearSolver::ConjugateGradient,
                    selection::CUDA_MINRES_ID => LinearSolver::MinimumResidual,
                    _ => LinearSolver::BiConjugateGradientStabilized,
                },
                preconditioner: if id == selection::CUDA_CG_ID {
                    PreconditionerPolicy::Jacobi
                } else {
                    PreconditionerPolicy::Identity
                },
                reduction: ReductionPolicy::Fast,
            }
        }
        REFERENCE_SPD_MINRES_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_SPD_MINRES_ID)
        },
        REFERENCE_SYMMETRIC_MINRES_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_SYMMETRIC_MINRES_ID)
        },
        REFERENCE_INDEFINITE_MINRES_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_MINRES_ID)
        },
        REFERENCE_HERMITIAN_MINRES_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_HERMITIAN_MINRES_ID)
        },
        REFERENCE_HPD_MINRES_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_HPD_MINRES_ID)
        },
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
        REFERENCE_CG_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_CG_ID)
        },
        FAER_CG_IDENTITY_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Identity,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(FAER_CG_ID)
        },
        REFERENCE_SPD_MINRES_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_MINRES_ID)
        },
        REFERENCE_SPD_BICGSTAB_ID | REFERENCE_SPD_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(if id == REFERENCE_SPD_BICGSTAB_ID {
                REFERENCE_IDENTITY_ID
            } else {
                REFERENCE_ID
            })
        },
        FAER_SPD_BICGSTAB_ID | FAER_SPD_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(if id == FAER_SPD_BICGSTAB_ID {
                FAER_IDENTITY_ID
            } else {
                FAER_BICGSTAB_ID
            })
        },
        REFERENCE_INDEFINITE_BICGSTAB_ID | REFERENCE_INDEFINITE_BICGSTAB_JACOBI_ID => {
            ExpectedCandidate {
                properties: LinearOperatorProperties::SymmetricIndefinite,
                evidence_case: PLANNING_EVIDENCE,
                ..expected_candidate(if id == REFERENCE_INDEFINITE_BICGSTAB_ID {
                    REFERENCE_IDENTITY_ID
                } else {
                    REFERENCE_ID
                })
            }
        }
        REFERENCE_SYMMETRIC_MINRES_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::Symmetric,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(REFERENCE_MINRES_ID)
        },
        REFERENCE_SYMMETRIC_BICGSTAB_ID | REFERENCE_SYMMETRIC_BICGSTAB_JACOBI_ID => {
            ExpectedCandidate {
                properties: LinearOperatorProperties::Symmetric,
                evidence_case: PLANNING_EVIDENCE,
                ..expected_candidate(if id == REFERENCE_SYMMETRIC_BICGSTAB_ID {
                    REFERENCE_IDENTITY_ID
                } else {
                    REFERENCE_ID
                })
            }
        }
        FAER_SYMMETRIC_BICGSTAB_ID | FAER_SYMMETRIC_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::Symmetric,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(if id == FAER_SYMMETRIC_BICGSTAB_ID {
                FAER_IDENTITY_ID
            } else {
                FAER_BICGSTAB_ID
            })
        },
        FAER_INDEFINITE_BICGSTAB_ID | FAER_INDEFINITE_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricIndefinite,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(if id == FAER_INDEFINITE_BICGSTAB_ID {
                FAER_IDENTITY_ID
            } else {
                FAER_BICGSTAB_ID
            })
        },
        FAER_SYMMETRIC_LU_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::Symmetric,
            evidence_case: PLANNING_EVIDENCE,
            ..expected_candidate(FAER_SPARSE_LU_ID)
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
            evidence_case: PLANNING_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        REFERENCE_COMPLEX_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            ..expected_candidate(REFERENCE_COMPLEX_ID)
        },
        REFERENCE_COMPLEX_SYMMETRIC_ID | REFERENCE_COMPLEX_SYMMETRIC_JACOBI_ID => {
            ExpectedCandidate {
                properties: LinearOperatorProperties::ComplexSymmetric,
                ..expected_candidate(if id == REFERENCE_COMPLEX_SYMMETRIC_ID {
                    REFERENCE_COMPLEX_ID
                } else {
                    REFERENCE_COMPLEX_JACOBI_ID
                })
            }
        }
        REFERENCE_HERMITIAN_MINRES_ID => ExpectedCandidate {
            algorithm: LinearSolver::MinimumResidual,
            ..expected_candidate(REFERENCE_HERMITIAN_ID)
        },
        REFERENCE_HPD_MINRES_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            ..expected_candidate(REFERENCE_HERMITIAN_MINRES_ID)
        },
        REFERENCE_HERMITIAN_ID | REFERENCE_HERMITIAN_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::Hermitian,
            ..expected_candidate(if id == REFERENCE_HERMITIAN_ID {
                REFERENCE_COMPLEX_ID
            } else {
                REFERENCE_COMPLEX_JACOBI_ID
            })
        },
        REFERENCE_HPD_BICGSTAB_ID | REFERENCE_HPD_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            ..expected_candidate(if id == REFERENCE_HPD_BICGSTAB_ID {
                REFERENCE_COMPLEX_ID
            } else {
                REFERENCE_COMPLEX_JACOBI_ID
            })
        },
        REFERENCE_HPD_JACOBI_ID => ExpectedCandidate {
            preconditioner: PreconditionerPolicy::Jacobi,
            ..expected_candidate(REFERENCE_HPD_ID)
        },
        FAER_COMPLEX_BICGSTAB_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_COMPLEX_ID)
        },
        FAER_COMPLEX_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_COMPLEX_JACOBI_ID)
        },
        FAER_COMPLEX_SYMMETRIC_BICGSTAB_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_COMPLEX_SYMMETRIC_ID)
        },
        FAER_COMPLEX_SYMMETRIC_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_COMPLEX_SYMMETRIC_JACOBI_ID)
        },
        FAER_HERMITIAN_BICGSTAB_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HERMITIAN_ID)
        },
        FAER_HERMITIAN_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HERMITIAN_JACOBI_ID)
        },
        FAER_HPD_BICGSTAB_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HPD_BICGSTAB_ID)
        },
        FAER_HPD_BICGSTAB_JACOBI_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HPD_BICGSTAB_JACOBI_ID)
        },
        FAER_HPD_CG_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HPD_ID)
        },
        FAER_HPD_CG_JACOBI_ID => ExpectedCandidate {
            provider: FAER_PROVIDER,
            reduction: ReductionPolicy::Fast,
            ..expected_candidate(REFERENCE_HPD_JACOBI_ID)
        },
        FAER_COMPLEX_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::General,
            evidence_case: PLANNING_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        FAER_COMPLEX_SYMMETRIC_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::ComplexSymmetric,
            evidence_case: PLANNING_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        FAER_HERMITIAN_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::Hermitian,
            evidence_case: PLANNING_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_HPD_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            evidence_case: PLANNING_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_HPD_LU_ID => ExpectedCandidate {
            scalar_domain: ScalarDomain::Complex,
            properties: LinearOperatorProperties::HermitianPositiveDefinite,
            evidence_case: PLANNING_EVIDENCE,
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

pub(super) fn catalog_ids(
    domain: ScalarDomain,
    properties: LinearOperatorProperties,
) -> &'static [&'static str] {
    match (domain, properties) {
        (ScalarDomain::Real, LinearOperatorProperties::General) => &[
            REFERENCE_ID,
            REFERENCE_IDENTITY_ID,
            FAER_BICGSTAB_ID,
            FAER_IDENTITY_ID,
            FAER_SPARSE_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::SymmetricPositiveDefinite) => &[
            REFERENCE_CG_ID,
            REFERENCE_CG_JACOBI_ID,
            REFERENCE_SPD_MINRES_ID,
            REFERENCE_SPD_MINRES_JACOBI_ID,
            REFERENCE_SPD_BICGSTAB_ID,
            REFERENCE_SPD_BICGSTAB_JACOBI_ID,
            FAER_CG_IDENTITY_ID,
            FAER_CG_ID,
            FAER_SPD_BICGSTAB_ID,
            FAER_SPD_BICGSTAB_JACOBI_ID,
            FAER_SPD_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::SymmetricIndefinite) => &[
            REFERENCE_MINRES_ID,
            REFERENCE_INDEFINITE_MINRES_JACOBI_ID,
            REFERENCE_INDEFINITE_BICGSTAB_ID,
            REFERENCE_INDEFINITE_BICGSTAB_JACOBI_ID,
            FAER_INDEFINITE_BICGSTAB_ID,
            FAER_INDEFINITE_BICGSTAB_JACOBI_ID,
            FAER_INDEFINITE_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::Symmetric) => &[
            REFERENCE_SYMMETRIC_MINRES_ID,
            REFERENCE_SYMMETRIC_MINRES_JACOBI_ID,
            REFERENCE_SYMMETRIC_BICGSTAB_ID,
            REFERENCE_SYMMETRIC_BICGSTAB_JACOBI_ID,
            FAER_SYMMETRIC_BICGSTAB_ID,
            FAER_SYMMETRIC_BICGSTAB_JACOBI_ID,
            FAER_SYMMETRIC_LU_ID,
        ],
        (ScalarDomain::Complex, LinearOperatorProperties::General) => &[
            FAER_COMPLEX_BICGSTAB_ID,
            FAER_COMPLEX_BICGSTAB_JACOBI_ID,
            REFERENCE_COMPLEX_ID,
            REFERENCE_COMPLEX_JACOBI_ID,
            FAER_COMPLEX_LU_ID,
        ],
        (ScalarDomain::Complex, LinearOperatorProperties::ComplexSymmetric) => &[
            FAER_COMPLEX_SYMMETRIC_BICGSTAB_ID,
            FAER_COMPLEX_SYMMETRIC_BICGSTAB_JACOBI_ID,
            REFERENCE_COMPLEX_SYMMETRIC_ID,
            REFERENCE_COMPLEX_SYMMETRIC_JACOBI_ID,
            FAER_COMPLEX_SYMMETRIC_LU_ID,
        ],
        (ScalarDomain::Complex, LinearOperatorProperties::Hermitian) => &[
            FAER_HERMITIAN_BICGSTAB_ID,
            FAER_HERMITIAN_BICGSTAB_JACOBI_ID,
            REFERENCE_HERMITIAN_MINRES_ID,
            REFERENCE_HERMITIAN_MINRES_JACOBI_ID,
            REFERENCE_HERMITIAN_ID,
            REFERENCE_HERMITIAN_JACOBI_ID,
            FAER_HERMITIAN_LU_ID,
        ],
        (ScalarDomain::Complex, LinearOperatorProperties::HermitianPositiveDefinite) => &[
            FAER_HPD_BICGSTAB_ID,
            FAER_HPD_BICGSTAB_JACOBI_ID,
            FAER_HPD_CG_ID,
            FAER_HPD_CG_JACOBI_ID,
            REFERENCE_HPD_MINRES_ID,
            REFERENCE_HPD_MINRES_JACOBI_ID,
            REFERENCE_HPD_ID,
            REFERENCE_HPD_JACOBI_ID,
            REFERENCE_HPD_BICGSTAB_ID,
            REFERENCE_HPD_BICGSTAB_JACOBI_ID,
            FAER_HPD_LU_ID,
        ],
        _ => &[],
    }
}
