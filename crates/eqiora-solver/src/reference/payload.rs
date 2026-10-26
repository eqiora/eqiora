//! Requested numeric storage of the admitted real serial reference kernels.
use crate::{
    FixedOrderInnerProduct, LinearOperatorOrientation, LinearSolver, PreconditionerPolicy,
    ReductionPolicy, SolverPlan,
};
use eqiora_core::{ScalarDomain, ScalarType};

/// Worst-case simultaneously live requested heap-backed numeric vector bytes,
/// including the owned candidate solution, for a real binary64 canonical CSR
/// action and serial
/// reference CG/BiCGSTAB. This is not a solvability or capability certificate.
/// Unknown kernels/domains/storage/policies and arithmetic overflow have no bound.
///
/// The CG recurrence retains solution, applied image, residual, preconditioned
/// residual and direction: five n-vectors. BiCGSTAB retains initial three,
/// correction base/candidate, three recurrence vectors and four work vectors:
/// twelve n-vectors. Jacobi adds one inverse-diagonal n-vector, constructed
/// in place before the remaining recurrence vectors exist (CG: at most 4n;
/// BiCG: 6n). It adds no second transient diagonal allocation.
///
/// Fixed serial reductions request ceil(n/1024) scalars. Canonical transpose
/// and real adjoint actions instead temporarily request n compensation scalars;
/// they never overlap a reduction's partials. Initial residual creation and
/// zero-update acceptance have smaller live sets, even with a caller guess:
/// its internal solution copy is counted, the borrowed caller guess is not.
///
/// Reference true-residual checks reuse `applied`. The common checked-backend
/// wrapper forwards without retaining numeric verifier buffers. Subsequent
/// finalized-system replay retains the solution and one residual; a separate
/// LinearAcceptanceWorkspace retains solution plus two n-vectors. Both phases,
/// including reduction/transpose scratch, are below the recurrence bound.
///
/// Excludes caller CSR/RHS and initial-guess storage, assembly, caller output
/// projections/copies, fixed stack/register state, Vec headers, nonnumeric
/// metadata, allocator rounding/overhead and RSS.
/// A caller retaining its own verifier buffers during a solve must add those
/// separately. Custom operator/execution allocations are not canonical serial
/// actions and are outside this bound. Zero dimension has zero numeric payload;
/// actual problem admission independently rejects unsupported shapes.
pub(crate) fn canonical_csr_numeric_workspace_bytes(
    plan: SolverPlan,
    scalar_domain: ScalarDomain,
    scalar_type: ScalarType,
    dimension: usize,
    orientation: LinearOperatorOrientation,
) -> Option<usize> {
    if scalar_domain != ScalarDomain::Real
        || scalar_type != ScalarType::F64
        || plan.reduction() != ReductionPolicy::Reproducible
    {
        return None;
    }
    let recurrence: usize = match plan.algorithm() {
        LinearSolver::ConjugateGradient => 5,
        LinearSolver::BiConjugateGradientStabilized => 12,
        _ => return None,
    };
    let jacobi = match plan.preconditioner() {
        PreconditionerPolicy::Identity => 0,
        PreconditionerPolicy::Jacobi => 1,
    };
    let vectors = recurrence.checked_add(jacobi)?;
    let scalars = match orientation {
        LinearOperatorOrientation::Normal => vectors
            .checked_mul(dimension)?
            .checked_add(dimension.div_ceil(FixedOrderInnerProduct::<f64>::CHUNK_LENGTH))?,
        LinearOperatorOrientation::Transposed | LinearOperatorOrientation::ConjugateTransposed => {
            vectors.checked_add(1)?.checked_mul(dimension)?
        }
    };
    scalars.checked_mul(size_of::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    fn plan(algorithm: LinearSolver, preconditioner: PreconditionerPolicy) -> SolverPlan {
        SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(17).unwrap())
            .unwrap()
            .with_preconditioner(preconditioner)
    }
    fn bound(
        n: usize,
        algorithm: LinearSolver,
        preconditioner: PreconditionerPolicy,
        orientation: LinearOperatorOrientation,
    ) -> Option<usize> {
        super::super::canonical_csr_numeric_workspace_bytes(
            plan(algorithm, preconditioner),
            ScalarDomain::Real,
            ScalarType::F64,
            n,
            orientation,
        )
    }
    #[test]
    fn real_normal_live_sets_include_small_and_partial_boundaries() {
        use LinearSolver::{BiConjugateGradientStabilized as BiCg, ConjugateGradient as Cg};
        use PreconditionerPolicy::{Identity, Jacobi};
        // Literal byte counts: five/twelve recurrence vectors, optional inverse
        // diagonal, and one partial through 1024 elements, then two partials.
        for (n, expected) in [
            (0, [0, 0, 0, 0]),
            (1, [48, 56, 104, 112]),
            (1023, [40928, 49112, 98216, 106400]),
            (1024, [40968, 49160, 98312, 106504]),
            (1025, [41016, 49216, 98416, 106616]),
        ] {
            for ((algorithm, preconditioner), bytes) in [
                (Cg, Identity),
                (Cg, Jacobi),
                (BiCg, Identity),
                (BiCg, Jacobi),
            ]
            .into_iter()
            .zip(expected)
            {
                assert_eq!(
                    bound(
                        n,
                        algorithm,
                        preconditioner,
                        LinearOperatorOrientation::Normal
                    ),
                    Some(bytes)
                );
            }
        }
    }
    #[test]
    fn transpose_scratch_replaces_the_reduction_stage_for_both_real_orientations() {
        use LinearSolver::{BiConjugateGradientStabilized as BiCg, ConjugateGradient as Cg};
        use PreconditionerPolicy::{Identity, Jacobi};
        for orientation in [
            LinearOperatorOrientation::Transposed,
            LinearOperatorOrientation::ConjugateTransposed,
        ] {
            for ((algorithm, preconditioner), bytes) in [
                (Cg, Identity),
                (Cg, Jacobi),
                (BiCg, Identity),
                (BiCg, Jacobi),
            ]
            .into_iter()
            .zip([49200, 57400, 106600, 114800])
            {
                assert_eq!(
                    bound(1025, algorithm, preconditioner, orientation),
                    Some(bytes)
                );
            }
        }
    }
    #[test]
    fn unsupported_policies_and_scalar_representations_have_no_invented_estimate() {
        let policy = plan(
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Identity,
        );
        for domain in [
            ScalarDomain::Boolean,
            ScalarDomain::Enum,
            ScalarDomain::Integer,
            ScalarDomain::Complex,
        ] {
            assert_eq!(
                canonical_csr_numeric_workspace_bytes(
                    policy,
                    domain,
                    ScalarType::F64,
                    1,
                    LinearOperatorOrientation::Normal
                ),
                None
            );
        }
        assert_eq!(
            canonical_csr_numeric_workspace_bytes(
                policy,
                ScalarDomain::Real,
                ScalarType::F32,
                1,
                LinearOperatorOrientation::Normal
            ),
            None
        );
        assert_eq!(
            canonical_csr_numeric_workspace_bytes(
                policy.with_reduction(ReductionPolicy::Fast),
                ScalarDomain::Real,
                ScalarType::F64,
                1,
                LinearOperatorOrientation::Normal
            ),
            None
        );
        for algorithm in [LinearSolver::MinimumResidual, LinearSolver::SparseLu] {
            assert_eq!(
                bound(
                    1,
                    algorithm,
                    PreconditionerPolicy::Identity,
                    LinearOperatorOrientation::Normal
                ),
                None
            );
        }
    }
    #[test]
    fn every_overflow_stage_fails_closed_without_allocating() {
        for n in [usize::MAX, usize::MAX / 5, usize::MAX / 40] {
            assert_eq!(
                bound(
                    n,
                    LinearSolver::ConjugateGradient,
                    PreconditionerPolicy::Identity,
                    LinearOperatorOrientation::Normal
                ),
                None
            );
        }
        // Fourteen simultaneously live n-vectors of binary64 scalars.
        let last = usize::MAX / 112;
        assert_eq!(
            bound(
                last,
                LinearSolver::BiConjugateGradientStabilized,
                PreconditionerPolicy::Jacobi,
                LinearOperatorOrientation::Transposed
            ),
            Some(last * 112)
        );
        assert_eq!(
            bound(
                last + 1,
                LinearSolver::BiConjugateGradientStabilized,
                PreconditionerPolicy::Jacobi,
                LinearOperatorOrientation::Transposed
            ),
            None
        );
    }
}
