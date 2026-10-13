use eqiora_core::Diagnostic;
use eqiora_solver::{CanonicalCsrSystemView, LinearSolver, PreconditionerPolicy, SolverPlan};

use crate::binding::invalid;

pub(super) fn minimum_device_payload_bytes(
    system: &CanonicalCsrSystemView,
    plan: SolverPlan,
) -> Result<usize, Diagnostic> {
    let vector_count = match plan.algorithm() {
        LinearSolver::ConjugateGradient => 3usize,
        LinearSolver::BiConjugateGradientStabilized => 7,
        LinearSolver::MinimumResidual => 6,
        LinearSolver::SparseLu => {
            return Err(invalid(
                "device payload estimation does not implement sparse LU",
            ));
        }
    };
    let diagonal_count = usize::from(plan.preconditioner() == PreconditionerPolicy::Jacobi);
    let dimension = system.columns();
    let index_elements = system
        .row_offsets()
        .len()
        .checked_add(system.column_indices().len())
        .ok_or_else(|| invalid("device index payload size overflowed"))?;
    let scalar_elements = system
        .values()
        .len()
        .checked_add(system.right_hand_side().len())
        .and_then(|count| {
            count.checked_add(dimension.checked_mul(2 + vector_count + diagonal_count)?)
        })
        .ok_or_else(|| invalid("device scalar payload size overflowed"))?;
    index_elements
        .checked_mul(size_of::<i64>())
        .and_then(|bytes| {
            scalar_elements
                .checked_mul(size_of::<f64>())
                .and_then(|scalars| bytes.checked_add(scalars))
        })
        .ok_or_else(|| invalid("known device payload byte count overflowed"))
}

#[cfg(test)]
mod tests {
    use eqiora_core::diagnostic::codes;
    use eqiora_solver::{CsrStorage, LinearOperatorProperties, ReductionPolicy};

    use super::*;

    struct OneByOne;

    impl CsrStorage for OneByOne {
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

        fn values(&self) -> &[f64] {
            &[1.0]
        }
    }

    impl OneByOne {
        pub(crate) fn rhs(&self) -> &[f64] {
            &[1.0]
        }
    }

    #[test]
    fn device_payload_rejects_sparse_lu_before_device_admission() {
        let system = CanonicalCsrSystemView::new(
            &OneByOne,
            OneByOne.rhs(),
            LinearOperatorProperties::General,
        )
        .unwrap();
        let plan = SolverPlan::new(
            LinearSolver::SparseLu,
            0.0,
            1.0e-12,
            std::num::NonZeroUsize::MIN,
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let error = minimum_device_payload_bytes(&system, plan).unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert_eq!(
            error.message(),
            "device payload estimation does not implement sparse LU"
        );
    }
}
