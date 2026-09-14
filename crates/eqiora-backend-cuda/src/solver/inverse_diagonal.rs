use eqiora_core::Diagnostic;
use eqiora_solver::{CanonicalCsrSystemView, LinearSolver, PreconditionerPolicy, SolverPlan};

use crate::runtime::solve_failed;

pub(super) fn inverse_diagonal(
    system: &CanonicalCsrSystemView,
    plan: SolverPlan,
) -> Result<Option<Vec<f64>>, Diagnostic> {
    if plan.preconditioner() == PreconditionerPolicy::Identity {
        return Ok(None);
    }
    let mut inverse = Vec::with_capacity(system.rows());
    for row in 0..system.rows() {
        let start = system.row_offsets()[row];
        let end = system.row_offsets()[row + 1];
        let columns = &system.column_indices()[start..end];
        let value = columns
            .binary_search(&row)
            .ok()
            .map_or(0.0, |offset| system.values()[start + offset]);
        let accepted = match plan.algorithm() {
            LinearSolver::ConjugateGradient => value.is_finite() && value > 0.0,
            LinearSolver::BiConjugateGradientStabilized => value.is_finite() && value != 0.0,
            LinearSolver::MinimumResidual | LinearSolver::SparseLu => false,
        };
        if !accepted {
            return Err(solve_failed(format!(
                "CUDA {:?} Jacobi preconditioning rejected diagonal entry {row}: {value:e}",
                plan.algorithm()
            )));
        }
        let reciprocal = 1.0 / value;
        if !reciprocal.is_finite() {
            return Err(solve_failed(format!(
                "CUDA Jacobi inverse diagonal overflowed at entry {row}"
            )));
        }
        inverse.push(reciprocal);
    }
    Ok(Some(inverse))
}

#[cfg(test)]
mod tests {
    use eqiora_core::ScalarType;
    use eqiora_core::diagnostic::codes;
    use eqiora_solver::{CompleteCsrStorage, LinearOperatorProperties, ReductionPolicy};

    use super::super::CudaLinearSolver;
    use super::*;

    struct OneByOne;

    impl CompleteCsrStorage for OneByOne {
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

        fn right_hand_side(&self) -> &[f64] {
            &[1.0]
        }
    }

    #[test]
    fn cuda_backend_rejects_sparse_lu_before_device_work() {
        let plan = SolverPlan::new(
            LinearSolver::SparseLu,
            0.0,
            1.0e-12,
            std::num::NonZeroUsize::MIN,
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let error = CudaLinearSolver::capabilities()
            .require_problem(plan, ScalarType::F64, LinearOperatorProperties::General)
            .unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("exact SolverCapability"));

        let system =
            CanonicalCsrSystemView::new(&OneByOne, LinearOperatorProperties::General).unwrap();
        let defensive_error = inverse_diagonal(
            &system,
            plan.with_preconditioner(PreconditionerPolicy::Jacobi),
        )
        .unwrap_err();
        assert_eq!(defensive_error.code(), codes::NUMERICAL_SOLVE_FAILED);
    }
}
