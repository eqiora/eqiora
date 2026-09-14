use eqiora_core::Diagnostic;
use eqiora_distributed::DistributedLinearProblem;
use eqiora_solver::{LinearSolver, SolverPlan};

use super::{CgWorkspace, KrylovWorkspace, MinresWorkspace, invalid_realization, zeroed};

impl KrylovWorkspace {
    pub(super) fn new(
        problem: &DistributedLinearProblem<'_>,
        plan: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        let dimension = problem.right_hand_side().len();
        let mut solution = zeroed(dimension, "distributed Krylov solution")?;
        if let Some(initial) = problem.initial_guess() {
            solution.copy_from_slice(initial);
        }
        match plan.algorithm() {
            LinearSolver::ConjugateGradient => Ok(Self::Cg(CgWorkspace {
                solution,
                applied: zeroed(dimension, "CG action")?,
                residual: zeroed(dimension, "CG residual")?,
                preconditioned: zeroed(dimension, "CG preconditioned residual")?,
                direction: zeroed(dimension, "CG direction")?,
                inverse_diagonal: zeroed(dimension, "Jacobi inverse diagonal")?,
            })),
            LinearSolver::MinimumResidual => Ok(Self::Minres(MinresWorkspace {
                solution,
                applied: zeroed(dimension, "MINRES action")?,
                previous_residual: zeroed(dimension, "MINRES previous residual")?,
                current_residual: zeroed(dimension, "MINRES current residual")?,
                lanczos_image: zeroed(dimension, "MINRES Lanczos image")?,
                basis: zeroed(dimension, "MINRES Lanczos basis")?,
                direction: zeroed(dimension, "MINRES direction")?,
                previous_direction: zeroed(dimension, "MINRES previous direction")?,
                older_direction: zeroed(dimension, "MINRES older direction")?,
            })),
            LinearSolver::BiConjugateGradientStabilized => Err(invalid_realization(
                "MPI distributed Krylov workspace does not implement BiCGSTAB",
            )),
            LinearSolver::SparseLu => Err(invalid_realization(
                "MPI distributed Krylov workspace does not implement sparse LU",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use eqiora_core::ScalarType;
    use eqiora_core::diagnostic::codes;
    use eqiora_distributed::{DistributedLinearSystem, GlobalVectorSpace, Partition, PartitionId};
    use eqiora_solver::{
        CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, ReductionPolicy,
    };

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
    fn mpi_workspace_rejects_sparse_lu_defensively() {
        let complete =
            CanonicalCsrSystemView::new(&OneByOne, LinearOperatorProperties::General).unwrap();
        let partition = Partition::new(
            GlobalVectorSpace::new(std::num::NonZeroUsize::MIN, ScalarType::F64),
            std::num::NonZeroUsize::MIN,
            vec![PartitionId::new(0)],
        )
        .unwrap();
        let distributed = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
        let problem = distributed.local_problem(PartitionId::new(0)).unwrap();
        let plan = SolverPlan::new(
            LinearSolver::SparseLu,
            0.0,
            1.0e-12,
            std::num::NonZeroUsize::MIN,
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let error = match KrylovWorkspace::new(&problem, plan) {
            Ok(_) => panic!("MPI workspace unexpectedly admitted sparse LU"),
            Err(error) => error,
        };

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert_eq!(
            error.message(),
            "MPI distributed Krylov workspace does not implement sparse LU"
        );
    }
}
