use super::*;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy};

#[test]
fn mini_solver_admission_separates_mathematics_from_provider_availability() {
    for (algorithm, preconditioner, reduction, admitted) in [
        (
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
            true,
        ),
        (
            LinearSolver::SparseLu,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            true,
        ),
        (
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            true,
        ),
        (
            LinearSolver::SparseLu,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
            true,
        ),
        (
            LinearSolver::SparseLu,
            PreconditionerPolicy::Jacobi,
            ReductionPolicy::Fast,
            false,
        ),
        (
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            true,
        ),
        (
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
            false,
        ),
        (
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Jacobi,
            ReductionPolicy::Reproducible,
            false,
        ),
    ] {
        let plan = SolverPlan::new(
            algorithm,
            1.0e-6,
            1.0e-13,
            NonZeroUsize::new(10).expect("ten is non-zero"),
        )
        .expect("test plan")
        .with_preconditioner(preconditioner)
        .with_reduction(reduction);
        assert_eq!(
            require_mini_solver(plan).is_ok(),
            admitted,
            "unexpected admission for {algorithm:?}/{preconditioner:?}/{reduction:?}",
        );
    }
}
