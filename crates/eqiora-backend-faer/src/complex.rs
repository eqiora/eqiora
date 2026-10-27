//! Native complex solves share Krylov kernels, LU factors and acceptance with real solves.
use eqiora_core::Diagnostic;
use eqiora_solver::{
    LinearProblem, LinearSolution, LinearSolverBackend, PreparedLinearSolver,
    ReplicatedLinearExecution, SolverCapabilities, SolverPlan, SolverProvider,
};
use faer::complex::Complex64;

use crate::FaerLinearSolver;

#[cfg(test)]
mod krylov_tests;

impl LinearSolverBackend<Complex64> for FaerLinearSolver {
    fn provider(&self) -> SolverProvider {
        Self::provider(self)
    }

    fn capabilities(&self) -> SolverCapabilities {
        Self::capabilities(self)
    }

    fn prepare_linear(
        &self,
        plan: SolverPlan,
    ) -> Result<Option<Box<dyn PreparedLinearSolver<Complex64>>>, Diagnostic> {
        crate::prepared_sparse_lu::prepare(plan)
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, Complex64>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution<Complex64>,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
        let algorithm = format!("{:?}", plan.algorithm());
        let _linear = eqiora_execution::telemetry_span!(linear_solve(
            &algorithm,
            self.provider().id().as_str()
        ))
        .entered();
        crate::require_serial_execution(execution, plan)?;
        self.capabilities().require_problem(
            plan,
            problem.scalar_domain(),
            problem.scalar_type(),
            problem.properties(),
        )?;
        match plan.algorithm() {
            eqiora_solver::LinearSolver::ConjugateGradient => crate::cg::solve(
                self.provider(),
                problem,
                plan,
                crate::inverse_diagonal(problem, plan)?,
            ),
            eqiora_solver::LinearSolver::BiConjugateGradientStabilized => crate::bicgstab::solve(
                self.provider(),
                problem,
                plan,
                crate::inverse_diagonal(problem, plan)?,
            ),
            eqiora_solver::LinearSolver::SparseLu => {
                crate::sparse_lu::solve_sparse_lu(problem, plan)
            }
            eqiora_solver::LinearSolver::MinimumResidual => Err(crate::invalid_realization(
                "the faer adapter does not implement MINRES",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_solver::{
        CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties as Properties, LinearSolver,
        ReductionPolicy,
    };

    struct Diagonal<S>([S; 2]);
    impl<S> CsrStorage<S> for Diagonal<S> {
        fn rows(&self) -> usize {
            2
        }
        fn columns(&self) -> usize {
            2
        }
        fn row_offsets(&self) -> &[usize] {
            &[0, 1, 2]
        }
        fn column_indices(&self) -> &[usize] {
            &[0, 1]
        }
        fn values(&self) -> &[S] {
            &self.0
        }
    }

    #[test]
    fn exact_complex_structures_admit_supported_native_methods() {
        use Complex64 as C;
        let plan = SolverPlan::new(LinearSolver::SparseLu, 0., 1e-12, 1.try_into().unwrap())
            .unwrap()
            .with_reduction(ReductionPolicy::Fast);
        for (properties, values, rhs) in [
            (
                Properties::General,
                [C::new(2., 1.), C::new(3., -1.)],
                [C::new(0., 5.), C::new(-5., 5.)],
            ),
            (
                Properties::ComplexSymmetric,
                [C::new(2., 1.), C::new(3., -1.)],
                [C::new(0., 5.), C::new(-5., 5.)],
            ),
            (
                Properties::Hermitian,
                [C::new(-2., 0.), C::new(3., 0.)],
                [C::new(-2., -4.), C::new(-6., 3.)],
            ),
            (
                Properties::HermitianPositiveDefinite,
                [C::new(2., 0.), C::new(3., 0.)],
                [C::new(2., 4.), C::new(-6., 3.)],
            ),
        ] {
            // These RHS values are the listed diagonal times [1+2i,-2+i].
            let source = CanonicalCsrSystemView::new(&Diagonal(values), &rhs, properties).unwrap();
            let solution = FaerLinearSolver
                .solve(&source.linear_problem().unwrap(), plan)
                .unwrap();
            for algorithm in [
                LinearSolver::ConjugateGradient,
                LinearSolver::BiConjugateGradientStabilized,
            ] {
                if algorithm == LinearSolver::ConjugateGradient
                    && properties != Properties::HermitianPositiveDefinite
                {
                    continue;
                }
                for preconditioner in [
                    eqiora_solver::PreconditionerPolicy::Identity,
                    eqiora_solver::PreconditionerPolicy::Jacobi,
                ] {
                    let request = SolverPlan::new(algorithm, 0., 1e-12, 16.try_into().unwrap())
                        .unwrap()
                        .with_reduction(ReductionPolicy::Fast)
                        .with_preconditioner(preconditioner);
                    let iterative = FaerLinearSolver
                        .solve(&source.linear_problem().unwrap(), request)
                        .unwrap();
                    for (actual, expected) in iterative
                        .values()
                        .iter()
                        .zip([C::new(1., 2.), C::new(-2., 1.)])
                    {
                        assert!((*actual - expected).norm() <= 1e-11);
                    }
                    assert_eq!(iterative.report().solver_plan(), request);
                }
            }
            for (actual, expected) in solution
                .values()
                .iter()
                .zip([C::new(1., 2.), C::new(-2., 1.)])
            {
                assert!((*actual - expected).norm() < 1e-12);
            }
        }
        let real =
            CanonicalCsrSystemView::new(&Diagonal([2., 3.]), &[2., -6.], Properties::Symmetric)
                .unwrap();
        let solution = FaerLinearSolver
            .solve(&real.linear_problem().unwrap(), plan)
            .unwrap();
        assert_eq!(solution.values(), &[1., -2.]);
        let unsupported = SolverPlan::new(
            LinearSolver::MinimumResidual,
            0.,
            1e-12,
            2.try_into().unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        assert!(
            FaerLinearSolver
                .capabilities()
                .require_problem(
                    unsupported,
                    eqiora_core::ScalarDomain::Complex,
                    eqiora_core::ScalarType::F64,
                    Properties::Hermitian,
                )
                .is_err()
        );
    }
}
