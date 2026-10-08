mod bicgstab;
mod complex;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    ConvergenceReason, DiagonalAvailability, FixedOrderInnerProduct, LinearProblem, LinearSolution,
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, ReplicatedLinearExecution,
    SolveReport, SolverCapabilities, SolverPlan, SolverProvider,
};

/// Deterministic one-worker reference linear solver.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceLinearSolver;

/// Shared stateless reference backend.
pub const REFERENCE_LINEAR_SOLVER: ReferenceLinearSolver = ReferenceLinearSolver;

/// Exact declared release identity of the deterministic reference solver.
pub const REFERENCE_SOLVER_PROVIDER: SolverProvider = SolverProvider::new(
    crate::BackendId::new("eqiora.reference"),
    env!("CARGO_PKG_VERSION"),
    &[],
);

impl ReferenceLinearSolver {
    /// Exact release identity shared by the admitted scalar specializations.
    #[must_use]
    pub const fn provider(&self) -> SolverProvider {
        REFERENCE_SOLVER_PROVIDER
    }

    /// Stable reference adapter identity.
    #[must_use]
    pub const fn id(&self) -> crate::BackendId {
        REFERENCE_SOLVER_PROVIDER.id()
    }

    /// Exact real and complex policies implemented by this adapter.
    #[must_use]
    pub fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::reference()
    }
}

impl LinearSolverBackend for ReferenceLinearSolver {
    fn provider(&self) -> SolverProvider {
        Self::provider(self)
    }

    fn capabilities(&self) -> SolverCapabilities {
        Self::capabilities(self)
    }

    fn complex_backend(&self) -> Option<&dyn LinearSolverBackend<num_complex::Complex64>> {
        Some(self)
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        self.capabilities().require_problem(
            plan,
            problem.scalar_domain(),
            problem.scalar_type(),
            problem.properties(),
        )?;
        execution.require_reduction(plan.reduction())?;
        match plan.algorithm() {
            LinearSolver::ConjugateGradient => {
                solve_preconditioned_conjugate_gradient(self.provider(), problem, plan, execution)
            }
            LinearSolver::MinimumResidual => {
                solve_minimum_residual(self.provider(), problem, plan, execution)
            }
            LinearSolver::BiConjugateGradientStabilized => {
                bicgstab::solve_preconditioned_bicgstab(self.provider(), problem, plan, execution)
            }
            LinearSolver::SparseLu => Err(invalid_realization(
                "the reference adapter does not implement sparse LU",
            )),
        }
    }
}

fn solve_minimum_residual(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut solution = problem
        .initial_guess()
        .map_or_else(|| vec![0.0; dimension], <[f64]>::to_vec);
    let mut applied = vec![0.0; dimension];
    execution.apply(problem.operator(), &solution, &mut applied)?;
    let previous_residual = problem
        .right_hand_side()
        .iter()
        .zip(&applied)
        .map(|(right, applied)| right - applied)
        .collect::<Vec<_>>();
    require_finite(&previous_residual, "initial residual")?;

    let right_hand_side_norm = norm(execution, problem.right_hand_side())?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(execution, &previous_residual)?;
    if initial_residual_norm <= target {
        let report = SolveReport::accepted(
            provider,
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            initial_residual_norm,
            initial_residual_norm,
            initial_residual_norm,
            target,
        )?;
        return LinearSolution::new(solution, report);
    }

    let iteration_limit = plan.maximum_iterations().get().min(dimension);
    let initial_solution = solution.clone();
    let mut krylov_basis = Vec::with_capacity(iteration_limit);
    krylov_basis.push(
        previous_residual
            .iter()
            .map(|value| value / initial_residual_norm)
            .collect::<Vec<_>>(),
    );
    let mut rotated_hessenberg = vec![vec![0.0; iteration_limit]; iteration_limit + 1];
    let mut cosines = vec![0.0; iteration_limit];
    let mut sines = vec![0.0; iteration_limit];
    let mut projected_right_hand_side = vec![0.0; iteration_limit + 1];
    projected_right_hand_side[0] = initial_residual_norm;
    let mut operator_scale = 0.0_f64;
    let mut reported_residual_norm = initial_residual_norm;

    for iteration in 1..=iteration_limit {
        let column = iteration - 1;
        let current_basis = &krylov_basis[column];
        execution.apply(problem.operator(), current_basis, &mut applied)?;
        operator_scale = operator_scale.max(norm(execution, &applied)?);

        // Full two-pass modified Gram-Schmidt retains every finite-precision
        // projection coefficient. For an asserted symmetric operator this
        // represents the same exact-arithmetic minimum-residual Krylov method
        // as tridiagonal Lanczos, without discarding global reorthogonalization
        // corrections that matter on ill-conditioned reference systems.
        for _ in 0..2 {
            for (row, existing_basis) in krylov_basis[..iteration].iter().enumerate() {
                let correction = dot(execution, existing_basis, &applied)?;
                rotated_hessenberg[row][column] += correction;
                for index in 0..dimension {
                    applied[index] -= correction * existing_basis[index];
                }
            }
        }
        require_finite(&applied, "MINRES reorthogonalized Krylov image")?;
        let next_beta = norm(execution, &applied)?;
        let closure_tolerance = (dimension as f64) * f64::EPSILON * operator_scale;
        let space_closed = iteration == dimension || next_beta <= closure_tolerance;
        rotated_hessenberg[column + 1][column] = if space_closed { 0.0 } else { next_beta };

        for rotation in 0..column {
            let upper = rotated_hessenberg[rotation][column];
            let lower = rotated_hessenberg[rotation + 1][column];
            rotated_hessenberg[rotation][column] =
                cosines[rotation] * upper + sines[rotation] * lower;
            rotated_hessenberg[rotation + 1][column] =
                -sines[rotation] * upper + cosines[rotation] * lower;
        }
        let diagonal = rotated_hessenberg[column][column];
        let subdiagonal = rotated_hessenberg[column + 1][column];
        let rotation_norm = diagonal.hypot(subdiagonal);
        if !rotation_norm.is_finite() || rotation_norm <= f64::MIN_POSITIVE {
            return Err(solve_failed(
                "MINRES orthogonal rotation broke down before convergence",
            ));
        }
        cosines[column] = diagonal / rotation_norm;
        sines[column] = subdiagonal / rotation_norm;
        rotated_hessenberg[column][column] = rotation_norm;
        rotated_hessenberg[column + 1][column] = 0.0;
        let projected_value = projected_right_hand_side[column];
        projected_right_hand_side[column] = cosines[column] * projected_value;
        projected_right_hand_side[column + 1] = -sines[column] * projected_value;
        reported_residual_norm = projected_right_hand_side[column + 1].abs();
        let next_basis = (iteration < iteration_limit && !space_closed)
            .then(|| applied.iter().map(|value| value / next_beta).collect());

        if reported_residual_norm <= target || space_closed || iteration == iteration_limit {
            solution = reconstruct_minimum_residual_solution(
                &initial_solution,
                &krylov_basis,
                &rotated_hessenberg,
                &projected_right_hand_side,
                iteration,
            )?;
            require_finite(&solution, "MINRES solution")?;
            let true_residual_norm =
                true_residual_norm(execution, problem, &solution, &mut applied)?;
            if reported_residual_norm <= target && true_residual_norm <= target {
                let report = SolveReport::accepted(
                    provider,
                    execution.provider(),
                    execution.report(),
                    problem.operator().orientation(),
                    plan,
                    ConvergenceReason::ResidualToleranceSatisfied,
                    iteration,
                    initial_residual_norm,
                    reported_residual_norm,
                    true_residual_norm,
                    target,
                )?;
                return LinearSolution::new(solution, report);
            }
            if space_closed {
                return Err(solve_failed(format!(
                    "MINRES Krylov space closed at iteration {iteration} with operator scale {operator_scale:e} and true residual {true_residual_norm:e} above target {target:e}"
                )));
            }
        }
        if let Some(next_basis) = next_basis {
            krylov_basis.push(next_basis);
        }
    }

    let true_residual_norm = true_residual_norm(execution, problem, &solution, &mut applied)?;
    Err(solve_failed(format!(
        "MINRES reached the plan limit of {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
        plan.maximum_iterations(),
    )))
}

fn reconstruct_minimum_residual_solution(
    initial_solution: &[f64],
    basis: &[Vec<f64>],
    rotated_hessenberg: &[Vec<f64>],
    projected_right_hand_side: &[f64],
    iterations: usize,
) -> Result<Vec<f64>, Diagnostic> {
    let mut coefficients = projected_right_hand_side[..iterations].to_vec();
    for row in (0..iterations).rev() {
        for column in row + 1..iterations {
            coefficients[row] -= rotated_hessenberg[row][column] * coefficients[column];
        }
        let diagonal = rotated_hessenberg[row][row];
        if !diagonal.is_finite() || diagonal.abs() <= f64::MIN_POSITIVE {
            return Err(solve_failed("MINRES projected triangular solve broke down"));
        }
        coefficients[row] /= diagonal;
    }
    require_finite(&coefficients, "MINRES projected coefficients")?;

    let mut solution = initial_solution.to_vec();
    for (coefficient, basis_vector) in coefficients.iter().zip(&basis[..iterations]) {
        for (solution, basis_value) in solution.iter_mut().zip(basis_vector) {
            *solution += coefficient * basis_value;
        }
    }
    Ok(solution)
}

fn solve_preconditioned_conjugate_gradient(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut solution = problem
        .initial_guess()
        .map_or_else(|| vec![0.0; dimension], <[f64]>::to_vec);
    let mut applied = vec![0.0; dimension];
    execution.apply(problem.operator(), &solution, &mut applied)?;
    let mut residual = problem
        .right_hand_side()
        .iter()
        .zip(&applied)
        .map(|(right, applied)| right - applied)
        .collect::<Vec<_>>();
    require_finite(&residual, "initial residual")?;

    let right_hand_side_norm = norm(execution, problem.right_hand_side())?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(execution, &residual)?;
    if initial_residual_norm <= target {
        let report = SolveReport::accepted(
            provider,
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            initial_residual_norm,
            initial_residual_norm,
            initial_residual_norm,
            target,
        )?;
        return LinearSolution::new(solution, report);
    }

    let inverse_diagonal = build_inverse_diagonal(
        problem,
        plan.preconditioner(),
        DiagonalRequirement::FinitePositive,
    )?;
    let mut preconditioned = vec![0.0; dimension];
    apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
    let mut direction = preconditioned.clone();
    let mut residual_product = dot(execution, &residual, &preconditioned)?;
    if residual_product <= 0.0 {
        return Err(solve_failed(
            "conjugate gradients requires a positive-definite preconditioner",
        ));
    }

    let mut reported_residual_norm = initial_residual_norm;
    for iteration in 1..=plan.maximum_iterations().get() {
        execution.apply(problem.operator(), &direction, &mut applied)?;
        let curvature = dot(execution, &direction, &applied)?;
        if curvature <= 0.0 {
            return Err(solve_failed(
                "conjugate gradients detected non-positive operator curvature",
            ));
        }
        let step = residual_product / curvature;
        for index in 0..dimension {
            solution[index] += step * direction[index];
            residual[index] -= step * applied[index];
        }
        require_finite(&solution, "conjugate-gradient solution")?;
        reported_residual_norm = norm(execution, &residual)?;
        if reported_residual_norm <= target {
            let true_residual_norm =
                true_residual_norm(execution, problem, &solution, &mut applied)?;
            if true_residual_norm <= target {
                let report = SolveReport::accepted(
                    provider,
                    execution.provider(),
                    execution.report(),
                    problem.operator().orientation(),
                    plan,
                    ConvergenceReason::ResidualToleranceSatisfied,
                    iteration,
                    initial_residual_norm,
                    reported_residual_norm,
                    true_residual_norm,
                    target,
                )?;
                return LinearSolution::new(solution, report);
            }
            residual.copy_from_slice(&applied);
            apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
            residual_product = dot(execution, &residual, &preconditioned)?;
            if residual_product <= 0.0 {
                return Err(solve_failed(
                    "conjugate gradients lost positive residual curvature after a true-residual restart",
                ));
            }
            direction.copy_from_slice(&preconditioned);
            continue;
        }

        apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
        let next_residual_product = dot(execution, &residual, &preconditioned)?;
        if next_residual_product <= 0.0 {
            return Err(solve_failed(
                "conjugate gradients lost positive preconditioned residual curvature",
            ));
        }
        let beta = next_residual_product / residual_product;
        for index in 0..dimension {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_product = next_residual_product;
    }

    let true_residual_norm = true_residual_norm(execution, problem, &solution, &mut applied)?;
    Err(solve_failed(format!(
        "conjugate gradients reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
        plan.maximum_iterations()
    )))
}

#[derive(Debug, Clone, Copy)]
enum DiagonalRequirement {
    FinitePositive,
    FiniteNonzero,
}

fn build_inverse_diagonal(
    problem: &LinearProblem<'_>,
    policy: PreconditionerPolicy,
    requirement: DiagonalRequirement,
) -> Result<Option<Vec<f64>>, Diagnostic> {
    if policy == PreconditionerPolicy::Identity {
        return Ok(None);
    }
    let mut diagonal = vec![0.0; problem.operator().rows()];
    if problem.operator().diagonal(&mut diagonal)? == DiagonalAvailability::Unavailable {
        return Err(invalid_realization(
            "Jacobi preconditioning requires an available operator diagonal",
        ));
    }
    let invalid = diagonal.iter().any(|value| match requirement {
        DiagonalRequirement::FinitePositive => !value.is_finite() || *value <= 0.0,
        DiagonalRequirement::FiniteNonzero => !value.is_finite() || *value == 0.0,
    });
    if invalid {
        let message = match requirement {
            DiagonalRequirement::FinitePositive => {
                "Jacobi-preconditioned CG requires a finite positive diagonal"
            }
            DiagonalRequirement::FiniteNonzero => {
                "Jacobi-preconditioned BiCGSTAB requires a finite nonzero diagonal"
            }
        };
        return Err(solve_failed(message));
    }
    let inverse = diagonal
        .into_iter()
        .map(|value| 1.0 / value)
        .collect::<Vec<_>>();
    require_finite(&inverse, "Jacobi inverse diagonal")?;
    Ok(Some(inverse))
}

fn apply_preconditioner(inverse: &Option<Vec<f64>>, residual: &[f64], output: &mut [f64]) {
    match inverse {
        Some(inverse) => {
            for ((output, residual), inverse) in output.iter_mut().zip(residual).zip(inverse) {
                *output = residual * inverse;
            }
        }
        None => output.copy_from_slice(residual),
    }
}

fn true_residual_norm(
    execution: &dyn ReplicatedLinearExecution,
    problem: &LinearProblem<'_>,
    solution: &[f64],
    applied: &mut [f64],
) -> Result<f64, Diagnostic> {
    execution.apply(problem.operator(), solution, applied)?;
    for (value, right) in applied.iter_mut().zip(problem.right_hand_side()) {
        *value = right - *value;
    }
    norm(execution, applied)
}

fn dot(
    execution: &dyn ReplicatedLinearExecution,
    left: &[f64],
    right: &[f64],
) -> Result<f64, Diagnostic> {
    execution.inner_product(FixedOrderInnerProduct::new(left, right)?)
}

fn norm(execution: &dyn ReplicatedLinearExecution, values: &[f64]) -> Result<f64, Diagnostic> {
    Ok(dot(execution, values, values)?.sqrt())
}

fn require_finite(values: &[f64], name: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(solve_failed(format!("{name} contains a non-finite value")))
    } else {
        Ok(())
    }
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{LinearOperator, LinearOperatorProperties, LinearSolver};

    #[derive(Debug)]
    struct DenseSpd;

    impl LinearOperator for DenseSpd {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if input.len() != 2 || output.len() != 2 {
                return Err(solve_failed("dense test operator shape mismatch"));
            }
            output[0] = 4.0 * input[0] + input[1];
            output[1] = input[0] + 3.0 * input[1];
            Ok(())
        }

        fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
            output.copy_from_slice(&[4.0, 3.0]);
            Ok(DiagonalAvailability::Available)
        }
    }

    #[derive(Debug)]
    struct DenseSymmetricIndefinite {
        scale: f64,
    }

    impl LinearOperator for DenseSymmetricIndefinite {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if input.len() != 2 || output.len() != 2 {
                return Err(solve_failed("dense indefinite test shape mismatch"));
            }
            output[0] = self.scale * (2.0 * input[0] + input[1]);
            output[1] = self.scale * (input[0] - input[1]);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct DenseGeneral;

    impl LinearOperator for DenseGeneral {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if input.len() != 2 || output.len() != 2 {
                return Err(solve_failed("dense general test shape mismatch"));
            }
            output[0] = 4.0 * input[0] + input[1];
            output[1] = 2.0 * input[0] + 3.0 * input[1];
            Ok(())
        }

        fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
            output.copy_from_slice(&[4.0, 3.0]);
            Ok(DiagonalAvailability::Available)
        }
    }

    #[derive(Debug)]
    struct SingularGeneral;

    impl LinearOperator for SingularGeneral {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            output[0] = input[1];
            output[1] = 0.0;
            Ok(())
        }
    }

    #[derive(Debug)]
    struct NonFiniteGeneral;

    impl LinearOperator for NonFiniteGeneral {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, _input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            output.fill(f64::NAN);
            Ok(())
        }
    }

    #[test]
    fn reference_cg_reports_an_independent_true_residual() {
        let problem = LinearProblem::new(
            &DenseSpd,
            &[1.0, 2.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Jacobi);
        let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
        assert!((solution.values()[0] - 1.0 / 11.0).abs() < 1.0e-14);
        assert!((solution.values()[1] - 7.0 / 11.0).abs() < 1.0e-14);
        assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
        assert_eq!(solution.report().backend().as_str(), "eqiora.reference");
    }

    #[test]
    fn cg_rejects_a_general_property_instead_of_guessing() {
        let problem =
            LinearProblem::new(&DenseSpd, &[1.0, 2.0], LinearOperatorProperties::General).unwrap();
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            0.0,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap();
        assert_eq!(
            REFERENCE_LINEAR_SOLVER
                .solve(&problem, plan)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn reference_backend_rejects_sparse_lu_before_numerical_work() {
        let problem = LinearProblem::new(
            &DenseGeneral,
            &[1.0, 2.0],
            LinearOperatorProperties::General,
        )
        .unwrap();
        let plan =
            SolverPlan::new(LinearSolver::SparseLu, 0.0, 1.0e-12, NonZeroUsize::MIN).unwrap();
        let error = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap_err();

        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("exact SolverCapability"));
    }

    #[test]
    fn reference_minres_solves_an_asserted_symmetric_indefinite_system() {
        let plan = SolverPlan::new(
            LinearSolver::MinimumResidual,
            1.0e-12,
            0.0,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap();
        for scale in [1.0e-20, 1.0, 1.0e20] {
            let operator = DenseSymmetricIndefinite { scale };
            let right_hand_side = [scale, 2.0 * scale];
            let problem = LinearProblem::new(
                &operator,
                &right_hand_side,
                LinearOperatorProperties::SymmetricIndefinite,
            )
            .unwrap();
            let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
            assert!((solution.values()[0] - 1.0).abs() < 1.0e-12);
            assert!((solution.values()[1] + 1.0).abs() < 1.0e-12);
            assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
        }

        let operator = DenseSymmetricIndefinite { scale: 1.0 };
        let problem = LinearProblem::new(
            &operator,
            &[1.0, 2.0],
            LinearOperatorProperties::SymmetricIndefinite,
        )
        .unwrap();
        let unsupported = plan.with_preconditioner(PreconditionerPolicy::Jacobi);
        assert_eq!(
            REFERENCE_LINEAR_SOLVER
                .solve(&problem, unsupported)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn reference_minres_also_accepts_symmetric_positive_definite_problems() {
        let problem = LinearProblem::new(
            &DenseSpd,
            &[1.0, 2.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_initial_guess(&[0.25, -0.5])
        .unwrap();
        let plan = SolverPlan::new(
            LinearSolver::MinimumResidual,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap();
        let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
        assert!((solution.values()[0] - 1.0 / 11.0).abs() < 1.0e-13);
        assert!((solution.values()[1] - 7.0 / 11.0).abs() < 1.0e-13);
        assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
    }

    #[test]
    fn reference_bicgstab_solves_a_general_operator_with_supported_preconditioners() {
        let problem = LinearProblem::new(
            &DenseGeneral,
            &[6.0, 8.0],
            LinearOperatorProperties::General,
        )
        .unwrap()
        .with_initial_guess(&[0.25, -0.5])
        .unwrap();
        for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
            let plan = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1.0e-12,
                1.0e-14,
                NonZeroUsize::new(20).unwrap(),
            )
            .unwrap()
            .with_preconditioner(preconditioner);
            let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
            assert!((solution.values()[0] - 1.0).abs() < 1.0e-13);
            assert!((solution.values()[1] - 2.0).abs() < 1.0e-13);
            assert_eq!(
                solution.report().algorithm(),
                LinearSolver::BiConjugateGradientStabilized
            );
            assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
        }
    }

    #[test]
    fn reference_bicgstab_fails_closed_on_breakdown_and_nonfinite_actions() {
        let plan = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-12,
            0.0,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap();
        for operator in [
            &SingularGeneral as &dyn LinearOperator<Scalar = f64>,
            &NonFiniteGeneral,
        ] {
            let problem =
                LinearProblem::new(operator, &[1.0, 0.0], LinearOperatorProperties::General)
                    .unwrap();
            assert_eq!(
                REFERENCE_LINEAR_SOLVER
                    .solve(&problem, plan)
                    .unwrap_err()
                    .code(),
                codes::NUMERICAL_SOLVE_FAILED
            );
        }
    }

    #[test]
    fn reference_bicgstab_rejects_an_unsupported_policy_tuple() {
        let problem = LinearProblem::new(
            &DenseGeneral,
            &[6.0, 8.0],
            LinearOperatorProperties::General,
        )
        .unwrap();
        let plan = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-12,
            0.0,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap()
        .with_reduction(crate::ReductionPolicy::Fast);
        assert_eq!(
            REFERENCE_LINEAR_SOLVER
                .solve(&problem, plan)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
}
