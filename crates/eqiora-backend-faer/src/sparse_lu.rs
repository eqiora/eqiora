use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    ConvergenceReason, FixedOrderInnerProduct, LinearOperatorOrientation, LinearProblem,
    LinearSolution, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverPlan, SolverProvider,
    accept_linear_solution,
};

use crate::sparse_lu_factor::{factor_numeric, factor_symbolic, solve_factored_oriented};

pub(super) fn solve_sparse_lu(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
) -> Result<LinearSolution, Diagnostic> {
    let system = problem.canonical_csr_system().ok_or_else(|| {
        if problem.operator().orientation() != LinearOperatorOrientation::Normal {
            invalid_realization(
                "faer sparse LU requires a normal-orientation canonical CSR problem or an oriented request with an exact canonical source",
            )
        } else {
            invalid_realization("faer sparse LU requires an exact canonical CSR coefficient source")
        }
    })?;
    if problem.operator().rows() != system.rows()
        || problem.operator().columns() != system.columns()
        || problem.properties() != system.properties()
    {
        return Err(invalid_realization(
            "faer sparse LU action, properties, and canonical CSR source disagree",
        ));
    }

    let initial = problem
        .initial_guess()
        .map_or_else(|| vec![0.0; system.columns()], <[f64]>::to_vec);
    let initial_residual_norm = fixed_residual_norm(problem, &initial)?;
    let right_hand_side_norm = fixed_norm(problem.right_hand_side())?;
    let residual_target = plan.residual_target(right_hand_side_norm)?;
    if initial_residual_norm <= residual_target {
        return accept_linear_solution(
            problem,
            plan,
            provider,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            initial_residual_norm,
            initial,
        );
    }

    let values = factor_and_solve(
        system,
        problem.right_hand_side(),
        problem.operator().orientation(),
    )?;
    let reported_residual_norm = fixed_residual_norm(problem, &values)?;
    accept_linear_solution(
        problem,
        plan,
        provider,
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        reported_residual_norm,
        values,
    )
}

fn factor_and_solve(
    system: &eqiora_solver::CanonicalCsrSystemView,
    right_hand_side: &[f64],
    orientation: LinearOperatorOrientation,
) -> Result<Vec<f64>, Diagnostic> {
    let symbolic = factor_symbolic(system)?;
    let numeric = factor_numeric(&symbolic, system)?;
    solve_factored_oriented(&symbolic, &numeric, right_hand_side, orientation)
}

pub(super) fn fixed_residual_norm(
    problem: &LinearProblem<'_>,
    values: &[f64],
) -> Result<f64, Diagnostic> {
    let mut residual = vec![0.0; problem.operator().rows()];
    SERIAL_LINEAR_EXECUTION.apply(problem.operator(), values, &mut residual)?;
    for (applied, right_hand_side) in residual.iter_mut().zip(problem.right_hand_side()) {
        *applied = right_hand_side - *applied;
    }
    fixed_norm(&residual)
}

pub(super) fn fixed_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let squared =
        SERIAL_LINEAR_EXECUTION.inner_product(FixedOrderInnerProduct::new(values, values)?)?;
    Ok(squared.sqrt())
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
