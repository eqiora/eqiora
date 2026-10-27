//! One MINRES recurrence in identity or symmetric Jacobi correction coordinates.
use super::*;

pub(super) fn solve(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    // Symmetric congruence preserves the asserted symmetry and inertia. A
    // positive diagonal does not itself establish either operator property.
    let inverse_sqrt = build_diagonal(
        problem,
        plan.preconditioner(),
        DiagonalRequirement::FinitePositive,
    )?
    .map(|mut diagonal| {
        for value in &mut diagonal {
            *value = 1. / value.sqrt();
        }
        diagonal
    });
    let InitialResidual {
        mut solution,
        mut applied,
        residual: mut previous_residual,
        initial_residual_norm,
        target,
        accepted_report,
    } = initial_residual(provider, problem, plan, execution)?;
    if let Some(report) = accepted_report {
        return LinearSolution::new(solution, report);
    }
    let dimension = solution.len();
    scale(&inverse_sqrt, &mut previous_residual)?;
    let beta0 = if inverse_sqrt.is_some() {
        FixedOrderInnerProduct::norm(execution, &previous_residual)?
    } else {
        initial_residual_norm
    };
    if beta0 == 0. {
        return Err(solve_failed(
            "MINRES scaled residual vanished before acceptance",
        ));
    }

    let iteration_limit = plan.maximum_iterations().get().min(dimension);
    let initial_solution = solution.clone();
    let mut krylov_basis = Vec::with_capacity(iteration_limit);
    krylov_basis.push(
        previous_residual
            .iter()
            .map(|value| value / beta0)
            .collect::<Vec<_>>(),
    );
    let mut rotated_hessenberg = vec![vec![0.0; iteration_limit]; iteration_limit + 1];
    let mut cosines = vec![0.0; iteration_limit];
    let mut sines = vec![0.0; iteration_limit];
    let mut projected_right_hand_side = vec![0.0; iteration_limit + 1];
    projected_right_hand_side[0] = beta0;
    let mut operator_scale = 0.0_f64;
    let mut reported_residual_norm = initial_residual_norm;
    let mut direction = vec![0.; dimension];

    for iteration in 1..=iteration_limit {
        let column = iteration - 1;
        let current_basis = &krylov_basis[column];
        apply_preconditioner(&inverse_sqrt, current_basis, &mut direction);
        require_finite(&direction, "MINRES scaled direction")?;
        execution.apply(problem.operator(), &direction, &mut applied)?;
        scale(&inverse_sqrt, &mut applied)?;
        operator_scale = operator_scale.max(FixedOrderInnerProduct::norm(execution, &applied)?);

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
        let next_beta = FixedOrderInnerProduct::norm(execution, &applied)?;
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
        let projected_residual_norm = projected_right_hand_side[column + 1].abs();
        let next_basis = (iteration < iteration_limit && !space_closed)
            .then(|| applied.iter().map(|value| value / next_beta).collect());

        if inverse_sqrt.is_some()
            || projected_residual_norm <= target
            || space_closed
            || iteration == iteration_limit
        {
            solution = reconstruct_minimum_residual_solution(
                &initial_solution,
                &inverse_sqrt,
                &krylov_basis,
                &rotated_hessenberg,
                &projected_right_hand_side,
                iteration,
            )?;
            require_finite(&solution, "MINRES solution")?;
            let true_residual_norm =
                true_residual_norm(execution, problem, &solution, &mut applied)?;
            // The scaled projected residual has different units and cannot
            // admit the caller's problem. Reapply the original action at every
            // Jacobi iterate, then require its physical candidate predicate.
            reported_residual_norm = if inverse_sqrt.is_some() {
                true_residual_norm
            } else {
                projected_residual_norm
            };
            if (inverse_sqrt.is_some() || projected_residual_norm <= target)
                && true_residual_norm <= target
                && problem.candidate_is_accepted(&solution)?
            {
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
                    "MINRES Krylov space closed at iteration {iteration} without joint acceptance: operator scale {operator_scale:e}, true residual {true_residual_norm:e}, target {target:e}"
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
    inverse_sqrt: &Option<Vec<f64>>,
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

    // Work in correction coordinates. Never form sqrt(D)*x0: that can
    // overflow even when the original initial guess and correction are finite.
    let mut solution = if inverse_sqrt.is_some() {
        vec![0.; initial_solution.len()]
    } else {
        initial_solution.to_vec()
    };
    for (coefficient, basis_vector) in coefficients.iter().zip(&basis[..iterations]) {
        for (solution, basis_value) in solution.iter_mut().zip(basis_vector) {
            *solution += coefficient * basis_value;
        }
    }
    if inverse_sqrt.is_some() {
        scale(inverse_sqrt, &mut solution)?;
        for (value, initial) in solution.iter_mut().zip(initial_solution) {
            *value += initial;
        }
    }
    Ok(solution)
}

fn scale(inverse_sqrt: &Option<Vec<f64>>, values: &mut [f64]) -> Result<(), Diagnostic> {
    if let Some(weights) = inverse_sqrt {
        for (value, weight) in values.iter_mut().zip(weights) {
            *value *= weight;
        }
    }
    require_finite(values, "MINRES symmetric Jacobi coordinates")
}

#[cfg(test)]
mod tests;
