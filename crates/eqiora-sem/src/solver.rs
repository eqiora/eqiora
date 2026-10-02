//! Deliberately small dense Newton solve used only by the semantic oracle.

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath};

#[derive(Debug, Clone, Copy)]
pub(crate) struct NonlinearSettings {
    pub(crate) absolute_tolerance: f64,
    pub(crate) relative_tolerance: f64,
    pub(crate) max_iterations: usize,
}

pub(crate) fn solve<F>(
    initial: Vec<f64>,
    settings: NonlinearSettings,
    path: GraphPath,
    residual: F,
) -> Result<Vec<f64>, Diagnostic>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, Diagnostic>,
{
    solve_checked(initial, settings, path, residual, false)
}

pub(crate) fn solve_initial<F>(
    initial: Vec<f64>,
    settings: NonlinearSettings,
    path: GraphPath,
    residual: F,
) -> Result<Vec<f64>, Diagnostic>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, Diagnostic>,
{
    solve_checked(initial, settings, path, residual, true)
}

fn solve_checked<F>(
    initial: Vec<f64>,
    settings: NonlinearSettings,
    path: GraphPath,
    mut residual: F,
    check_rank: bool,
) -> Result<Vec<f64>, Diagnostic>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, Diagnostic>,
{
    let size = initial.len();
    let mut values = initial;
    let mut residuals = residual(&values)?;
    if residuals.len() != size {
        return Err(Diagnostic::error(
            codes::NONSQUARE_SYSTEM,
            format!(
                "executable-kernel v0 requires a square implicit system; found {} equations and {size} unknowns",
                residuals.len()
            ),
        )
        .with_graph_path(path));
    }
    require_finite(&values, &residuals, &path)?;
    if size == 0 {
        return Ok(values);
    }

    let initial_norm = max_norm(&residuals);
    let tolerance =
        settings.absolute_tolerance + settings.relative_tolerance * initial_norm.max(1.0);

    for _ in 0..settings.max_iterations {
        if !check_rank && max_norm(&residuals) <= tolerance {
            return Ok(values);
        }

        let mut jacobian = vec![vec![0.0; size]; size];
        for column in 0..size {
            let step = 1.490_116_119_384_765_6e-8 * values[column].abs().max(1.0);
            let mut perturbed = values.clone();
            perturbed[column] += step;
            let perturbed_residuals = residual(&perturbed)?;
            if perturbed_residuals.len() != size {
                return Err(Diagnostic::error(
                    codes::NONSQUARE_SYSTEM,
                    "residual equation count changed during nonlinear evaluation",
                )
                .with_graph_path(path));
            }
            require_finite(&perturbed, &perturbed_residuals, &path)?;
            for row in 0..size {
                jacobian[row][column] = (perturbed_residuals[row] - residuals[row]) / step;
            }
        }

        let right_hand_side = residuals.iter().map(|value| -value).collect::<Vec<_>>();
        let update = solve_linear(jacobian, right_hand_side).ok_or_else(|| {
            Diagnostic::error(
                codes::NONLINEAR_SOLVE_FAILED,
                "reference Newton Jacobian is singular",
            )
            .with_graph_path(path.clone())
        })?;
        if max_norm(&residuals) <= tolerance {
            return Ok(values);
        }
        for (value, delta) in values.iter_mut().zip(update) {
            *value += delta;
        }
        residuals = residual(&values)?;
        require_finite(&values, &residuals, &path)?;
    }

    Err(Diagnostic::error(
        codes::NONLINEAR_SOLVE_FAILED,
        format!(
            "reference Newton solve did not converge in {} iterations; residual infinity norm is {}",
            settings.max_iterations,
            max_norm(&residuals)
        ),
    )
    .with_graph_path(path))
}

pub(crate) fn solve_linear(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let size = right.len();
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let pivot_tolerance = f64::EPSILON * scale * (size as f64).max(1.0);
    for pivot_column in 0..size {
        let pivot_row = (pivot_column..size).max_by(|&left, &right_row| {
            matrix[left][pivot_column]
                .abs()
                .total_cmp(&matrix[right_row][pivot_column].abs())
        })?;
        let pivot = matrix[pivot_row][pivot_column];
        if !pivot.is_finite() || pivot.abs() <= pivot_tolerance {
            return None;
        }
        matrix.swap(pivot_column, pivot_row);
        right.swap(pivot_column, pivot_row);

        let pivot_values = matrix[pivot_column].clone();
        let pivot_right = right[pivot_column];
        for (row_values, row_right) in matrix
            .iter_mut()
            .zip(right.iter_mut())
            .skip(pivot_column + 1)
        {
            let factor = row_values[pivot_column] / pivot_values[pivot_column];
            row_values[pivot_column] = 0.0;
            for (value, pivot_value) in row_values
                .iter_mut()
                .zip(&pivot_values)
                .skip(pivot_column + 1)
            {
                *value -= factor * pivot_value;
            }
            *row_right -= factor * pivot_right;
        }
    }

    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let known = ((row + 1)..size)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        let pivot = matrix[row][row];
        if !pivot.is_finite() || pivot.abs() <= pivot_tolerance {
            return None;
        }
        solution[row] = (right[row] - known) / pivot;
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn require_finite(values: &[f64], residuals: &[f64], path: &GraphPath) -> Result<(), Diagnostic> {
    if values
        .iter()
        .chain(residuals)
        .all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(Diagnostic::error(
            codes::NONFINITE_EVALUATION,
            "reference evaluation produced NaN or infinity",
        )
        .with_graph_path(path.clone()))
    }
}

fn max_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_newton_solves_a_small_nonlinear_system() {
        let settings = NonlinearSettings {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-12,
            max_iterations: 32,
        };
        let solution = solve(
            vec![1.0, 1.0],
            settings,
            GraphPath::new(["test"]),
            |values| {
                Ok(vec![
                    values[0] * values[0] + values[1] - 5.0,
                    values[0] + values[1] * values[1] - 5.0,
                ])
            },
        )
        .expect("converges");

        assert!((solution[0] - 1.791_287_847_477_92).abs() < 1.0e-9);
        assert!((solution[1] - 1.791_287_847_477_92).abs() < 1.0e-9);
    }
}
