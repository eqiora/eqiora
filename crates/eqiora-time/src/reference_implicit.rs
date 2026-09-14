use crate::diagnostic::{invalid_plan, time_solve_failed};
use crate::{
    DaeVariableKind, ImplicitDaeInitialization, ImplicitDaeProblem, InitialConditionPolicy,
    TimeBackendIdentity, TimeEquationClass, TimeExecutionReport, TimeMethod, TimePlan,
    TimeSolution,
};
use eqiora_core::Diagnostic;

const MAX_NEWTON_ITERATIONS: usize = 12;
const MAX_LINE_SEARCH_STEPS: usize = 10;
const MAX_INTERNAL_STEPS: usize = 1_000_000;

/// Stable identity of the deterministic residual-native reference oracle.
const REFERENCE_IMPLICIT_TIME_BACKEND: TimeBackendIdentity = TimeBackendIdentity::new(
    "eqiora.time.reference-implicit-euler",
    env!("CARGO_PKG_VERSION"),
);

/// Small deterministic implicit-Euler/Newton oracle for general residuals.
///
/// This backend exists to falsify the residual/JVP and initialization
/// contracts. It is not an adaptive production DAE solver and does not replace
/// a future IDA adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceImplicitTimeBackend;

impl ReferenceImplicitTimeBackend {
    /// Construct the stateless reference oracle.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Solve or accept a consistent initial `(y0, y_dot0)` pair.
    ///
    /// # Errors
    /// Returns stable plan, callback, singular-Jacobian, or nonlinear
    /// convergence diagnostics.
    pub fn initialize(
        &self,
        problem: &ImplicitDaeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<ImplicitDaeInitialization, Diagnostic> {
        admit(problem, plan)?;
        if problem.initial_condition() == InitialConditionPolicy::Provided {
            return ImplicitDaeInitialization::accepted(
                problem.initial_state().to_vec(),
                problem.initial_derivative().to_vec(),
            );
        }

        let dimension = problem.dimension();
        let mut unknown = problem
            .variable_kinds()
            .iter()
            .enumerate()
            .map(|(coordinate, kind)| match kind {
                DaeVariableKind::Differential => problem.initial_derivative()[coordinate],
                DaeVariableKind::Algebraic => problem.initial_state()[coordinate],
            })
            .collect::<Vec<_>>();
        newton(
            &mut unknown,
            plan.absolute_tolerances(),
            plan.relative_tolerance(),
            |point, residual| {
                let (state, derivative) = initial_pair(problem, point);
                problem
                    .system()
                    .residual(plan.start_time(), &state, &derivative, residual)
            },
            |point, direction, output| {
                let (state, derivative) = initial_pair(problem, point);
                let mut state_direction = vec![0.0; dimension];
                let mut derivative_direction = vec![0.0; dimension];
                for (coordinate, kind) in problem.variable_kinds().iter().enumerate() {
                    match kind {
                        DaeVariableKind::Differential => {
                            derivative_direction[coordinate] = direction[coordinate];
                        }
                        DaeVariableKind::Algebraic => {
                            state_direction[coordinate] = direction[coordinate];
                        }
                    }
                }
                problem.system().residual_jvp(
                    plan.start_time(),
                    &state,
                    &derivative,
                    &state_direction,
                    &derivative_direction,
                    output,
                )
            },
        )?;
        let (state, derivative) = initial_pair(problem, &unknown);
        ImplicitDaeInitialization::accepted(state, derivative)
    }

    /// Integrate one residual-native problem with deterministic BDF1 steps.
    ///
    /// `TimePlan::initial_step` is the maximum internal BDF1 step; steps are
    /// shortened to land exactly on requested output times.
    ///
    /// # Errors
    /// Returns stable admission, callback, singular-Jacobian, or nonlinear
    /// convergence diagnostics.
    pub fn solve(
        &self,
        problem: &ImplicitDaeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<TimeSolution, Diagnostic> {
        let initialization = self.initialize(problem, plan)?;
        let dimension = problem.dimension();
        let mut state = initialization.state().to_vec();
        let mut derivative = initialization.derivative().to_vec();
        let mut time = plan.start_time();
        let mut values = Vec::with_capacity(dimension * plan.output_times().len());

        for &output_time in plan.output_times() {
            let interval = output_time - time;
            let step_count = (interval / plan.initial_step()).ceil();
            if !step_count.is_finite() || step_count < 1.0 || step_count > MAX_INTERNAL_STEPS as f64
            {
                return Err(time_solve_failed(
                    "implicit-Euler output interval exceeds the bounded internal-step count",
                ));
            }
            let step_count = step_count as usize;
            let step = interval / step_count as f64;
            if !(step > 0.0 && time + step > time) {
                return Err(time_solve_failed(
                    "implicit-Euler step cannot advance representable model time",
                ));
            }
            for index in 0..step_count {
                let next_time = if index + 1 == step_count {
                    output_time
                } else {
                    time + step
                };
                let (next_state, next_derivative) =
                    implicit_euler_step(problem, plan, time, &state, &derivative, next_time)?;
                state = next_state;
                derivative = next_derivative;
                time = next_time;
            }
            values.extend_from_slice(&state);
        }

        TimeSolution::accepted(
            dimension,
            plan.output_times().to_vec(),
            values,
            TimeExecutionReport::new(
                REFERENCE_IMPLICIT_TIME_BACKEND,
                TimeMethod::ImplicitEuler,
                TimeEquationClass::GeneralImplicitDae,
                problem.initial_condition(),
            ),
        )
    }
}

fn admit(problem: &ImplicitDaeProblem<'_>, plan: &TimePlan) -> Result<(), Diagnostic> {
    plan.validate_for_implicit(problem)?;
    if plan.method() != TimeMethod::ImplicitEuler {
        return Err(invalid_plan(
            "the residual-native reference backend admits only ImplicitEuler",
        ));
    }
    Ok(())
}

fn initial_pair(problem: &ImplicitDaeProblem<'_>, unknown: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut state = problem.initial_state().to_vec();
    let mut derivative = problem.initial_derivative().to_vec();
    for (coordinate, kind) in problem.variable_kinds().iter().enumerate() {
        match kind {
            DaeVariableKind::Differential => derivative[coordinate] = unknown[coordinate],
            DaeVariableKind::Algebraic => state[coordinate] = unknown[coordinate],
        }
    }
    (state, derivative)
}

fn implicit_euler_step(
    problem: &ImplicitDaeProblem<'_>,
    plan: &TimePlan,
    time: f64,
    previous_state: &[f64],
    previous_derivative: &[f64],
    next_time: f64,
) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
    let step = next_time - time;
    let mut state = previous_state
        .iter()
        .zip(previous_derivative)
        .map(|(state, derivative)| state + step * derivative)
        .collect::<Vec<_>>();
    newton(
        &mut state,
        plan.absolute_tolerances(),
        plan.relative_tolerance(),
        |candidate, residual| {
            let derivative = candidate
                .iter()
                .zip(previous_state)
                .map(|(next, previous)| (next - previous) / step)
                .collect::<Vec<_>>();
            problem
                .system()
                .residual(next_time, candidate, &derivative, residual)
        },
        |candidate, direction, output| {
            let derivative = candidate
                .iter()
                .zip(previous_state)
                .map(|(next, previous)| (next - previous) / step)
                .collect::<Vec<_>>();
            let derivative_direction = direction
                .iter()
                .map(|direction| direction / step)
                .collect::<Vec<_>>();
            problem.system().residual_jvp(
                next_time,
                candidate,
                &derivative,
                direction,
                &derivative_direction,
                output,
            )
        },
    )?;
    let derivative = state
        .iter()
        .zip(previous_state)
        .map(|(next, previous)| (next - previous) / step)
        .collect();
    Ok((state, derivative))
}

fn newton<E, J>(
    point: &mut [f64],
    absolute_tolerances: &[f64],
    relative_tolerance: f64,
    mut evaluate: E,
    mut jvp: J,
) -> Result<(), Diagnostic>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<(), Diagnostic>,
    J: FnMut(&[f64], &[f64], &mut [f64]) -> Result<(), Diagnostic>,
{
    let dimension = point.len();
    let mut residual = vec![0.0; dimension];
    evaluate(point, &mut residual)?;
    require_finite(&residual, "implicit residual")?;
    if infinity_norm(&residual) == 0.0 {
        return Ok(());
    }

    for _ in 0..MAX_NEWTON_ITERATIONS {
        let mut jacobian = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            let mut direction = vec![0.0; dimension];
            direction[column] = 1.0;
            let mut action = vec![0.0; dimension];
            jvp(point, &direction, &mut action)?;
            require_finite(&action, "implicit residual JVP")?;
            for row in 0..dimension {
                jacobian[row * dimension + column] = action[row];
            }
        }
        let right_hand_side = residual.iter().map(|value| -*value).collect::<Vec<_>>();
        let correction = solve_dense(jacobian, right_hand_side)?;
        let correction_norm =
            weighted_rms(&correction, point, relative_tolerance, absolute_tolerances);
        if correction_norm <= 1.0 {
            for (point, correction) in point.iter_mut().zip(&correction) {
                *point += correction;
            }
            require_finite(point, "implicit Newton iterate")?;
            evaluate(point, &mut residual)?;
            require_finite(&residual, "implicit residual")?;
            return Ok(());
        }
        let previous_norm = euclidean_norm(&residual);
        let mut accepted = None;
        let mut scale = 1.0;
        for _ in 0..=MAX_LINE_SEARCH_STEPS {
            let candidate = point
                .iter()
                .zip(&correction)
                .map(|(point, correction)| point + scale * correction)
                .collect::<Vec<_>>();
            if candidate.iter().all(|value| value.is_finite()) {
                let mut candidate_residual = vec![0.0; dimension];
                evaluate(&candidate, &mut candidate_residual)?;
                require_finite(&candidate_residual, "implicit residual")?;
                if euclidean_norm(&candidate_residual) < previous_norm {
                    accepted = Some((candidate, candidate_residual, scale));
                    break;
                }
            }
            scale *= 0.5;
        }
        let Some((candidate, candidate_residual, scale)) = accepted else {
            return Err(time_solve_failed(
                "residual-native Newton line search failed to decrease the residual",
            ));
        };
        point.copy_from_slice(&candidate);
        residual = candidate_residual;
        if infinity_norm(&residual) == 0.0 || scale * correction_norm <= 1.0 {
            return Ok(());
        }
    }
    Err(time_solve_failed(
        "residual-native Newton iteration did not converge within its reference bound",
    ))
}

fn solve_dense(
    mut matrix: Vec<f64>,
    mut right_hand_side: Vec<f64>,
) -> Result<Vec<f64>, Diagnostic> {
    let dimension = right_hand_side.len();
    let scale = matrix
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0);
    let pivot_tolerance = f64::EPSILON * scale * (dimension as f64).max(1.0);
    for column in 0..dimension {
        let pivot_row = (column..dimension)
            .max_by(|left, right| {
                matrix[*left * dimension + column]
                    .abs()
                    .total_cmp(&matrix[*right * dimension + column].abs())
            })
            .expect("a non-empty trailing pivot range exists");
        let pivot = matrix[pivot_row * dimension + column];
        if !pivot.is_finite() || pivot.abs() <= pivot_tolerance {
            return Err(time_solve_failed(
                "residual-native Newton Jacobian is singular at the current iterate",
            ));
        }
        if pivot_row != column {
            for entry in 0..dimension {
                matrix.swap(column * dimension + entry, pivot_row * dimension + entry);
            }
            right_hand_side.swap(column, pivot_row);
        }
        for row in (column + 1)..dimension {
            let factor = matrix[row * dimension + column] / matrix[column * dimension + column];
            matrix[row * dimension + column] = 0.0;
            for trailing in (column + 1)..dimension {
                matrix[row * dimension + trailing] -=
                    factor * matrix[column * dimension + trailing];
            }
            right_hand_side[row] -= factor * right_hand_side[column];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let remainder = ((row + 1)..dimension)
            .map(|column| matrix[row * dimension + column] * solution[column])
            .sum::<f64>();
        let pivot = matrix[row * dimension + row];
        if !pivot.is_finite() || pivot.abs() <= pivot_tolerance {
            return Err(time_solve_failed(
                "residual-native Newton Jacobian is singular during back substitution",
            ));
        }
        solution[row] = (right_hand_side[row] - remainder) / pivot;
    }
    require_finite(&solution, "implicit Newton correction")?;
    Ok(solution)
}

fn weighted_rms(
    correction: &[f64],
    point: &[f64],
    relative_tolerance: f64,
    absolute_tolerances: &[f64],
) -> f64 {
    let sum = correction
        .iter()
        .zip(point)
        .zip(absolute_tolerances)
        .map(|((correction, point), absolute)| {
            let scaled = correction / (relative_tolerance * point.abs() + absolute);
            scaled * scaled
        })
        .sum::<f64>();
    (sum / correction.len() as f64).sqrt()
}

fn euclidean_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

fn require_finite(values: &[f64], name: &str) -> Result<(), Diagnostic> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(time_solve_failed(format!(
            "{name} contains a non-finite value"
        )))
    }
}
