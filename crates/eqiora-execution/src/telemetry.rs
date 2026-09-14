//! Solver-agnostic process-local execution telemetry.
//!
//! The fixed macro arms keep tracing callsites static and keep telemetry out of
//! Model, Plan, Result, and artifact identity.

/// Create one structured Eqiora execution phase span.
#[macro_export]
macro_rules! telemetry_span {
    (run($family:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "run", family = $family)
    };
    (setup($role:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "setup", role = $role)
    };
    (solve($index:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "solve", solve = $index)
    };
    (time_step($step:expr, $time_s:expr, $dt_s:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "time_step", step = $step, time_s = $time_s, dt_s = $dt_s)
    };
    (assembly($role:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "assembly", role = $role)
    };
    (nonlinear_iteration($solver:expr, $iteration:expr, $residual_norm:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "nonlinear_iteration", nonlinear_solver = $solver, iteration = $iteration, residual_norm = $residual_norm)
    };
    (linear_solve($solver:expr, $provider:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "linear_solve", linear_solver = $solver, solver_provider = $provider)
    };
    (backend($name:expr, $backend:expr)) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = $name, backend = $backend)
    };
    (postprocess) => {
        tracing::span!(target: "eqiora::execution", tracing::Level::INFO, "eqiora_phase", phase = "postprocess")
    };
}

/// Emit one structured Eqiora execution event.
#[macro_export]
macro_rules! telemetry_event {
    ($solver:expr, $iteration:expr, $residual_norm:expr, $residual_target:expr, $converged:expr $(,)?) => {
        tracing::event!(target: "eqiora::execution", tracing::Level::INFO, event = "nonlinear_status", nonlinear_solver = $solver, iteration = $iteration, residual_norm = $residual_norm, residual_target = $residual_target, converged = $converged)
    };
}
