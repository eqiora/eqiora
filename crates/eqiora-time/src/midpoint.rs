//! Implicit midpoint in the common real-coordinate time lifecycle.
use crate::diagnostic::{invalid_plan, time_solve_failed};
use crate::{
    AcceptedTimeHistory, InitialConditionPolicy, TimeBackendIdentity, TimeEquationClass,
    TimeExecutionReport, TimeHistoryStep, TimeMethod, TimePlan, TimeSolution,
};
use eqiora_core::Diagnostic;

const MAX_STEPS: usize = 1_000_000;
/// Fixed-step implicit-midpoint clocks and history with caller-admitted stage solves.
///
/// Complex values use their declared real coordinates; no norm projection or
/// rescaling is performed. Requested observations use the linear collocation
/// polynomial of the accepted step and never select an internal step boundary.
#[derive(Debug, Default, Clone, Copy)]
pub struct ImplicitMidpointTimeBackend;
impl ImplicitMidpointTimeBackend {
    /// Identity of the common host implicit-midpoint implementation.
    pub const IDENTITY: TimeBackendIdentity =
        TimeBackendIdentity::new("eqiora.time.implicit-midpoint", env!("CARGO_PKG_VERSION"));

    /// Domains and real-coordinate precision supported by this adapter.
    pub const CAPABILITIES: crate::TimeBackendCapabilities = crate::TimeBackendCapabilities::new(
        Self::IDENTITY,
        &[
            eqiora_core::ScalarDomain::Real,
            eqiora_core::ScalarDomain::Complex,
        ],
        &[eqiora_core::ScalarType::F64],
    );

    /// Construct the stateless host backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Integrate an already admitted residual formulation with its stage solver.
    ///
    /// The callback solves one midpoint equation from the immutable accepted State
    /// at `time` to `next_time`. It owns nonlinear and linear solver selection;
    /// this owner alone selects internal clocks, accepts stencils and samples
    /// outputs. Initial data must already be consistent with the formulation.
    ///
    /// # Errors
    /// Rejects invalid controls, nonfinite or incomplete stage results, failed
    /// stages, and unrepresentable clocks without returning a partial solution.
    pub fn solve_with_stages(
        &self,
        initial_state: &[f64],
        plan: &TimePlan,
        equation_class: TimeEquationClass,
        mut solve_stage: impl FnMut(f64, f64, &[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<TimeSolution, Diagnostic> {
        if initial_state.is_empty() || initial_state.iter().any(|value| !value.is_finite()) {
            return Err(invalid_plan(
                "midpoint requires a nonempty finite accepted initial state",
            ));
        }
        plan.validate_dimension(initial_state.len())?;
        if plan.method() != TimeMethod::ImplicitMidpoint {
            return Err(invalid_plan(
                "midpoint stage execution requires ImplicitMidpoint",
            ));
        }
        let n = initial_state.len();
        let horizon = *plan
            .output_times()
            .last()
            .expect("validated nonempty output times");
        let count = step_count(plan)?;
        let capacity = n
            .checked_mul(plan.output_times().len())
            .ok_or_else(|| time_solve_failed("implicit midpoint output cardinality overflows"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| time_solve_failed("implicit midpoint output exceeds available storage"))?;
        let mut steps = Vec::new();
        let mut state = initial_state.to_vec();
        let mut time = plan.start_time();
        let mut sample = 0;
        for index in 0..count {
            if time == horizon {
                break;
            }
            let next_time = step_end(plan, index, count);
            if !next_time.is_finite() || next_time <= time {
                return Err(time_solve_failed(
                    "implicit midpoint cannot advance representable time",
                ));
            }
            let candidate = solve_stage(time, next_time, &state)?;
            // Publish a whole finite step only after the solve and stencil validate.
            let accepted = TimeHistoryStep::accepted(
                time,
                next_time,
                state.clone(),
                midpoint(&state, &candidate),
                candidate.clone(),
                vec![time, next_time],
                vec![state.clone(), candidate.clone()],
            )?;
            while sample < plan.output_times().len() && plan.output_times()[sample] <= next_time {
                let output = plan.output_times()[sample];
                values.extend(Self::sample_step(&accepted, output)?);
                sample += 1;
            }
            steps.push(accepted);
            state = candidate;
            time = next_time;
        }
        TimeSolution::accepted_with_history(
            n,
            plan.output_times().to_vec(),
            values,
            TimeExecutionReport::new(
                ImplicitMidpointTimeBackend::IDENTITY,
                TimeMethod::ImplicitMidpoint,
                equation_class,
                InitialConditionPolicy::Provided,
            ),
            AcceptedTimeHistory::accepted(n, steps, Vec::new())?,
        )
    }

    /// Recheck the exact fixed clock and collocation stencils without solving equations.
    /// The formulation owner separately validates every stage's numerical acceptance.
    /// # Errors
    /// Rejects foreign initial data, clocks, methods, dimensions or stencils.
    pub fn validate_history(
        initial: &[f64],
        plan: &TimePlan,
        history: &AcceptedTimeHistory,
    ) -> Result<(), Diagnostic> {
        plan.validate_dimension(initial.len())?;
        let count = step_count(plan)?;
        if plan.method() != TimeMethod::ImplicitMidpoint
            || history.dimension() != initial.len()
            || !history.events().is_empty()
            || history.steps().len() != count
            || history.steps()[0].start_time().to_bits() != plan.start_time().to_bits()
            || history.steps()[0].start_state() != initial
        {
            return Err(invalid_plan(
                "midpoint history differs from its initial state, method or clock",
            ));
        }
        for (index, step) in history.steps().iter().enumerate() {
            if step.end_time().to_bits() != step_end(plan, index, count).to_bits()
                || !has_linear_dense_output(step)
                || step.midpoint_state() != midpoint(step.start_state(), step.end_state())
            {
                return Err(invalid_plan(
                    "midpoint history changed an internal clock or collocation stencil",
                ));
            }
        }
        Ok(())
    }

    /// Evaluate the midpoint method's linear collocation polynomial.
    /// # Errors
    /// Rejects times outside the step or a non-midpoint stencil.
    pub fn sample_step(step: &TimeHistoryStep, time: f64) -> Result<Vec<f64>, Diagnostic> {
        if !time.is_finite()
            || time < step.start_time()
            || time > step.end_time()
            || !has_linear_dense_output(step)
            || step.midpoint_state() != midpoint(step.start_state(), step.end_state())
        {
            return Err(invalid_plan(
                "midpoint sample requires its exact interval and linear stencil",
            ));
        }
        step.sample(time)
    }
}
fn has_linear_dense_output(step: &TimeHistoryStep) -> bool {
    step.dense_times() == [step.start_time(), step.end_time()]
        && step.dense_states()[0] == step.start_state()
        && step.dense_states()[1] == step.end_state()
}
pub(crate) fn midpoint(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| 0.5 * a + 0.5 * b).collect()
}

fn step_count(plan: &TimePlan) -> Result<usize, Diagnostic> {
    let horizon = *plan.output_times().last().expect("validated output times");
    let count = ((horizon - plan.start_time()) / plan.initial_step()).ceil();
    if !count.is_finite() || count < 1. || count > MAX_STEPS as f64 {
        return Err(time_solve_failed(
            "implicit midpoint exceeds its internal-step work budget",
        ));
    }
    // Rounding the quotient upward can add an endpoint already reached by the
    // preceding grid point. Execution and replay share the effective count.
    let mut count = count as usize;
    if count > 1 && plan.start_time() + (count - 1) as f64 * plan.initial_step() >= horizon {
        count -= 1;
    }
    Ok(count)
}
fn step_end(plan: &TimePlan, index: usize, count: usize) -> f64 {
    let horizon = *plan.output_times().last().expect("validated output times");
    if index + 1 == count {
        horizon
    } else {
        (plan.start_time() + (index + 1) as f64 * plan.initial_step()).min(horizon)
    }
}
