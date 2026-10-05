//! Implicit midpoint in the common real-coordinate time lifecycle.
use crate::diagnostic::{invalid_plan, time_solve_failed};
use crate::{
    AcceptedTimeHistory, InitialConditionPolicy, MassMatrixRank, TimeBackendIdentity,
    TimeEquationClass, TimeExecutionReport, TimeHistoryStep, TimeMethod, TimePlan, TimeProblem,
    TimeSolution,
};
use eqiora_core::Diagnostic;

const MAX_STEPS: usize = 1_000_000;
/// Fixed-step implicit midpoint over the ordinary RHS/JVP and mass actions.
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

    /// Integrate from the supplied accepted state with the Plan's fixed step.
    /// The final step alone is shortened to the last requested time (the Run
    /// horizon). Tolerances control Newton corrections, not temporal truncation
    /// error. The returned history owns every accepted collocation stencil.
    ///
    /// # Errors
    /// Rejects unsupported methods, singular mass/DAE profiles, nonfinite
    /// callbacks, failed Newton solves, and unrepresentable times or storage.
    pub fn solve(
        &self,
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<TimeSolution, Diagnostic> {
        plan.validate_for(problem)?;
        if plan.method() != TimeMethod::ImplicitMidpoint {
            return Err(invalid_plan(
                "implicit-midpoint backend requires the ImplicitMidpoint method",
            ));
        }
        if !matches!(
            problem.equation_class(),
            TimeEquationClass::ExplicitOde
                | TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::Full
                }
        ) || problem.initial_condition() != InitialConditionPolicy::Provided
        {
            return Err(invalid_plan(
                "implicit midpoint requires an ODE with a provided consistent state",
            ));
        }
        let n = problem.dimension();
        let horizon = *plan
            .output_times()
            .last()
            .expect("validated nonempty output times");
        let count = ((horizon - plan.start_time()) / plan.initial_step()).ceil();
        if !count.is_finite() || count < 1. || count > MAX_STEPS as f64 {
            return Err(time_solve_failed(
                "implicit midpoint exceeds its internal-step work budget",
            ));
        }
        let capacity = n
            .checked_mul(plan.output_times().len())
            .ok_or_else(|| time_solve_failed("implicit midpoint output cardinality overflows"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| time_solve_failed("implicit midpoint output exceeds available storage"))?;
        let mut steps = Vec::new();
        let mut state = problem.initial_state().to_vec();
        let mut time = plan.start_time();
        let mut sample = 0;
        for index in 0..count as usize {
            if time == horizon {
                break;
            }
            let next_time = if index + 1 == count as usize {
                horizon
            } else {
                (plan.start_time() + (index + 1) as f64 * plan.initial_step()).min(horizon)
            };
            if !next_time.is_finite() || next_time <= time {
                return Err(time_solve_failed(
                    "implicit midpoint cannot advance representable time",
                ));
            }
            let step = next_time - time;
            let middle_time = time + step / 2.;
            let mut candidate = state.clone();
            crate::reference_implicit::newton(
                &mut candidate,
                plan.absolute_tolerances(),
                plan.relative_tolerance(),
                |next, residual| {
                    let middle = midpoint(&state, next);
                    let difference = next
                        .iter()
                        .zip(&state)
                        .map(|(b, a)| b - a)
                        .collect::<Vec<_>>();
                    mass(problem, middle_time, &difference, residual)?;
                    let mut rhs = vec![0.; n];
                    problem.system().rhs(middle_time, &middle, &mut rhs)?;
                    for (value, rhs) in residual.iter_mut().zip(rhs) {
                        *value -= step * rhs;
                    }
                    Ok(())
                },
                |next, direction, output| {
                    mass(problem, middle_time, direction, output)?;
                    let mut rhs = vec![0.; n];
                    problem.system().rhs_jvp(
                        middle_time,
                        &midpoint(&state, next),
                        direction,
                        &mut rhs,
                    )?;
                    for (value, rhs) in output.iter_mut().zip(rhs) {
                        *value -= step / 2. * rhs;
                    }
                    Ok(())
                },
            )?;
            // Publish a whole finite step only after the solve and stencil validate.
            let accepted = TimeHistoryStep::accepted(
                time,
                next_time,
                state.clone(),
                midpoint(&state, &candidate),
                candidate.clone(),
            )?;
            while sample < plan.output_times().len() && plan.output_times()[sample] <= next_time {
                let output = plan.output_times()[sample];
                if output == next_time {
                    values.extend_from_slice(&candidate);
                } else if output == middle_time {
                    values.extend_from_slice(accepted.midpoint_state());
                } else {
                    let weight = (output - time) / step;
                    values.extend(
                        state
                            .iter()
                            .zip(&candidate)
                            .map(|(a, b)| (1. - weight) * a + weight * b),
                    );
                }
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
                problem.equation_class(),
                problem.initial_condition(),
            ),
            AcceptedTimeHistory::accepted(n, steps, Vec::new())?,
        )
    }
}
fn midpoint(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| 0.5 * a + 0.5 * b).collect()
}
fn mass(
    problem: &TimeProblem<'_>,
    time: f64,
    direction: &[f64],
    output: &mut [f64],
) -> Result<(), Diagnostic> {
    if problem.equation_class() == TimeEquationClass::ExplicitOde {
        output.copy_from_slice(direction);
        Ok(())
    } else {
        problem.system().mass_action(time, direction, output)
    }
}
