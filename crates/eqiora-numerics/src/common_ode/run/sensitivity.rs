//! Forward products across the same accepted registered-event execution.
use super::*;
use crate::common_ode::parameter_system::GlobalParameterSystem;
use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use eqiora_time::{ForwardSensitivityProblem, ParametricTimeSystem, TimeRootOutcome};

impl CommonTimeRunRequest {
    /// Execute registered events with native continuous forward sensitivity history.
    ///
    /// The canonical event owner supplies guard-time and reset products. This
    /// bounded method rejects resumed initial data and events at either fixed
    /// integration endpoint. Adaptive-controller, adjoint, and delay products
    /// are not provided by this method.
    pub fn run_with_event_forward_sensitivities<F>(
        &self,
        mut solve: F,
    ) -> Result<
        (
            crate::CommonTrajectory,
            CommonTrajectoryParameterSensitivity,
        ),
        Diagnostic,
    >
    where
        F: FnMut(
            &ForwardSensitivityProblem<'_>,
            &RegisteredRootProblem<'_>,
            &TimePlan,
        ) -> Result<TimeRootOutcome, Diagnostic>,
    {
        if self.state != self.ode_plan()?.initial_state(self.state.time_s())? {
            return Err(invalid(
                "event Parameter sensitivity requires the exact Model initial State",
            ));
        }
        let roots = self.ode_plan()?.root_set()?.ok_or_else(|| {
            invalid("event Parameter sensitivity requires the explicit registered-event policy")
        })?;
        let mut system = GlobalParameterSystem::new(self.ode_plan()?, &roots)?;
        let parameters = system.parameter_ids().to_vec();
        let parameter_dimension = parameters.len();
        let state_dimension = self.state.values().len();
        system.initial_parameter_jvp(
            self.state.time_s(),
            &vec![0.; parameter_dimension],
            &mut vec![0.; state_dimension],
        )?;
        let mut steps = Vec::new();
        let mut event_time_gradients = Vec::new();
        let mut output_products = Vec::new();
        let trajectory = self.run_with_events(|problem, registered_roots, plan| {
            let problem = ForwardSensitivityProblem::new(
                &system, TimeEquationClass::ExplicitOde,
                problem.initialization().clone(),
            )?;
            let outcome = solve(&problem, registered_roots, plan)?;
            if outcome.parameter_dimension() != parameter_dimension {
                return Err(invalid("event sensitivity backend returned a foreign Parameter layout"));
            }
            let tangent = outcome.sensitivity_history().ok_or_else(|| invalid("event sensitivity backend omitted native sensitivity history"))?;
            let primal = &outcome;
            if let Some(samples) = primal.samples() {
                for (sample, &time) in samples.times().iter().enumerate() {
                    let values = (0..parameter_dimension).flat_map(|parameter|
                        outcome.sensitivity(parameter, sample).expect("accepted output product layout").iter().copied()).collect::<Vec<_>>();
                    output_products.push((time, values));
                }
            }
            steps.extend_from_slice(tangent.steps());
            if let Some(proposal) = primal.proposal() {
                if proposal.time() <= self.state.time_s() || proposal.time() >= self.until_s {
                    return Err(invalid("event functional sensitivity is not admitted at a fixed integration endpoint"));
                }
                let tolerance = self.ode_plan()?.guard_tolerance(proposal.root_index())?;
                let canonical = roots.linearize_proposal(proposal, tolerance.value())?;
                let derivatives = system.lift_event(&roots.events()[proposal.root_index()], canonical.derivatives())?;
                let fixed_time = tangent.steps().last().expect("accepted native sensitivity prefix").end_state();
                let mut before = vec![0.0; fixed_time.len()];
                for state in 0..state_dimension {
                    for parameter in 0..parameter_dimension {
                        before[state * parameter_dimension + parameter] = fixed_time[parameter * state_dimension + state];
                    }
                }
                let accepted = derivatives.propagate_forward(&before)?;
                event_time_gradients.push(accepted.event_time().to_vec());
                if self.execution_times_s.contains(&proposal.time()) {
                    let mut values = vec![0.; state_dimension * parameter_dimension];
                    for parameter in 0..parameter_dimension {
                        for state in 0..state_dimension {
                            values[parameter * state_dimension + state] = accepted.post_state()[state * parameter_dimension + parameter];
                        }
                    }
                    output_products.push((proposal.time(), values));
                }
                system = system.with_initial_jacobian(accepted.post_state().to_vec())?;
            }
            Ok(outcome)
        })?;
        let states = trajectory.time_states().expect("accepted ODE");
        if output_products.len() != self.execution_times_s.len()
            || output_products
                .iter()
                .zip(&self.execution_times_s)
                .any(|((time, _), expected)| time != expected)
        {
            return Err(invalid(
                "event sensitivities omitted or substituted requested output times",
            ));
        }
        let outputs = output_products
            .into_iter()
            .take(states.len())
            .map(|(_, values)| values)
            .collect();
        let sensitivity = CommonTrajectoryParameterSensitivity::accept_event_history(
            &trajectory,
            parameters,
            steps,
            outputs,
            event_time_gradients,
        )?;
        Ok((trajectory, sensitivity))
    }
}
