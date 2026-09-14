//! Forward products across the same accepted registered-event execution.
use super::*;
use crate::common_ode::parameter_system::GlobalParameterSystem;
use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use eqiora_time::{ForwardSensitivityProblem, TimeRootSensitivityOutcome};

impl CommonOdeRunRequest {
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
        ) -> Result<TimeRootSensitivityOutcome, Diagnostic>,
    {
        if self.state.time_s() != 0.0 || self.state.values() != self.plan.initial_state()?.values()
        {
            return Err(invalid(
                "event Parameter sensitivity requires the exact Model initial State",
            ));
        }
        let roots = self.plan.root_set()?.ok_or_else(|| {
            invalid("event Parameter sensitivity requires the explicit registered-event policy")
        })?;
        let mut system = GlobalParameterSystem::new(&self.plan, &roots)?;
        let parameters = system.parameter_ids().to_vec();
        let parameter_dimension = parameters.len();
        let state_dimension = self.state.values().len();
        let mut steps = Vec::new();
        let mut event_time_gradients = Vec::new();
        let trajectory = self.run_with_events(|problem, registered_roots, plan| {
            let problem = ForwardSensitivityProblem::new(
                &system, TimeEquationClass::ExplicitOde, InitialConditionPolicy::Provided,
                problem.initial_state().to_vec(),
            )?;
            let outcome = solve(&problem, registered_roots, plan)?;
            if outcome.parameter_dimension() != parameter_dimension {
                return Err(invalid("event sensitivity backend returned a foreign Parameter layout"));
            }
            let tangent = outcome.sensitivity_history();
            let primal = outcome.primal();
            steps.extend_from_slice(tangent.steps());
            if let Some(proposal) = primal.proposal() {
                if proposal.time() <= self.state.time_s() || proposal.time() >= self.until_s {
                    return Err(invalid("event functional sensitivity is not admitted at a fixed integration endpoint"));
                }
                let tolerance = self.plan.guard_tolerance(proposal.root_index())?;
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
                system = system.with_initial_jacobian(accepted.post_state().to_vec())?;
            }
            Ok(primal.clone())
        })?;
        let sensitivity = CommonTrajectoryParameterSensitivity::accept_event_history(
            &trajectory,
            parameters,
            steps,
            event_time_gradients,
        )?;
        Ok((trajectory, sensitivity))
    }
}
