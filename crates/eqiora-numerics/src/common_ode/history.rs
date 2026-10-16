//! Model-bound history admission shared by completed trajectories and checkpoints.
use super::*;
use eqiora_time::AcceptedTimeHistory;
impl CommonOdeRunRequest {
    pub(crate) fn validate_history(
        &self,
        history: &AcceptedTimeHistory,
        until_s: f64,
    ) -> Result<(), Diagnostic> {
        let first = history.steps().first().expect("history is nonempty");
        let last = history.steps().last().expect("history is nonempty");
        if history.dimension() != self.plan().state_dimensions().len()
            || first.start_time().to_bits() != self.state().time_s().to_bits()
            || first.start_state() != self.state().values()
            || last.end_time().to_bits() != until_s.to_bits()
        {
            return Err(invalid(
                "accepted ODE history differs from its exact Run interval, initial State, or activation profile",
            ));
        }
        if !history.events().is_empty() {
            let roots = self.plan().root_set()?.ok_or_else(|| {
                invalid("Trajectory events require an admitted registered root set")
            })?;
            let policy = self
                .plan()
                .event_policy()
                .expect("admitted root set has policy");
            if history.events().len() > policy.max_events() {
                return Err(invalid("Trajectory exceeds its exact Plan event budget"));
            }
            for event in history.events() {
                let proposal = event.proposal();
                if proposal.report().backend_identity() != self.plan().backend()
                    || proposal.report().equation_class() != TimeEquationClass::ExplicitOde
                    || proposal.report().method() != TimeMethod::Tsitouras45
                    || proposal.report().initial_condition() != InitialConditionPolicy::Provided
                {
                    return Err(invalid(
                        "Trajectory event report differs from its exact Plan",
                    ));
                }
                let tolerance = self.plan().guard_tolerance(proposal.root_index())?;
                let accepted = roots.linearize_proposal(proposal, tolerance.value())?;
                if accepted.post_state() != event.after_state() {
                    return Err(invalid(
                        "Trajectory event post-State differs from the canonical registered reset",
                    ));
                }
            }
        }
        Ok(())
    }
}
