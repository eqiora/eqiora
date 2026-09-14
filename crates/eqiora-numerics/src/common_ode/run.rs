//! Ordinary registered-event execution consumes native smooth prefixes exactly once.
use super::*;
use eqiora_time::{
    AcceptedTimeHistory, RegisteredRootProblem, TimeEventDiscontinuity, TimeExecutionReport,
    TimeRootOutcome, TimeSolution,
};

mod sensitivity;

impl CommonOdeRunRequest {
    /// Execute the Plan's admitted registered events through a numerical backend.
    ///
    /// The callback localizes the first root and retains its native smooth prefix.
    /// This owner checks the exact registration, commits the canonical reset,
    /// enforces the explicit event budget and restarts at the accepted post-state.
    /// Requested samples at an event use that post-state. No output sampling
    /// operation chooses or reconstructs the accepted integration history.
    pub fn run_with_events<F>(&self, mut solve: F) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnMut(
            &TimeProblem<'_>,
            &RegisteredRootProblem<'_>,
            &TimePlan,
        ) -> Result<TimeRootOutcome, Diagnostic>,
    {
        let roots = self.plan.root_set()?.ok_or_else(|| {
            invalid("registered-event Run requires the Plan's explicit event policy")
        })?;
        let policy = self.plan.event_policy().ok_or_else(|| {
            invalid("registered-event Run has no admitted numerical event policy")
        })?;
        let root_problem = roots.root_problem()?;
        let expected_report = TimeExecutionReport::new(
            self.plan.backend(),
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        );
        let mut time = self.state.time_s();
        let mut state = self.state.values().to_vec();
        let mut samples = Vec::new();
        let mut cursor = 0;
        let mut steps = Vec::new();
        let mut events = Vec::new();
        while time < self.until_s {
            let remaining = &self.execution_times_s[cursor..];
            let segment_plan = TimePlan::new(
                TimeMethod::Tsitouras45,
                time,
                self.plan.temporal.initial_step_s(),
                self.plan.temporal.relative_tolerance(),
                self.plan.ordered_absolute_tolerances.clone(),
                remaining.to_vec(),
            )?;
            let problem = TimeProblem::new(
                &self.plan.program,
                TimeEquationClass::ExplicitOde,
                InitialConditionPolicy::Provided,
                state.clone(),
            )?;
            let outcome = solve(&problem, &root_problem, &segment_plan)?;
            let history = outcome.history();
            let first = &history.steps()[0];
            let last = history.steps().last().expect("accepted prefix is nonempty");
            let report = outcome.proposal().map_or_else(
                || {
                    outcome
                        .samples()
                        .expect("horizon has native samples")
                        .report()
                },
                |proposal| proposal.report(),
            );
            if report != expected_report
                || first.start_time().to_bits() != time.to_bits()
                || first.start_state() != state
                || history.dimension() != state.len()
                || last.end_time() > self.until_s
                || last.end_time() <= time
            {
                return Err(invalid(
                    "native root-search prefix differs from the exact Run State, interval, or backend",
                ));
            }
            let count = if outcome.proposal().is_some() {
                remaining
                    .iter()
                    .take_while(|output| **output < last.end_time())
                    .count()
            } else {
                remaining.len()
            };
            if outcome.samples().is_none() != (count == 0)
                || outcome
                    .samples()
                    .is_some_and(|samples| samples.times() != &remaining[..count])
            {
                return Err(invalid(
                    "native root-search prefix omitted or substituted requested output times",
                ));
            }
            if let Some(segment_samples) = outcome.samples() {
                for sample in 0..count {
                    samples.extend_from_slice(
                        segment_samples
                            .state(sample)
                            .expect("accepted sample shape"),
                    );
                }
            }
            cursor += count;
            steps.extend_from_slice(history.steps());
            if let Some(proposal) = outcome.proposal() {
                if events.len() >= policy.max_events() {
                    return Err(invalid(
                        "registered-event Run exceeded its explicit accepted-event budget",
                    ));
                }
                let tolerance = self.plan.guard_tolerance(proposal.root_index())?;
                let event = roots.linearize_proposal(proposal, tolerance.value())?;
                state = event.post_state().to_vec();
                time = proposal.time();
                events.push(TimeEventDiscontinuity::accepted(
                    proposal.clone(),
                    state.clone(),
                )?);
                if self
                    .execution_times_s
                    .get(cursor)
                    .is_some_and(|sample| sample.to_bits() == time.to_bits())
                {
                    samples.extend_from_slice(&state);
                    cursor += 1;
                }
            } else {
                if last.end_time().to_bits() != self.until_s.to_bits() {
                    return Err(invalid(
                        "root-search horizon differs from the exact Run terminal time",
                    ));
                }
                time = last.end_time();
            }
        }
        if cursor != self.execution_times_s.len() {
            return Err(invalid(
                "registered-event Run did not produce every exact requested sample",
            ));
        }
        let history = AcceptedTimeHistory::accepted(state.len(), steps, events)?;
        let solution = TimeSolution::accepted_with_history(
            state.len(),
            self.execution_times_s.clone(),
            samples,
            expected_report,
            history,
        )?;
        crate::CommonTrajectory::accept_ode(self.clone(), solution)
    }
}
