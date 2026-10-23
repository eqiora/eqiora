//! Ordinary registered-event execution consumes native smooth prefixes exactly once.
use super::*;
use eqiora_time::{
    AcceptedTimeHistory, RegisteredRootProblem, TimeCheckpoint, TimeEventDiscontinuity,
    TimeExecutionReport, TimeRootOutcome, TimeSolution,
};

mod sensitivity;
use std::ops::ControlFlow;

impl CommonTimeRunRequest {
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
        if self.ode_plan()?.event_policy().is_none() {
            return Err(invalid(
                "registered-event Run requires the Plan's explicit event policy",
            ));
        }
        self.execute_segments(None, |problem, roots, plan, _, _| {
            solve(
                problem,
                roots.expect("admitted registered-event request"),
                plan,
            )
            .map(ControlFlow::Continue)
        })?
        .continue_value()
        .ok_or_else(|| invalid("ordinary event execution returned an unfinished checkpoint"))
    }

    /// Execute or resume the same adaptive Run through accepted native boundaries.
    /// The adapter callback runs once per smooth segment; it is not a progress callback.
    /// Its final argument counts accepted steps in earlier committed segments.
    /// # Errors
    /// Rejects foreign checkpoints, unsupported continuation policies or invalid native prefixes.
    pub fn advance<F>(
        &self,
        checkpoint: Option<&CommonTimeCheckpoint>,
        solve: F,
    ) -> Result<ControlFlow<CommonTimeCheckpoint, crate::CommonTrajectory>, Diagnostic>
    where
        F: FnMut(
            &TimeProblem<'_>,
            Option<&RegisteredRootProblem<'_>>,
            &TimePlan,
            Option<&TimeCheckpoint>,
            usize,
        ) -> Result<ControlFlow<TimeCheckpoint, TimeRootOutcome>, Diagnostic>,
    {
        self.require_continuation()?;
        if checkpoint.is_some_and(|checkpoint| checkpoint.request().identity() != self.identity()) {
            return Err(invalid(
                "checkpoint belongs to a different exact ODE Run request",
            ));
        }
        let checkpoint = checkpoint.map(CommonTimeCheckpoint::adaptive).transpose()?;
        match self.execute_segments(checkpoint, solve)? {
            ControlFlow::Break(value) => Ok(ControlFlow::Break(
                CommonTimeCheckpoint::from_adaptive(value),
            )),
            ControlFlow::Continue(value) => Ok(ControlFlow::Continue(value)),
        }
    }

    /// Validate the exact continuation profile before scheduling a finite-time Run.
    /// # Errors
    /// Rejects methods and sensitivity policies without complete continuation data.
    pub fn require_continuation(&self) -> Result<(), Diagnostic> {
        let Some(plan) = self.plan().as_ode() else {
            return Ok(());
        };
        if plan.temporal.method() != TimeMethod::Tsitouras45
            || self.ode_plan()?.equation_class() != TimeEquationClass::ExplicitOde
            || self.ode_plan()?.forward_sensitivity_plan().is_some()
        {
            return Err(invalid(
                "exact ODE continuation currently requires explicit Tsitouras45 without sensitivity execution",
            ));
        }
        Ok(())
    }

    pub(super) fn smooth_segment(
        &self,
        time: f64,
        state: Vec<f64>,
        cursor: usize,
    ) -> Result<(TimeProblem<'_>, TimePlan), Diagnostic> {
        let remaining = self
            .execution_times_s
            .get(cursor..)
            .ok_or_else(|| invalid("ODE segment output cursor exceeds the request"))?;
        let plan = TimePlan::new(
            self.ode_plan()?.temporal.method(),
            time,
            self.ode_plan()?.temporal.initial_step_s(),
            self.ode_plan()?.temporal.relative_tolerance(),
            self.ode_plan()?.ordered_absolute_tolerances.clone(),
            remaining.to_vec(),
        )?;
        let problem = TimeProblem::from_state(
            &self.ode_plan()?.program,
            TimeEquationClass::ExplicitOde,
            time,
            state,
        )?;
        Ok((problem, plan))
    }

    fn execute_segments<F>(
        &self,
        checkpoint: Option<&AdaptiveCheckpoint>,
        mut solve: F,
    ) -> Result<ControlFlow<AdaptiveCheckpoint, crate::CommonTrajectory>, Diagnostic>
    where
        F: FnMut(
            &TimeProblem<'_>,
            Option<&RegisteredRootProblem<'_>>,
            &TimePlan,
            Option<&TimeCheckpoint>,
            usize,
        ) -> Result<ControlFlow<TimeCheckpoint, TimeRootOutcome>, Diagnostic>,
    {
        let roots = self.ode_plan()?.root_set()?;
        let root_problem = roots
            .as_ref()
            .map(|roots| roots.root_problem())
            .transpose()?;
        let expected_report = TimeExecutionReport::new(
            self.ode_plan()?.backend(),
            self.ode_plan()?.temporal.method(),
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        );
        let (mut time, mut state, mut cursor) = AdaptiveCheckpoint::segment_origin(
            self,
            checkpoint.and_then(|checkpoint| checkpoint.prefix()),
            checkpoint.map_or(&[], |checkpoint| checkpoint.samples()),
        )?;
        let mut samples =
            checkpoint.map_or_else(Vec::new, |checkpoint| checkpoint.samples().to_vec());
        let mut steps = checkpoint
            .and_then(|checkpoint| checkpoint.prefix())
            .map_or_else(Vec::new, |history| history.steps().to_vec());
        let mut events = checkpoint
            .and_then(|checkpoint| checkpoint.prefix())
            .map_or_else(Vec::new, |history| history.events().to_vec());
        let mut native_checkpoint = checkpoint.map(|checkpoint| checkpoint.segment());
        while time < self.until_s {
            let remaining = &self.execution_times_s[cursor..];
            let (problem, segment_plan) = self.smooth_segment(time, state.clone(), cursor)?;
            let outcome = match solve(
                &problem,
                root_problem.as_ref(),
                &segment_plan,
                native_checkpoint.take(),
                steps.len(),
            )? {
                ControlFlow::Continue(outcome) => outcome,
                ControlFlow::Break(segment) => {
                    let prefix = if steps.is_empty() {
                        None
                    } else {
                        Some(AcceptedTimeHistory::accepted(state.len(), steps, events)?)
                    };
                    return AdaptiveCheckpoint::accept(self.clone(), prefix, samples, segment)
                        .map(ControlFlow::Break);
                }
            };
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
                let roots = roots
                    .as_ref()
                    .ok_or_else(|| invalid("root-free ODE Run received a root proposal"))?;
                let policy = self
                    .ode_plan()?
                    .event_policy()
                    .expect("registered roots have a policy");
                if events.len() >= policy.max_events() {
                    return Err(invalid(
                        "registered-event Run exceeded its explicit accepted-event budget",
                    ));
                }
                let tolerance = self.ode_plan()?.guard_tolerance(proposal.root_index())?;
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
        crate::CommonTrajectory::accept_ode(self.clone(), solution).map(ControlFlow::Continue)
    }
}
