//! Owned native stencils from accepted steps, independent of output sampling.
use crate::RootProposal;
use crate::diagnostic::time_solve_failed;
use eqiora_core::Diagnostic;

/// One accepted interval with its solver-native midpoint value.
///
/// These three states are a quadrature stencil, not a general dense interpolant.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeHistoryStep {
    start_time: f64,
    end_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    end_state: Vec<f64>,
}

impl TimeHistoryStep {
    /// Accept a finite, consistently shaped native step stencil.
    /// # Errors
    /// Rejects empty or inconsistent state shapes and non-increasing times.
    pub fn accepted(
        start_time: f64,
        end_time: f64,
        start_state: Vec<f64>,
        midpoint_state: Vec<f64>,
        end_state: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if !start_time.is_finite()
            || !end_time.is_finite()
            || start_time >= end_time
            || start_state.is_empty()
            || start_state.len() != midpoint_state.len()
            || start_state.len() != end_state.len()
            || start_state
                .iter()
                .chain(&midpoint_state)
                .chain(&end_state)
                .any(|value| !value.is_finite())
        {
            return Err(time_solve_failed(
                "accepted time step has invalid interval, state shape, or values",
            ));
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            midpoint_state,
            end_state,
        })
    }
    /// Exact beginning of the accepted step.
    #[must_use]
    pub const fn start_time(&self) -> f64 {
        self.start_time
    }
    /// Exact end of the accepted step.
    #[must_use]
    pub const fn end_time(&self) -> f64 {
        self.end_time
    }
    /// State at the start, after any reset at that instant.
    #[must_use]
    pub fn start_state(&self) -> &[f64] {
        &self.start_state
    }
    /// State evaluated by native dense output at the interval midpoint.
    #[must_use]
    pub fn midpoint_state(&self) -> &[f64] {
        &self.midpoint_state
    }
    /// State at the end, before any reset at that instant.
    #[must_use]
    pub fn end_state(&self) -> &[f64] {
        &self.end_state
    }
}

/// Both sides of a committed reset at an exactly registered root proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeEventDiscontinuity {
    proposal: RootProposal,
    after_state: Vec<f64>,
}
impl TimeEventDiscontinuity {
    /// Retain a checked proposal and the state accepted by the reset owner.
    /// # Errors
    /// Rejects inconsistent or non-finite post-reset state data.
    pub fn accepted(proposal: RootProposal, after_state: Vec<f64>) -> Result<Self, Diagnostic> {
        if proposal.state().is_empty()
            || after_state.len() != proposal.state().len()
            || after_state.iter().any(|value| !value.is_finite())
        {
            return Err(time_solve_failed(
                "time event has an invalid post-reset state",
            ));
        }
        Ok(Self {
            proposal,
            after_state,
        })
    }
    /// Root identity, exact time, and pre-reset state supplied by localization.
    #[must_use]
    pub const fn proposal(&self) -> &RootProposal {
        &self.proposal
    }
    /// Exact committed post-reset state.
    #[must_use]
    pub fn after_state(&self) -> &[f64] {
        &self.after_state
    }
}

/// Cadence-independent accepted intervals and explicit event sides.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedTimeHistory {
    dimension: usize,
    steps: Vec<TimeHistoryStep>,
    events: Vec<TimeEventDiscontinuity>,
}
impl AcceptedTimeHistory {
    /// Accept contiguous intervals whose joins are continuous or explicitly reset.
    /// # Errors
    /// Rejects empty, inconsistent, unordered, or unaccounted discontinuous history.
    pub fn accepted(
        dimension: usize,
        steps: Vec<TimeHistoryStep>,
        events: Vec<TimeEventDiscontinuity>,
    ) -> Result<Self, Diagnostic> {
        if dimension == 0
            || steps.is_empty()
            || steps.iter().any(|step| step.start_state.len() != dimension)
            || events
                .iter()
                .any(|event| event.after_state.len() != dimension)
            || events
                .windows(2)
                .any(|pair| pair[0].proposal.time() >= pair[1].proposal.time())
        {
            return Err(time_solve_failed(
                "accepted history has invalid shape or event order",
            ));
        }
        let mut event_index = 0;
        for pair in steps.windows(2) {
            if pair[0].end_time != pair[1].start_time {
                return Err(time_solve_failed(
                    "accepted history intervals are not contiguous",
                ));
            }
            if let Some(event) = events
                .get(event_index)
                .filter(|event| event.proposal.time() == pair[0].end_time)
            {
                if event.proposal.state() != pair[0].end_state
                    || event.after_state != pair[1].start_state
                {
                    return Err(time_solve_failed(
                        "accepted event sides do not match their adjoining steps",
                    ));
                }
                event_index += 1;
            } else if pair[0].end_state != pair[1].start_state {
                return Err(time_solve_failed(
                    "accepted history has an unrecorded state discontinuity",
                ));
            }
        }
        // A reset at the final instant has no following positive-duration interval.
        if let Some(event) = events.get(event_index)
            && event.proposal.time() == steps.last().unwrap().end_time
            && event.proposal.state() == steps.last().unwrap().end_state
        {
            event_index += 1;
        }
        if event_index != events.len() {
            return Err(time_solve_failed(
                "accepted history contains an event outside an exact step boundary",
            ));
        }
        Ok(Self {
            dimension,
            steps,
            events,
        })
    }
    /// Number of scalar coordinates in every stencil state.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Native accepted intervals in execution order.
    #[must_use]
    pub fn steps(&self) -> &[TimeHistoryStep] {
        &self.steps
    }
    /// Exact accepted reset sides in time order.
    #[must_use]
    pub fn events(&self) -> &[TimeEventDiscontinuity] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InitialConditionPolicy, RootRegistrationId, TimeBackendIdentity, TimeEquationClass,
        TimeExecutionReport, TimeMethod,
    };

    fn step(start: f64, end: f64, left: f64, right: f64) -> TimeHistoryStep {
        TimeHistoryStep::accepted(
            start,
            end,
            vec![left],
            vec![(left + right) * 0.5],
            vec![right],
        )
        .unwrap()
    }

    #[test]
    fn reset_sides_join_exactly_and_unrecorded_jumps_fail() {
        // x'=1, x(0)=0, reset to zero at x=1; horizon lies before the second reset.
        let steps = vec![step(0.0, 1.0, 0.0, 1.0), step(1.0, 1.5, 0.0, 0.5)];
        assert!(AcceptedTimeHistory::accepted(1, steps.clone(), vec![]).is_err());
        let report = TimeExecutionReport::new(
            TimeBackendIdentity::new("test.history", "1"),
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        );
        let proposal = RootProposal::accepted(
            RootRegistrationId::from_sha256([1; 32]),
            1.0,
            0,
            1,
            vec![1.0],
            1,
            report,
        )
        .unwrap();
        let event = TimeEventDiscontinuity::accepted(proposal.clone(), vec![0.0]).unwrap();
        let history = AcceptedTimeHistory::accepted(1, steps.clone(), vec![event]).unwrap();
        assert_eq!(history.events()[0].proposal().state(), &[1.0]);
        assert_eq!(history.events()[0].after_state(), &[0.0]);
        let wrong = TimeEventDiscontinuity::accepted(proposal, vec![0.1]).unwrap();
        assert!(AcceptedTimeHistory::accepted(1, steps, vec![wrong]).is_err());
        assert!(
            AcceptedTimeHistory::accepted(
                1,
                vec![step(0.0, 0.5, 0.0, 0.5), step(0.6, 1.0, 0.5, 1.0)],
                vec![]
            )
            .is_err()
        );
    }
}
