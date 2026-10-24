//! Owned native stencils from accepted steps, independent of output sampling.
use crate::RootProposal;
use crate::diagnostic::time_solve_failed;
use eqiora_core::Diagnostic;
mod dense;
use dense::DenseOutput;

/// One accepted interval with its solver-native midpoint value.
///
/// The three states preserve exact native boundary/quadrature values; the separate
/// polynomial supplies off-stencil interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeHistoryStep {
    start_time: f64,
    end_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    end_state: Vec<f64>,
    dense_output: DenseOutput,
}

impl TimeHistoryStep {
    /// Capture a native polynomial while retaining exact accepted stencil values.
    ///
    /// The provider supplies its interpolation degree and an evaluator valid on
    /// this interval. Output-request cadence does not select the capture nodes.
    /// # Errors
    /// Rejects a zero/overflowing degree, invalid samples or provider failures.
    pub fn capture(
        start_time: f64,
        end_time: f64,
        degree: usize,
        start_state: Vec<f64>,
        midpoint_state: Vec<f64>,
        end_state: Vec<f64>,
        mut sample: impl FnMut(f64) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<Self, Diagnostic> {
        if degree == 0
            || !start_time.is_finite()
            || !end_time.is_finite()
            || start_time >= end_time
            || !(end_time - start_time).is_finite()
        {
            return Err(time_solve_failed(
                "native dense output requires a positive degree and finite increasing interval",
            ));
        }
        let count = degree
            .checked_add(1)
            .ok_or_else(|| time_solve_failed("native interpolation degree overflows"))?;
        let mut times = Vec::new();
        times.try_reserve_exact(count).map_err(|_| {
            time_solve_failed("native interpolation nodes exceed allocation capacity")
        })?;
        for index in 0..count {
            times.push(if index == degree {
                end_time
            } else {
                start_time + (end_time - start_time) * (index as f64 / degree as f64)
            });
        }
        let states = times
            .iter()
            .map(|&time| sample(time))
            .collect::<Result<Vec<_>, _>>()?;
        Self::accepted(
            start_time,
            end_time,
            start_state,
            midpoint_state,
            end_state,
            times,
            states,
        )
    }

    /// Accept a native step stencil and its captured dense polynomial.
    /// # Errors
    /// Rejects empty or inconsistent state shapes and non-increasing times.
    pub fn accepted(
        start_time: f64,
        end_time: f64,
        start_state: Vec<f64>,
        midpoint_state: Vec<f64>,
        end_state: Vec<f64>,
        dense_times: Vec<f64>,
        dense_states: Vec<Vec<f64>>,
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
        let dense_output = DenseOutput::from_samples(dense_times, dense_states)?;
        if dense_output.times()[0] != start_time
            || dense_output.times().last().copied() != Some(end_time)
            || dense_output.states()[0].len() != start_state.len()
        {
            return Err(time_solve_failed(
                "dense output differs from its accepted step interval or coordinates",
            ));
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            midpoint_state,
            end_state,
            dense_output,
        })
    }
    /// Physical times defining the native polynomial, including both endpoints.
    #[must_use]
    pub fn dense_times(&self) -> &[f64] {
        self.dense_output.times()
    }

    /// Native polynomial values at the retained sampling times.
    #[must_use]
    pub fn dense_states(&self) -> &[Vec<f64>] {
        self.dense_output.states()
    }

    /// Exact retained quadrature or endpoint value, if this is its timestamp.
    #[must_use]
    fn stencil_state(&self, time: f64) -> Option<&[f64]> {
        if time == self.start_time {
            Some(&self.start_state)
        } else if time == self.end_time {
            Some(&self.end_state)
        } else if time == self.start_time + (self.end_time - self.start_time) * 0.5 {
            Some(&self.midpoint_state)
        } else {
            None
        }
    }

    /// Sample the accepted step, preserving exact retained boundary/stencil values.
    /// # Errors
    /// Rejects extrapolation or non-finite polynomial evaluation.
    pub fn sample(&self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        if let Some(state) = self.stencil_state(time) {
            return Ok(state.to_vec());
        }
        self.dense_output.sample(time)
    }

    /// Differentiate the retained dense polynomial with respect to physical time.
    /// Exact retained State overrides at stencil points do not change this polynomial.
    /// # Errors
    /// Rejects extrapolation or non-finite differentiation.
    pub fn polynomial_derivative(&self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        self.dense_output.derivative(time)
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
            vec![start, start + (end - start) * 0.5, end],
            vec![vec![left], vec![(left + right) * 0.5], vec![right]],
        )
        .unwrap()
    }

    #[test]
    fn native_capture_retains_cubic_polynomial_and_rejects_invalid_degree() {
        let value = |t: f64| 1. + 2. * t - 3. * t * t + 4. * t * t * t;
        let step = TimeHistoryStep::capture(0., 1., 3, vec![1.], vec![1.75], vec![4.], |t| {
            Ok(vec![value(t)])
        })
        .unwrap();
        for time in [0.13, 0.37, 0.83] {
            assert!((step.sample(time).unwrap()[0] - value(time)).abs() < 1e-13);
        }
        assert_eq!(step.sample(0.5).unwrap(), vec![1.75]);
        for degree in [0, usize::MAX] {
            assert!(
                TimeHistoryStep::capture(
                    0.,
                    1.,
                    degree,
                    vec![1.],
                    vec![1.75],
                    vec![4.],
                    |_| panic!("invalid degree must fail before evaluation")
                )
                .is_err()
            );
        }
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
