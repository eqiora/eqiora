//! An unfinished adaptive occurrence bound to its original Model, Plan and Run.
use super::*;
use eqiora_time::{AcceptedTimeHistory, TimeCheckpoint};

mod artifact;

/// Durable continuation of one exact common ODE Run, including committed resets.
///
/// Physical State remains independently reusable for a fresh restart. This
/// record instead preserves the numerical history and original output schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonOdeCheckpoint {
    request: Box<CommonOdeRunRequest>,
    prefix: Option<AcceptedTimeHistory>,
    samples: Vec<f64>,
    segment: TimeCheckpoint,
    identity: String,
}

impl CommonOdeCheckpoint {
    pub(super) fn accept(
        request: CommonOdeRunRequest,
        prefix: Option<AcceptedTimeHistory>,
        samples: Vec<f64>,
        segment: TimeCheckpoint,
    ) -> Result<Self, Diagnostic> {
        request.require_continuation()?;
        let (time, state, cursor) = Self::segment_origin(&request, prefix.as_ref(), &samples)?;
        let (problem, plan) = request.smooth_segment(time, state, cursor)?;
        let roots = request.plan.root_set()?;
        segment.validate_for(
            &problem,
            &plan,
            request.plan.backend(),
            roots.as_ref().map(|roots| roots.registration()),
        )?;
        if let Some(history) = segment.history() {
            for (sample, &time) in plan
                .output_times()
                .iter()
                .take(segment.samples().len() / problem.dimension())
                .enumerate()
            {
                crate::common_trajectory::history_boundary::validate(
                    history,
                    time,
                    &segment.samples()
                        [sample * problem.dimension()..(sample + 1) * problem.dimension()],
                )?;
            }
        }
        // Reuse physical State admission, including the Plan's conserved norms.
        CommonOdeState::new(
            &request.plan,
            segment.time(),
            segment.state().to_vec(),
            "checkpoint",
        )?;
        let mut checkpoint = Self {
            request: Box::new(request),
            prefix,
            samples,
            segment,
            identity: String::new(),
        };
        checkpoint.identity = artifact::identity(&checkpoint)?;
        Ok(checkpoint)
    }

    pub(super) fn segment_origin(
        request: &CommonOdeRunRequest,
        prefix: Option<&AcceptedTimeHistory>,
        samples: &[f64],
    ) -> Result<(f64, Vec<f64>, usize), Diagnostic> {
        let Some(history) = prefix else {
            if !samples.is_empty() {
                return Err(invalid(
                    "checkpoint has output samples without their committed history",
                ));
            }
            return Ok((request.state.time_s(), request.state.values().to_vec(), 0));
        };
        let end = history.steps().last().expect("accepted history").end_time();
        request.validate_history(history, end)?;
        let event = history
            .events()
            .last()
            .filter(|event| event.proposal().time().to_bits() == end.to_bits())
            .ok_or_else(|| {
                invalid("checkpoint committed prefix must end at its last accepted reset")
            })?;
        let cursor = request
            .execution_times_s
            .partition_point(|time| *time <= end);
        let dimension = request.state.values().len();
        if end >= request.until_s
            || cursor.checked_mul(dimension) != Some(samples.len())
            || samples.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "checkpoint committed samples differ from the exact unfinished Run",
            ));
        }
        for (sample, &time) in request.execution_times_s[..cursor].iter().enumerate() {
            crate::common_trajectory::history_boundary::validate(
                history,
                time,
                &samples[sample * dimension..(sample + 1) * dimension],
            )?;
        }
        Ok((end, event.after_state().to_vec(), cursor))
    }

    /// Exact original request, retaining the original initial State and horizon.
    #[must_use]
    pub fn request(&self) -> &CommonOdeRunRequest {
        &self.request
    }
    /// Identity of the bound request, controller state and accepted history.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Last accepted physical State, suitable for an explicitly fresh restart.
    /// # Errors
    /// Propagates the owning Plan's physical State admission diagnostic.
    pub fn state(&self) -> Result<CommonOdeState, Diagnostic> {
        CommonOdeState::new(
            self.request.plan(),
            self.segment.time(),
            self.segment.state().to_vec(),
            "checkpoint",
        )
    }
    /// Total accepted native intervals, including earlier reset segments.
    #[must_use]
    pub fn accepted_steps(&self) -> usize {
        self.prefix
            .as_ref()
            .map_or(0, |history| history.steps().len())
            + self
                .segment
                .history()
                .map_or(0, |history| history.steps().len())
    }
    /// Number of canonical resets already committed against the Run's event budget.
    #[must_use]
    pub fn accepted_events(&self) -> usize {
        self.prefix
            .as_ref()
            .map_or(0, |history| history.events().len())
    }

    pub(super) fn prefix(&self) -> Option<&AcceptedTimeHistory> {
        self.prefix.as_ref()
    }
    pub(super) fn samples(&self) -> &[f64] {
        &self.samples
    }
    pub(super) fn segment(&self) -> &TimeCheckpoint {
        &self.segment
    }
}
