//! Canonical persistence of the accepted prefix and exact adaptive controller.
use super::*;
use crate::common_time::request_artifact::WireTimeRequest;
use crate::time_history_artifact::{WireReport, WireTimeStep, events};
use eqiora_time::RootRegistrationId;
use serde::{Deserialize, Serialize};

use crate::common_time::checkpoint;
const PROFILE: &str = "adaptive-ode";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    request: WireTimeRequest,
    committed_steps: Vec<WireTimeStep>,
    committed_events: Vec<events::WireEvent>,
    committed_samples: Vec<f64>,
    segment: WireSegment,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSegment {
    report: WireReport,
    registration: Option<[u8; 32]>,
    time: f64,
    next_step: f64,
    derivative: Vec<f64>,
    steps: Vec<WireTimeStep>,
    samples: Vec<f64>,
}

impl AdaptiveCheckpoint {
    /// Encode the original request, committed resets and unfinished adaptive segment.
    /// # Errors
    /// Rejects an artifact exceeding the bounded canonical representation.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        checkpoint::encode(PROFILE, &Payload::encode(self)?)
    }

    /// Reopen and reaccept a checkpoint against the exact owning Plan.
    /// # Errors
    /// Rejects changed requests/providers, omitted history/controller data,
    /// noncanonical/truncated input and resets that differ from the Model.
    pub fn from_bytes(bytes: &[u8], plan: &CommonOdePlan) -> Result<Self, Diagnostic> {
        let checkpoint =
            checkpoint::decode(bytes, PROFILE, |payload: Payload| payload.replay(plan))?;
        if checkpoint.to_bytes()? != bytes {
            return Err(invalid("time checkpoint differs from its admitted replay"));
        }
        Ok(checkpoint)
    }
}

pub(super) fn identity(value: &AdaptiveCheckpoint) -> Result<String, Diagnostic> {
    checkpoint::identity(PROFILE, &Payload::encode(value)?)
}

impl Payload {
    fn encode(checkpoint: &AdaptiveCheckpoint) -> Result<Self, Diagnostic> {
        let segment = &checkpoint.segment;
        Ok(Self {
            request: WireTimeRequest::from_request(&checkpoint.request)?,
            committed_steps: checkpoint.prefix.as_ref().map_or_else(Vec::new, |history| {
                history.steps().iter().map(WireTimeStep::from).collect()
            }),
            committed_events: events::encode(
                checkpoint
                    .prefix
                    .as_ref()
                    .map_or(&[], |history| history.events()),
            )?,
            committed_samples: checkpoint.samples.clone(),
            segment: WireSegment {
                report: WireReport::from_native(segment.report())?,
                registration: segment
                    .registration()
                    .map(|registration| registration.as_sha256()),
                time: segment.time(),
                next_step: segment.next_step(),
                derivative: segment.derivative().to_vec(),
                steps: segment.history().map_or_else(Vec::new, |history| {
                    history.steps().iter().map(WireTimeStep::from).collect()
                }),
                samples: segment.samples().to_vec(),
            },
        })
    }

    fn replay(self, plan: &CommonOdePlan) -> Result<AdaptiveCheckpoint, Diagnostic> {
        let request = self.request.replay(&plan.clone().into())?;
        request.require_continuation()?;
        let steps = self
            .committed_steps
            .iter()
            .map(WireTimeStep::replay)
            .collect::<Result<Vec<_>, _>>()?;
        let events = events::replay(&self.committed_events, plan)?;
        let prefix = if steps.is_empty() {
            if !events.is_empty() {
                return Err(invalid(
                    "checkpoint reset receipts lack their accepted prefix",
                ));
            }
            None
        } else {
            Some(AcceptedTimeHistory::accepted(
                plan.state_dimensions().len(),
                steps,
                events,
            )?)
        };
        let (time, state, cursor) =
            AdaptiveCheckpoint::segment_origin(&request, prefix.as_ref(), &self.committed_samples)?;
        let (problem, time_plan) = request.smooth_segment(time, state, cursor)?;
        self.segment.report.replay(plan)?;
        let mut segment = TimeCheckpoint::initial(
            &problem,
            &time_plan,
            plan.backend(),
            self.segment
                .registration
                .map(RootRegistrationId::from_sha256),
            self.segment.next_step,
            self.segment.derivative.clone(),
        )?;
        if !self.segment.steps.is_empty() {
            let steps = self
                .segment
                .steps
                .iter()
                .map(WireTimeStep::replay)
                .collect::<Result<Vec<_>, _>>()?;
            segment = segment.with_history(
                AcceptedTimeHistory::accepted(problem.dimension(), steps, Vec::new())?,
                self.segment.samples,
                self.segment.next_step,
                self.segment.derivative,
            )?;
        } else if !self.segment.samples.is_empty() {
            return Err(invalid(
                "checkpoint samples lack their native accepted history",
            ));
        }
        if segment.time().to_bits() != self.segment.time.to_bits() {
            return Err(invalid(
                "checkpoint time differs from its retained native history",
            ));
        }
        AdaptiveCheckpoint::accept(request, prefix, self.committed_samples, segment)
    }
}
