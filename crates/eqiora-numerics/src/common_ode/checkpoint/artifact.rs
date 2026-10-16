//! Canonical persistence of the accepted prefix and exact adaptive controller.
use super::*;
use crate::common_ode::request_artifact::WireOdeRequest;
use crate::time_history_artifact::{WireReport, WireTimeStep, events};
use eqiora_time::RootRegistrationId;
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "eqiora.common-ode-checkpoint/v1";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCheckpoint {
    schema: String,
    encoding: String,
    identity: String,
    payload: Payload,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    request: WireOdeRequest,
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

impl CommonOdeCheckpoint {
    /// Encode the original request, committed resets and unfinished adaptive segment.
    /// # Errors
    /// Rejects an artifact exceeding the bounded canonical representation.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes = serde_json::to_vec(&WireCheckpoint {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            identity: self.identity.clone(),
            payload: Payload::encode(self)?,
        })
        .map_err(|error| invalid(format!("cannot encode ODE checkpoint: {error}")))?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid("ODE checkpoint exceeds its artifact size limit"));
        }
        Ok(bytes)
    }

    /// Reopen and reaccept a checkpoint against the exact owning Plan.
    /// # Errors
    /// Rejects changed requests/providers, omitted history/controller data,
    /// noncanonical/truncated input and resets that differ from the Model.
    pub fn from_bytes(bytes: &[u8], plan: &CommonOdePlan) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid("ODE checkpoint exceeds its artifact size limit"));
        }
        let wire: WireCheckpoint = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid ODE checkpoint JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid("ODE checkpoint has an unknown schema or encoding"));
        }
        let checkpoint = wire.payload.replay(plan)?;
        if checkpoint.identity != wire.identity || checkpoint.to_bytes()? != bytes {
            return Err(invalid(
                "ODE checkpoint identity or canonical content differs from its replay",
            ));
        }
        Ok(checkpoint)
    }
}

pub(super) fn identity(checkpoint: &CommonOdeCheckpoint) -> Result<String, Diagnostic> {
    let bytes = serde_json::to_vec(&Payload::encode(checkpoint)?)
        .map_err(|error| invalid(format!("cannot identify ODE checkpoint: {error}")))?;
    Ok(digest(b"eqiora.common-ode-checkpoint/v1\0", &bytes))
}

impl Payload {
    fn encode(checkpoint: &CommonOdeCheckpoint) -> Result<Self, Diagnostic> {
        let segment = &checkpoint.segment;
        Ok(Self {
            request: WireOdeRequest::from_request(&checkpoint.request)?,
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

    fn replay(self, plan: &CommonOdePlan) -> Result<CommonOdeCheckpoint, Diagnostic> {
        let request = self.request.replay(plan)?;
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
        let (time, state, cursor) = CommonOdeCheckpoint::segment_origin(
            &request,
            prefix.as_ref(),
            &self.committed_samples,
        )?;
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
        CommonOdeCheckpoint::accept(request, prefix, self.committed_samples, segment)
    }
}
