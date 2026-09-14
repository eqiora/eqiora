//! Canonical persistence for one accepted common Trajectory.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};

use super::*;
use crate::ResolvedCommonPlan;
mod event_history;

const SCHEMA: &str = "eqiora.common-trajectory/v3";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTrajectoryPayload {
    Ode {
        plan_identity: String,
        request_identity: String,
        initial_state_base64: String,
        until_s: f64,
        output_times_s: Vec<f64>,
        states_base64: Vec<String>,
        history: Vec<WireTimeStep>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        events: Vec<event_history::WireEvent>,
    },
    TransientFlow {
        plan_identity: String,
        request_identity: String,
        initial_state_base64: String,
        accepted_steps: u64,
        output_steps: Vec<u64>,
        states: Vec<WireSpatialState>,
    },
    FixedReferenceFsi {
        plan_identity: String,
        request_identity: String,
        initial_state_base64: String,
        accepted_steps: u64,
        output_steps: Vec<u64>,
        states: Vec<WireSpatialState>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTimeStep {
    start_time: f64,
    end_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    end_state: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialState {
    step: u64,
    state_base64: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonTrajectoryV3 {
    schema: String,
    encoding: String,
    identity: String,
    payload: WireTrajectoryPayload,
}

impl CommonTrajectory {
    /// Encode the immutable Run request and every requested State canonically.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&WireCommonTrajectoryV3::from_trajectory(self)?)
            .map_err(|error| invalid(format!("cannot encode common Trajectory artifact: {error}")))
    }

    /// Decode a producer-independent Trajectory against its exact owning Plan.
    pub fn from_bytes(bytes: &[u8], plan: &ResolvedCommonPlan) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common Trajectory has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireCommonTrajectoryV3 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid common Trajectory JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid(
                "common Trajectory has an unknown schema or encoding",
            ));
        }
        let trajectory = wire.replay(plan)?;
        if trajectory.to_bytes()? != bytes {
            return Err(invalid(
                "common Trajectory bytes are not the canonical encoding of their content",
            ));
        }
        Ok(trajectory)
    }
}

impl WireCommonTrajectoryV3 {
    fn from_trajectory(trajectory: &CommonTrajectory) -> Result<Self, Diagnostic> {
        let payload = match trajectory {
            CommonTrajectory::Ode {
                request,
                states,
                history,
                ..
            } => WireTrajectoryPayload::Ode {
                plan_identity: request.plan().identity().to_owned(),
                request_identity: request.identity().to_owned(),
                initial_state_base64: encode(&request.state().to_bytes()?),
                until_s: request.until_s(),
                output_times_s: request.output_times_s().to_vec(),
                events: event_history::encode(history.events())?,
                history: history
                    .steps()
                    .iter()
                    .map(|step| WireTimeStep {
                        start_time: step.start_time(),
                        end_time: step.end_time(),
                        start_state: step.start_state().to_vec(),
                        midpoint_state: step.midpoint_state().to_vec(),
                        end_state: step.end_state().to_vec(),
                    })
                    .collect(),
                states_base64: states
                    .iter()
                    .map(|state| state.to_bytes().map(|bytes| encode(&bytes)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            CommonTrajectory::TransientFlow {
                request, states, ..
            } => WireTrajectoryPayload::TransientFlow {
                plan_identity: request.plan().identity().to_owned(),
                request_identity: request.identity().to_owned(),
                initial_state_base64: encode(&request.state().to_bytes()?),
                accepted_steps: to_u64(request.accepted_steps().get(), "accepted step count")?,
                output_steps: request
                    .output_steps()
                    .iter()
                    .map(|step| to_u64(*step, "output step"))
                    .collect::<Result<Vec<_>, _>>()?,
                states: encode_spatial_states(states)?,
            },
            CommonTrajectory::Fsi {
                request, states, ..
            } => WireTrajectoryPayload::FixedReferenceFsi {
                plan_identity: request.plan().identity().to_owned(),
                request_identity: request.identity().to_owned(),
                initial_state_base64: encode(&request.state().to_bytes()?),
                accepted_steps: to_u64(request.accepted_steps().get(), "accepted step count")?,
                output_steps: request
                    .output_steps()
                    .iter()
                    .map(|step| to_u64(*step, "output step"))
                    .collect::<Result<Vec<_>, _>>()?,
                states: encode_spatial_states(states)?,
            },
        };
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            identity: trajectory.identity().to_owned(),
            payload,
        })
    }

    fn replay(&self, plan: &ResolvedCommonPlan) -> Result<CommonTrajectory, Diagnostic> {
        let trajectory = match (&self.payload, plan) {
            (
                WireTrajectoryPayload::Ode {
                    plan_identity,
                    request_identity,
                    initial_state_base64,
                    until_s,
                    output_times_s,
                    states_base64,
                    history,
                    events,
                },
                ResolvedCommonPlan::Ode(plan),
            ) => {
                require_plan_identity(plan.identity(), plan_identity)?;
                let initial = CommonOdeState::from_bytes(
                    &decode(initial_state_base64, "initial State")?,
                    plan,
                )?;
                let request = CommonOdeRunRequest::new(
                    plan.as_ref().clone(),
                    initial,
                    *until_s,
                    output_times_s.clone(),
                )?;
                require_request_identity(request.identity(), request_identity)?;
                let states = states_base64
                    .iter()
                    .map(|bytes| CommonOdeState::from_bytes(&decode(bytes, "output State")?, plan))
                    .collect::<Result<Vec<_>, _>>()?;
                let events = event_history::replay(events, plan)?;
                let steps = history
                    .iter()
                    .map(|step| {
                        eqiora_time::TimeHistoryStep::accepted(
                            step.start_time,
                            step.end_time,
                            step.start_state.clone(),
                            step.midpoint_state.clone(),
                            step.end_state.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let history =
                    AcceptedTimeHistory::accepted(plan.field_dimensions().len(), steps, events)?;
                CommonTrajectory::accept_ode_states(request, states, history)?
            }
            (
                WireTrajectoryPayload::TransientFlow {
                    plan_identity,
                    request_identity,
                    initial_state_base64,
                    accepted_steps,
                    output_steps,
                    states,
                },
                ResolvedCommonPlan::TransientFlow(plan),
            ) => {
                require_plan_identity(plan.identity(), plan_identity)?;
                let initial = CommonState::from_bytes(
                    &decode(initial_state_base64, "initial State")?,
                    &ResolvedCommonPlan::TransientFlow(plan.clone()),
                )?;
                let request = CommonTransientRunRequest::from_steps(
                    plan.as_ref().clone(),
                    initial,
                    to_usize(*accepted_steps, "accepted step count")?,
                    decode_steps(output_steps)?,
                )?;
                require_request_identity(request.identity(), request_identity)?;
                let states = decode_spatial_states(
                    states,
                    &ResolvedCommonPlan::TransientFlow(plan.clone()),
                )?;
                CommonTrajectory::accept_transient_flow(request, states)?
            }
            (
                WireTrajectoryPayload::FixedReferenceFsi {
                    plan_identity,
                    request_identity,
                    initial_state_base64,
                    accepted_steps,
                    output_steps,
                    states,
                },
                ResolvedCommonPlan::Fsi(plan),
            ) => {
                require_plan_identity(plan.identity(), plan_identity)?;
                let initial = CommonState::from_bytes(
                    &decode(initial_state_base64, "initial State")?,
                    &ResolvedCommonPlan::Fsi(plan.clone()),
                )?;
                let request = CommonFsiRunRequest::from_steps(
                    plan.as_ref().clone(),
                    initial,
                    to_usize(*accepted_steps, "accepted step count")?,
                    decode_steps(output_steps)?,
                )?;
                require_request_identity(request.identity(), request_identity)?;
                let states = decode_spatial_states(states, &ResolvedCommonPlan::Fsi(plan.clone()))?;
                CommonTrajectory::accept_fsi(request, states)?
            }
            _ => {
                return Err(invalid(
                    "common Trajectory crossed an incompatible Plan family",
                ));
            }
        };
        if trajectory.identity() != self.identity {
            return Err(invalid(
                "common Trajectory replay differs from its persisted identity",
            ));
        }
        Ok(trajectory)
    }
}

fn encode_spatial_states(
    states: &[(usize, CommonState)],
) -> Result<Vec<WireSpatialState>, Diagnostic> {
    states
        .iter()
        .map(|(step, state)| {
            Ok(WireSpatialState {
                step: to_u64(*step, "output step")?,
                state_base64: encode(&state.to_bytes()?),
            })
        })
        .collect()
}

fn decode_spatial_states(
    states: &[WireSpatialState],
    plan: &ResolvedCommonPlan,
) -> Result<Vec<(usize, CommonState)>, Diagnostic> {
    states
        .iter()
        .map(|state| {
            Ok((
                to_usize(state.step, "output step")?,
                CommonState::from_bytes(&decode(&state.state_base64, "output State")?, plan)?,
            ))
        })
        .collect()
}

fn decode_steps(steps: &[u64]) -> Result<Vec<usize>, Diagnostic> {
    steps
        .iter()
        .map(|step| to_usize(*step, "output step"))
        .collect()
}

fn require_plan_identity(actual: &str, persisted: &str) -> Result<(), Diagnostic> {
    if actual != persisted {
        return Err(invalid(
            "common Trajectory belongs to a different exact Plan",
        ));
    }
    Ok(())
}

fn require_request_identity(actual: &str, persisted: &str) -> Result<(), Diagnostic> {
    if actual != persisted {
        return Err(invalid(
            "common Trajectory Run request differs from its persisted identity",
        ));
    }
    Ok(())
}

fn encode(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

fn decode(value: &str, label: &str) -> Result<Vec<u8>, Diagnostic> {
    BASE64_STANDARD
        .decode(value)
        .map_err(|error| invalid(format!("invalid common Trajectory {label} base64: {error}")))
}

fn to_u64(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid(format!("common Trajectory {label} exceeds u64")))
}

fn to_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value).map_err(|_| invalid(format!("common Trajectory {label} exceeds usize")))
}
