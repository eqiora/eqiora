//! Canonical persistence for one accepted common Trajectory.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};

use super::*;
use crate::ResolvedCommonPlan;
use crate::common_time::request_artifact::WireTimeRequest;
use crate::time_history_artifact::{WireTimeStep, events as event_history};

const SCHEMA: &str = "eqiora.common-trajectory/v8";
const ENCODING: &str = "canonical-json-rfc8259-v1";
use eqiora_artifact::{
    MAX_EXECUTION_ARTIFACT_BYTES as MAX_BYTES, encode_execution_artifact as to_vec,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTrajectoryPayload {
    Process {
        checkpoint_base64: String,
    },
    Ode {
        request: WireTimeRequest,
        states_base64: Vec<String>,
        history: Vec<WireTimeStep>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        events: Vec<event_history::WireEvent>,
    },
    SpatialTransient {
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
struct WireSpatialState {
    step: u64,
    previous_state_base64: String,
    state_base64: String,
    execution: Option<WireSpatialExecution>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialExecution {
    solve: crate::common_result::artifact::WireSolveHistory,
    assembly: crate::common_result::artifact::WireAssembly,
}

impl WireSpatialExecution {
    fn from_execution(
        value: &crate::numerical_admission::CommonSpatialExecution,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            solve: crate::common_result::artifact::WireSolveHistory::from_evidence(&value.solve)?,
            assembly: crate::common_result::artifact::WireAssembly::from_assembly(&value.assembly)?,
        })
    }
    fn replay(&self) -> Result<crate::numerical_admission::CommonSpatialExecution, Diagnostic> {
        Ok(crate::numerical_admission::CommonSpatialExecution {
            solve: self.solve.replay_metadata()?,
            assembly: self.assembly.replay()?,
        })
    }
}

pub(super) fn execution_identity_bytes(
    value: Option<&crate::numerical_admission::CommonSpatialExecution>,
) -> Result<Vec<u8>, Diagnostic> {
    let wire = value
        .map(WireSpatialExecution::from_execution)
        .transpose()?;
    to_vec(&wire).map_err(|error| invalid(format!("cannot encode step execution: {error}")))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonTrajectory {
    schema: String,
    encoding: String,
    identity: String,
    payload: WireTrajectoryPayload,
}

impl CommonTrajectory {
    /// Encode the immutable Run request and every requested State canonically.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        to_vec(&WireCommonTrajectory::from_trajectory(self)?)
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
        let wire: WireCommonTrajectory = serde_json::from_slice(bytes)
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

impl WireCommonTrajectory {
    fn from_trajectory(trajectory: &CommonTrajectory) -> Result<Self, Diagnostic> {
        let payload = match &trajectory.payload {
            TrajectoryPayload::Process { checkpoint, .. } => WireTrajectoryPayload::Process {
                checkpoint_base64: encode(&checkpoint.to_bytes()?),
            },
            TrajectoryPayload::Ode {
                request,
                states,
                history,
                ..
            } => WireTrajectoryPayload::Ode {
                request: WireTimeRequest::from_request(request)?,
                events: event_history::encode(history.events())?,
                history: history.steps().iter().map(WireTimeStep::from).collect(),
                states_base64: states
                    .iter()
                    .map(|state| state.to_bytes().map(|bytes| encode(&bytes)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            TrajectoryPayload::SpatialTransient {
                request, states, ..
            } => WireTrajectoryPayload::SpatialTransient {
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
                WireTrajectoryPayload::Process { checkpoint_base64 },
                ResolvedCommonPlan::Time(plan),
            ) => CommonTrajectory::accept_process(crate::CommonTimeCheckpoint::from_bytes(
                &decode(checkpoint_base64, "process checkpoint")?,
                plan,
            )?)?,
            (
                WireTrajectoryPayload::Ode {
                    request,
                    states_base64,
                    history,
                    events,
                },
                ResolvedCommonPlan::Time(plan),
            ) => {
                let request = request.replay(plan)?;
                let plan = plan.as_ode().ok_or_else(|| {
                    invalid("ODE trajectory requires its admitted ODE capability")
                })?;
                let states = states_base64
                    .iter()
                    .map(|bytes| plan.state_from_bytes(&decode(bytes, "output State")?))
                    .collect::<Result<Vec<_>, _>>()?;
                let events = event_history::replay(events, plan)?;
                let steps = history
                    .iter()
                    .map(WireTimeStep::replay)
                    .collect::<Result<Vec<_>, _>>()?;
                let history =
                    AcceptedTimeHistory::accepted(plan.state_dimensions().len(), steps, events)?;
                CommonTrajectory::accept_ode_states(request, states, history)?
            }
            (
                WireTrajectoryPayload::SpatialTransient {
                    plan_identity,
                    request_identity,
                    initial_state_base64,
                    accepted_steps,
                    output_steps,
                    states,
                },
                plan @ (ResolvedCommonPlan::TransientFlow(_) | ResolvedCommonPlan::Spatial(_)),
            ) => {
                require_plan_identity(plan.identity(), plan_identity)?;
                let initial =
                    CommonState::from_bytes(&decode(initial_state_base64, "initial State")?, plan)?;
                let request = CommonTransientRunRequest::from_steps(
                    plan.clone(),
                    initial,
                    to_usize(*accepted_steps, "accepted step count")?,
                    decode_steps(output_steps)?,
                )?;
                require_request_identity(request.identity(), request_identity)?;
                let states = decode_spatial_states(states, plan)?;
                CommonTrajectory::accept_spatial_transient(request, states)?
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
    states: &[(usize, CommonAcceptedSpatialStep)],
) -> Result<Vec<WireSpatialState>, Diagnostic> {
    states
        .iter()
        .map(|(step, state)| {
            Ok(WireSpatialState {
                step: to_u64(*step, "output step")?,
                previous_state_base64: encode(&state.previous_state().to_bytes()?),
                state_base64: encode(&state.state().to_bytes()?),
                execution: state
                    .execution
                    .as_ref()
                    .map(WireSpatialExecution::from_execution)
                    .transpose()?,
            })
        })
        .collect()
}

fn decode_spatial_states(
    states: &[WireSpatialState],
    plan: &ResolvedCommonPlan,
) -> Result<Vec<(usize, CommonAcceptedSpatialStep)>, Diagnostic> {
    states
        .iter()
        .map(|state| {
            let mut accepted = CommonAcceptedSpatialStep::new(
                &CommonState::from_bytes(
                    &decode(&state.previous_state_base64, "previous State")?,
                    plan,
                )?,
                CommonState::from_bytes(&decode(&state.state_base64, "output State")?, plan)?,
            )?;
            accepted.execution = state
                .execution
                .as_ref()
                .map(WireSpatialExecution::replay)
                .transpose()?;
            Ok((to_usize(state.step, "output step")?, accepted))
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
