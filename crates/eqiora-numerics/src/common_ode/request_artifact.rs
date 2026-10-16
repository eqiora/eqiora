//! One exact ODE request encoding shared by trajectories and checkpoints.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireOdeRequest {
    plan_identity: String,
    request_identity: String,
    initial_state_base64: String,
    until_s: f64,
    output_times_s: Vec<f64>,
}

impl WireOdeRequest {
    pub(crate) fn from_request(request: &CommonOdeRunRequest) -> Result<Self, Diagnostic> {
        Ok(Self {
            plan_identity: request.plan().identity().to_owned(),
            request_identity: request.identity().to_owned(),
            initial_state_base64: STANDARD.encode(request.state().to_bytes()?),
            until_s: request.until_s(),
            output_times_s: request.output_times_s().to_vec(),
        })
    }
    pub(crate) fn replay(&self, plan: &CommonOdePlan) -> Result<CommonOdeRunRequest, Diagnostic> {
        if self.plan_identity != plan.identity() {
            return Err(invalid("ODE request belongs to a different exact Plan"));
        }
        let bytes = STANDARD
            .decode(&self.initial_state_base64)
            .map_err(|error| {
                invalid(format!(
                    "invalid ODE request initial State encoding: {error}"
                ))
            })?;
        let state = CommonOdeState::from_bytes(&bytes, plan)?;
        let request = CommonOdeRunRequest::new(
            plan.clone(),
            state,
            self.until_s,
            self.output_times_s.clone(),
        )?;
        if request.identity() != self.request_identity {
            return Err(invalid(
                "ODE request identity differs from its canonical content",
            ));
        }
        Ok(request)
    }
}
