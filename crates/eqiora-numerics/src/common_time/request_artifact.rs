//! One exact finite-time request encoding shared by trajectories and checkpoints.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTimeRequest {
    plan_identity: String,
    request_identity: String,
    initial_state_base64: String,
    until_s: f64,
    output_times_s: Vec<f64>,
    sampling: Option<([u8; 32], crate::SamplingIdentity)>,
}

impl WireTimeRequest {
    pub(crate) fn from_request(request: &CommonTimeRunRequest) -> Result<Self, Diagnostic> {
        Ok(Self {
            plan_identity: request.plan().identity().to_owned(),
            request_identity: request.identity().to_owned(),
            initial_state_base64: STANDARD.encode(request.state().to_bytes()?),
            until_s: request.until_s(),
            output_times_s: request.output_times_s().to_vec(),
            sampling: request.sampling().cloned(),
        })
    }
    pub(crate) fn replay(&self, plan: &CommonTimePlan) -> Result<CommonTimeRunRequest, Diagnostic> {
        if self.plan_identity != plan.identity() {
            return Err(invalid("time request belongs to a different exact Plan"));
        }
        let bytes = STANDARD
            .decode(&self.initial_state_base64)
            .map_err(|error| {
                invalid(format!(
                    "invalid time request initial State encoding: {error}"
                ))
            })?;
        let state = plan.state_from_bytes(&bytes)?;
        let request = CommonTimeRunRequest::admit(
            plan.clone(),
            state,
            self.until_s,
            self.output_times_s.clone(),
            self.sampling.clone(),
        )?;
        if request.identity() != self.request_identity {
            return Err(invalid(
                "time request identity differs from its canonical content",
            ));
        }
        Ok(request)
    }
}
