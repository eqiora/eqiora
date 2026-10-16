//! Canonical persistence for one exact no-Mesh ODE State.

use serde::{Deserialize, Serialize};

use super::*;

const SCHEMA: &str = "eqiora.common-ode-state/v1";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonOdeStateV1 {
    schema: String,
    encoding: String,
    state_space_identity: String,
    identity: String,
    model_digest: String,
    time_s: f64,
    values: Vec<f64>,
    source_kind: String,
}

impl CommonOdeState {
    /// Encode this exact no-Mesh ODE State as bounded canonical bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&WireCommonOdeStateV1::from_state(self))
            .map_err(|error| invalid(format!("cannot encode common ODE State artifact: {error}")))
    }

    /// Decode and reauthenticate one exact ODE State against its owning Plan.
    pub fn from_bytes(bytes: &[u8], plan: &CommonOdePlan) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common ODE State has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireCommonOdeStateV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid common ODE State JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid(
                "common ODE State has an unknown schema or encoding",
            ));
        }
        let state = wire.replay(plan)?;
        if state.to_bytes()? != bytes {
            return Err(invalid(
                "common ODE State bytes are not the canonical encoding of their content",
            ));
        }
        Ok(state)
    }
}

impl WireCommonOdeStateV1 {
    fn from_state(state: &CommonOdeState) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            state_space_identity: state.state_space_identity.clone(),
            identity: state.identity.clone(),
            model_digest: state.model_digest.clone(),
            time_s: state.time_s,
            values: state.values.clone(),
            source_kind: state.source_kind.to_owned(),
        }
    }

    fn replay(&self, plan: &CommonOdePlan) -> Result<CommonOdeState, Diagnostic> {
        let source_kind = match self.source_kind.as_str() {
            "initial" => "initial",
            "result" => "result",
            "checkpoint" => "checkpoint",
            _ => return Err(invalid("common ODE State has an unknown source kind")),
        };
        let state = CommonOdeState::new(plan, self.time_s, self.values.clone(), source_kind)?;
        if state.state_space_identity() != self.state_space_identity
            || state.identity() != self.identity
            || state.model_digest() != self.model_digest
        {
            return Err(invalid(
                "common ODE State replay differs from its persisted identities",
            ));
        }
        Ok(state)
    }
}
