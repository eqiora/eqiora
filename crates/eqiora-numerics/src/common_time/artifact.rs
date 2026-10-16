//! Canonical persistence for one exact finite time State.

use serde::{Deserialize, Serialize};

use super::*;

const SCHEMA: &str = "eqiora.common-time-state/v1";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonTimeStateV1 {
    schema: String,
    encoding: String,
    state_space_identity: String,
    identity: String,
    model_digest: String,
    time_s: f64,
    values: Vec<f64>,
    source_kind: String,
}

impl CommonTimeState {
    /// Encode this exact finite time State as bounded canonical bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes =
            serde_json::to_vec(&WireCommonTimeStateV1::from_state(self)).map_err(|error| {
                invalid(format!("cannot encode common time State artifact: {error}"))
            })?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid("common time State exceeds its artifact size limit"));
        }
        Ok(bytes)
    }

    /// Decode and reauthenticate one exact time State against its owning state space.
    pub(crate) fn from_bytes(bytes: &[u8], space: Arc<TimeStateSpace>) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common time State has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireCommonTimeStateV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid common time State JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid(
                "common time State has an unknown schema or encoding",
            ));
        }
        let state = wire.replay(space)?;
        if state.to_bytes()? != bytes {
            return Err(invalid(
                "common time State bytes are not the canonical encoding of their content",
            ));
        }
        Ok(state)
    }
}

impl WireCommonTimeStateV1 {
    fn from_state(state: &CommonTimeState) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            state_space_identity: state.space.identity.clone(),
            identity: state.identity.clone(),
            model_digest: state.space.model_digest.clone(),
            time_s: state.time_s,
            values: state.values.clone(),
            source_kind: state.source_kind.to_owned(),
        }
    }

    fn replay(&self, space: Arc<TimeStateSpace>) -> Result<CommonTimeState, Diagnostic> {
        let source_kind = match self.source_kind.as_str() {
            "initial" => "initial",
            "result" => "result",
            "checkpoint" => "checkpoint",
            _ => return Err(invalid("common time State has an unknown source kind")),
        };
        let state = CommonTimeState::new(space, self.time_s, self.values.clone(), source_kind)?;
        if state.state_space_identity() != self.state_space_identity
            || state.identity() != self.identity
            || state.model_digest() != self.model_digest
        {
            return Err(invalid(
                "common time State replay differs from its persisted identities",
            ));
        }
        Ok(state)
    }
}
