//! One bounded, canonical envelope for complete time-execution continuations.
use super::{digest, invalid};
use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

const SCHEMA: &str = "eqiora.common-time-checkpoint/v1";
const ENCODING: &str = "canonical-json-rfc8259-v1";
use eqiora_artifact::{
    MAX_EXECUTION_ARTIFACT_BYTES as MAX_BYTES, encode_execution_artifact as to_vec,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope<T> {
    schema: String,
    encoding: String,
    profile: String,
    identity: String,
    payload: T,
}

pub(crate) fn identity<T: Serialize>(profile: &str, payload: &T) -> Result<String, Diagnostic> {
    let bytes = to_vec(&(profile, payload))
        .map_err(|error| invalid(format!("cannot identify time checkpoint: {error}")))?;
    Ok(digest(&[SCHEMA.as_bytes(), b"\0"].concat(), &bytes))
}

pub(crate) fn encode<T: Serialize>(profile: &str, payload: &T) -> Result<Vec<u8>, Diagnostic> {
    let bytes = to_vec(&Envelope {
        schema: SCHEMA.to_owned(),
        encoding: ENCODING.to_owned(),
        profile: profile.to_owned(),
        identity: identity(profile, payload)?,
        payload,
    })
    .map_err(|error| invalid(format!("cannot encode time checkpoint: {error}")))?;
    Ok(bytes)
}

pub(crate) fn decode<T: Serialize + DeserializeOwned, R>(
    bytes: &[u8],
    profile: &str,
    accept: impl FnOnce(T) -> Result<R, Diagnostic>,
) -> Result<R, Diagnostic> {
    if bytes.len() > MAX_BYTES {
        return Err(invalid("time checkpoint exceeds its artifact size limit"));
    }
    let wire: Envelope<T> = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("invalid time checkpoint JSON: {error}")))?;
    if wire.schema != SCHEMA || wire.encoding != ENCODING || wire.profile != profile {
        return Err(invalid(
            "time checkpoint has an unknown schema, encoding or execution profile",
        ));
    }
    let identified = wire.identity == identity(profile, &wire.payload)?;
    let canonical = encode(profile, &wire.payload)? == bytes;
    // Admission constructs a new candidate, never mutating a live execution.
    // Preserve semantic diagnostics for missing controller/path information;
    // every successful return still requires exact canonical content identity.
    let candidate = accept(wire.payload)?;
    if !identified || !canonical {
        return Err(invalid(
            "time checkpoint identity or canonical content differs from its payload",
        ));
    }
    Ok(candidate)
}
