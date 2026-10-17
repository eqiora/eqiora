use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{
    CONTROL_PROTOCOL_V2, DiagnosticRecord, MAX_COMPILE_FILENAME_BYTES_V2,
    MAX_COMPILE_REQUEST_BYTES_V2, MAX_COMPILE_SOURCE_BYTES_V2,
    MAX_CONTROL_DISPATCH_IDENTITY_BYTES_V2, MAX_CONTROL_REQUEST_ID_BYTES_V2,
};

/// Exact command identity of the bounded compile/check operation.
pub const COMPILE_COMMAND_V1: &str = "model.compile-check/v1";

/// Validated compile/check request for the sole current Model contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileRequestV2 {
    request_id: String,
    filename: String,
    source: String,
}

impl CompileRequestV2 {
    /// Construct an ordinary current compile/check request.
    ///
    /// # Errors
    /// Returns a structured control diagnostic for an invalid request ID,
    /// filename, or source bound.
    pub fn new(
        request_id: impl Into<String>,
        filename: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<Self, DiagnosticRecord> {
        Self::validate_parts(request_id.into(), filename.into(), source.into())
    }

    /// Decode one bounded request through protocol dispatch and the closed v2
    /// DTO.
    ///
    /// # Errors
    /// Returns a standalone diagnostic before compilation. Dispatch does not
    /// retry or reinterpret the request under another protocol.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DiagnosticRecord> {
        dispatch_compile_v2(bytes)
    }

    /// Canonical compact JSON in frozen member order.
    ///
    /// # Errors
    /// Returns a diagnostic if serialization unexpectedly fails or the
    /// encoded request exceeds the control-plane resource bound.
    pub fn canonical_json(&self) -> Result<Vec<u8>, DiagnosticRecord> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            DiagnosticRecord::invalid_request(format!(
                "cannot encode compile/check v2 request: {error}"
            ))
        })?;
        if bytes.len() > MAX_COMPILE_REQUEST_BYTES_V2 {
            return Err(DiagnosticRecord::invalid_request(format!(
                "compile/check request exceeds {MAX_COMPILE_REQUEST_BYTES_V2} encoded bytes"
            )));
        }
        Ok(bytes)
    }

    /// Opaque caller-chosen identity echoed exactly by an admitted response.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Source filename used by compiler diagnostics.
    #[must_use]
    pub fn filename(&self) -> &str {
        &self.filename
    }

    /// Eqiora Language source.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    fn try_from_wire(wire: WireCompileRequestV2) -> Result<Self, DiagnosticRecord> {
        debug_assert_eq!(wire.protocol, CONTROL_PROTOCOL_V2);
        debug_assert_eq!(wire.command, COMPILE_COMMAND_V1);
        Self::validate_parts(wire.request_id, wire.filename, wire.source)
    }

    fn validate_parts(
        request_id: String,
        filename: String,
        source: String,
    ) -> Result<Self, DiagnosticRecord> {
        validate_request_id(&request_id)?;
        if filename.is_empty()
            || filename.chars().count() > MAX_COMPILE_FILENAME_BYTES_V2
            || filename.len() > MAX_COMPILE_FILENAME_BYTES_V2
            || filename.chars().any(char::is_control)
        {
            return Err(DiagnosticRecord::invalid_request(format!(
                "source filename must contain 1 to {MAX_COMPILE_FILENAME_BYTES_V2} non-control UTF-8 bytes"
            )));
        }
        if source.chars().count() > MAX_COMPILE_SOURCE_BYTES_V2
            || source.len() > MAX_COMPILE_SOURCE_BYTES_V2
        {
            return Err(DiagnosticRecord::invalid_request(format!(
                "source exceeds the {MAX_COMPILE_SOURCE_BYTES_V2}-byte compile/check v2 limit"
            )));
        }
        Ok(Self {
            request_id,
            filename,
            source,
        })
    }
}

impl Serialize for CompileRequestV2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WireCompileRequestRefV2 {
            protocol: CONTROL_PROTOCOL_V2,
            command: COMPILE_COMMAND_V1,
            request_id: &self.request_id,
            filename: &self.filename,
            source: &self.source,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CompileRequestV2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WireCompileRequestV2::deserialize(deserializer)?;
        if wire.protocol != CONTROL_PROTOCOL_V2 || wire.command != COMPILE_COMMAND_V1 {
            return Err(serde::de::Error::custom(
                "compile/check request has the wrong protocol or command",
            ));
        }
        Self::try_from_wire(wire).map_err(serde::de::Error::custom)
    }
}

/// Dispatch a bounded JSON request without protocol sniffing or fallback.
///
/// # Errors
/// Returns a standalone control diagnostic for request-size, JSON, dispatch,
/// or closed-DTO admission failure.
fn dispatch_compile_v2(bytes: &[u8]) -> Result<CompileRequestV2, DiagnosticRecord> {
    if bytes.len() > MAX_COMPILE_REQUEST_BYTES_V2 {
        return Err(DiagnosticRecord::invalid_request(format!(
            "compile/check request exceeds {MAX_COMPILE_REQUEST_BYTES_V2} encoded bytes"
        )));
    }
    let prelude: WireDispatchPrelude = serde_json::from_slice(bytes).map_err(|error| {
        DiagnosticRecord::invalid_request(format!("invalid compile/check v2 request JSON: {error}"))
    })?;
    validate_dispatch_identity("protocol", &prelude.protocol)?;
    validate_dispatch_identity("command", &prelude.command)?;
    if prelude.protocol != CONTROL_PROTOCOL_V2 {
        return Err(DiagnosticRecord::unsupported(format!(
            "unsupported control protocol `{}`; expected `{CONTROL_PROTOCOL_V2}`",
            prelude.protocol
        )));
    }
    if prelude.command != COMPILE_COMMAND_V1 {
        return Err(DiagnosticRecord::unsupported(format!(
            "unsupported control command `{}`; expected `{COMPILE_COMMAND_V1}`",
            prelude.command
        )));
    }
    let wire: WireCompileRequestV2 = serde_json::from_slice(bytes).map_err(|error| {
        DiagnosticRecord::invalid_request(format!("invalid compile/check v2 request JSON: {error}"))
    })?;
    CompileRequestV2::try_from_wire(wire)
}

#[derive(Deserialize)]
struct WireDispatchPrelude {
    protocol: String,
    command: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WireCompileRequestV2 {
    protocol: String,
    command: String,
    request_id: String,
    filename: String,
    source: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WireCompileRequestRefV2<'a> {
    protocol: &'static str,
    command: &'static str,
    request_id: &'a str,
    filename: &'a str,
    source: &'a str,
}

pub(super) fn validate_request_id(request_id: &str) -> Result<(), DiagnosticRecord> {
    if request_id.is_empty()
        || request_id.chars().count() > MAX_CONTROL_REQUEST_ID_BYTES_V2
        || request_id.len() > MAX_CONTROL_REQUEST_ID_BYTES_V2
        || !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(DiagnosticRecord::invalid_request(format!(
            "request ID must contain 1 to {MAX_CONTROL_REQUEST_ID_BYTES_V2} ASCII letters, digits, period, colon, underscore, or hyphen"
        )));
    }
    Ok(())
}

fn validate_dispatch_identity(name: &str, value: &str) -> Result<(), DiagnosticRecord> {
    if value.chars().count() > MAX_CONTROL_DISPATCH_IDENTITY_BYTES_V2
        || value.len() > MAX_CONTROL_DISPATCH_IDENTITY_BYTES_V2
    {
        return Err(DiagnosticRecord::invalid_request(format!(
            "control {name} exceeds the dispatch identity limit"
        )));
    }
    Ok(())
}
