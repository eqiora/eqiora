//! Exact, execution-occurrence-independent common transient Run request.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_numerics::{CommonState, CommonTransientRunRequest, ResolvedCommonPlan};
use eqiora_solver::LinearSolverBackend;
use eqiora_time::TimeBackendCapabilities;
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "eqiora.common-transient-run-request/v2";
const ENCODING: &str = "canonical-json-rfc8259-v1";
// A canonical Plan admits 256 MiB and a canonical spatial State admits 512 MiB.
// Base64 expansion plus the small request header stays below this envelope bound.
const MAX_BYTES: usize = 1_100_000_000;

/// Self-contained exact Plan, initial State, and normalized common transient schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct RunRequest {
    native: CommonTransientRunRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRunRequest {
    schema: String,
    encoding: String,
    identity: String,
    plan_base64: String,
    state_base64: String,
    until_s: f64,
    output_times_s: Vec<f64>,
}

impl RunRequest {
    /// Prepare one fixed-grid spatial request and normalize it to physical times.
    pub fn from_steps(
        plan: ResolvedCommonPlan,
        state: CommonState,
        steps: usize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        CommonTransientRunRequest::from_steps(plan, state, steps, output_steps).map(Self::from)
    }

    /// Prepare one spatial request with a physical horizon and observation times.
    pub fn from_times(
        plan: ResolvedCommonPlan,
        state: CommonState,
        until_s: f64,
        output_times_s: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        CommonTransientRunRequest::from_times(plan, state, until_s, output_times_s).map(Self::from)
    }

    /// Stable identity of the normalized request meaning.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.native.identity()
    }

    /// Borrow the canonical worker request without creating an execution occurrence.
    #[must_use]
    pub const fn native(&self) -> &CommonTransientRunRequest {
        &self.native
    }

    /// Consume the artifact into the ordinary common transient worker request.
    #[must_use]
    pub fn into_native(self) -> CommonTransientRunRequest {
        self.native
    }

    /// Encode the complete Plan, initial State, and normalized schedule.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes =
            serde_json::to_vec(&WireRunRequest::from_request(&self.native)?).map_err(|error| {
                invalid(format!(
                    "cannot encode common transient RunRequest artifact: {error}"
                ))
            })?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common transient RunRequest has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    /// Replay one self-contained request through the ordinary Plan, State, and request decoders.
    pub fn from_bytes(
        bytes: &[u8],
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendCapabilities,
    ) -> Result<Self, Diagnostic> {
        Self::from_bytes_with_limit(bytes, linear_backend, time_backend, MAX_BYTES)
    }

    fn from_bytes_with_limit(
        bytes: &[u8],
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendCapabilities,
        max_bytes: usize,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > max_bytes {
            return Err(invalid(format!(
                "common transient RunRequest has {} bytes, exceeding the {max_bytes} byte limit",
                bytes.len()
            )));
        }
        let wire: WireRunRequest = serde_json::from_slice(bytes).map_err(|error| {
            invalid(format!("invalid common transient RunRequest JSON: {error}"))
        })?;
        wire.validate_header()?;
        let request = wire.replay(linear_backend, time_backend)?;
        let artifact = Self::from(request);
        if artifact.to_bytes()? != bytes {
            return Err(invalid(
                "common transient RunRequest bytes are not the canonical encoding of their content",
            ));
        }
        Ok(artifact)
    }
}

impl From<CommonTransientRunRequest> for RunRequest {
    fn from(native: CommonTransientRunRequest) -> Self {
        Self { native }
    }
}

impl WireRunRequest {
    fn from_request(request: &CommonTransientRunRequest) -> Result<Self, Diagnostic> {
        let plan = request.plan();
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            identity: request.identity().to_owned(),
            plan_base64: BASE64_STANDARD.encode(plan.to_bytes()?),
            state_base64: BASE64_STANDARD.encode(request.state().to_bytes()?),
            until_s: request.until_s(),
            output_times_s: request.output_times_s().to_vec(),
        })
    }

    fn validate_header(&self) -> Result<(), Diagnostic> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(invalid(
                "common transient RunRequest has an unknown schema or encoding",
            ));
        }
        Ok(())
    }

    fn replay(
        &self,
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendCapabilities,
    ) -> Result<CommonTransientRunRequest, Diagnostic> {
        let plan_bytes = decode(&self.plan_base64, "Plan")?;
        let resolved = ResolvedCommonPlan::from_bytes(&plan_bytes, linear_backend, time_backend)?;

        let state = CommonState::from_bytes(&decode(&self.state_base64, "State")?, &resolved)?;
        let request = CommonTransientRunRequest::from_times(
            resolved,
            state,
            self.until_s,
            self.output_times_s.clone(),
        )?;
        if request.identity() != self.identity {
            return Err(invalid(
                "common transient RunRequest identity does not match its canonical content",
            ));
        }
        Ok(request)
    }
}

fn decode(value: &str, label: &str) -> Result<Vec<u8>, Diagnostic> {
    BASE64_STANDARD.decode(value).map_err(|error| {
        invalid(format!(
            "RunRequest {label} is not canonical base64: {error}"
        ))
    })
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[cfg(test)]
mod tests;
