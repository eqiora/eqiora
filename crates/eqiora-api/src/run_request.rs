//! Exact, execution-occurrence-independent common transient Run request.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_numerics::{CommonState, CommonTransientRunRequest, ResolvedCommonPlan};
use eqiora_solver::LinearSolverBackend;
use eqiora_time::TimeBackendIdentity;
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "eqiora.common-transient-run-request/v1";
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
struct WireRunRequestV1 {
    schema: String,
    encoding: String,
    identity: String,
    plan_base64: String,
    state_base64: String,
    accepted_steps: u64,
    output_steps: Vec<u64>,
}

impl RunRequest {
    /// Prepare one normalized accepted-step request for common transient flow.
    pub fn from_steps(
        plan: ResolvedCommonPlan,
        state: CommonState,
        steps: usize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        CommonTransientRunRequest::from_steps(plan, state, steps, output_steps).map(Self::from)
    }

    /// Prepare one normalized exact-time request for common transient flow.
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
        let bytes = serde_json::to_vec(&WireRunRequestV1::from_request(&self.native)?).map_err(
            |error| {
                invalid(format!(
                    "cannot encode common transient RunRequest artifact: {error}"
                ))
            },
        )?;
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
        time_backend: TimeBackendIdentity,
    ) -> Result<Self, Diagnostic> {
        Self::from_bytes_with_limit(bytes, linear_backend, time_backend, MAX_BYTES)
    }

    fn from_bytes_with_limit(
        bytes: &[u8],
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendIdentity,
        max_bytes: usize,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > max_bytes {
            return Err(invalid(format!(
                "common transient RunRequest has {} bytes, exceeding the {max_bytes} byte limit",
                bytes.len()
            )));
        }
        let wire: WireRunRequestV1 = serde_json::from_slice(bytes).map_err(|error| {
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

impl WireRunRequestV1 {
    fn from_request(request: &CommonTransientRunRequest) -> Result<Self, Diagnostic> {
        let plan = request.plan();
        let accepted_steps = u64::try_from(request.accepted_steps().get())
            .map_err(|_| invalid("RunRequest horizon exceeds canonical u64 range"))?;
        let output_steps = request
            .output_steps()
            .iter()
            .copied()
            .map(|step| {
                u64::try_from(step)
                    .map_err(|_| invalid("RunRequest output index exceeds canonical u64 range"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            identity: request.identity().to_owned(),
            plan_base64: BASE64_STANDARD.encode(plan.to_bytes()?),
            state_base64: BASE64_STANDARD.encode(request.state().to_bytes()?),
            accepted_steps,
            output_steps,
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
        time_backend: TimeBackendIdentity,
    ) -> Result<CommonTransientRunRequest, Diagnostic> {
        let plan_bytes = decode(&self.plan_base64, "Plan")?;
        let resolved = ResolvedCommonPlan::from_bytes(&plan_bytes, linear_backend, time_backend)?;

        let state = CommonState::from_bytes(&decode(&self.state_base64, "State")?, &resolved)?;
        let accepted_steps = usize::try_from(self.accepted_steps)
            .map_err(|_| invalid("RunRequest horizon exceeds this platform's usize range"))?;
        let output_steps = self
            .output_steps
            .iter()
            .copied()
            .map(|step| {
                usize::try_from(step).map_err(|_| {
                    invalid("RunRequest output index exceeds this platform's usize range")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let request =
            CommonTransientRunRequest::from_steps(resolved, state, accepted_steps, output_steps)?;
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
