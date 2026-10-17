//! Canonical checkpoint content is a reference, not permission to resume a Run.
use super::*;

/// Read-only checkpoint bytes and their exact original request.
///
/// Reopening checks the existing bounded, typed checkpoint envelope and its
/// canonical content identity. It does not instantiate a controller or random
/// path, attest physical/path consistency, or establish restart compatibility.
/// [`Self::restore`] performs the original numerical continuation admission.
/// No additional archive format or checkpoint decoder is introduced.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedTimeCheckpoint {
    request: CommonTimeRunRequest,
    identity: String,
    bytes: Arc<[u8]>,
}

impl CommonTimeCheckpoint {
    /// Retain this checkpoint as immutable content without its live continuation.
    pub fn record(&self) -> Result<RecordedTimeCheckpoint, Diagnostic> {
        Ok(RecordedTimeCheckpoint {
            request: self.request().clone(),
            identity: self.identity().to_owned(),
            bytes: self.to_bytes()?.into(),
        })
    }
}

impl RecordedTimeCheckpoint {
    /// Inspect canonical content without solving, drawing, or resuming execution.
    /// Numerical admissibility remains the responsibility of [`Self::restore`].
    pub fn from_bytes(bytes: &[u8], plan: &CommonTimePlan) -> Result<Self, Diagnostic> {
        let (request, identity) = match plan.as_ode() {
            Some(ode) => AdaptiveCheckpoint::inspect_bytes(bytes, ode)?,
            None => ProcessCursor::inspect_checkpoint_bytes(bytes, plan)?,
        };
        Ok(Self {
            request,
            identity,
            bytes: bytes.into(),
        })
    }

    /// Original Plan, initial State, horizon, output schedule and sample identity.
    pub fn request(&self) -> &CommonTimeRunRequest {
        &self.request
    }

    /// Content identity owned by the existing checkpoint envelope.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The exact original checkpoint encoding, without a second envelope.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Reaccept the original numerical continuation before it can resume.
    /// Stochastic checkpoints verify their original retained paths and physical
    /// partition here; an inspectable record can still fail this admission.
    pub fn restore(&self) -> Result<CommonTimeCheckpoint, Diagnostic> {
        CommonTimeCheckpoint::from_bytes(&self.bytes, self.request.plan())
    }
}
