//! Atomic physical/path continuation using the common time-checkpoint envelope.
//!
//! Replay is exact for the same sampler versions, accepted partition, binary64
//! execution and unchanged native runtime/libm environment. Target metadata is
//! a necessary compatibility check, not proof of equivalence between machines.
use super::super::retained;
use super::provenance::RunBinding;
use super::*;
use crate::common_time::checkpoint;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

const PROFILE: &str = "serial-stochastic";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    request: RunBinding,
    // A sorted nominal map: never serialize HashMap iteration order. Poisson
    // entries retain every exact threshold including the pending next arrival.
    paths: Vec<(String, RealizedNoise)>,
    #[serde(with = "retained::values")]
    accepted_positions: Vec<BigRational>,
    state_base64: String,
}

fn paths(run: &ProcessRun) -> Vec<(String, RealizedNoise)> {
    let mut paths = run
        .paths
        .iter()
        .map(|(id, path)| (id.ulid().to_string(), path.clone()))
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    paths
}

impl ProcessCursor {
    pub(crate) fn identity(&self) -> Result<String, Diagnostic> {
        checkpoint::identity(PROFILE, &self.payload()?)
    }
    fn payload(&self) -> Result<Payload, Diagnostic> {
        Ok(Payload {
            request: RunBinding::new(&self.run.request)?,
            paths: paths(&self.run),
            accepted_positions: self
                .history
                .iter()
                .map(|step| step.elapsed.clone())
                .collect(),
            state_base64: STANDARD.encode(self.state.to_bytes()?),
        })
    }
    pub(crate) fn from_bytes(
        bytes: &[u8],
        plan: &crate::CommonTimePlan,
    ) -> Result<Self, Diagnostic> {
        let payload: Payload = checkpoint::decode(bytes, PROFILE, Ok)?;
        let request = payload.request.run.replay(plan)?;
        ProcessRun::new(request)?.resume_payload(bytes, payload)
    }
    pub(crate) fn checkpoint_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        checkpoint::encode(PROFILE, &self.payload()?)
    }
}

impl ProcessRun {
    #[cfg(test)]
    pub(crate) fn resume_bytes(
        self: &Arc<Self>,
        bytes: &[u8],
    ) -> Result<ProcessCursor, Diagnostic> {
        let payload: Payload = checkpoint::decode(bytes, PROFILE, Ok)?;
        self.resume_payload(bytes, payload)
    }
    fn resume_payload(
        self: &Arc<Self>,
        bytes: &[u8],
        payload: Payload,
    ) -> Result<ProcessCursor, Diagnostic> {
        if payload.request != RunBinding::new(&self.request)? || payload.paths != paths(self) {
            return Err(invalid(
                "stochastic checkpoint differs from its exact Run, sampler or retained paths",
            ));
        }
        let state_bytes = STANDARD
            .decode(&payload.state_base64)
            .map_err(|_| invalid("invalid stochastic checkpoint physical State encoding"))?;
        let physical = self.program().state_from_bytes(&state_bytes)?;
        // Reaccept the retained partition using the original immutable paths.
        // This bounded serial profile spends linear replay work rather than
        // trusting a physical State unrelated to its claimed stochastic position.
        let mut cursor = self.start();
        for elapsed in &payload.accepted_positions {
            cursor.accept(cursor.trial(elapsed)?)?;
        }
        if cursor.state != physical || cursor.checkpoint_bytes()? != bytes {
            return Err(invalid(
                "stochastic checkpoint physical State differs from its accepted path partition",
            ));
        }
        Ok(cursor)
    }
}

#[cfg(test)]
mod tests;
