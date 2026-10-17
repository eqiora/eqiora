//! Public finite-time continuation with method-owned private execution state.
use super::*;
use crate::common_ode::AdaptiveCheckpoint;
use crate::stochastic_path::run::ProcessCursor;

mod record;
pub use record::RecordedTimeCheckpoint;

#[derive(Debug, Clone, PartialEq)]
enum Continuation {
    Adaptive(Box<AdaptiveCheckpoint>),
    Process(Box<ProcessCursor>),
}

/// Atomic physical and numerical continuation of one exact original time Run.
/// A physical State alone instead starts a fresh Run with new numerical history.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTimeCheckpoint {
    continuation: Continuation,
    identity: String,
}

impl CommonTimeCheckpoint {
    pub(crate) fn from_adaptive(value: AdaptiveCheckpoint) -> Self {
        Self {
            identity: value.identity().to_owned(),
            continuation: Continuation::Adaptive(Box::new(value)),
        }
    }

    pub(crate) fn from_process(value: ProcessCursor) -> Result<Self, Diagnostic> {
        Ok(Self {
            identity: value.identity()?,
            continuation: Continuation::Process(Box::new(value)),
        })
    }

    pub(crate) fn adaptive(&self) -> Result<&AdaptiveCheckpoint, Diagnostic> {
        match &self.continuation {
            Continuation::Adaptive(value) => Ok(value),
            Continuation::Process(_) => Err(invalid(
                "stochastic checkpoint has no adaptive ODE controller",
            )),
        }
    }

    pub(crate) fn process(&self) -> Result<&ProcessCursor, Diagnostic> {
        match &self.continuation {
            Continuation::Process(value) => Ok(value),
            Continuation::Adaptive(_) => {
                Err(invalid("adaptive ODE checkpoint has no stochastic path"))
            }
        }
    }

    /// Original Plan, initial State, horizon, outputs and sample identity.
    pub fn request(&self) -> &CommonTimeRunRequest {
        match &self.continuation {
            Continuation::Adaptive(value) => value.request(),
            Continuation::Process(value) => value.request(),
        }
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Complete last accepted physical State, independently reusable in a fresh Run.
    pub fn state(&self) -> Result<CommonTimeState, Diagnostic> {
        match &self.continuation {
            Continuation::Adaptive(value) => value.state(),
            Continuation::Process(value) => {
                let state = value.state();
                CommonTimeState::new(
                    state.space.clone(),
                    state.time_s(),
                    state.values().to_vec(),
                    "checkpoint",
                )
            }
        }
    }
    pub fn accepted_steps(&self) -> usize {
        match &self.continuation {
            Continuation::Adaptive(value) => value.accepted_steps(),
            Continuation::Process(value) => value.accepted_steps(),
        }
    }
    /// Committed deterministic resets or exact Poisson arrivals, by execution method.
    pub fn accepted_events(&self) -> usize {
        match &self.continuation {
            Continuation::Adaptive(value) => value.accepted_events(),
            Continuation::Process(value) => value.accepted_events(),
        }
    }

    /// Persist the original request and complete method-owned continuation.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        match &self.continuation {
            Continuation::Adaptive(value) => value.to_bytes(),
            Continuation::Process(value) => value.checkpoint_bytes(),
        }
    }

    /// Reaccept canonical bytes against the owning Plan, including physical/path
    /// consistency and the sampler/environment boundary for a stochastic Run.
    pub fn from_bytes(bytes: &[u8], plan: &CommonTimePlan) -> Result<Self, Diagnostic> {
        match plan.as_ode() {
            Some(ode) => AdaptiveCheckpoint::from_bytes(bytes, ode).map(Self::from_adaptive),
            None => Self::from_process(ProcessCursor::from_bytes(bytes, plan)?),
        }
    }
}
