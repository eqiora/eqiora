//! Exact state edges shared by states and immutable trajectory projections.
use super::{SpatialStateEnvelope, WireOrigin};
use crate::{ArtifactDigest, invalid_artifact};
use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateLineage {
    pub state_sha256: String,
    pub geometry_state_sha256: String,
    pub step: u64,
    pub time_s: f64,
    pub origin: WireOrigin,
}
impl StateLineage {
    pub fn from_state(state: &SpatialStateEnvelope) -> Result<Self, Diagnostic> {
        Ok(Self {
            state_sha256: state.digest()?.to_string(),
            geometry_state_sha256: state.geometry_state_artifact().to_string(),
            step: state.step(),
            time_s: state.time_s(),
            origin: state.wire.origin.clone(),
        })
    }
    pub fn validate(&self, dimension: usize) -> Result<(), Diagnostic> {
        ArtifactDigest::from_hex(self.state_sha256.clone())?;
        ArtifactDigest::from_hex(self.geometry_state_sha256.clone())?;
        self.origin.validate(self.step, self.time_s, dimension)
    }
    pub fn require_continuation(&self, next: &Self) -> Result<(), Diagnostic> {
        crate::geometry_state::validation::validate_successor(
            next.step,
            next.time_s,
            self.step,
            self.time_s,
        )?;
        match &next.origin {
            WireOrigin::Continuous {
                predecessor_spatial_state_sha256,
                predecessor_geometry_state_sha256,
                remesh_anchor,
            } if predecessor_spatial_state_sha256 == &self.state_sha256
                && predecessor_geometry_state_sha256 == &self.geometry_state_sha256
                && remesh_anchor.as_ref() == self.origin.anchor() =>
            {
                Ok(())
            }
            _ => Err(invalid_artifact(
                "moving spatial continuation breaks its exact state, geometry or transfer edge",
            )),
        }
    }
    pub fn require_remesh(&self, next: &Self) -> Result<(), Diagnostic> {
        match &next.origin {
            WireOrigin::Remesh { anchor }
                if anchor.source_spatial_state().as_str() == self.state_sha256
                    && next.step == self.step
                    && next.time_s == self.time_s =>
            {
                Ok(())
            }
            _ => Err(invalid_artifact(
                "remesh does not replace the exact source state at the same coordinate",
            )),
        }
    }
}
