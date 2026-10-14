//! Closed remeshing-trajectory projections and their local admission.
use super::*;
use serde::{Deserialize, Serialize};

pub(super) fn segment_reference(
    segment: &SpatialTrajectorySegmentEnvelopeV3,
) -> Result<WireSegmentReferenceV3, Diagnostic> {
    Ok(WireSegmentReferenceV3 {
        segment_sha256: segment.digest()?.to_string(),
        origin: WireSegmentOriginTagV3::encode(segment.origin()),
        source_prefix_v2_sha256: segment.wire.source_prefix_v2_sha256.clone(),
        source_state_v2_sha256: segment.wire.source_state_v2_sha256.clone(),
        target_reference: segment.wire.target_reference.clone(),
        overlap_sha256: segment.wire.overlap_sha256.clone(),
        transfer_receipt_sha256: segment.wire.transfer_receipt_sha256.clone(),
        first_step: segment.first_step(),
        last_step: segment.last_step(),
        first_time_s: segment.first_time_s(),
        last_time_s: segment.last_time_s(),
        first_predecessor_sha256: segment.wire.states[0].predecessor_sha256.clone(),
        last_state_sha256: segment.last_state().to_string(),
        state_count: segment.wire.states.len(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTrajectorySegmentV3 {
    pub(super) schema: String,
    pub(super) encoding: String,
    pub(super) source_prefix_v2_sha256: String,
    pub(super) source_state_v2_sha256: String,
    pub(super) target_reference: WireTargetReferenceV3,
    pub(super) overlap_sha256: String,
    pub(super) transfer_receipt_sha256: String,
    pub(super) origin: WireSegmentOriginV3,
    pub(super) states: Vec<WireTargetStateReferenceV3>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireSegmentOriginV3 {
    Remesh {
        source_step: u64,
        source_time_s: f64,
    },
    Continuation {
        predecessor_spatial_state_v3_sha256: String,
    },
}

impl WireSegmentOriginV3 {
    pub(super) fn predecessor(&self) -> Option<&str> {
        match self {
            Self::Remesh { .. } => None,
            Self::Continuation {
                predecessor_spatial_state_v3_sha256,
            } => Some(predecessor_spatial_state_v3_sha256),
        }
    }

    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        match self {
            Self::Remesh { source_time_s, .. }
                if source_time_s.is_finite()
                    && *source_time_s >= 0.0
                    && !is_negative_zero(*source_time_s) =>
            {
                Ok(())
            }
            Self::Continuation {
                predecessor_spatial_state_v3_sha256,
            } => ArtifactDigest::from_hex(predecessor_spatial_state_v3_sha256.clone()).map(drop),
            _ => Err(invalid_artifact("V3 remesh segment source time is invalid")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTargetReferenceV3 {
    pub(super) model_sha256: String,
    pub(super) semantic_revision: u64,
    pub(super) realization_sha256: String,
    pub(super) geometry_sha256: String,
    pub(super) correspondence_sha256: String,
    pub(super) mesh_sha256: String,
}

impl WireTargetReferenceV3 {
    pub(super) fn from_state(state: &SpatialStateEnvelope) -> Self {
        Self {
            model_sha256: state.model_artifact().to_string(),
            semantic_revision: state.semantic_revision(),
            realization_sha256: state.realization_artifact().to_string(),
            geometry_sha256: state.reference_geometry_artifact().to_string(),
            correspondence_sha256: state.correspondence_artifact().to_string(),
            mesh_sha256: state.reference_mesh_artifact().to_string(),
        }
    }

    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [
            &self.model_sha256,
            &self.realization_sha256,
            &self.geometry_sha256,
            &self.correspondence_sha256,
            &self.mesh_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.semantic_revision == 0 {
            return Err(invalid_artifact(
                "V3 trajectory target semantic revision must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTargetStateReferenceV3 {
    pub(super) state_sha256: String,
    pub(super) origin: WireStateOriginV3,
    pub(super) step: u64,
    pub(super) time_s: f64,
    pub(super) geometry_state_sha256: String,
    pub(super) predecessor_sha256: Option<String>,
    pub(super) source_state_v2_sha256: String,
    pub(super) overlap_sha256: String,
    pub(super) transfer_receipt_sha256: String,
}

impl WireTargetStateReferenceV3 {
    pub(super) fn from_state(state: &SpatialStateEnvelope) -> Result<Self, Diagnostic> {
        let anchor = state
            .remesh_anchor()
            .ok_or_else(|| invalid_artifact("target trajectory state has no remesh anchor"))?;
        Ok(Self {
            state_sha256: state.digest()?.to_string(),
            origin: WireStateOriginV3::encode(state.origin())?,
            step: state.step(),
            time_s: state.time_s(),
            geometry_state_sha256: state.geometry_state_artifact().to_string(),
            predecessor_sha256: state.predecessor().map(|digest| digest.to_string()),
            source_state_v2_sha256: anchor.source_spatial_state().to_string(),
            overlap_sha256: anchor.overlap_artifact().to_string(),
            transfer_receipt_sha256: anchor.transfer_receipt_artifact().to_string(),
        })
    }

    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [
            &self.state_sha256,
            &self.geometry_state_sha256,
            &self.source_state_v2_sha256,
            &self.overlap_sha256,
            &self.transfer_receipt_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if let Some(predecessor) = &self.predecessor_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        if !self.time_s.is_finite()
            || self.time_s < 0.0
            || is_negative_zero(self.time_s)
            || matches!(self.origin, WireStateOriginV3::Remesh) != self.predecessor_sha256.is_none()
        {
            return Err(invalid_artifact(
                "V3 trajectory state coordinate or predecessor shape is invalid",
            ));
        }
        Ok(())
    }

    pub(super) fn require_adjacent(&self, next: &Self) -> Result<(), Diagnostic> {
        if next.origin != WireStateOriginV3::Continuous
            || next.predecessor_sha256.as_deref() != Some(&self.state_sha256)
            || next.step
                != self
                    .step
                    .checked_add(1)
                    .ok_or_else(|| invalid_artifact("V3 trajectory step overflows"))?
            || next.time_s <= self.time_s
            || next.source_state_v2_sha256 != self.source_state_v2_sha256
            || next.overlap_sha256 != self.overlap_sha256
            || next.transfer_receipt_sha256 != self.transfer_receipt_sha256
        {
            return Err(invalid_artifact(
                "V3 trajectory states do not form an exact acyclic append edge",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireStateOriginV3 {
    Remesh,
    Continuous,
}

impl WireStateOriginV3 {
    pub(super) fn encode(value: SpatialStateOriginKind) -> Result<Self, Diagnostic> {
        match value {
            SpatialStateOriginKind::Remesh => Ok(Self::Remesh),
            SpatialStateOriginKind::Continuous => Ok(Self::Continuous),
            SpatialStateOriginKind::Initial => Err(invalid_artifact(
                "initial state cannot appear in a remeshed target trajectory",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireSpatialTrajectoryV3 {
    pub(super) schema: String,
    pub(super) encoding: String,
    pub(super) generation: u64,
    pub(super) previous_root_sha256: Option<String>,
    pub(super) source_prefix_v2_sha256: String,
    pub(super) source_state_v2_sha256: String,
    pub(super) target_reference: WireTargetReferenceV3,
    pub(super) overlap_sha256: String,
    pub(super) transfer_receipt_sha256: String,
    pub(super) segments: Vec<WireSegmentReferenceV3>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireSegmentReferenceV3 {
    pub(super) segment_sha256: String,
    pub(super) origin: WireSegmentOriginTagV3,
    pub(super) source_prefix_v2_sha256: String,
    pub(super) source_state_v2_sha256: String,
    pub(super) target_reference: WireTargetReferenceV3,
    pub(super) overlap_sha256: String,
    pub(super) transfer_receipt_sha256: String,
    pub(super) first_step: u64,
    pub(super) last_step: u64,
    pub(super) first_time_s: f64,
    pub(super) last_time_s: f64,
    pub(super) first_predecessor_sha256: Option<String>,
    pub(super) last_state_sha256: String,
    pub(super) state_count: usize,
}

impl WireSegmentReferenceV3 {
    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [
            &self.segment_sha256,
            &self.source_prefix_v2_sha256,
            &self.source_state_v2_sha256,
            &self.overlap_sha256,
            &self.transfer_receipt_sha256,
            &self.last_state_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if let Some(predecessor) = &self.first_predecessor_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        self.target_reference.validate()?;
        if self.state_count == 0
            || self.first_step > self.last_step
            || !self.first_time_s.is_finite()
            || !self.last_time_s.is_finite()
            || self.first_time_s < 0.0
            || self.last_time_s < self.first_time_s
            || is_negative_zero(self.first_time_s)
            || is_negative_zero(self.last_time_s)
        {
            return Err(invalid_artifact("V3 trajectory segment summary is invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireSegmentOriginTagV3 {
    Remesh,
    Continuation,
}

impl WireSegmentOriginTagV3 {
    pub(super) const fn encode(value: SpatialTrajectorySegmentOriginKindV3) -> Self {
        match value {
            SpatialTrajectorySegmentOriginKindV3::Remesh => Self::Remesh,
            SpatialTrajectorySegmentOriginKindV3::Continuation => Self::Continuation,
        }
    }
}

pub(super) fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}
