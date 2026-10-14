//! Bounded projections of exact state edges, shared across topology changes.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FieldIdentity {
    pub support_domain_ulid: String,
    pub field_ulid: String,
}
impl FieldIdentity {
    pub fn from_state(state: &SpatialStateEnvelope) -> Vec<Self> {
        state
            .fields()
            .into_iter()
            .map(|(domain, field, _)| Self {
                support_domain_ulid: domain.ulid().to_string(),
                field_ulid: field.ulid().to_string(),
            })
            .collect()
    }
    pub fn validate_inventory(fields: &[Self], limit: usize) -> Result<(), Diagnostic> {
        if fields.is_empty() || fields.len() > limit {
            return Err(invalid_artifact(
                "trajectory Field inventory is empty or exceeds its bound",
            ));
        }
        let mut previous = None;
        for field in fields {
            crate::spatial_state::wire::parse_id::<kinds::Domain>(
                &field.support_domain_ulid,
                "support Domain",
            )?;
            let id =
                crate::spatial_state::wire::parse_id::<kinds::Field>(&field.field_ulid, "Field")?;
            if previous.is_some_and(|previous| previous >= id.ulid()) {
                return Err(invalid_artifact(
                    "trajectory Fields must be unique and canonically ordered",
                ));
            }
            previous = Some(id.ulid());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireSegment {
    pub schema: String,
    pub encoding: String,
    pub spatial_dimension: usize,
    pub reference: MovingSpatialReference,
    pub fields: Vec<FieldIdentity>,
    pub states: Vec<StateLineage>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SegmentReference {
    pub segment_sha256: String,
    pub spatial_dimension: usize,
    pub reference: MovingSpatialReference,
    pub fields: Vec<FieldIdentity>,
    pub first: StateLineage,
    pub last: StateLineage,
    pub state_count: usize,
}
impl SegmentReference {
    pub fn from_segment(segment: &SpatialTrajectorySegmentEnvelope) -> Result<Self, Diagnostic> {
        Ok(Self {
            segment_sha256: segment.digest()?.to_string(),
            spatial_dimension: segment.wire.spatial_dimension,
            reference: segment.wire.reference.clone(),
            fields: segment.wire.fields.clone(),
            first: segment.wire.states[0].clone(),
            last: segment.wire.states[segment.state_count() - 1].clone(),
            state_count: segment.state_count(),
        })
    }
    pub fn validate(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        ArtifactDigest::from_hex(self.segment_sha256.clone())?;
        self.reference.validate()?;
        FieldIdentity::validate_inventory(&self.fields, limits.max_spatial_state_fields)?;
        if !matches!(self.spatial_dimension, 2 | 3)
            || self.state_count == 0
            || self.state_count > limits.max_trajectory_segment_states
        {
            return Err(invalid_artifact(
                "trajectory segment dimension or state count is invalid",
            ));
        }
        self.first.validate(self.spatial_dimension)?;
        self.last.validate(self.spatial_dimension)?;
        if self.state_count == 1 {
            if self.first != self.last {
                return Err(invalid_artifact("single-state segment endpoints differ"));
            }
        } else if self.first.step.checked_add((self.state_count - 1) as u64) != Some(self.last.step)
            || self.first.time_s >= self.last.time_s
            || !matches!(self.last.origin, WireOrigin::Continuous { .. })
            || self.first.origin.anchor() != self.last.origin.anchor()
            || self.first.state_sha256 == self.last.state_sha256
        {
            return Err(invalid_artifact(
                "trajectory segment endpoints do not describe consecutive states",
            ));
        }
        Ok(())
    }
    pub fn require_next(&self, next: &Self) -> Result<(), Diagnostic> {
        if self.reference.model_sha256 != next.reference.model_sha256
            || self.reference.semantic_revision != next.reference.semantic_revision
            || self.spatial_dimension != next.spatial_dimension
            || self.fields != next.fields
        {
            return Err(invalid_artifact(
                "trajectory changes its Model, dimension or semantic Field inventory",
            ));
        }
        match next.first.origin {
            WireOrigin::Remesh { .. } => {
                if self.reference.mesh_sha256 == next.reference.mesh_sha256 {
                    return Err(invalid_artifact(
                        "trajectory remesh requires a different reference mesh",
                    ));
                }
                self.last.require_remesh(&next.first)
            }
            _ => {
                if self.reference != next.reference {
                    return Err(invalid_artifact(
                        "trajectory continuation changes its reference context",
                    ));
                }
                self.last.require_continuation(&next.first)
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTrajectory {
    pub schema: String,
    pub encoding: String,
    pub generation: u64,
    pub previous_root_sha256: Option<String>,
    pub segments: Vec<SegmentReference>,
}
