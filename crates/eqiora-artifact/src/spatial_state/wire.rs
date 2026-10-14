//! Closed origins and bounded local admission for the current moving-state wire.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireMovingSpatialState {
    pub schema: String,
    pub encoding: String,
    pub spatial_dimension: u64,
    pub reference: MovingSpatialReference,
    pub accepted: WireAccepted,
    pub geometry_state_sha256: String,
    pub driver_snapshot_sha256: String,
    pub origin: WireOrigin,
    pub fields: Vec<WireStateField>,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireAccepted {
    pub step: u64,
    pub time_s: f64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireOrigin {
    Initial {},
    Continuous {
        predecessor_spatial_state_sha256: String,
        predecessor_geometry_state_sha256: String,
        remesh_anchor: Option<RemeshStateAnchor>,
    },
    Remesh {
        anchor: RemeshStateAnchor,
    },
}
/// Exact source, overlap and transfer evidence for one representation change.
///
/// These identities form one indivisible anchor, retained across continuation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemeshStateAnchor {
    pub(super) source_spatial_state_sha256: String,
    pub(super) overlap_sha256: String,
    pub(super) transfer_receipt_sha256: String,
}
impl RemeshStateAnchor {
    /// Exact accepted source state.
    #[must_use]
    pub fn source_spatial_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.source_spatial_state_sha256.clone())
    }
    /// Exact material/current overlap.
    #[must_use]
    pub fn overlap_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.overlap_sha256.clone())
    }
    /// Exact field-aware transfer receipt.
    #[must_use]
    pub fn transfer_receipt_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.transfer_receipt_sha256.clone())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireStateField {
    pub support_domain_ulid: String,
    pub field_ulid: String,
    pub snapshot_sha256: String,
}

impl SpatialStateEnvelope {
    pub(super) fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SPATIAL_STATE_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || !matches!(self.spatial_dimension(), 2 | 3)
        {
            return Err(invalid_artifact(
                "unsupported moving spatial schema, encoding or dimension",
            ));
        }
        self.wire.reference.validate()?;
        for digest in [
            &self.wire.geometry_state_sha256,
            &self.wire.driver_snapshot_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        self.wire
            .origin
            .validate(self.step(), self.time_s(), self.spatial_dimension())?;
        if self.wire.fields.is_empty() || self.wire.fields.len() > limits.max_spatial_state_fields {
            return Err(invalid_artifact(
                "moving spatial Field inventory is empty or exceeds its bound",
            ));
        }
        let mut previous = None;
        let mut driver_count = 0;
        for entry in &self.wire.fields {
            parse_id::<kinds::Domain>(&entry.support_domain_ulid, "support Domain")?;
            let field = parse_id::<kinds::Field>(&entry.field_ulid, "Field")?;
            ArtifactDigest::from_hex(entry.snapshot_sha256.clone())?;
            if previous.is_some_and(|previous| previous >= field.ulid()) {
                return Err(invalid_artifact(
                    "moving spatial Fields must be unique and canonically ordered",
                ));
            }
            previous = Some(field.ulid());
            if entry.snapshot_sha256 == self.wire.driver_snapshot_sha256 {
                driver_count += 1;
            }
        }
        if driver_count != 1 {
            return Err(invalid_artifact(
                "moving spatial driver must occur exactly once",
            ));
        }
        Ok(())
    }
}
pub(super) fn validate_coordinate(step: u64, time_s: f64) -> Result<(), Diagnostic> {
    if step > (1_u64 << 53)
        || !time_s.is_finite()
        || time_s < 0.0
        || (time_s == 0.0 && time_s.is_sign_negative())
    {
        return Err(invalid_artifact(
            "moving spatial coordinate must be finite, nonnegative, canonical and exactly representable",
        ));
    }
    Ok(())
}
pub(crate) fn parse_id<K: eqiora_core::Entity>(
    value: &str,
    label: &str,
) -> Result<Id<K>, Diagnostic> {
    let id = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("{label} ULID is malformed")))?;
    if id.to_string() != value {
        return Err(invalid_artifact(format!("{label} ULID is not canonical")));
    }
    Ok(Id::from_ulid(id))
}

impl WireOrigin {
    pub(crate) const fn kind(&self) -> SpatialStateOriginKind {
        match self {
            Self::Initial {} => SpatialStateOriginKind::Initial,
            Self::Continuous { .. } => SpatialStateOriginKind::Continuous,
            Self::Remesh { .. } => SpatialStateOriginKind::Remesh,
        }
    }
    pub(crate) const fn anchor(&self) -> Option<&RemeshStateAnchor> {
        match self {
            Self::Initial {} => None,
            Self::Continuous { remesh_anchor, .. } => remesh_anchor.as_ref(),
            Self::Remesh { anchor } => Some(anchor),
        }
    }
    pub(crate) fn validate(
        &self,
        step: u64,
        time_s: f64,
        dimension: usize,
    ) -> Result<(), Diagnostic> {
        validate_coordinate(step, time_s)?;
        match self {
            WireOrigin::Initial {} if step != 0 || time_s != 0.0 => {
                return Err(invalid_artifact(
                    "initial moving state requires zero step and time",
                ));
            }
            WireOrigin::Continuous {
                predecessor_spatial_state_sha256,
                predecessor_geometry_state_sha256,
                ..
            } => {
                ArtifactDigest::from_hex(predecessor_spatial_state_sha256.clone())?;
                ArtifactDigest::from_hex(predecessor_geometry_state_sha256.clone())?;
                if step == 0 || time_s == 0.0 {
                    return Err(invalid_artifact(
                        "continuous moving state requires positive step and time",
                    ));
                }
            }
            _ => {}
        }
        if let Some(anchor) = self.anchor() {
            if dimension != 2 {
                return Err(invalid_artifact(
                    "remesh transfer currently requires two dimensions",
                ));
            }
            for digest in [
                &anchor.source_spatial_state_sha256,
                &anchor.overlap_sha256,
                &anchor.transfer_receipt_sha256,
            ] {
                ArtifactDigest::from_hex(digest.clone())?;
            }
        }
        Ok(())
    }
}
