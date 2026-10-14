//! Remeshing-aware moving spatial states over one target topology.

use std::str::FromStr;

use eqiora_core::entity::Entity;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, FieldDecoderLimits, FieldSnapshotEnvelopeV2,
    GeometryStateEnvelope, GeometryStateOriginKind, MeshRevisionOverlapEnvelopeV1,
    RemeshTransferReceiptEnvelopeV1, ReplayableCanonicalModelArtifact,
    ValidatedMovingSpatialContextV2, ValidatedRemeshGeometrySource, check_json_limits,
    invalid_artifact,
};

const SPATIAL_STATE_SCHEMA: &str = "eqiora.spatial-state-envelope/v3";

/// Closed origin of one remeshing-aware target state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialStateOriginKindV3 {
    /// Same-time transition from the finalized source representation.
    Remesh,
    /// Positive-duration continuation on the target topology.
    Continuous,
}

/// One complete target state retaining its immutable remesh transition anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialStateEnvelopeV3 {
    wire: WireSpatialStateEnvelopeV3,
}

impl SpatialStateEnvelopeV3 {
    /// Construct the first target state at an exact same-time remesh seam.
    ///
    /// # Errors
    /// Returns `EQ0901` for any stale resource, incomplete Field inventory,
    /// invalid geometry/overlap/receipt replay, or changed step/time.
    #[allow(clippy::too_many_arguments)]
    pub fn remesh<M: ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelope,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        receipt: &RemeshTransferReceiptEnvelopeV1,
        source_snapshots: &[FieldSnapshotEnvelopeV2],
        target_snapshots: &[FieldSnapshotEnvelopeV2],
    ) -> Result<Self, Diagnostic> {
        if target_geometry_state.origin() != GeometryStateOriginKind::Remesh
            || target_geometry_state.step() != source.state().step()
            || target_geometry_state.time_s() != source.state().time_s()
        {
            return Err(invalid_artifact(
                "first target state must be a same-coordinate remesh representation",
            ));
        }
        let (fields, target_solid_displacement) =
            validate_target_snapshots(target_context, target_geometry_state, target_snapshots)?;
        overlap.validate_against(
            source,
            target_context,
            target_geometry_state,
            target_solid_displacement,
        )?;
        receipt.validate_against(
            source,
            overlap,
            target_context,
            target_geometry_state,
            source_snapshots,
            target_snapshots,
        )?;
        Self::finish(
            target_context,
            target_geometry_state,
            fields,
            WireSpatialStateOriginV3::Remesh {
                source_spatial_state_v2_sha256: source.state().digest()?.to_string(),
                overlap_sha256: overlap.digest()?.to_string(),
                transfer_receipt_sha256: receipt.digest()?.to_string(),
            },
        )
    }

    /// Construct an immediate positive-duration continuation on the target mesh.
    ///
    /// The original remesh anchor is retained verbatim rather than summarized
    /// or replaced by a newer receipt.
    ///
    /// # Errors
    /// Returns `EQ0901` for non-adjacent state/geometry coordinates, stale
    /// resources, incomplete snapshots, or changed transition ancestry.
    pub fn continuous<'a, M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        geometry_state: &GeometryStateEnvelope,
        predecessor_geometry_state: &GeometryStateEnvelope,
        predecessor: &Self,
        snapshots: &[FieldSnapshotEnvelopeV2],
        driver_blocks: impl IntoIterator<Item = &'a crate::DiscreteFieldEnvelopeV1>,
    ) -> Result<Self, Diagnostic> {
        let (fields, driver) = validate_target_snapshots(context, geometry_state, snapshots)?;
        geometry_state.validate_against(
            context,
            Some(predecessor_geometry_state),
            driver,
            driver_blocks,
        )?;
        predecessor.require_context(context)?;
        if predecessor.geometry_state_artifact() != predecessor_geometry_state.digest()?
            || geometry_state.step()
                != predecessor
                    .step()
                    .checked_add(1)
                    .ok_or_else(|| invalid_artifact("spatial-state/v3 step overflows"))?
            || geometry_state.time_s() <= predecessor.time_s()
        {
            return Err(invalid_artifact(
                "continuous spatial-state/v3 must immediately follow its target predecessor",
            ));
        }
        Self::finish(
            context,
            geometry_state,
            fields,
            WireSpatialStateOriginV3::Continuous {
                predecessor_spatial_state_v3_sha256: predecessor.digest()?.to_string(),
                source_spatial_state_v2_sha256: predecessor
                    .remesh_source_spatial_state()
                    .to_string(),
                overlap_sha256: predecessor.overlap_artifact().to_string(),
                transfer_receipt_sha256: predecessor.transfer_receipt_artifact().to_string(),
            },
        )
    }

    fn finish<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        geometry_state: &GeometryStateEnvelope,
        fields: Vec<WireSpatialFieldV3>,
        origin: WireSpatialStateOriginV3,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireSpatialStateEnvelopeV3 {
                schema: SPATIAL_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: WireSpatialReferenceV3 {
                    model_sha256: context.model_reference().artifact().to_string(),
                    semantic_revision: context.model_reference().semantic_revision().get(),
                    realization_sha256: context.realization().digest()?.to_string(),
                    geometry_sha256: context.geometry().digest()?.to_string(),
                    correspondence_sha256: context.correspondence().digest()?.to_string(),
                    mesh_sha256: context.mesh().digest()?.to_string(),
                },
                accepted_step: geometry_state.step(),
                accepted_time_s: geometry_state.time_s(),
                geometry_state_v2_sha256: geometry_state.digest()?.to_string(),
                origin,
                fields,
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded state data without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid spatial-state/v3 JSON: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize spatial-state/v3: {error}"))
        })
    }

    /// Domain-separated state identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SPATIAL_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Closed state origin.
    #[must_use]
    pub const fn origin(&self) -> SpatialStateOriginKindV3 {
        match self.wire.origin {
            WireSpatialStateOriginV3::Remesh { .. } => SpatialStateOriginKindV3::Remesh,
            WireSpatialStateOriginV3::Continuous { .. } => SpatialStateOriginKindV3::Continuous,
        }
    }

    /// Accepted step ordinal.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.wire.accepted_step
    }

    /// Accepted coherent-SI time.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.wire.accepted_time_s
    }

    /// Exact target geometry state.
    #[must_use]
    pub fn geometry_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.geometry_state_v2_sha256.clone())
    }

    /// Exact finalized source V2 state retained through all continuations.
    #[must_use]
    pub fn remesh_source_spatial_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.origin.source_spatial_state().to_owned())
    }

    /// Exact material/current overlap retained through all continuations.
    #[must_use]
    pub fn overlap_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.origin.overlap().to_owned())
    }

    /// Exact field-aware transfer receipt retained through all continuations.
    #[must_use]
    pub fn transfer_receipt_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.origin.receipt().to_owned())
    }

    /// Immediate V3 predecessor, absent only at the remesh seam.
    #[must_use]
    pub fn predecessor(&self) -> Option<ArtifactDigest> {
        self.wire
            .origin
            .predecessor()
            .map(|value| ArtifactDigest(value.to_owned()))
    }

    /// Exact target Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Exact target semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Exact target Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
    }

    /// Exact target Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.geometry_sha256.clone())
    }

    /// Exact target correspondence artifact.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact target mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.mesh_sha256.clone())
    }

    /// Canonically ordered `(Domain, Field, snapshot)` references.
    #[must_use]
    pub fn fields(&self) -> Vec<(Id<kinds::Domain>, Id<kinds::Field>, ArtifactDigest)> {
        self.wire
            .fields
            .iter()
            .map(|entry| {
                (
                    parse_id(&entry.support_domain_ulid, "support Domain")
                        .expect("validated spatial-state/v3 Domain"),
                    parse_id(&entry.field_ulid, "Field").expect("validated spatial-state/v3 Field"),
                    ArtifactDigest(entry.snapshot_sha256.clone()),
                )
            })
            .collect()
    }

    /// Exact snapshot for one Semantic Field.
    #[must_use]
    pub fn field_snapshot(&self, field: Id<kinds::Field>) -> Option<ArtifactDigest> {
        self.wire
            .fields
            .binary_search_by_key(&field.ulid(), |entry| {
                Ulid::from_str(&entry.field_ulid).expect("validated spatial-state/v3 Field")
            })
            .ok()
            .map(|index| ArtifactDigest(self.wire.fields[index].snapshot_sha256.clone()))
    }

    /// Replay and compare a remesh-origin state.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted dependency.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against_remesh<M: ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelope,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        receipt: &RemeshTransferReceiptEnvelopeV1,
        source_snapshots: &[FieldSnapshotEnvelopeV2],
        target_snapshots: &[FieldSnapshotEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        require_equal(
            self,
            &Self::remesh(
                source,
                target_context,
                target_geometry_state,
                overlap,
                receipt,
                source_snapshots,
                target_snapshots,
            )?,
        )
    }

    /// Replay and compare a continuous target state.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted dependency.
    pub fn validate_against_continuous<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        geometry_state: &GeometryStateEnvelope,
        predecessor_geometry_state: &GeometryStateEnvelope,
        predecessor: &Self,
        snapshots: &[FieldSnapshotEnvelopeV2],
        driver_blocks: impl IntoIterator<Item = &'a crate::DiscreteFieldEnvelopeV1>,
    ) -> Result<(), Diagnostic> {
        require_equal(
            self,
            &Self::continuous(
                context,
                geometry_state,
                predecessor_geometry_state,
                predecessor,
                snapshots,
                driver_blocks,
            )?,
        )
    }

    pub(crate) fn require_adjacent(&self, next: &Self) -> Result<(), Diagnostic> {
        if next.predecessor() != Some(self.digest()?)
            || next.step()
                != self
                    .step()
                    .checked_add(1)
                    .ok_or_else(|| invalid_artifact("spatial-state/v3 adjacency overflows"))?
            || next.time_s() <= self.time_s()
            || next.remesh_source_spatial_state() != self.remesh_source_spatial_state()
            || next.overlap_artifact() != self.overlap_artifact()
            || next.transfer_receipt_artifact() != self.transfer_receipt_artifact()
            || next.wire.reference != self.wire.reference
        {
            return Err(invalid_artifact(
                "spatial-state/v3 transition ancestry is not an immediate continuation",
            ));
        }
        Ok(())
    }

    pub(crate) fn require_context<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
    ) -> Result<(), Diagnostic> {
        if self.model_artifact() != *context.model_reference().artifact()
            || self.wire.reference.semantic_revision
                != context.model_reference().semantic_revision().get()
            || self.realization_artifact() != context.realization().digest()?
            || self.reference_geometry_artifact() != context.geometry().digest()?
            || self.correspondence_artifact() != context.correspondence().digest()?
            || self.reference_mesh_artifact() != context.mesh().digest()?
        {
            return Err(invalid_artifact(
                "spatial-state/v3 differs from its exact target context",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SPATIAL_STATE_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-state/v3 schema or encoding",
            ));
        }
        self.wire.reference.validate()?;
        ArtifactDigest::from_hex(self.wire.geometry_state_v2_sha256.clone())?;
        self.wire.origin.validate()?;
        if !self.time_s().is_finite()
            || self.time_s() < 0.0
            || is_negative_zero(self.time_s())
            || self.wire.fields.is_empty()
            || self.wire.fields.len() > limits.max_spatial_state_fields
        {
            return Err(invalid_artifact(
                "spatial-state/v3 coordinate or Field inventory is invalid",
            ));
        }
        for entry in &self.wire.fields {
            parse_id::<kinds::Domain>(&entry.support_domain_ulid, "support Domain")?;
            parse_id::<kinds::Field>(&entry.field_ulid, "Field")?;
            ArtifactDigest::from_hex(entry.snapshot_sha256.clone())?;
        }
        if self
            .wire
            .fields
            .windows(2)
            .any(|pair| pair[0].field_ulid >= pair[1].field_ulid)
        {
            return Err(invalid_artifact(
                "spatial-state/v3 Fields must be unique and canonical",
            ));
        }
        Ok(())
    }
}

fn validate_target_snapshots<'a, M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
    geometry_state: &GeometryStateEnvelope,
    snapshots: &'a [FieldSnapshotEnvelopeV2],
) -> Result<(Vec<WireSpatialFieldV3>, &'a FieldSnapshotEnvelopeV2), Diagnostic> {
    if geometry_state.model_artifact() != *context.model_reference().artifact()
        || geometry_state.realization_artifact() != context.realization().digest()?
        || geometry_state.reference_geometry_artifact() != context.geometry().digest()?
        || geometry_state.reference_correspondence_artifact()
            != context.correspondence().digest()?
        || geometry_state.reference_mesh_artifact() != context.mesh().digest()?
    {
        return Err(invalid_artifact(
            "spatial-state/v3 geometry differs from the target context",
        ));
    }
    let mut fields = Vec::with_capacity(snapshots.len());
    for snapshot in snapshots {
        context.validate_snapshot(snapshot)?;
        fields.push(WireSpatialFieldV3 {
            support_domain_ulid: snapshot.support_domain().ulid().to_string(),
            field_ulid: snapshot.field().ulid().to_string(),
            snapshot_sha256: snapshot.digest()?.to_string(),
        });
    }
    fields.sort_by(|left, right| left.field_ulid.cmp(&right.field_ulid));
    let expected = context
        .represented_fields()
        .iter()
        .map(|entry| (entry.domain().ulid(), entry.field().ulid()))
        .collect::<Vec<_>>();
    let actual = fields
        .iter()
        .map(|entry| {
            Ok((
                Ulid::from_str(&entry.support_domain_ulid)
                    .map_err(|_| invalid_artifact("invalid target Domain ULID"))?,
                Ulid::from_str(&entry.field_ulid)
                    .map_err(|_| invalid_artifact("invalid target Field ULID"))?,
            ))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let mut expected = expected;
    expected.sort_by_key(|entry| entry.1);
    if actual != expected
        || fields
            .windows(2)
            .any(|pair| pair[0].field_ulid >= pair[1].field_ulid)
    {
        return Err(invalid_artifact(
            "spatial-state/v3 must contain the complete exact target Field inventory and driver",
        ));
    }
    let driver = snapshots
        .iter()
        .find(|snapshot| snapshot.field() == context.driver())
        .ok_or_else(|| invalid_artifact("spatial-state/v3 omits its exact geometry driver"))?;
    if driver.digest()? != geometry_state.driver_snapshot() {
        return Err(invalid_artifact(
            "spatial-state/v3 substitutes its geometry driver",
        ));
    }
    Ok((fields, driver))
}

fn require_equal(
    actual: &SpatialStateEnvelopeV3,
    expected: &SpatialStateEnvelopeV3,
) -> Result<(), Diagnostic> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_artifact(
            "spatial-state/v3 differs from exact dependency replay",
        ))
    }
}

fn parse_id<K: Entity>(value: &str, label: &str) -> Result<Id<K>, Diagnostic> {
    let ulid = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("spatial-state/v3 {label} ULID is malformed")))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(format!(
            "spatial-state/v3 {label} ULID spelling is noncanonical",
        )));
    }
    Ok(Id::from_ulid(ulid))
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialStateEnvelopeV3 {
    schema: String,
    encoding: String,
    reference: WireSpatialReferenceV3,
    accepted_step: u64,
    accepted_time_s: f64,
    geometry_state_v2_sha256: String,
    origin: WireSpatialStateOriginV3,
    fields: Vec<WireSpatialFieldV3>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialReferenceV3 {
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: String,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
}

impl WireSpatialReferenceV3 {
    fn validate(&self) -> Result<(), Diagnostic> {
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
                "spatial-state/v3 semantic revision must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSpatialStateOriginV3 {
    Remesh {
        source_spatial_state_v2_sha256: String,
        overlap_sha256: String,
        transfer_receipt_sha256: String,
    },
    Continuous {
        predecessor_spatial_state_v3_sha256: String,
        source_spatial_state_v2_sha256: String,
        overlap_sha256: String,
        transfer_receipt_sha256: String,
    },
}

impl WireSpatialStateOriginV3 {
    fn source_spatial_state(&self) -> &str {
        match self {
            Self::Remesh {
                source_spatial_state_v2_sha256,
                ..
            }
            | Self::Continuous {
                source_spatial_state_v2_sha256,
                ..
            } => source_spatial_state_v2_sha256,
        }
    }

    fn overlap(&self) -> &str {
        match self {
            Self::Remesh { overlap_sha256, .. } | Self::Continuous { overlap_sha256, .. } => {
                overlap_sha256
            }
        }
    }

    fn receipt(&self) -> &str {
        match self {
            Self::Remesh {
                transfer_receipt_sha256,
                ..
            }
            | Self::Continuous {
                transfer_receipt_sha256,
                ..
            } => transfer_receipt_sha256,
        }
    }

    fn predecessor(&self) -> Option<&str> {
        match self {
            Self::Remesh { .. } => None,
            Self::Continuous {
                predecessor_spatial_state_v3_sha256,
                ..
            } => Some(predecessor_spatial_state_v3_sha256),
        }
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [self.source_spatial_state(), self.overlap(), self.receipt()] {
            ArtifactDigest::from_hex(digest.to_owned())?;
        }
        if let Some(predecessor) = self.predecessor() {
            ArtifactDigest::from_hex(predecessor.to_owned())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialFieldV3 {
    support_domain_ulid: String,
    field_ulid: String,
    snapshot_sha256: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> SpatialStateEnvelopeV3 {
        SpatialStateEnvelopeV3 {
            wire: WireSpatialStateEnvelopeV3 {
                schema: SPATIAL_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: WireSpatialReferenceV3 {
                    model_sha256: "00".repeat(32),
                    semantic_revision: 1,
                    realization_sha256: "11".repeat(32),
                    geometry_sha256: "22".repeat(32),
                    correspondence_sha256: "33".repeat(32),
                    mesh_sha256: "44".repeat(32),
                },
                accepted_step: 5,
                accepted_time_s: 0.5,
                geometry_state_v2_sha256: "55".repeat(32),
                origin: WireSpatialStateOriginV3::Remesh {
                    source_spatial_state_v2_sha256: "66".repeat(32),
                    overlap_sha256: "77".repeat(32),
                    transfer_receipt_sha256: "88".repeat(32),
                },
                fields: vec![
                    WireSpatialFieldV3 {
                        support_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned(),
                        field_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAX".to_owned(),
                        snapshot_sha256: "99".repeat(32),
                    },
                    WireSpatialFieldV3 {
                        support_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAW".to_owned(),
                        field_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAY".to_owned(),
                        snapshot_sha256: "aa".repeat(32),
                    },
                ],
            },
        }
    }

    #[test]
    fn state_v3_roundtrip_and_digest_are_frozen() {
        let value = state();
        value.validate_local(FieldDecoderLimits::default()).unwrap();
        let bytes = value.canonical_json().unwrap();
        assert_eq!(
            SpatialStateEnvelopeV3::from_json(&bytes, FieldDecoderLimits::default()).unwrap(),
            value
        );
        assert_eq!(
            value.digest().unwrap().to_string(),
            "fbd8274347041c49bd03c7e4031a318714579531cfd8221dcd8d76ac9145b35b"
        );
    }

    #[test]
    fn state_v3_rejects_duplicate_field_and_resource_excess() {
        let value = state();
        let mut duplicate = value.clone();
        duplicate.wire.fields[1].field_ulid = duplicate.wire.fields[0].field_ulid.clone();
        assert!(
            duplicate
                .validate_local(FieldDecoderLimits::default())
                .is_err()
        );

        let limits = FieldDecoderLimits {
            max_spatial_state_fields: 1,
            ..FieldDecoderLimits::default()
        };
        assert!(
            SpatialStateEnvelopeV3::from_json(&value.canonical_json().unwrap(), limits,).is_err()
        );
    }
}
