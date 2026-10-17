//! One moving spatial state for initial, continuous and remeshed representations.
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, FieldDecoderLimits,
    FieldSnapshotEnvelopeV2, GeometryStateEnvelope, GeometryStateOriginKind,
    ReplayableCanonicalModelArtifact, invalid_artifact,
};
use eqiora_core::{Diagnostic, Id, entity::kinds};
use std::str::FromStr;
use ulid::Ulid;

pub(crate) mod context;
use context::MovingSpatialReference;
pub(crate) mod lineage;
mod remesh;
pub(crate) mod wire;
pub use context::ValidatedMovingSpatialContext;
pub use wire::RemeshStateAnchor;
use wire::*;
const SPATIAL_STATE_SCHEMA: &str = "eqiora.spatial-state-envelope/v4";

/// Immediate origin of an accepted moving spatial state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialStateOriginKind {
    /// Initial state on the reference topology.
    Initial,
    /// Positive-duration continuation on the same topology.
    Continuous,
    /// Same-time representation change with exact overlap and transfer evidence.
    Remesh,
}

/// Topology lineage of the current representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpatialStateTopology {
    /// No representation change precedes this state.
    Reference,
    /// This state is a remesh origin or its same-topology continuation.
    Remeshed,
}

/// Complete physical Field inventory bound to one exact current geometry.
///
/// Continuous states retain both immediate predecessor identities and any
/// immutable remesh anchor. Coordinates are owned only by GeometryState.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialStateEnvelope {
    wire: WireMovingSpatialState,
}

impl SpatialStateEnvelope {
    /// Capture an initial state or immediate same-topology continuation.
    ///
    /// # Errors
    /// Rejects stale resources, incomplete Fields, invalid driver evidence,
    /// non-adjacent states or coordinates inconsistent with the exact time step.
    pub fn new<'a, M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        geometry: &GeometryStateEnvelope,
        predecessor: Option<(&Self, &GeometryStateEnvelope)>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        driver_blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
    ) -> Result<Self, Diagnostic> {
        let (fields, driver) = state_inputs(context, geometry, snapshots)?;
        geometry.validate_against(
            context,
            predecessor.map(|(_, geometry)| geometry),
            driver,
            driver_blocks,
        )?;
        let origin = match predecessor {
            None if geometry.origin() == GeometryStateOriginKind::Initial => WireOrigin::Initial {},
            Some((previous, previous_geometry)) => {
                previous.require_context(context)?;
                if geometry.origin() != GeometryStateOriginKind::Continuous
                    || previous.geometry_state_artifact() != previous_geometry.digest()?
                    || previous.step() != previous_geometry.step()
                    || previous.time_s() != previous_geometry.time_s()
                {
                    return Err(invalid_artifact(
                        "moving spatial predecessor differs from its exact geometry",
                    ));
                }
                WireOrigin::Continuous {
                    predecessor_spatial_state_sha256: previous.digest()?.to_string(),
                    predecessor_geometry_state_sha256: previous_geometry.digest()?.to_string(),
                    remesh_anchor: previous.remesh_anchor().cloned(),
                }
            }
            None => {
                return Err(invalid_artifact(
                    "moving spatial state requires its exact predecessor",
                ));
            }
        };
        Self::finish(context, geometry, fields, origin)
    }

    fn finish<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        geometry: &GeometryStateEnvelope,
        fields: Vec<WireStateField>,
        origin: WireOrigin,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireMovingSpatialState {
                schema: SPATIAL_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                spatial_dimension: context.mesh().dimension() as u64,
                reference: MovingSpatialReference::from_context(context)?,
                accepted: WireAccepted {
                    step: geometry.step(),
                    time_s: geometry.time_s(),
                },
                geometry_state_sha256: geometry.digest()?.to_string(),
                driver_snapshot_sha256: geometry.driver_snapshot().to_string(),
                origin,
                fields,
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode the sole current wire; exact dependency replay remains required.
    /// # Errors
    /// Rejects retired schemas, invalid origins, noncanonical data or exceeded bounds.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid moving spatial state: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }
    /// Deterministic canonical bytes.
    /// # Errors
    /// Preserves serialization failures.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| invalid_artifact(error.to_string()))
    }
    /// Schema-domain-separated identity.
    /// # Errors
    /// Preserves canonical encoding failures.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SPATIAL_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }
    /// Intrinsic dimension of the reference mesh.
    #[must_use]
    pub const fn spatial_dimension(&self) -> usize {
        self.wire.spatial_dimension as usize
    }
    /// Closed immediate origin.
    #[must_use]
    pub const fn origin(&self) -> SpatialStateOriginKind {
        self.wire.origin.kind()
    }
    /// Immediate same-topology state predecessor; absent at initial/remesh origins.
    #[must_use]
    pub fn predecessor(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Continuous {
                predecessor_spatial_state_sha256,
                ..
            } => Some(ArtifactDigest(predecessor_spatial_state_sha256.clone())),
            _ => None,
        }
    }
    /// Immediate geometry predecessor; absent at initial/remesh origins.
    #[must_use]
    pub fn predecessor_geometry_state(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Continuous {
                predecessor_geometry_state_sha256,
                ..
            } => Some(ArtifactDigest(predecessor_geometry_state_sha256.clone())),
            _ => None,
        }
    }
    /// Topology lineage derived from the state's complete origin evidence.
    #[must_use]
    pub const fn topology(&self) -> SpatialStateTopology {
        if self.remesh_anchor().is_some() {
            SpatialStateTopology::Remeshed
        } else {
            SpatialStateTopology::Reference
        }
    }
    /// Immutable representation-change evidence, retained by every continuation.
    #[must_use]
    pub const fn remesh_anchor(&self) -> Option<&RemeshStateAnchor> {
        self.wire.origin.anchor()
    }
    /// Accepted step ordinal.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.wire.accepted.step
    }

    /// Accepted coherent-SI time in seconds.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.wire.accepted.time_s
    }

    /// Exact current GeometryState artifact.
    #[must_use]
    pub fn geometry_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.geometry_state_sha256.clone())
    }

    /// Exact solid-displacement snapshot driving the GeometryState.
    #[must_use]
    pub fn geometry_driver_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.driver_snapshot_sha256.clone())
    }

    /// Exact reference Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Reference Model semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Exact ALE Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
    }

    /// Exact reference Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.geometry_sha256.clone())
    }

    /// Exact geometry-to-mesh correspondence artifact.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact immutable reference mesh artifact.
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
                        .expect("validated moving state Domain ULID"),
                    parse_id(&entry.field_ulid, "Field")
                        .expect("validated moving state Field ULID"),
                    ArtifactDigest(entry.snapshot_sha256.clone()),
                )
            })
            .collect()
    }

    /// Look up one exact Field snapshot without loading unrelated Fields.
    #[must_use]
    pub fn field_snapshot(&self, field: Id<kinds::Field>) -> Option<ArtifactDigest> {
        self.wire
            .fields
            .binary_search_by_key(&field.ulid(), |entry| {
                Ulid::from_str(&entry.field_ulid).expect("validated moving state Field ULID")
            })
            .ok()
            .map(|index| ArtifactDigest(self.wire.fields[index].snapshot_sha256.clone()))
    }

    /// Replay an initial or continuous state from exact immediate dependencies.
    /// # Errors
    /// Rejects changed state, geometry, driver or predecessor evidence.
    pub fn validate_against<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContext<'_, M>,
        geometry: &GeometryStateEnvelope,
        predecessor: Option<(&Self, &GeometryStateEnvelope)>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        driver_blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
    ) -> Result<(), Diagnostic> {
        self.require_equal(&Self::new(
            context,
            geometry,
            predecessor,
            snapshots,
            driver_blocks,
        )?)
    }
    fn require_equal(&self, expected: &Self) -> Result<(), Diagnostic> {
        if self == expected {
            Ok(())
        } else {
            Err(invalid_artifact(
                "moving spatial state differs from exact dependency replay",
            ))
        }
    }
    pub(crate) fn require_context<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContext<'_, M>,
    ) -> Result<(), Diagnostic> {
        if self.wire.reference != MovingSpatialReference::from_context(context)?
            || self.spatial_dimension() != context.mesh().dimension()
        {
            return Err(invalid_artifact(
                "moving spatial state differs from its exact context",
            ));
        }
        Ok(())
    }
}

fn state_inputs<'a, M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    geometry: &GeometryStateEnvelope,
    snapshots: &'a [FieldSnapshotEnvelopeV2],
) -> Result<(Vec<WireStateField>, &'a FieldSnapshotEnvelopeV2), Diagnostic> {
    validate_coordinate(geometry.step(), geometry.time_s())?;
    if geometry.time_s() != (geometry.step() as f64) * context.duration()
        || geometry.spatial_dimension() != context.mesh().dimension()
        || geometry.reference() != &MovingSpatialReference::from_context(context)?
    {
        return Err(invalid_artifact(
            "moving spatial geometry differs from its exact context or fixed-step time",
        ));
    }
    if snapshots.len() != context.represented_fields().len() {
        return Err(invalid_artifact(
            "moving spatial state omits or adds a represented physical Field",
        ));
    }
    let mut ordered = snapshots.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|snapshot| snapshot.field().ulid());
    if ordered
        .windows(2)
        .any(|pair| pair[0].field() == pair[1].field())
    {
        return Err(invalid_artifact(
            "moving spatial state contains duplicate Field snapshots",
        ));
    }
    for snapshot in &ordered {
        context.validate_snapshot(snapshot)?;
    }
    let driver = ordered
        .iter()
        .copied()
        .find(|snapshot| snapshot.field() == context.driver())
        .ok_or_else(|| invalid_artifact("moving spatial state omits its exact geometry driver"))?;
    if geometry.driver_snapshot() != driver.digest()? {
        return Err(invalid_artifact(
            "moving spatial state substitutes its geometry driver",
        ));
    }
    let fields = ordered
        .into_iter()
        .map(|snapshot| {
            Ok(WireStateField {
                support_domain_ulid: snapshot.support_domain().ulid().to_string(),
                field_ulid: snapshot.field().ulid().to_string(),
                snapshot_sha256: snapshot.digest()?.to_string(),
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    Ok((fields, driver))
}
