//! Exact source validation and a distinct zero-duration representation-change origin.

use super::*;
use crate::{GeometryRevisionAssociationEnvelopeV1, SpatialStateEnvelopeV2};

/// Accepted source state and exact semantic association at one remesh boundary.
#[derive(Debug)]
pub struct ValidatedRemeshGeometrySource<'a, M: ReplayableCanonicalModelArtifact> {
    context: &'a ValidatedMovingSpatialContextV2<'a, M>,
    state: &'a SpatialStateEnvelopeV2,
    geometry_state: &'a GeometryStateEnvelope,
    association: &'a GeometryRevisionAssociationEnvelopeV1,
}

impl<'a, M: ReplayableCanonicalModelArtifact> ValidatedRemeshGeometrySource<'a, M> {
    /// Replay the source moving state, including its exact geometry driver leaves.
    ///
    /// # Errors
    /// Rejects stale dependencies or invalid source replay.
    pub fn new<'b>(
        context: &'a ValidatedMovingSpatialContextV2<'a, M>,
        state: &'a SpatialStateEnvelopeV2,
        geometry_state: &'a GeometryStateEnvelope,
        predecessor_geometry_state: Option<&GeometryStateEnvelope>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        driver_blocks: impl IntoIterator<Item = &'b DiscreteFieldEnvelopeV1>,
        association: &'a GeometryRevisionAssociationEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        state.validate_against(
            context,
            geometry_state,
            predecessor_geometry_state,
            snapshots,
            driver_blocks,
        )?;
        Ok(Self {
            context,
            state,
            geometry_state,
            association,
        })
    }

    pub(crate) const fn context(&self) -> &'a ValidatedMovingSpatialContextV2<'a, M> {
        self.context
    }
    /// Exact accepted source state.
    #[must_use]
    pub const fn state(&self) -> &SpatialStateEnvelopeV2 {
        self.state
    }
    /// Exact accepted source geometry.
    #[must_use]
    pub const fn geometry_state(&self) -> &GeometryStateEnvelope {
        self.geometry_state
    }
    /// Exact semantic geometry-revision association.
    #[must_use]
    pub const fn association(&self) -> &GeometryRevisionAssociationEnvelopeV1 {
        self.association
    }
}

impl GeometryStateEnvelope {
    /// Construct the same-time origin of a distinct two-dimensional mesh revision.
    ///
    /// This verifies semantic retention and target coordinate quality. Material and
    /// current-spatial overlap and transfer correctness belong to their receipts.
    /// No same-topology path or velocity is fabricated for a representation change.
    ///
    /// # Errors
    /// Rejects changed Model/time, identical meshes, incomplete association,
    /// wrong driver lineage or unsupported dimensions.
    pub fn remesh<M: ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target: &ValidatedMovingSpatialContextV2<'_, M>,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
        mut coordinates: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        if source.state.model_artifact() != *target.model_reference().artifact()
            || source.state.semantic_revision()
                != target.model_reference().semantic_revision().get()
            || source.context.mesh().digest()? == target.mesh().digest()?
            || source.context.mesh().dimension() != 2
            || target.mesh().dimension() != 2
        {
            return Err(invalid_artifact(
                "remesh geometry-state requires the same Model and distinct two-dimensional meshes",
            ));
        }
        source
            .association
            .validate_against(
                source.context.model(),
                source.context.geometry(),
                source.context.correspondence(),
                source.context.mesh(),
                target.model(),
                target.geometry(),
                target.correspondence(),
                target.mesh(),
            )
            .map_err(|error| {
                invalid_artifact(format!(
                    "remesh semantic geometry association failed: {error}"
                ))
            })?;
        target.validate_snapshot(driver_snapshot)?;
        if driver_snapshot.field() != target.driver() {
            return Err(invalid_artifact(
                "remesh geometry-state snapshot is not the exact geometry driver",
            ));
        }
        normalize_geometry_coordinates(&mut coordinates)?;
        let current = FixedTopologyGeometryState::<2>::new(target.mesh().mesh(), coordinates)
            .map_err(|error| invalid_artifact(error.message()))?;
        Self::finish(
            target,
            source.state.step(),
            source.state.time_s(),
            driver_snapshot,
            current,
            WireOrigin::Remesh {
                source_spatial_state_sha256: source.state.digest()?.to_string(),
                source_geometry_state_sha256: source.geometry_state.digest()?.to_string(),
                semantic_association_sha256: source.association.digest()?.to_string(),
            },
        )
    }

    /// Replay a remesh origin from its exact dependencies.
    ///
    /// # Errors
    /// Rejects changed source, target or derived geometry evidence.
    pub fn validate_against_remesh<M: ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target: &ValidatedMovingSpatialContextV2<'_, M>,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        self.require_equal(&Self::remesh(
            source,
            target,
            driver_snapshot,
            self.current_coordinates_m().to_vec(),
        )?)
    }

    /// Exact source state of a remesh origin; absent for fixed-topology origins.
    #[must_use]
    pub fn remesh_source_spatial_state(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Remesh {
                source_spatial_state_sha256,
                ..
            } => Some(ArtifactDigest(source_spatial_state_sha256.clone())),
            _ => None,
        }
    }
    /// Exact source geometry at a remesh origin.
    #[must_use]
    pub fn remesh_source_geometry_state(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Remesh {
                source_geometry_state_sha256,
                ..
            } => Some(ArtifactDigest(source_geometry_state_sha256.clone())),
            _ => None,
        }
    }
    /// Semantic association at a remesh origin.
    #[must_use]
    pub fn remesh_semantic_association(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Remesh {
                semantic_association_sha256,
                ..
            } => Some(ArtifactDigest(semantic_association_sha256.clone())),
            _ => None,
        }
    }
}
