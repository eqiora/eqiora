//! Version-neutral replay projection for fixed-topology Geometry State artifacts.

use eqiora_core::Diagnostic;

use crate::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, GeometryStateEnvelopeV1,
    GeometryStateEnvelopeV3, ReplayableCanonicalModelArtifact, ValidatedMovingSpatialContextV2,
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::GeometryStateEnvelopeV1 {}
    impl Sealed for super::GeometryStateEnvelopeV3 {}
}

/// One canonical fixed-topology Geometry State generation replayable by the
/// dimension-neutral Spatial State v2 wire.
///
/// The contract is intentionally closed to Geometry State v1 (triangles) and
/// v3 (tetrahedra). Each generation retains its own schema and digest domain;
/// this projection shares only the exact lineage needed by downstream state
/// publication.
pub trait ReplayableFixedTopologyGeometryStateArtifact: sealed::Sealed {
    /// Complete generation-specific evidence needed to replay the geometry
    /// driver rather than trusting the coordinate payload.
    ///
    /// V1 retains its legacy lineage replay. V3 requires the exact normalized
    /// solid-displacement leaves named by its snapshot.
    type DriverReplayEvidence<'a>: Copy;

    /// Domain-separated identity in the concrete Geometry State generation.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    fn geometry_state_digest(&self) -> Result<ArtifactDigest, Diagnostic>;

    /// Exact reference Model artifact.
    fn model_artifact(&self) -> ArtifactDigest;

    /// Reference Model semantic revision.
    fn semantic_revision(&self) -> u64;

    /// Exact reference Geometry Identity artifact.
    fn reference_geometry_artifact(&self) -> ArtifactDigest;

    /// Exact reference geometry-to-mesh correspondence artifact.
    fn reference_correspondence_artifact(&self) -> ArtifactDigest;

    /// Exact immutable reference mesh artifact.
    fn reference_mesh_artifact(&self) -> ArtifactDigest;

    /// Exact Realization artifact.
    fn realization_artifact(&self) -> ArtifactDigest;

    /// Accepted step ordinal.
    fn step(&self) -> u64;

    /// Accepted coherent-SI time in seconds.
    fn time_s(&self) -> f64;

    /// Exact predecessor Geometry State, absent only at state zero.
    fn predecessor(&self) -> Option<ArtifactDigest>;

    /// Exact solid-displacement snapshot driving this state.
    fn solid_displacement_snapshot(&self) -> ArtifactDigest;

    /// Replay the concrete generation from exact dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for any dependency or derived-evidence drift.
    fn validate_fixed_topology_replay<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        evidence: Self::DriverReplayEvidence<'a>,
    ) -> Result<(), Diagnostic>;
}

macro_rules! impl_geometry_state_projection {
    ($state:ty) => {
        fn geometry_state_digest(&self) -> Result<ArtifactDigest, Diagnostic> {
            <$state>::digest(self)
        }

        fn model_artifact(&self) -> ArtifactDigest {
            <$state>::model_artifact(self)
        }

        fn semantic_revision(&self) -> u64 {
            <$state>::semantic_revision(self)
        }

        fn reference_geometry_artifact(&self) -> ArtifactDigest {
            <$state>::reference_geometry_artifact(self)
        }

        fn reference_correspondence_artifact(&self) -> ArtifactDigest {
            <$state>::reference_correspondence_artifact(self)
        }

        fn reference_mesh_artifact(&self) -> ArtifactDigest {
            <$state>::reference_mesh_artifact(self)
        }

        fn realization_artifact(&self) -> ArtifactDigest {
            <$state>::realization_artifact(self)
        }

        fn step(&self) -> u64 {
            <$state>::step(self)
        }

        fn time_s(&self) -> f64 {
            <$state>::time_s(self)
        }

        fn predecessor(&self) -> Option<ArtifactDigest> {
            <$state>::predecessor(self)
        }

        fn solid_displacement_snapshot(&self) -> ArtifactDigest {
            <$state>::solid_displacement_snapshot(self)
        }
    };
}

impl ReplayableFixedTopologyGeometryStateArtifact for GeometryStateEnvelopeV1 {
    type DriverReplayEvidence<'a> = ();

    impl_geometry_state_projection!(GeometryStateEnvelopeV1);

    fn validate_fixed_topology_replay<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        (): Self::DriverReplayEvidence<'a>,
    ) -> Result<(), Diagnostic> {
        self.validate_against(
            context.model(),
            context.geometry(),
            context.correspondence(),
            context.mesh(),
            context.realization(),
            predecessor,
            solid_displacement,
        )
    }
}

impl ReplayableFixedTopologyGeometryStateArtifact for GeometryStateEnvelopeV3 {
    type DriverReplayEvidence<'a> = &'a [DiscreteFieldEnvelopeV1];

    impl_geometry_state_projection!(GeometryStateEnvelopeV3);

    fn validate_fixed_topology_replay<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        evidence: Self::DriverReplayEvidence<'a>,
    ) -> Result<(), Diagnostic> {
        self.validate_against(context, predecessor, solid_displacement, evidence)
    }
}
