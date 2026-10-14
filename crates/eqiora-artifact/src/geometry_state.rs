//! Current dimension-explicit coordinate states with closed initial, continuous and remesh origins.

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2,
    MeshDecoderLimits, ReplayableCanonicalModelArtifact, ValidatedMovingSpatialContext,
    check_json_limits, invalid_artifact,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyGeometryState;

mod driver;
mod remesh;
pub(crate) mod validation;
mod wire;

use crate::spatial_state::context::MovingSpatialReference;
use driver::validate_coordinate_derivation;
pub use remesh::ValidatedRemeshGeometrySource;
use validation::{
    normalize_geometry_coordinates, normalize_geometry_zero, replay_geometry_action,
    validate_successor,
};
use wire::*;

const GEOMETRY_STATE_SCHEMA: &str = "eqiora.geometry-state-envelope/v4";

/// Meaning of an accepted geometry's immediate origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryStateOriginKind {
    /// Initial coordinates replayed along the path from the reference mesh.
    Initial,
    /// Positive-duration advance from the exact same-topology predecessor.
    Continuous,
    /// Same-time representation change from an accepted source topology.
    Remesh,
}

/// One accepted coordinate state over an exact immutable reference simplex mesh.
///
/// Dimensions two and three share the same wire and validation. Fixed-topology
/// states replay their graph-selected harmonic action from exact driver leaves.
/// A remesh origin retains its source and semantic association; overlap and
/// transfer receipts separately prove the representation change. No connectivity
/// or independently authored mesh velocity is admitted.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryStateEnvelope {
    wire: WireGeometryState,
}

impl GeometryStateEnvelope {
    pub(crate) const fn reference(&self) -> &MovingSpatialReference {
        &self.wire.reference
    }
    /// Capture an initial state or immediate positive-duration continuation.
    ///
    /// # Errors
    /// Returns EQ0901 for stale dependencies, invalid driver leaves or coordinates,
    /// non-adjacent step/time, or failed endpoint/path quality replay.
    pub fn new<'a, M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        step: u64,
        time_s: f64,
        predecessor: Option<&Self>,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
        driver_blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
        mut coordinates: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid_artifact(
                "geometry-state time must be finite and nonnegative",
            ));
        }
        let time_s = normalize_geometry_zero(time_s);
        normalize_geometry_coordinates(&mut coordinates)?;
        let driver_blocks = driver_blocks.into_iter().collect::<Vec<_>>();
        driver_snapshot.validate_against_moving(context, driver_blocks.iter().copied())?;
        match context.mesh().dimension() {
            2 => Self::fixed_topology::<2, M>(
                context,
                step,
                time_s,
                predecessor,
                driver_snapshot,
                &driver_blocks,
                coordinates,
            ),
            3 => Self::fixed_topology::<3, M>(
                context,
                step,
                time_s,
                predecessor,
                driver_snapshot,
                &driver_blocks,
                coordinates,
            ),
            _ => Err(invalid_artifact(
                "geometry-state requires dimension two or three",
            )),
        }
    }

    fn fixed_topology<const D: usize, M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        step: u64,
        time_s: f64,
        predecessor: Option<&Self>,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
        driver_blocks: &[&DiscreteFieldEnvelopeV1],
        coordinates: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        validate_coordinate_derivation::<D, M>(
            context,
            driver_snapshot,
            driver_blocks,
            &coordinates,
        )?;
        let current = FixedTopologyGeometryState::<D>::new(context.mesh().mesh(), coordinates)
            .map_err(|error| invalid_artifact(error.message()))?;
        let reference = MovingSpatialReference::from_context(context)?;
        let previous = match predecessor {
            None if step == 0 && time_s == 0.0 => None,
            None => {
                return Err(invalid_artifact(
                    "geometry-state zero is the only initial state",
                ));
            }
            Some(previous) => {
                previous.validate_reference::<D>(&reference, context.mesh().mesh())?;
                validate_successor(step, time_s, previous.step(), previous.time_s())?;
                Some((previous.current_coordinates_m(), time_s - previous.time_s()))
            }
        };
        let action = replay_geometry_action(context.mesh().mesh(), &current, previous)?;
        let origin = match predecessor {
            None => WireOrigin::Initial {
                minimum_path_signed_measure_scale: action.minimum_path_signed_measure_scale,
            },
            Some(previous) => WireOrigin::Continuous {
                predecessor_geometry_state_sha256: previous.digest()?.to_string(),
                mesh_velocity_unit: WireMeshVelocityUnit::MetrePerSecond,
                mesh_velocity_m_per_s: action.velocity.expect("a continuous action has velocity"),
                minimum_path_signed_measure_scale: action.minimum_path_signed_measure_scale,
            },
        };
        Self::finish(context, step, time_s, driver_snapshot, current, origin)
    }

    fn finish<const D: usize, M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        step: u64,
        time_s: f64,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
        current: FixedTopologyGeometryState<D>,
        origin: WireOrigin,
    ) -> Result<Self, Diagnostic> {
        let reference = FixedTopologyGeometryState::<D>::reference(context.mesh().mesh())
            .map_err(|error| invalid_artifact(error.message()))?;
        let value = Self {
            wire: WireGeometryState {
                schema: GEOMETRY_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                spatial_dimension: D as u64,
                reference: MovingSpatialReference::from_context(context)?,
                accepted: WireAccepted { step, time_s },
                driver_snapshot_sha256: driver_snapshot.digest()?.to_string(),
                coordinates: WireCoordinates {
                    scalar: WireScalar::F64,
                    unit: WireCoordinateUnit::Metre,
                    ordering: WireCoordinateOrdering::ReferenceMeshVertex,
                    values: current.coordinates().to_vec(),
                },
                origin,
                quality_evidence: WireQuality {
                    reference: WireEndpointQuality::from_report(reference.quality_report()),
                    current: WireEndpointQuality::from_report(current.quality_report()),
                },
            },
        };
        value.validate_local(MeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode the sole current wire; dependency replay remains required.
    ///
    /// # Errors
    /// Rejects retired schemas, unknown fields, malformed origins or exceeded budgets.
    pub fn from_json(bytes: &[u8], limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid geometry-state JSON: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical bytes.
    ///
    /// # Errors
    /// Returns EQ0901 if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| invalid_artifact(error.to_string()))
    }

    /// Schema-domain-separated identity.
    ///
    /// # Errors
    /// Preserves canonical encoding failures.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GEOMETRY_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Intrinsic reference dimension.
    #[must_use]
    pub const fn spatial_dimension(&self) -> usize {
        self.wire.spatial_dimension as usize
    }

    /// Closed origin of these coordinates.
    #[must_use]
    pub const fn origin(&self) -> GeometryStateOriginKind {
        match self.wire.origin {
            WireOrigin::Initial { .. } => GeometryStateOriginKind::Initial,
            WireOrigin::Continuous { .. } => GeometryStateOriginKind::Continuous,
            WireOrigin::Remesh { .. } => GeometryStateOriginKind::Remesh,
        }
    }

    /// Reference Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Reference Model semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Reference Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.geometry_sha256.clone())
    }

    /// Reference geometry-to-mesh correspondence artifact.
    #[must_use]
    pub fn reference_correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Immutable reference simplex mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.mesh_sha256.clone())
    }

    /// Fixed-topology ALE Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
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

    /// Same-topology predecessor; absent at an initial or remesh origin.
    #[must_use]
    pub fn predecessor(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOrigin::Continuous {
                predecessor_geometry_state_sha256,
                ..
            } => Some(ArtifactDigest(predecessor_geometry_state_sha256.clone())),
            _ => None,
        }
    }

    /// Exact Field snapshot selected as the geometry driver.
    #[must_use]
    pub fn driver_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.driver_snapshot_sha256.clone())
    }

    /// Absolute coherent-SI coordinates in immutable reference-vertex order.
    #[must_use]
    pub fn current_coordinates_m(&self) -> &[Vec<f64>] {
        &self.wire.coordinates.values
    }

    /// Derived velocity; absent at initial and zero-duration remesh origins.
    #[must_use]
    pub fn mesh_velocity_m_per_s(&self) -> Option<&[Vec<f64>]> {
        match &self.wire.origin {
            WireOrigin::Continuous {
                mesh_velocity_m_per_s,
                ..
            } => Some(mesh_velocity_m_per_s),
            _ => None,
        }
    }

    /// Replayed minimum mean ratio of the immutable reference mesh.
    #[must_use]
    pub const fn reference_minimum_mean_ratio(&self) -> f64 {
        self.wire.quality_evidence.reference.minimum_mean_ratio
    }

    /// Replayed minimum signed measure scale of the immutable reference mesh.
    #[must_use]
    pub const fn reference_minimum_signed_measure_scale(&self) -> f64 {
        self.wire
            .quality_evidence
            .reference
            .minimum_signed_measure_scale
    }

    /// Replayed minimum mean ratio of the accepted current coordinates.
    #[must_use]
    pub const fn current_minimum_mean_ratio(&self) -> f64 {
        self.wire.quality_evidence.current.minimum_mean_ratio
    }

    /// Replayed minimum signed measure scale of the accepted current coordinates.
    #[must_use]
    pub const fn current_minimum_signed_measure_scale(&self) -> f64 {
        self.wire
            .quality_evidence
            .current
            .minimum_signed_measure_scale
    }

    /// Replayed path quality; a zero-duration remesh has no same-topology path.
    #[must_use]
    pub const fn minimum_path_signed_measure_scale(&self) -> Option<f64> {
        match self.wire.origin {
            WireOrigin::Initial {
                minimum_path_signed_measure_scale,
            }
            | WireOrigin::Continuous {
                minimum_path_signed_measure_scale,
                ..
            } => Some(minimum_path_signed_measure_scale),
            WireOrigin::Remesh { .. } => None,
        }
    }

    /// Replay an initial or continuous state from its exact dependencies.
    ///
    /// # Errors
    /// Rejects dependency, driver, predecessor or derived-evidence drift.
    pub fn validate_against<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContext<'_, M>,
        predecessor: Option<&Self>,
        driver_snapshot: &FieldSnapshotEnvelopeV2,
        driver_blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            context,
            self.step(),
            self.time_s(),
            predecessor,
            driver_snapshot,
            driver_blocks,
            self.current_coordinates_m().to_vec(),
        )?;
        self.require_equal(&expected)
    }

    fn require_equal(&self, expected: &Self) -> Result<(), Diagnostic> {
        if self != expected {
            return Err(invalid_artifact(
                "geometry-state differs from exact dependency replay",
            ));
        }
        Ok(())
    }

    fn validate_reference<const D: usize>(
        &self,
        reference: &MovingSpatialReference,
        mesh: &eqiora_meshing::SimplicialMesh,
    ) -> Result<(), Diagnostic> {
        if &self.wire.reference != reference || self.spatial_dimension() != D {
            return Err(invalid_artifact(
                "geometry-state predecessor has stale reference lineage or dimension",
            ));
        }
        let state =
            FixedTopologyGeometryState::<D>::new(mesh, self.current_coordinates_m().to_vec())
                .map_err(|error| invalid_artifact(error.message()))?;
        let quality = state.quality_report();
        if quality.minimum_mean_ratio().to_bits() != self.current_minimum_mean_ratio().to_bits()
            || quality.minimum_signed_measure_scale().to_bits()
                != self.current_minimum_signed_measure_scale().to_bits()
        {
            return Err(invalid_artifact(
                "geometry-state predecessor quality differs from reference-topology replay",
            ));
        }
        Ok(())
    }
}
