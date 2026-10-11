//! Tetrahedral fixed-topology geometry states with explicit dimension.

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    CellId, DiscreteFieldAssociation, FacetId, FixedTopologyGeometryAction,
    FixedTopologyGeometryState, P1HarmonicCoordinateRelation,
};
use serde::{Deserialize, Serialize};

use crate::geometry_state::{
    is_negative_geometry_zero, normalize_geometry_coordinates, normalize_geometry_zero,
    validate_geometry_coordinate_array,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2,
    MeshDecoderLimits, ReplayableCanonicalModelArtifact,
    ReplayableFixedTopologyAleRealizationArtifact, SimplicialMeshEnvelopeV1,
    ValidatedMovingSpatialContextV2, check_json_limits, invalid_artifact,
};

const GEOMETRY_STATE_SCHEMA: &str = "eqiora.geometry-state-envelope/v3";
const SPATIAL_DIMENSION: usize = 3;

/// One accepted tetrahedral coordinate state over immutable reference topology.
///
/// Version 3 is deliberately narrower than a dimension-generic wire format. It
/// admits only intrinsic three-dimensional affine tetrahedra and carries that
/// dimension explicitly. Connectivity, mesh velocity, and quality are never
/// authored by a caller: connectivity comes from the exact reference mesh,
/// velocity comes from consecutive accepted coordinates, and every quality
/// value is replayed with [`FixedTopologyGeometryState<3>`] and
/// [`FixedTopologyGeometryAction<3>`].
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryStateEnvelopeV3 {
    wire: WireGeometryStateEnvelopeV3,
}

impl GeometryStateEnvelopeV3 {
    /// Capture one accepted fixed-topology tetrahedral coordinate state.
    ///
    /// State zero is exactly `(step = 0, time_s = 0)` and is checked along the
    /// complete linear path from the reference coordinates. Every later state
    /// immediately follows an exact predecessor and derives its mesh velocity
    /// and path evidence from that positive-duration action.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale Model, Geometry, correspondence, mesh,
    /// Realization, predecessor, or displacement identities; a non-tetrahedral
    /// reference; invalid coordinates; non-adjacent accepted coordinates; or
    /// failed current/path quality replay.
    pub fn new<
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        step: u64,
        time_s: f64,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        solid_displacement_blocks: &[DiscreteFieldEnvelopeV1],
        mut current_coordinates_m: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let model_reference = context.model_reference();
        let realization_reference = context.realization().artifact_reference()?;
        let geometry = context.geometry();
        let correspondence = context.correspondence();
        let reference_mesh = context.mesh();
        if geometry.dimension() != SPATIAL_DIMENSION
            || reference_mesh.dimension() != SPATIAL_DIMENSION
        {
            return Err(invalid_artifact(
                "geometry-state/v3 admits only three-dimensional affine-tetrahedral references",
            ));
        }
        solid_displacement.validate_against_moving(context, solid_displacement_blocks.iter())?;

        if !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid_artifact(
                "geometry-state/v3 accepted time must be finite and nonnegative",
            ));
        }
        let time_s = normalize_geometry_zero(time_s);
        normalize_geometry_coordinates(&mut current_coordinates_m)?;
        validate_coordinate_derivation(
            context,
            solid_displacement,
            solid_displacement_blocks,
            &current_coordinates_m,
        )?;
        let current = FixedTopologyGeometryState::<SPATIAL_DIMENSION>::new(
            reference_mesh.mesh(),
            current_coordinates_m,
        )
        .map_err(|error| invalid_artifact(error.message()))?;
        let reference =
            FixedTopologyGeometryState::<SPATIAL_DIMENSION>::reference(reference_mesh.mesh())
                .map_err(|error| invalid_artifact(error.message()))?;
        let reference_lineage = WireReferenceLineageV3 {
            model_sha256: model_reference.artifact().to_string(),
            semantic_revision: model_reference.semantic_revision().get(),
            geometry_sha256: geometry.digest()?.to_string(),
            correspondence_sha256: correspondence.digest()?.to_string(),
            mesh_sha256: reference_mesh.digest()?.to_string(),
            realization_sha256: realization_reference.artifact().to_string(),
        };

        let (predecessor_sha256, path_origin, mut mesh_velocity_m_per_s, path_quality) =
            match predecessor {
                None if step == 0 && time_s == 0.0 => {
                    let action = FixedTopologyGeometryAction::<SPATIAL_DIMENSION>::new(
                        reference_mesh.mesh(),
                        &reference,
                        &current,
                        1.0,
                    )
                    .map_err(|error| invalid_artifact(error.message()))?;
                    (
                        None,
                        WirePathOriginV3::ReferenceMesh,
                        None,
                        action.minimum_path_signed_measure_scale(),
                    )
                }
                None => {
                    return Err(invalid_artifact(
                        "geometry-state/v3 zero is the only state without a predecessor",
                    ));
                }
                Some(previous) => {
                    previous.validate_reference_lineage(&reference_lineage, reference_mesh)?;
                    if step
                        != previous.step().checked_add(1).ok_or_else(|| {
                            invalid_artifact("geometry-state/v3 predecessor step overflows u64")
                        })?
                        || time_s <= previous.time_s()
                    {
                        return Err(invalid_artifact(
                            "geometry-state/v3 must immediately follow its predecessor in step and time",
                        ));
                    }
                    let previous_state = FixedTopologyGeometryState::<SPATIAL_DIMENSION>::new(
                        reference_mesh.mesh(),
                        previous.current_coordinates_m().to_vec(),
                    )
                    .map_err(|error| invalid_artifact(error.message()))?;
                    let action = FixedTopologyGeometryAction::<SPATIAL_DIMENSION>::new(
                        reference_mesh.mesh(),
                        &previous_state,
                        &current,
                        time_s - previous.time_s(),
                    )
                    .map_err(|error| invalid_artifact(error.message()))?;
                    (
                        Some(previous.digest()?.to_string()),
                        WirePathOriginV3::PreviousAcceptedState,
                        Some(action.vertex_velocities().to_vec()),
                        action.minimum_path_signed_measure_scale(),
                    )
                }
            };
        if let Some(velocity) = &mut mesh_velocity_m_per_s {
            normalize_geometry_coordinates(velocity)?;
        }

        let reference_quality = reference.quality_report();
        let current_quality = current.quality_report();
        let value = Self {
            wire: WireGeometryStateEnvelopeV3 {
                schema: GEOMETRY_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                spatial_dimension: SPATIAL_DIMENSION as u64,
                reference: reference_lineage,
                accepted: WireAcceptedCoordinateV3 { step, time_s },
                predecessor_geometry_state_sha256: predecessor_sha256,
                solid_displacement_snapshot_sha256: solid_displacement.digest()?.to_string(),
                coordinates: WireCoordinatesV3 {
                    scalar: WireScalarV3::F64,
                    unit: WireCoordinateUnitV3::Metre,
                    ordering: WireCoordinateOrderingV3::ReferenceMeshVertex,
                    values: current.coordinates().to_vec(),
                },
                action_evidence: WireGeometryActionEvidenceV3 {
                    path_origin,
                    mesh_velocity_unit: WireMeshVelocityUnitV3::MetrePerSecond,
                    mesh_velocity_m_per_s,
                    minimum_path_signed_measure_scale: path_quality,
                },
                quality_evidence: WireQualityEvidenceV3 {
                    reference: WireEndpointQualityV3 {
                        minimum_mean_ratio: reference_quality.minimum_mean_ratio(),
                        minimum_signed_measure_scale: reference_quality
                            .minimum_signed_measure_scale(),
                    },
                    current: WireEndpointQualityV3 {
                        minimum_mean_ratio: current_quality.minimum_mean_ratio(),
                        minimum_signed_measure_scale: current_quality
                            .minimum_signed_measure_scale(),
                    },
                },
            },
        };
        value.validate_local(MeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded wire data without trusting referenced artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, noncanonical, or
    /// non-three-dimensional wire data. Exact replay remains pending until
    /// [`Self::validate_against`] succeeds.
    pub fn from_json(bytes: &[u8], limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid geometry-state/v3 JSON: {error}"))
        })?;
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
            invalid_artifact(format!("cannot serialize geometry state v3: {error}"))
        })
    }

    /// Domain-separated identity of the complete state and replay evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GEOMETRY_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Explicit admitted spatial dimension, always three.
    #[must_use]
    pub const fn spatial_dimension(&self) -> usize {
        SPATIAL_DIMENSION
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

    /// Exact reference Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.geometry_sha256.clone())
    }

    /// Exact reference geometry-to-mesh correspondence artifact.
    #[must_use]
    pub fn reference_correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact immutable tetrahedral reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.mesh_sha256.clone())
    }

    /// Exact fixed-topology ALE Realization artifact.
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

    /// Exact predecessor GeometryState, absent only at state zero.
    #[must_use]
    pub fn predecessor(&self) -> Option<ArtifactDigest> {
        self.wire
            .predecessor_geometry_state_sha256
            .clone()
            .map(ArtifactDigest)
    }

    /// Exact accepted solid-displacement Field snapshot driving this state.
    #[must_use]
    pub fn solid_displacement_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.solid_displacement_snapshot_sha256.clone())
    }

    /// Absolute coherent-SI coordinates in immutable reference-vertex order.
    #[must_use]
    pub fn current_coordinates_m(&self) -> &[Vec<f64>] {
        &self.wire.coordinates.values
    }

    /// Mesh velocity derived from consecutive accepted coordinates.
    ///
    /// State zero has no positive time interval and therefore no velocity.
    #[must_use]
    pub fn mesh_velocity_m_per_s(&self) -> Option<&[Vec<f64>]> {
        self.wire.action_evidence.mesh_velocity_m_per_s.as_deref()
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

    /// Exact minimum signed measure scale over the complete admitted path.
    #[must_use]
    pub const fn minimum_path_signed_measure_scale(&self) -> f64 {
        self.wire.action_evidence.minimum_path_signed_measure_scale
    }

    /// Rebuild and compare the complete state from exact dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for dependency, predecessor, displacement-driver,
    /// coordinate, velocity, reference/current quality, or path-evidence drift.
    pub fn validate_against<
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        solid_displacement_blocks: &[DiscreteFieldEnvelopeV1],
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            context,
            self.step(),
            self.time_s(),
            predecessor,
            solid_displacement,
            solid_displacement_blocks,
            self.current_coordinates_m().to_vec(),
        )?;
        if self != &expected {
            return Err(invalid_artifact(
                "geometry-state/v3 differs from exact tetrahedral fixed-topology replay",
            ));
        }
        Ok(())
    }

    fn validate_reference_lineage(
        &self,
        reference: &WireReferenceLineageV3,
        reference_mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        if &self.wire.reference != reference {
            return Err(invalid_artifact(
                "geometry-state/v3 predecessor has stale reference lineage",
            ));
        }
        let state = FixedTopologyGeometryState::<SPATIAL_DIMENSION>::new(
            reference_mesh.mesh(),
            self.current_coordinates_m().to_vec(),
        )
        .map_err(|error| invalid_artifact(error.message()))?;
        let quality = state.quality_report();
        if quality.minimum_mean_ratio().to_bits() != self.current_minimum_mean_ratio().to_bits()
            || quality.minimum_signed_measure_scale().to_bits()
                != self.current_minimum_signed_measure_scale().to_bits()
        {
            return Err(invalid_artifact(
                "geometry-state/v3 predecessor quality differs from reference-topology replay",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: MeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != GEOMETRY_STATE_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.spatial_dimension != SPATIAL_DIMENSION as u64
            || self.wire.coordinates.scalar != WireScalarV3::F64
            || self.wire.coordinates.unit != WireCoordinateUnitV3::Metre
            || self.wire.coordinates.ordering != WireCoordinateOrderingV3::ReferenceMeshVertex
            || self.wire.action_evidence.mesh_velocity_unit
                != WireMeshVelocityUnitV3::MetrePerSecond
        {
            return Err(invalid_artifact(
                "unsupported geometry-state/v3 schema, dimension, encoding, scalar, unit, or ordering",
            ));
        }
        for digest in [
            &self.wire.reference.model_sha256,
            &self.wire.reference.geometry_sha256,
            &self.wire.reference.correspondence_sha256,
            &self.wire.reference.mesh_sha256,
            &self.wire.reference.realization_sha256,
            &self.wire.solid_displacement_snapshot_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if let Some(predecessor) = &self.wire.predecessor_geometry_state_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        if !self.wire.accepted.time_s.is_finite()
            || self.wire.accepted.time_s < 0.0
            || is_negative_geometry_zero(self.wire.accepted.time_s)
        {
            return Err(invalid_artifact(
                "geometry-state/v3 accepted time must be finite, nonnegative, and canonical",
            ));
        }

        let initial = self.wire.predecessor_geometry_state_sha256.is_none();
        if initial
            != (self.wire.accepted.step == 0
                && self.wire.accepted.time_s == 0.0
                && self.wire.action_evidence.path_origin == WirePathOriginV3::ReferenceMesh
                && self.wire.action_evidence.mesh_velocity_m_per_s.is_none())
            || (!initial
                && (self.wire.accepted.step == 0
                    || self.wire.accepted.time_s == 0.0
                    || self.wire.action_evidence.path_origin
                        != WirePathOriginV3::PreviousAcceptedState
                    || self.wire.action_evidence.mesh_velocity_m_per_s.is_none()))
        {
            return Err(invalid_artifact(
                "geometry-state/v3 predecessor, accepted coordinate, path origin, and velocity roles are inconsistent",
            ));
        }

        let coordinate_values = validate_geometry_coordinate_array(
            "geometry-state/v3 coordinates",
            &self.wire.coordinates.values,
            limits,
        )?;
        if self.wire.coordinates.values[0].len() != SPATIAL_DIMENSION {
            return Err(invalid_artifact(
                "geometry-state/v3 coordinates must be three-dimensional",
            ));
        }
        if let Some(velocity) = &self.wire.action_evidence.mesh_velocity_m_per_s {
            let velocity_values = validate_geometry_coordinate_array(
                "geometry-state/v3 mesh velocity",
                velocity,
                limits,
            )?;
            if coordinate_values
                .checked_add(velocity_values)
                .is_none_or(|total| total > limits.max_mesh_coordinate_values)
            {
                return Err(invalid_artifact(
                    "geometry-state/v3 coordinate and velocity aggregate exceeds the decoder limit",
                ));
            }
            if velocity.len() != self.wire.coordinates.values.len()
                || velocity
                    .iter()
                    .zip(&self.wire.coordinates.values)
                    .any(|(left, right)| left.len() != right.len())
            {
                return Err(invalid_artifact(
                    "geometry-state/v3 mesh velocity shape differs from current coordinates",
                ));
            }
        }

        for quality in [
            &self.wire.quality_evidence.reference,
            &self.wire.quality_evidence.current,
        ] {
            if !quality.minimum_mean_ratio.is_finite()
                || quality.minimum_mean_ratio <= 0.0
                || quality.minimum_mean_ratio > 1.0
                || !quality.minimum_signed_measure_scale.is_finite()
                || quality.minimum_signed_measure_scale <= 0.0
                || is_negative_geometry_zero(quality.minimum_mean_ratio)
                || is_negative_geometry_zero(quality.minimum_signed_measure_scale)
            {
                return Err(invalid_artifact(
                    "geometry-state/v3 endpoint quality must be finite, canonical, and strictly positive",
                ));
            }
        }
        let path_quality = self.wire.action_evidence.minimum_path_signed_measure_scale;
        if !path_quality.is_finite()
            || path_quality <= 0.0
            || is_negative_geometry_zero(path_quality)
        {
            return Err(invalid_artifact(
                "geometry-state/v3 path quality must be finite, canonical, and strictly positive",
            ));
        }
        Ok(())
    }
}

fn validate_coordinate_derivation<
    M: ReplayableCanonicalModelArtifact,
    R: ReplayableFixedTopologyAleRealizationArtifact,
>(
    context: &ValidatedMovingSpatialContextV2<'_, M, R>,
    solid_displacement: &FieldSnapshotEnvelopeV2,
    solid_displacement_blocks: &[DiscreteFieldEnvelopeV1],
    current_coordinates_m: &[Vec<f64>],
) -> Result<(), Diagnostic> {
    let requirements = context.realization().ale_requirements()?;
    let plan = context.realization().ale_plan()?;
    let motion = plan.mesh_motion();
    if solid_displacement.field() != requirements.solid_displacement()
        || solid_displacement.support_domain() != requirements.solid_domain()
        || motion.fluid_domain() != requirements.fluid_domain()
        || motion.solid_domain() != requirements.solid_domain()
        || motion.solid_displacement() != requirements.solid_displacement()
    {
        return Err(invalid_artifact(
            "geometry-state/v3 driver or material roles differ from the exact ALE Realization",
        ));
    }

    let vertex_block = solid_displacement_blocks
        .iter()
        .find(|block| block.association() == DiscreteFieldAssociation::Vertex)
        .ok_or_else(|| {
            invalid_artifact(
                "geometry-state/v3 driver snapshot has no complete vertex coefficient block",
            )
        })?;
    if solid_displacement_blocks.len() != 1
        || vertex_block
            .component_shape()
            .component_count()
            .map_err(|error| invalid_artifact(error.message()))?
            != SPATIAL_DIMENSION
    {
        return Err(invalid_artifact(
            "geometry-state/v3 driver must be one P1 spatial-vector vertex block",
        ));
    }
    let solid_values = vertex_block
        .values()
        .as_chunks::<SPATIAL_DIMENSION>()
        .0
        .iter()
        .map(|values| [values[0], values[1], values[2]])
        .collect::<Vec<_>>();

    let interface = context.correspondence().derive_conserving_interface(
        context.geometry(),
        context.model(),
        context.mesh(),
        motion.interface(),
    )?;
    let fluid_cells = context
        .correspondence()
        .body_cells(motion.fluid_domain())
        .ok_or_else(|| invalid_artifact("ALE fluid Domain has no exact mesh-cell inventory"))?
        .into_iter()
        .map(CellId::new)
        .collect();
    let solid_cells = context
        .correspondence()
        .body_cells(motion.solid_domain())
        .ok_or_else(|| invalid_artifact("ALE solid Domain has no exact mesh-cell inventory"))?
        .into_iter()
        .map(CellId::new)
        .collect();
    let interface_facets = interface
        .facet_indices()
        .iter()
        .copied()
        .map(FacetId::new)
        .collect();
    let solver = motion.solver();
    let relation = P1HarmonicCoordinateRelation::<SPATIAL_DIMENSION>::new(
        context.mesh().mesh(),
        fluid_cells,
        solid_cells,
        interface_facets,
    )
    .map_err(|error| invalid_artifact(error.message()))?;
    let driver_residual_targets = relation
        .driver_rhs_norms()
        .iter()
        .map(|&right_hand_side_norm| {
            solver
                .residual_target(right_hand_side_norm)
                .map_err(|error| invalid_artifact(error.message()))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    relation
        .validate_current_coordinates(
            &solid_values,
            current_coordinates_m,
            &driver_residual_targets,
        )
        .map_err(|error| invalid_artifact(error.message()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryStateEnvelopeV3 {
    schema: String,
    encoding: String,
    spatial_dimension: u64,
    reference: WireReferenceLineageV3,
    accepted: WireAcceptedCoordinateV3,
    predecessor_geometry_state_sha256: Option<String>,
    solid_displacement_snapshot_sha256: String,
    coordinates: WireCoordinatesV3,
    action_evidence: WireGeometryActionEvidenceV3,
    quality_evidence: WireQualityEvidenceV3,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReferenceLineageV3 {
    model_sha256: String,
    semantic_revision: u64,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    realization_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedCoordinateV3 {
    step: u64,
    time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCoordinatesV3 {
    scalar: WireScalarV3,
    unit: WireCoordinateUnitV3,
    ordering: WireCoordinateOrderingV3,
    values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryActionEvidenceV3 {
    path_origin: WirePathOriginV3,
    mesh_velocity_unit: WireMeshVelocityUnitV3,
    mesh_velocity_m_per_s: Option<Vec<Vec<f64>>>,
    minimum_path_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQualityEvidenceV3 {
    reference: WireEndpointQualityV3,
    current: WireEndpointQualityV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEndpointQualityV3 {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarV3 {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateUnitV3 {
    Metre,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMeshVelocityUnitV3 {
    MetrePerSecond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateOrderingV3 {
    ReferenceMeshVertex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePathOriginV3 {
    ReferenceMesh,
    PreviousAcceptedState,
}
