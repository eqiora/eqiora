//! Connectivity-free geometry states over one immutable reference mesh.

use eqiora_core::ValueFrame;
use eqiora_core::{Diagnostic, DimExponents};
use eqiora_meshing::FixedTopologyGeometryState2d;

pub(crate) mod validation;
use serde::{Deserialize, Serialize};
use validation::{
    is_negative_geometry_zero, normalize_geometry_coordinates, normalize_geometry_zero,
    replay_geometry_action, validate_coordinate_payload, validate_endpoint_quality,
    validate_path_quality, validate_successor,
};

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalRealizationArtifact, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, MeshDecoderLimits,
    ReplayableCanonicalModelArtifact, SimplicialMeshEnvelopeV1, check_json_limits,
    invalid_artifact,
};

const GEOMETRY_STATE_SCHEMA: &str = "eqiora.geometry-state-envelope/v1";

/// One accepted coordinate state over an immutable reference simplex mesh.
///
/// The wire deliberately contains no cells, facets, or other topology. Vertex
/// order comes exclusively from the exact reference mesh digest. Current
/// quality and path-orientation evidence are replayed with that reference
/// connectivity; mesh velocity is derived from consecutive accepted states
/// and cannot be supplied to [`Self::new`]. Version 1 is the bounded 2D
/// fixed-topology ALE contract from RFC 0064.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryStateEnvelopeV1 {
    wire: WireGeometryStateEnvelopeV1,
}

impl GeometryStateEnvelopeV1 {
    /// Capture one accepted 2D fixed-topology coordinate state.
    ///
    /// The initial state is exactly `(step = 0, time_s = 0)` and has no
    /// predecessor or mesh velocity. Every later state must immediately
    /// follow its predecessor in step and time. The solid-displacement
    /// snapshot must share the complete reference lineage and have coherent-SI
    /// length-valued spatial-vector type.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale dependencies, invalid step/time lineage,
    /// wrong driver type, changed vertex inventory, non-finite coordinates,
    /// an inverted or low-quality current mesh, or an orientation loss along
    /// the complete linear path from the predecessor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &impl ReplayableCanonicalModelArtifact,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        reference_mesh: &SimplicialMeshEnvelopeV1,
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
        step: u64,
        time_s: f64,
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        mut current_coordinates_m: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let model_reference = model.artifact_reference()?;
        let realization_reference = realization.artifact_reference()?;
        if realization_reference.model_artifact() != model_reference.artifact()
            || realization_reference.semantic_revision() != model_reference.semantic_revision()
        {
            return Err(invalid_artifact(
                "geometry-state Realization does not select the exact reference Model",
            ));
        }
        geometry.validate_against(model)?;
        correspondence.validate_against(geometry, model, reference_mesh)?;
        if reference_mesh.dimension() != 2 {
            return Err(invalid_artifact(
                "geometry-state/v1 admits only two-dimensional affine-simplex reference meshes",
            ));
        }
        validate_geometry_driver(
            solid_displacement,
            model_reference.artifact(),
            realization_reference.artifact(),
            &geometry.digest()?,
            &correspondence.digest()?,
            &reference_mesh.digest()?,
            reference_mesh.dimension(),
        )?;

        if !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid_artifact(
                "geometry-state accepted time must be finite and nonnegative",
            ));
        }
        let time_s = normalize_geometry_zero(time_s);
        let reference = WireReferenceLineageV1 {
            model_sha256: model_reference.artifact().to_string(),
            semantic_revision: model_reference.semantic_revision().get(),
            geometry_sha256: geometry.digest()?.to_string(),
            correspondence_sha256: correspondence.digest()?.to_string(),
            mesh_sha256: reference_mesh.digest()?.to_string(),
            realization_sha256: realization_reference.artifact().to_string(),
        };
        normalize_geometry_coordinates(&mut current_coordinates_m)?;
        let current_state =
            FixedTopologyGeometryState2d::new(reference_mesh.mesh(), current_coordinates_m.clone())
                .map_err(|error| invalid_artifact(error.message()))?;
        let (predecessor_sha256, path_origin, previous) = match predecessor {
            None if step == 0 && time_s == 0.0 => (None, WirePathOriginV1::ReferenceMesh, None),
            None => {
                return Err(invalid_artifact(
                    "geometry-state zero is the only state without a predecessor",
                ));
            }
            Some(previous) => {
                previous.validate_reference_lineage(&reference, reference_mesh)?;
                validate_successor(step, time_s, previous.step(), previous.time_s())?;
                (
                    Some(previous.digest()?.to_string()),
                    WirePathOriginV1::PreviousAcceptedState,
                    Some((previous.current_coordinates_m(), time_s - previous.time_s())),
                )
            }
        };
        let action = replay_geometry_action(reference_mesh.mesh(), &current_state, previous)?;
        let quality = current_state.quality_report();
        let value = Self {
            wire: WireGeometryStateEnvelopeV1 {
                schema: GEOMETRY_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference,
                accepted: WireAcceptedCoordinateV1 { step, time_s },
                predecessor_geometry_state_sha256: predecessor_sha256,
                solid_displacement_snapshot_sha256: solid_displacement.digest()?.to_string(),
                coordinates: WireCoordinatesV1 {
                    scalar: WireScalarV1::F64,
                    unit: WireCoordinateUnitV1::Metre,
                    ordering: WireCoordinateOrderingV1::ReferenceMeshVertex,
                    values: current_coordinates_m,
                },
                action_evidence: WireGeometryActionEvidenceV1 {
                    path_origin,
                    mesh_velocity_unit: WireMeshVelocityUnitV1::MetrePerSecond,
                    mesh_velocity_m_per_s: action.velocity,
                    minimum_path_signed_measure_scale: action.minimum_path_signed_measure_scale,
                },
                quality_evidence: WireQualityEvidenceV1 {
                    minimum_mean_ratio: quality.minimum_mean_ratio(),
                    minimum_signed_measure_scale: quality.minimum_signed_measure_scale(),
                },
            },
        };
        value.validate_local(MeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded wire data without trusting referenced artifacts.
    ///
    /// Exact topology, quality, path, velocity, and dependency replay remains
    /// pending until [`Self::validate_against`] succeeds.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, non-finite, or
    /// noncanonical wire data, including any topology-bearing unknown field.
    pub fn from_json(bytes: &[u8], limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid geometry-state JSON: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize geometry state: {error}")))
    }

    /// Domain-separated identity of this coordinate state and its evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GEOMETRY_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
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

    /// Exact immutable reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.mesh_sha256.clone())
    }

    /// Exact ALE Realization artifact.
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

    /// Exact predecessor GeometryState, absent only at step zero.
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

    /// Absolute coherent-SI coordinates in immutable reference vertex order.
    #[must_use]
    pub fn current_coordinates_m(&self) -> &[Vec<f64>] {
        &self.wire.coordinates.values
    }

    /// Derived mesh velocity in immutable reference vertex order.
    ///
    /// The initial state has no time interval and therefore no velocity.
    #[must_use]
    pub fn mesh_velocity_m_per_s(&self) -> Option<&[Vec<f64>]> {
        self.wire.action_evidence.mesh_velocity_m_per_s.as_deref()
    }

    /// Recomputed minimum current-cell mean-ratio quality.
    #[must_use]
    pub const fn minimum_mean_ratio(&self) -> f64 {
        self.wire.quality_evidence.minimum_mean_ratio
    }

    /// Recomputed minimum current-cell signed measure scale.
    #[must_use]
    pub const fn minimum_signed_measure_scale(&self) -> f64 {
        self.wire.quality_evidence.minimum_signed_measure_scale
    }

    /// Exact minimum signed measure scale along the admitted linear path.
    #[must_use]
    pub const fn minimum_path_signed_measure_scale(&self) -> f64 {
        self.wire.action_evidence.minimum_path_signed_measure_scale
    }

    /// Rebuild and compare the complete state from exact dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for dependency, predecessor, driver, coordinate,
    /// velocity, path, or quality-evidence drift.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        reference_mesh: &SimplicialMeshEnvelopeV1,
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
        predecessor: Option<&Self>,
        solid_displacement: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            model,
            geometry,
            correspondence,
            reference_mesh,
            realization,
            self.step(),
            self.time_s(),
            predecessor,
            solid_displacement,
            self.current_coordinates_m().to_vec(),
        )?;
        if self != &expected {
            return Err(invalid_artifact(
                "geometry state differs from exact fixed-topology replay",
            ));
        }
        Ok(())
    }

    fn validate_reference_lineage(
        &self,
        reference: &WireReferenceLineageV1,
        reference_mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        if &self.wire.reference != reference {
            return Err(invalid_artifact(
                "geometry-state predecessor has stale reference lineage",
            ));
        }
        let state = FixedTopologyGeometryState2d::new(
            reference_mesh.mesh(),
            self.current_coordinates_m().to_vec(),
        )
        .map_err(|error| invalid_artifact(error.message()))?;
        let report = state.quality_report();
        if report.minimum_mean_ratio().to_bits() != self.minimum_mean_ratio().to_bits()
            || report.minimum_signed_measure_scale().to_bits()
                != self.minimum_signed_measure_scale().to_bits()
        {
            return Err(invalid_artifact(
                "geometry-state predecessor quality evidence differs from reference-topology replay",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: MeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != GEOMETRY_STATE_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.coordinates.scalar != WireScalarV1::F64
            || self.wire.coordinates.unit != WireCoordinateUnitV1::Metre
            || self.wire.coordinates.ordering != WireCoordinateOrderingV1::ReferenceMeshVertex
            || self.wire.action_evidence.mesh_velocity_unit
                != WireMeshVelocityUnitV1::MetrePerSecond
        {
            return Err(invalid_artifact(
                "unsupported geometry-state schema, encoding, scalar, unit, or ordering",
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
                "geometry-state accepted time must be finite, nonnegative, and canonical",
            ));
        }
        let initial = self.wire.predecessor_geometry_state_sha256.is_none();
        if initial
            != (self.wire.accepted.step == 0
                && self.wire.accepted.time_s == 0.0
                && self.wire.action_evidence.path_origin == WirePathOriginV1::ReferenceMesh
                && self.wire.action_evidence.mesh_velocity_m_per_s.is_none())
            || (!initial
                && (self.wire.accepted.step == 0
                    || self.wire.accepted.time_s == 0.0
                    || self.wire.action_evidence.path_origin
                        != WirePathOriginV1::PreviousAcceptedState
                    || self.wire.action_evidence.mesh_velocity_m_per_s.is_none()))
        {
            return Err(invalid_artifact(
                "geometry-state predecessor, coordinate, path origin, and velocity roles are inconsistent",
            ));
        }
        validate_coordinate_payload(
            &self.wire.coordinates.values,
            self.wire.action_evidence.mesh_velocity_m_per_s.as_deref(),
            2,
            limits,
        )?;
        let quality = &self.wire.quality_evidence;
        validate_endpoint_quality(
            quality.minimum_mean_ratio,
            quality.minimum_signed_measure_scale,
        )?;
        validate_path_quality(self.wire.action_evidence.minimum_path_signed_measure_scale)?;
        Ok(())
    }
}

pub(crate) fn validate_geometry_driver(
    driver: &FieldSnapshotEnvelopeV2,
    model: &ArtifactDigest,
    realization: &ArtifactDigest,
    geometry: &ArtifactDigest,
    correspondence: &ArtifactDigest,
    mesh: &ArtifactDigest,
    dimension: usize,
) -> Result<(), Diagnostic> {
    if driver.model_artifact() != *model
        || driver.realization_artifact() != *realization
        || driver.geometry_artifact() != *geometry
        || driver.correspondence_artifact() != *correspondence
        || driver.mesh_artifact() != *mesh
    {
        return Err(invalid_artifact(
            "geometry-state displacement driver has stale reference lineage",
        ));
    }
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let shape = driver.value_shape();
    if driver.dimension() != length
        || shape.rank() != 1
        || shape.component_count() != Some(dimension)
        || driver.frame() != ValueFrame::SpatialCartesian
    {
        return Err(invalid_artifact(
            "geometry-state driver must be a coherent-SI spatial displacement vector",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryStateEnvelopeV1 {
    schema: String,
    encoding: String,
    reference: WireReferenceLineageV1,
    accepted: WireAcceptedCoordinateV1,
    predecessor_geometry_state_sha256: Option<String>,
    solid_displacement_snapshot_sha256: String,
    coordinates: WireCoordinatesV1,
    action_evidence: WireGeometryActionEvidenceV1,
    quality_evidence: WireQualityEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReferenceLineageV1 {
    model_sha256: String,
    semantic_revision: u64,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    realization_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedCoordinateV1 {
    step: u64,
    time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCoordinatesV1 {
    scalar: WireScalarV1,
    unit: WireCoordinateUnitV1,
    ordering: WireCoordinateOrderingV1,
    values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryActionEvidenceV1 {
    path_origin: WirePathOriginV1,
    mesh_velocity_unit: WireMeshVelocityUnitV1,
    mesh_velocity_m_per_s: Option<Vec<Vec<f64>>>,
    minimum_path_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQualityEvidenceV1 {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarV1 {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateUnitV1 {
    Metre,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMeshVelocityUnitV1 {
    MetrePerSecond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateOrderingV1 {
    ReferenceMeshVertex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePathOriginV1 {
    ReferenceMesh,
    PreviousAcceptedState,
}
