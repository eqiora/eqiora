//! Geometry states across an explicit fixed-topology/remesh seam.

use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyGeometryState2d;
use serde::{Deserialize, Serialize};

use crate::geometry_state::validate_geometry_driver;
use crate::geometry_state::validation::{
    is_negative_geometry_zero, normalize_geometry_coordinates, normalize_geometry_zero,
    replay_geometry_action, validate_coordinate_payload, validate_endpoint_quality,
    validate_path_quality, validate_successor,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalRealizationArtifact, FieldSnapshotEnvelopeV2,
    GeometryAssociationArtifactError, GeometryIdentityEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, GeometryRevisionAssociationEnvelopeV1,
    GeometryStateEnvelopeV1, MeshDecoderLimits, ReplayableCanonicalModelArtifact,
    SimplicialMeshEnvelopeV1, SpatialStateEnvelopeV2, ValidatedMovingSpatialContextV2,
    check_json_limits, invalid_artifact,
};

const GEOMETRY_STATE_SCHEMA: &str = "eqiora.geometry-state-envelope/v2";

/// Fully replayed source side of one remesh-origin geometry state.
///
/// Construction validates the source moving state against its exact context,
/// predecessor, geometry state, and complete Field snapshots before any
/// target resource can consume it.
#[derive(Debug)]
pub struct ValidatedRemeshGeometrySourceV2<'a, M: ReplayableCanonicalModelArtifact> {
    context: &'a ValidatedMovingSpatialContextV2<'a, M>,
    state: &'a SpatialStateEnvelopeV2,
    geometry_state: &'a GeometryStateEnvelopeV1,
    association: &'a GeometryRevisionAssociationEnvelopeV1,
}

impl<'a, M: ReplayableCanonicalModelArtifact> ValidatedRemeshGeometrySourceV2<'a, M> {
    pub(crate) const fn context(&self) -> &'a ValidatedMovingSpatialContextV2<'a, M> {
        self.context
    }

    /// Replay the complete source state and retain one semantic association.
    ///
    /// # Errors
    /// Returns `EQ0901` for any stale source dependency.
    pub fn new(
        context: &'a ValidatedMovingSpatialContextV2<'a, M>,
        state: &'a SpatialStateEnvelopeV2,
        geometry_state: &'a GeometryStateEnvelopeV1,
        predecessor_geometry_state: Option<&GeometryStateEnvelopeV1>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        association: &'a GeometryRevisionAssociationEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        state.validate_against(
            context,
            geometry_state,
            predecessor_geometry_state,
            snapshots,
            (),
        )?;
        Ok(Self {
            context,
            state,
            geometry_state,
            association,
        })
    }

    /// Exact accepted source moving state.
    #[must_use]
    pub const fn state(&self) -> &SpatialStateEnvelopeV2 {
        self.state
    }

    /// Exact accepted source geometry state.
    #[must_use]
    pub const fn geometry_state(&self) -> &GeometryStateEnvelopeV1 {
        self.geometry_state
    }

    /// Exact semantic geometry-revision association.
    #[must_use]
    pub const fn association(&self) -> &GeometryRevisionAssociationEnvelopeV1 {
        self.association
    }
}

/// Closed origin of one geometry-state/v2 artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryStateOriginKindV2 {
    /// Positive-duration continuation on one immutable reference topology.
    Continuous,
    /// Zero-duration representation change from an accepted source topology.
    Remesh,
}

/// One accepted current geometry whose origin is explicit and closed.
///
/// A continuous state derives mesh velocity from a same-topology predecessor.
/// A remesh state retains the source state and semantic-association identities
/// at the same exact model coordinate and contains no mesh velocity.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryStateEnvelopeV2 {
    wire: WireGeometryStateEnvelopeV2,
}

impl GeometryStateEnvelopeV2 {
    /// Construct a positive-duration continuation on one reference topology.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale resources, a non-adjacent predecessor,
    /// invalid geometry, driver drift, or failed path-quality replay.
    #[allow(clippy::too_many_arguments)]
    pub fn continuous(
        model: &impl ReplayableCanonicalModelArtifact,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        reference_mesh: &SimplicialMeshEnvelopeV1,
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
        step: u64,
        time_s: f64,
        predecessor: &Self,
        solid_displacement: &FieldSnapshotEnvelopeV2,
        current_coordinates_m: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let common = validate_common(
            model,
            geometry,
            correspondence,
            reference_mesh,
            realization,
            step,
            time_s,
            solid_displacement,
            current_coordinates_m,
        )?;
        require_reference_lineage(predecessor, &common.reference, reference_mesh)?;
        validate_successor(
            step,
            common.time_s,
            predecessor.step(),
            predecessor.time_s(),
        )?;
        let action = replay_geometry_action(
            reference_mesh.mesh(),
            &common.current,
            Some((
                predecessor.current_coordinates_m(),
                common.time_s - predecessor.time_s(),
            )),
        )?;
        Self::finish(
            common,
            WireOriginV2::Continuous {
                predecessor_geometry_state_sha256: predecessor.digest()?.to_string(),
                mesh_velocity_unit: WireMeshVelocityUnitV2::MetrePerSecond,
                mesh_velocity_m_per_s: action.velocity.expect("a continuous action has velocity"),
                minimum_path_signed_measure_scale: action.minimum_path_signed_measure_scale,
            },
        )
    }

    /// Construct a zero-duration remesh origin at the source state coordinate.
    ///
    /// The target reference mesh must differ from the source mesh. Semantic
    /// retention is independently replayed, and no mesh velocity is created.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale/crossed resources, changed model time,
    /// identical mesh revisions, invalid target geometry, or a non-bijective
    /// semantic association. Typed association failures are flattened only at
    /// this artifact boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn remesh<M: ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        target_model: &impl ReplayableCanonicalModelArtifact,
        target_geometry: &GeometryIdentityEnvelopeV1,
        target_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        target_reference_mesh: &SimplicialMeshEnvelopeV1,
        target_realization: &(impl CanonicalRealizationArtifact + ?Sized),
        target_solid_displacement: &FieldSnapshotEnvelopeV2,
        target_current_coordinates_m: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let target_model_reference = target_model.artifact_reference()?;
        if source.state.model_artifact() != *target_model_reference.artifact()
            || source.state.semantic_revision() != target_model_reference.semantic_revision().get()
        {
            return Err(invalid_artifact(
                "remesh geometry-state/v2 must retain the exact source Model revision",
            ));
        }
        if source.context.mesh().digest()? == target_reference_mesh.digest()? {
            return Err(invalid_artifact(
                "remesh geometry-state/v2 requires a distinct target mesh revision",
            ));
        }
        source
            .association
            .validate_against(
                source.context.model(),
                source.context.geometry(),
                source.context.correspondence(),
                source.context.mesh(),
                target_model,
                target_geometry,
                target_correspondence,
                target_reference_mesh,
            )
            .map_err(association_error)?;
        let common = validate_common(
            target_model,
            target_geometry,
            target_correspondence,
            target_reference_mesh,
            target_realization,
            source.state.step(),
            source.state.time_s(),
            target_solid_displacement,
            target_current_coordinates_m,
        )?;
        Self::finish(
            common,
            WireOriginV2::Remesh {
                source_spatial_state_sha256: source.state.digest()?.to_string(),
                source_geometry_state_sha256: source.geometry_state.digest()?.to_string(),
                semantic_association_sha256: source.association.digest()?.to_string(),
            },
        )
    }

    fn finish(common: CommonConstruction, origin: WireOriginV2) -> Result<Self, Diagnostic> {
        let quality = common.current.quality_report();
        let value = Self {
            wire: WireGeometryStateEnvelopeV2 {
                schema: GEOMETRY_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: common.reference,
                accepted: WireAcceptedCoordinateV2 {
                    step: common.step,
                    time_s: common.time_s,
                },
                solid_displacement_snapshot_sha256: common.driver_sha256,
                coordinates: WireCoordinatesV2 {
                    scalar: WireScalarV2::F64,
                    unit: WireCoordinateUnitV2::Metre,
                    ordering: WireCoordinateOrderingV2::ReferenceMeshVertex,
                    values: common.current.coordinates().to_vec(),
                },
                origin,
                quality_evidence: WireQualityEvidenceV2 {
                    minimum_mean_ratio: quality.minimum_mean_ratio(),
                    minimum_signed_measure_scale: quality.minimum_signed_measure_scale(),
                },
            },
        };
        value.validate_local(MeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded wire data without resolving referenced artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical
    /// data, including any origin outside the closed grammar.
    pub fn from_json(bytes: &[u8], limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid geometry-state/v2 JSON: {error}"))
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
            invalid_artifact(format!("cannot serialize geometry state v2: {error}"))
        })
    }

    /// Domain-separated identity of the complete state and origin evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GEOMETRY_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Closed origin kind.
    #[must_use]
    pub const fn origin(&self) -> GeometryStateOriginKindV2 {
        match self.wire.origin {
            WireOriginV2::Continuous { .. } => GeometryStateOriginKindV2::Continuous,
            WireOriginV2::Remesh { .. } => GeometryStateOriginKindV2::Remesh,
        }
    }

    /// Accepted step ordinal.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.wire.accepted.step
    }

    /// Accepted coherent-SI model time.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.wire.accepted.time_s
    }

    /// Exact Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Exact semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Exact target Geometry Identity.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.geometry_sha256.clone())
    }

    /// Exact target geometry-to-mesh correspondence.
    #[must_use]
    pub fn reference_correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact target reference mesh.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.mesh_sha256.clone())
    }

    /// Exact target Realization.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
    }

    /// Exact target solid-displacement snapshot.
    #[must_use]
    pub fn solid_displacement_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.solid_displacement_snapshot_sha256.clone())
    }

    /// Current coordinates in target reference-vertex order.
    #[must_use]
    pub fn current_coordinates_m(&self) -> &[Vec<f64>] {
        &self.wire.coordinates.values
    }

    /// Derived mesh velocity, present only for continuous origins.
    #[must_use]
    pub fn mesh_velocity_m_per_s(&self) -> Option<&[Vec<f64>]> {
        match &self.wire.origin {
            WireOriginV2::Continuous {
                mesh_velocity_m_per_s,
                ..
            } => Some(mesh_velocity_m_per_s),
            WireOriginV2::Remesh { .. } => None,
        }
    }

    /// Same-topology predecessor, when this is a continuous origin.
    #[must_use]
    pub fn predecessor(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOriginV2::Continuous {
                predecessor_geometry_state_sha256,
                ..
            } => Some(ArtifactDigest(predecessor_geometry_state_sha256.clone())),
            WireOriginV2::Remesh { .. } => None,
        }
    }

    /// Source moving state, when this is a remesh origin.
    #[must_use]
    pub fn remesh_source_spatial_state(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOriginV2::Remesh {
                source_spatial_state_sha256,
                ..
            } => Some(ArtifactDigest(source_spatial_state_sha256.clone())),
            WireOriginV2::Continuous { .. } => None,
        }
    }

    /// Source geometry state, when this is a remesh origin.
    #[must_use]
    pub fn remesh_source_geometry_state(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOriginV2::Remesh {
                source_geometry_state_sha256,
                ..
            } => Some(ArtifactDigest(source_geometry_state_sha256.clone())),
            WireOriginV2::Continuous { .. } => None,
        }
    }

    /// Semantic association, when this is a remesh origin.
    #[must_use]
    pub fn remesh_semantic_association(&self) -> Option<ArtifactDigest> {
        match &self.wire.origin {
            WireOriginV2::Remesh {
                semantic_association_sha256,
                ..
            } => Some(ArtifactDigest(semantic_association_sha256.clone())),
            WireOriginV2::Continuous { .. } => None,
        }
    }

    /// Minimum current-cell mean-ratio quality.
    #[must_use]
    pub const fn minimum_mean_ratio(&self) -> f64 {
        self.wire.quality_evidence.minimum_mean_ratio
    }

    /// Minimum current signed measure scale.
    #[must_use]
    pub const fn minimum_signed_measure_scale(&self) -> f64 {
        self.wire.quality_evidence.minimum_signed_measure_scale
    }

    /// Rebuild and compare a continuous-origin state.
    ///
    /// # Errors
    /// Returns `EQ0901` for any dependency or replay drift.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against_continuous(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        reference_mesh: &SimplicialMeshEnvelopeV1,
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
        predecessor: &Self,
        solid_displacement: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        let expected = Self::continuous(
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
        require_equal(self, &expected)
    }

    /// Rebuild and compare a remesh-origin state.
    ///
    /// # Errors
    /// Returns `EQ0901` for any dependency or replay drift.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against_remesh<M: ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        target_model: &impl ReplayableCanonicalModelArtifact,
        target_geometry: &GeometryIdentityEnvelopeV1,
        target_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        target_reference_mesh: &SimplicialMeshEnvelopeV1,
        target_realization: &(impl CanonicalRealizationArtifact + ?Sized),
        target_solid_displacement: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        let expected = Self::remesh(
            source,
            target_model,
            target_geometry,
            target_correspondence,
            target_reference_mesh,
            target_realization,
            target_solid_displacement,
            self.current_coordinates_m().to_vec(),
        )?;
        require_equal(self, &expected)
    }

    fn validate_local(&self, limits: MeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != GEOMETRY_STATE_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.coordinates.scalar != WireScalarV2::F64
            || self.wire.coordinates.unit != WireCoordinateUnitV2::Metre
            || self.wire.coordinates.ordering != WireCoordinateOrderingV2::ReferenceMeshVertex
        {
            return Err(invalid_artifact(
                "unsupported geometry-state/v2 schema, encoding, scalar, unit, or ordering",
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
        if !self.time_s().is_finite()
            || self.time_s() < 0.0
            || is_negative_geometry_zero(self.time_s())
        {
            return Err(invalid_artifact(
                "geometry-state/v2 time must be finite, nonnegative, and canonical",
            ));
        }
        validate_coordinate_payload(
            self.current_coordinates_m(),
            self.mesh_velocity_m_per_s(),
            2,
            limits,
        )?;
        match &self.wire.origin {
            WireOriginV2::Continuous {
                predecessor_geometry_state_sha256,
                mesh_velocity_unit,
                minimum_path_signed_measure_scale,
                ..
            } => {
                ArtifactDigest::from_hex(predecessor_geometry_state_sha256.clone())?;
                if *mesh_velocity_unit != WireMeshVelocityUnitV2::MetrePerSecond
                    || self.step() == 0
                    || self.time_s() == 0.0
                {
                    return Err(invalid_artifact(
                        "continuous geometry-state/v2 origin has invalid coordinate or path evidence",
                    ));
                }
                validate_path_quality(*minimum_path_signed_measure_scale)?;
            }
            WireOriginV2::Remesh {
                source_spatial_state_sha256,
                source_geometry_state_sha256,
                semantic_association_sha256,
            } => {
                for digest in [
                    source_spatial_state_sha256,
                    source_geometry_state_sha256,
                    semantic_association_sha256,
                ] {
                    ArtifactDigest::from_hex(digest.clone())?;
                }
            }
        }
        let quality = &self.wire.quality_evidence;
        validate_endpoint_quality(
            quality.minimum_mean_ratio,
            quality.minimum_signed_measure_scale,
        )?;
        Ok(())
    }
}

struct CommonConstruction {
    reference: WireReferenceLineageV2,
    step: u64,
    time_s: f64,
    driver_sha256: String,
    current: FixedTopologyGeometryState2d,
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    model: &impl ReplayableCanonicalModelArtifact,
    geometry: &GeometryIdentityEnvelopeV1,
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    reference_mesh: &SimplicialMeshEnvelopeV1,
    realization: &(impl CanonicalRealizationArtifact + ?Sized),
    step: u64,
    time_s: f64,
    solid_displacement: &FieldSnapshotEnvelopeV2,
    mut current_coordinates_m: Vec<Vec<f64>>,
) -> Result<CommonConstruction, Diagnostic> {
    let model_reference = model.artifact_reference()?;
    let realization_reference = realization.artifact_reference()?;
    if realization_reference.model_artifact() != model_reference.artifact()
        || realization_reference.semantic_revision() != model_reference.semantic_revision()
    {
        return Err(invalid_artifact(
            "geometry-state/v2 Realization does not select the exact Model revision",
        ));
    }
    geometry.validate_against(model)?;
    correspondence.validate_against(geometry, model, reference_mesh)?;
    if reference_mesh.dimension() != 2 || !time_s.is_finite() || time_s < 0.0 {
        return Err(invalid_artifact(
            "geometry-state/v2 requires a 2D mesh and finite nonnegative time",
        ));
    }
    let geometry_digest = geometry.digest()?;
    let correspondence_digest = correspondence.digest()?;
    let mesh_digest = reference_mesh.digest()?;
    validate_geometry_driver(
        solid_displacement,
        model_reference.artifact(),
        realization_reference.artifact(),
        &geometry_digest,
        &correspondence_digest,
        &mesh_digest,
        2,
    )?;
    normalize_geometry_coordinates(&mut current_coordinates_m)?;
    let current = FixedTopologyGeometryState2d::new(reference_mesh.mesh(), current_coordinates_m)
        .map_err(|error| invalid_artifact(error.message()))?;
    Ok(CommonConstruction {
        reference: WireReferenceLineageV2 {
            model_sha256: model_reference.artifact().to_string(),
            semantic_revision: model_reference.semantic_revision().get(),
            geometry_sha256: geometry_digest.to_string(),
            correspondence_sha256: correspondence_digest.to_string(),
            mesh_sha256: mesh_digest.to_string(),
            realization_sha256: realization_reference.artifact().to_string(),
        },
        step,
        time_s: normalize_geometry_zero(time_s),
        driver_sha256: solid_displacement.digest()?.to_string(),
        current,
    })
}

fn require_reference_lineage(
    predecessor: &GeometryStateEnvelopeV2,
    reference: &WireReferenceLineageV2,
    mesh: &SimplicialMeshEnvelopeV1,
) -> Result<(), Diagnostic> {
    if predecessor.model_artifact().as_str() != reference.model_sha256
        || predecessor.semantic_revision() != reference.semantic_revision
        || predecessor.reference_geometry_artifact().as_str() != reference.geometry_sha256
        || predecessor.reference_correspondence_artifact().as_str()
            != reference.correspondence_sha256
        || predecessor.reference_mesh_artifact().as_str() != reference.mesh_sha256
        || predecessor.realization_artifact().as_str() != reference.realization_sha256
    {
        return Err(invalid_artifact(
            "continuous geometry-state/v2 predecessor has stale reference lineage",
        ));
    }
    let replayed = FixedTopologyGeometryState2d::new(
        mesh.mesh(),
        predecessor.current_coordinates_m().to_vec(),
    )
    .map_err(|error| invalid_artifact(error.message()))?;
    let report = replayed.quality_report();
    if report.minimum_mean_ratio().to_bits() != predecessor.minimum_mean_ratio().to_bits()
        || report.minimum_signed_measure_scale().to_bits()
            != predecessor.minimum_signed_measure_scale().to_bits()
    {
        return Err(invalid_artifact(
            "continuous geometry-state/v2 predecessor quality evidence drifted",
        ));
    }
    Ok(())
}

fn association_error(error: GeometryAssociationArtifactError) -> Diagnostic {
    invalid_artifact(format!(
        "remesh semantic geometry association failed: {error}"
    ))
}

fn require_equal(
    actual: &GeometryStateEnvelopeV2,
    expected: &GeometryStateEnvelopeV2,
) -> Result<(), Diagnostic> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_artifact(
            "geometry-state/v2 differs from exact dependency replay",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryStateEnvelopeV2 {
    schema: String,
    encoding: String,
    reference: WireReferenceLineageV2,
    accepted: WireAcceptedCoordinateV2,
    solid_displacement_snapshot_sha256: String,
    coordinates: WireCoordinatesV2,
    origin: WireOriginV2,
    quality_evidence: WireQualityEvidenceV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReferenceLineageV2 {
    model_sha256: String,
    semantic_revision: u64,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    realization_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedCoordinateV2 {
    step: u64,
    time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCoordinatesV2 {
    scalar: WireScalarV2,
    unit: WireCoordinateUnitV2,
    ordering: WireCoordinateOrderingV2,
    values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireOriginV2 {
    Continuous {
        predecessor_geometry_state_sha256: String,
        mesh_velocity_unit: WireMeshVelocityUnitV2,
        mesh_velocity_m_per_s: Vec<Vec<f64>>,
        minimum_path_signed_measure_scale: f64,
    },
    Remesh {
        source_spatial_state_sha256: String,
        source_geometry_state_sha256: String,
        semantic_association_sha256: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQualityEvidenceV2 {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarV2 {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateUnitV2 {
    Metre,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMeshVelocityUnitV2 {
    MetrePerSecond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateOrderingV2 {
    ReferenceMeshVertex,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remesh_state() -> GeometryStateEnvelopeV2 {
        GeometryStateEnvelopeV2 {
            wire: WireGeometryStateEnvelopeV2 {
                schema: GEOMETRY_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: WireReferenceLineageV2 {
                    model_sha256: "00".repeat(32),
                    semantic_revision: 1,
                    geometry_sha256: "11".repeat(32),
                    correspondence_sha256: "22".repeat(32),
                    mesh_sha256: "33".repeat(32),
                    realization_sha256: "44".repeat(32),
                },
                accepted: WireAcceptedCoordinateV2 {
                    step: 4,
                    time_s: 0.4,
                },
                solid_displacement_snapshot_sha256: "55".repeat(32),
                coordinates: WireCoordinatesV2 {
                    scalar: WireScalarV2::F64,
                    unit: WireCoordinateUnitV2::Metre,
                    ordering: WireCoordinateOrderingV2::ReferenceMeshVertex,
                    values: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                },
                origin: WireOriginV2::Remesh {
                    source_spatial_state_sha256: "66".repeat(32),
                    source_geometry_state_sha256: "77".repeat(32),
                    semantic_association_sha256: "88".repeat(32),
                },
                quality_evidence: WireQualityEvidenceV2 {
                    minimum_mean_ratio: 1.0,
                    minimum_signed_measure_scale: 1.0,
                },
            },
        }
    }

    #[test]
    fn remesh_origin_roundtrip_has_no_fictitious_velocity_and_frozen_digest() {
        let value = remesh_state();
        value.validate_local(MeshDecoderLimits::default()).unwrap();
        assert_eq!(value.origin(), GeometryStateOriginKindV2::Remesh);
        assert!(value.mesh_velocity_m_per_s().is_none());
        let bytes = value.canonical_json().unwrap();
        assert_eq!(
            GeometryStateEnvelopeV2::from_json(&bytes, MeshDecoderLimits::default()).unwrap(),
            value
        );
        assert_eq!(
            value.digest().unwrap().to_string(),
            "b4de12f2fd78b7c572f2968efd777ef37f661645ca65902258d3c5a54987cbee"
        );
    }

    #[test]
    fn remesh_origin_rejects_velocity_injection_and_resource_excess() {
        let value = remesh_state();
        let mut json: serde_json::Value =
            serde_json::from_slice(&value.canonical_json().unwrap()).unwrap();
        json["origin"]["mesh_velocity_m_per_s"] = serde_json::json!([[0.0, 0.0]]);
        assert!(
            GeometryStateEnvelopeV2::from_json(
                &serde_json::to_vec(&json).unwrap(),
                MeshDecoderLimits::default(),
            )
            .is_err()
        );

        let limits = MeshDecoderLimits {
            max_mesh_coordinate_values: 5,
            ..MeshDecoderLimits::default()
        };
        assert!(
            GeometryStateEnvelopeV2::from_json(&value.canonical_json().unwrap(), limits,).is_err()
        );
    }
}
