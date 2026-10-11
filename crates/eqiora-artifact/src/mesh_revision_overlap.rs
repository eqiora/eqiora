//! Content-bound material and current-spatial overlap evidence for one remesh.

use std::collections::BTreeSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_meshing::{
    CellId, FacetId, FixedTopologyGeometryState2d, MeshEntity, MeshTopology,
    OverlapCoordinateChart2d, RetainedFacetSide2d, SimplicialMesh, SimplicialRevisionOverlap2d,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, FieldSnapshotEnvelopeV2, GeometryStateEnvelopeV2,
    RemeshDecoderLimits, ReplayableCanonicalModelArtifact, ValidatedMovingSpatialContextV2,
    ValidatedRemeshGeometrySourceV2, check_json_limits, invalid_artifact,
};

const OVERLAP_SCHEMA: &str = "eqiora.mesh-revision-overlap-envelope/v1";

/// Material and current-spatial common refinements for every retained body.
///
/// The two charts are deliberately carried together. Material overlap is the
/// integration measure for absolute solid state, while current-spatial overlap
/// is the integration measure for fluid state at the same accepted model time.
/// Neither chart is inferred later from a Field name.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshRevisionOverlapEnvelopeV1 {
    wire: WireMeshRevisionOverlapEnvelopeV1,
}

impl MeshRevisionOverlapEnvelopeV1 {
    /// Derive both overlap charts from exact source and target dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale lineage, an invalid remesh-origin geometry
    /// state, incomplete semantic retention, invalid boundary parent incidence,
    /// or any failure of bidirectional overlap coverage.
    pub fn new<M: ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelopeV2,
        target_solid_displacement: &FieldSnapshotEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        target_geometry_state.validate_against_remesh(
            source,
            target_context.model(),
            target_context.geometry(),
            target_context.correspondence(),
            target_context.mesh(),
            target_context.realization(),
            target_solid_displacement,
        )?;

        let source_context = source.context();
        let source_reference = source_context.mesh().mesh();
        let target_reference = target_context.mesh().mesh();
        let source_current = FixedTopologyGeometryState2d::new(
            source_reference,
            source.geometry_state().current_coordinates_m().to_vec(),
        )
        .and_then(|state| state.reconstruct_mesh(source_reference))
        .map_err(|error| invalid_artifact(error.message()))?;
        let target_current = FixedTopologyGeometryState2d::new(
            target_reference,
            target_geometry_state.current_coordinates_m().to_vec(),
        )
        .and_then(|state| state.reconstruct_mesh(target_reference))
        .map_err(|error| invalid_artifact(error.message()))?;

        let mut bodies = Vec::new();
        for source_body in source_context.geometry().bodies() {
            let target_body = source
                .association()
                .retained_body_target(source_body.domain())
                .ok_or_else(|| invalid_artifact("remesh overlap omits one retained body"))?;
            let source_cells = body_cells(source_context, source_body.domain())?;
            let target_cells = body_cells(target_context, target_body)?;
            let source_sides =
                body_boundary_sides(source_context, source_body.domain(), &source_cells)?;
            let target_sides = body_boundary_sides(target_context, target_body, &target_cells)?;
            require_retained_boundaries(
                source,
                source_body.domain(),
                target_body,
                &source_sides,
                &target_sides,
            )?;

            let material = derive_overlap(
                OverlapCoordinateChart2d::Material,
                source_reference,
                &source_cells,
                &source_sides,
                target_reference,
                &target_cells,
                &target_sides,
            )?;
            let current = derive_overlap(
                OverlapCoordinateChart2d::CurrentSpatial,
                &source_current,
                &source_cells,
                &source_sides,
                &target_current,
                &target_cells,
                &target_sides,
            )?;
            bodies.push(WireBodyOverlapV1 {
                source_domain_ulid: source_body.domain().ulid().to_string(),
                target_domain_ulid: target_body.ulid().to_string(),
                material: WireOverlap2d::encode(&material)?,
                current_spatial: WireOverlap2d::encode(&current)?,
            });
        }
        bodies.sort_by(|left, right| left.source_domain_ulid.cmp(&right.source_domain_ulid));

        let value = Self {
            wire: WireMeshRevisionOverlapEnvelopeV1 {
                schema: OVERLAP_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source: WireOverlapEndpointV1 {
                    model_sha256: source.state().model_artifact().to_string(),
                    semantic_revision: source.state().semantic_revision(),
                    realization_sha256: source.state().realization_artifact().to_string(),
                    geometry_sha256: source.state().reference_geometry_artifact().to_string(),
                    correspondence_sha256: source.state().correspondence_artifact().to_string(),
                    mesh_sha256: source.state().reference_mesh_artifact().to_string(),
                    geometry_state_sha256: source.geometry_state().digest()?.to_string(),
                },
                target: WireOverlapEndpointV1 {
                    model_sha256: target_geometry_state.model_artifact().to_string(),
                    semantic_revision: target_geometry_state.semantic_revision(),
                    realization_sha256: target_geometry_state.realization_artifact().to_string(),
                    geometry_sha256: target_geometry_state
                        .reference_geometry_artifact()
                        .to_string(),
                    correspondence_sha256: target_geometry_state
                        .reference_correspondence_artifact()
                        .to_string(),
                    mesh_sha256: target_geometry_state.reference_mesh_artifact().to_string(),
                    geometry_state_sha256: target_geometry_state.digest()?.to_string(),
                },
                source_spatial_state_sha256: source.state().digest()?.to_string(),
                semantic_association_sha256: source.association().digest()?.to_string(),
                accepted_step: source.state().step(),
                accepted_time_s: source.state().time_s(),
                bodies,
            },
        };
        value.validate_local(RemeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded overlap data without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: RemeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid remesh overlap JSON: {error}")))?;
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
            .map_err(|error| invalid_artifact(format!("cannot serialize remesh overlap: {error}")))
    }

    /// Domain-separated identity of both overlap charts and all dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            OVERLAP_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact accepted source spatial state.
    #[must_use]
    pub fn source_spatial_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_spatial_state_sha256.clone())
    }

    /// Exact source geometry state.
    #[must_use]
    pub fn source_geometry_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source.geometry_state_sha256.clone())
    }

    /// Exact target remesh-origin geometry state.
    #[must_use]
    pub fn target_geometry_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.target.geometry_state_sha256.clone())
    }

    /// Exact semantic revision-association proof.
    #[must_use]
    pub fn semantic_association_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.semantic_association_sha256.clone())
    }

    /// Accepted model step shared by both representations.
    #[must_use]
    pub const fn accepted_step(&self) -> u64 {
        self.wire.accepted_step
    }

    /// Accepted model time shared by both representations.
    #[must_use]
    pub const fn accepted_time_s(&self) -> f64 {
        self.wire.accepted_time_s
    }

    /// Recompute both charts and compare exact canonical content.
    ///
    /// # Errors
    /// Returns `EQ0901` for any resource substitution or replay drift.
    pub fn validate_against<M: ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelopeV2,
        target_solid_displacement: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            source,
            target_context,
            target_geometry_state,
            target_solid_displacement,
        )?;
        if self == &expected {
            Ok(())
        } else {
            Err(invalid_artifact(
                "remesh overlap differs from exact dependency replay",
            ))
        }
    }

    fn validate_local(&self, limits: RemeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != OVERLAP_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported remesh overlap schema or encoding",
            ));
        }
        self.wire.source.validate()?;
        self.wire.target.validate()?;
        for digest in [
            &self.wire.source_spatial_state_sha256,
            &self.wire.semantic_association_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.wire.source.model_sha256 != self.wire.target.model_sha256
            || self.wire.source.semantic_revision != self.wire.target.semantic_revision
            || self.wire.source.mesh_sha256 == self.wire.target.mesh_sha256
            || self.wire.source.realization_sha256 == self.wire.target.realization_sha256
            || self.wire.source.geometry_state_sha256 == self.wire.target.geometry_state_sha256
            || !self.wire.accepted_time_s.is_finite()
            || self.wire.accepted_time_s < 0.0
            || is_negative_zero(self.wire.accepted_time_s)
            || self.wire.bodies.is_empty()
            || self.wire.bodies.len() > limits.max_geometry_revision_associations
        {
            return Err(invalid_artifact(
                "remesh overlap has invalid shared coordinate or revision lineage",
            ));
        }
        let mut cell_fragments = 0_usize;
        let mut facet_fragments = 0_usize;
        let mut source_domains = BTreeSet::new();
        let mut target_domains = BTreeSet::new();
        for body in &self.wire.bodies {
            parse_domain(&body.source_domain_ulid)?;
            parse_domain(&body.target_domain_ulid)?;
            if !source_domains.insert(&body.source_domain_ulid)
                || !target_domains.insert(&body.target_domain_ulid)
            {
                return Err(invalid_artifact(
                    "remesh overlap body association is not one-to-one",
                ));
            }
            body.material.validate(WireOverlapChartV1::Material)?;
            body.current_spatial
                .validate(WireOverlapChartV1::CurrentSpatial)?;
            cell_fragments = cell_fragments
                .checked_add(body.material.cell_fragments.len())
                .and_then(|count| count.checked_add(body.current_spatial.cell_fragments.len()))
                .ok_or_else(|| invalid_artifact("remesh overlap cell-fragment count overflows"))?;
            facet_fragments = facet_fragments
                .checked_add(body.material.facet_fragments.len())
                .and_then(|count| count.checked_add(body.current_spatial.facet_fragments.len()))
                .ok_or_else(|| invalid_artifact("remesh overlap facet-fragment count overflows"))?;
        }
        if self
            .wire
            .bodies
            .windows(2)
            .any(|pair| pair[0].source_domain_ulid >= pair[1].source_domain_ulid)
            || cell_fragments > limits.max_mesh_overlap_cell_fragments
            || facet_fragments > limits.max_mesh_overlap_facet_fragments
        {
            return Err(invalid_artifact(
                "remesh overlap is reordered or exceeds its fragment budget",
            ));
        }
        Ok(())
    }
}

fn derive_overlap(
    chart: OverlapCoordinateChart2d,
    source_mesh: &SimplicialMesh,
    source_cells: &[CellId],
    source_sides: &[RetainedFacetSide2d],
    target_mesh: &SimplicialMesh,
    target_cells: &[CellId],
    target_sides: &[RetainedFacetSide2d],
) -> Result<SimplicialRevisionOverlap2d, Diagnostic> {
    SimplicialRevisionOverlap2d::new(chart, source_mesh, source_cells, target_mesh, target_cells)
        .and_then(|overlap| {
            overlap.with_retained_facets(source_mesh, source_sides, target_mesh, target_sides)
        })
        .map_err(|error| invalid_artifact(error.message()))
}

fn body_cells<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
    body: Id<kinds::Domain>,
) -> Result<Vec<CellId>, Diagnostic> {
    context
        .correspondence()
        .body_cells(body)
        .ok_or_else(|| invalid_artifact("remesh overlap body has no correspondence cells"))
        .map(|cells| cells.into_iter().map(CellId::new).collect())
}

fn body_boundary_sides<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
    body: Id<kinds::Domain>,
    body_cells: &[CellId],
) -> Result<Vec<RetainedFacetSide2d>, Diagnostic> {
    let selected = body_cells.iter().copied().collect::<BTreeSet<_>>();
    let mut sides = Vec::new();
    for boundary in context
        .geometry()
        .boundaries()
        .into_iter()
        .filter(|boundary| boundary.parent() == body)
    {
        for facet in context
            .correspondence()
            .boundary_facets(boundary.domain())
            .ok_or_else(|| {
                invalid_artifact("remesh overlap boundary has no correspondence facets")
            })?
        {
            let parents = context
                .mesh()
                .mesh()
                .incidence(MeshEntity::new(1, facet), 2)
                .ok_or_else(|| invalid_artifact("remesh overlap facet incidence is invalid"))?
                .into_iter()
                .map(|entry| CellId::new(entry.entity.index()))
                .filter(|cell| selected.contains(cell))
                .collect::<Vec<_>>();
            if parents.len() != 1 {
                return Err(invalid_artifact(
                    "retained boundary facet must have exactly one parent in its body",
                ));
            }
            sides.push(RetainedFacetSide2d::new(FacetId::new(facet), parents[0]));
        }
    }
    sides.sort_unstable();
    if sides.is_empty()
        || sides
            .windows(2)
            .any(|pair| pair[0].facet() == pair[1].facet())
    {
        return Err(invalid_artifact(
            "remesh overlap requires a complete unique retained boundary frontier",
        ));
    }
    Ok(sides)
}

fn require_retained_boundaries<M: ReplayableCanonicalModelArtifact>(
    source: &ValidatedRemeshGeometrySourceV2<'_, M>,
    source_body: Id<kinds::Domain>,
    target_body: Id<kinds::Domain>,
    source_sides: &[RetainedFacetSide2d],
    target_sides: &[RetainedFacetSide2d],
) -> Result<(), Diagnostic> {
    let source_boundaries = source
        .context()
        .geometry()
        .boundaries()
        .into_iter()
        .filter(|boundary| boundary.parent() == source_body)
        .collect::<Vec<_>>();
    let target_boundaries = source_boundaries
        .iter()
        .map(|boundary| {
            source
                .association()
                .retained_boundary_target(boundary.domain())
                .map(|target| target.ulid())
                .ok_or_else(|| invalid_artifact("remesh overlap omits one retained boundary"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if target_boundaries.len() != source_boundaries.len()
        || source_sides.is_empty()
        || target_sides.is_empty()
        || source.association().retained_body_target(source_body) != Some(target_body)
    {
        return Err(invalid_artifact(
            "remesh overlap boundary retention differs from its body association",
        ));
    }
    Ok(())
}

fn parse_domain(value: &str) -> Result<Id<kinds::Domain>, Diagnostic> {
    let ulid = Ulid::from_str(value)
        .map_err(|_| invalid_artifact("remesh overlap Domain ULID is malformed"))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(
            "remesh overlap Domain ULID spelling is noncanonical",
        ));
    }
    Ok(Id::from_ulid(ulid))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMeshRevisionOverlapEnvelopeV1 {
    schema: String,
    encoding: String,
    source: WireOverlapEndpointV1,
    target: WireOverlapEndpointV1,
    source_spatial_state_sha256: String,
    semantic_association_sha256: String,
    accepted_step: u64,
    accepted_time_s: f64,
    bodies: Vec<WireBodyOverlapV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOverlapEndpointV1 {
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: String,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    geometry_state_sha256: String,
}

impl WireOverlapEndpointV1 {
    fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [
            &self.model_sha256,
            &self.realization_sha256,
            &self.geometry_sha256,
            &self.correspondence_sha256,
            &self.mesh_sha256,
            &self.geometry_state_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.semantic_revision == 0 {
            return Err(invalid_artifact(
                "remesh overlap semantic revision must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBodyOverlapV1 {
    source_domain_ulid: String,
    target_domain_ulid: String,
    material: WireOverlap2d,
    current_spatial: WireOverlap2d,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOverlap2d {
    chart: WireOverlapChartV1,
    source_cells: Vec<u64>,
    target_cells: Vec<u64>,
    source_retained_facets: Vec<WireRetainedFacetSideV1>,
    target_retained_facets: Vec<WireRetainedFacetSideV1>,
    cell_fragments: Vec<WireCellFragmentV1>,
    facet_fragments: Vec<WireFacetFragmentV1>,
}

impl WireOverlap2d {
    fn encode(value: &SimplicialRevisionOverlap2d) -> Result<Self, Diagnostic> {
        Ok(Self {
            chart: WireOverlapChartV1::encode(value.chart()),
            source_cells: encode_indices(value.source_cells().iter().map(|cell| cell.index()))?,
            target_cells: encode_indices(value.target_cells().iter().map(|cell| cell.index()))?,
            source_retained_facets: value
                .source_retained_facets()
                .iter()
                .map(|side| WireRetainedFacetSideV1::encode(*side))
                .collect::<Result<_, _>>()?,
            target_retained_facets: value
                .target_retained_facets()
                .iter()
                .map(|side| WireRetainedFacetSideV1::encode(*side))
                .collect::<Result<_, _>>()?,
            cell_fragments: value
                .cell_fragments()
                .iter()
                .map(|fragment| {
                    Ok(WireCellFragmentV1 {
                        source_cell: encode_index(fragment.source_cell().index())?,
                        target_cell: encode_index(fragment.target_cell().index())?,
                        triangle_m: *fragment.triangle(),
                        area_m2: fragment.area(),
                        first_moment_m3: fragment.first_moment(),
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
            facet_fragments: value
                .retained_facet_fragments()
                .iter()
                .map(|fragment| {
                    Ok(WireFacetFragmentV1 {
                        source: WireRetainedFacetSideV1 {
                            facet: encode_index(fragment.source_facet().index())?,
                            parent: encode_index(fragment.source_parent().index())?,
                        },
                        target: WireRetainedFacetSideV1 {
                            facet: encode_index(fragment.target_facet().index())?,
                            parent: encode_index(fragment.target_parent().index())?,
                        },
                        segment_m: *fragment.segment(),
                        length_m: fragment.length(),
                        first_moment_m2: fragment.first_moment(),
                        source_outward_normal: fragment.source_outward_normal(),
                        target_outward_normal: fragment.target_outward_normal(),
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
        })
    }

    fn validate(&self, expected_chart: WireOverlapChartV1) -> Result<(), Diagnostic> {
        if self.chart != expected_chart
            || self.source_cells.is_empty()
            || self.target_cells.is_empty()
            || self.source_retained_facets.is_empty()
            || self.target_retained_facets.is_empty()
            || self.cell_fragments.is_empty()
            || self.facet_fragments.is_empty()
            || !strictly_increasing(&self.source_cells)
            || !strictly_increasing(&self.target_cells)
        {
            return Err(invalid_artifact(
                "remesh overlap chart or bounded inventories are incomplete",
            ));
        }
        for side in self
            .source_retained_facets
            .iter()
            .chain(&self.target_retained_facets)
        {
            side.validate()?;
        }
        for fragment in &self.cell_fragments {
            fragment.validate()?;
        }
        for fragment in &self.facet_fragments {
            fragment.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOverlapChartV1 {
    Material,
    CurrentSpatial,
}

impl WireOverlapChartV1 {
    const fn encode(value: OverlapCoordinateChart2d) -> Self {
        match value {
            OverlapCoordinateChart2d::Material => Self::Material,
            OverlapCoordinateChart2d::CurrentSpatial => Self::CurrentSpatial,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRetainedFacetSideV1 {
    facet: u64,
    parent: u64,
}

impl WireRetainedFacetSideV1 {
    fn encode(value: RetainedFacetSide2d) -> Result<Self, Diagnostic> {
        Ok(Self {
            facet: encode_index(value.facet().index())?,
            parent: encode_index(value.parent().index())?,
        })
    }

    fn validate(self) -> Result<(), Diagnostic> {
        usize::try_from(self.facet)
            .and_then(|_| usize::try_from(self.parent))
            .map(drop)
            .map_err(|_| invalid_artifact("remesh overlap index exceeds platform usize"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCellFragmentV1 {
    source_cell: u64,
    target_cell: u64,
    triangle_m: [[f64; 2]; 3],
    area_m2: f64,
    first_moment_m3: [f64; 2],
}

impl WireCellFragmentV1 {
    fn validate(self) -> Result<(), Diagnostic> {
        usize::try_from(self.source_cell)
            .and_then(|_| usize::try_from(self.target_cell))
            .map_err(|_| invalid_artifact("remesh overlap cell index exceeds platform usize"))?;
        if !self.area_m2.is_finite()
            || self.area_m2 <= 0.0
            || finite_values(self.triangle_m.into_iter().flatten()).is_err()
            || finite_values(self.first_moment_m3).is_err()
        {
            return Err(invalid_artifact(
                "remesh overlap cell fragment is non-finite or non-positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFacetFragmentV1 {
    source: WireRetainedFacetSideV1,
    target: WireRetainedFacetSideV1,
    segment_m: [[f64; 2]; 2],
    length_m: f64,
    first_moment_m2: [f64; 2],
    source_outward_normal: [f64; 2],
    target_outward_normal: [f64; 2],
}

impl WireFacetFragmentV1 {
    fn validate(self) -> Result<(), Diagnostic> {
        self.source.validate()?;
        self.target.validate()?;
        if !self.length_m.is_finite()
            || self.length_m <= 0.0
            || finite_values(self.segment_m.into_iter().flatten()).is_err()
            || finite_values(self.first_moment_m2).is_err()
            || finite_values(self.source_outward_normal).is_err()
            || finite_values(self.target_outward_normal).is_err()
        {
            return Err(invalid_artifact(
                "remesh overlap facet fragment is non-finite or non-positive",
            ));
        }
        Ok(())
    }
}

fn finite_values(values: impl IntoIterator<Item = f64>) -> Result<(), Diagnostic> {
    if values
        .into_iter()
        .any(|value| !value.is_finite() || is_negative_zero(value))
    {
        Err(invalid_artifact(
            "remesh overlap floating-point evidence must be finite and canonical",
        ))
    } else {
        Ok(())
    }
}

fn encode_indices(values: impl IntoIterator<Item = usize>) -> Result<Vec<u64>, Diagnostic> {
    values.into_iter().map(encode_index).collect()
}

fn encode_index(value: usize) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact("remesh overlap index exceeds u64"))
}

fn strictly_increasing(values: &[u64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(mesh: &str, realization: &str, state: &str) -> WireOverlapEndpointV1 {
        WireOverlapEndpointV1 {
            model_sha256: "00".repeat(32),
            semantic_revision: 1,
            realization_sha256: realization.repeat(32),
            geometry_sha256: "33".repeat(32),
            correspondence_sha256: "44".repeat(32),
            mesh_sha256: mesh.repeat(32),
            geometry_state_sha256: state.repeat(32),
        }
    }

    fn chart(chart: WireOverlapChartV1) -> WireOverlap2d {
        WireOverlap2d {
            chart,
            source_cells: vec![0],
            target_cells: vec![0],
            source_retained_facets: vec![WireRetainedFacetSideV1 {
                facet: 0,
                parent: 0,
            }],
            target_retained_facets: vec![WireRetainedFacetSideV1 {
                facet: 0,
                parent: 0,
            }],
            cell_fragments: vec![WireCellFragmentV1 {
                source_cell: 0,
                target_cell: 0,
                triangle_m: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                area_m2: 0.5,
                first_moment_m3: [1.0 / 6.0, 1.0 / 6.0],
            }],
            facet_fragments: vec![WireFacetFragmentV1 {
                source: WireRetainedFacetSideV1 {
                    facet: 0,
                    parent: 0,
                },
                target: WireRetainedFacetSideV1 {
                    facet: 0,
                    parent: 0,
                },
                segment_m: [[0.0, 0.0], [1.0, 0.0]],
                length_m: 1.0,
                first_moment_m2: [0.5, 0.0],
                source_outward_normal: [0.0, -1.0],
                target_outward_normal: [0.0, -1.0],
            }],
        }
    }

    fn overlap() -> MeshRevisionOverlapEnvelopeV1 {
        MeshRevisionOverlapEnvelopeV1 {
            wire: WireMeshRevisionOverlapEnvelopeV1 {
                schema: OVERLAP_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source: endpoint("55", "11", "77"),
                target: endpoint("66", "22", "88"),
                source_spatial_state_sha256: "99".repeat(32),
                semantic_association_sha256: "aa".repeat(32),
                accepted_step: 4,
                accepted_time_s: 0.4,
                bodies: vec![WireBodyOverlapV1 {
                    source_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned(),
                    target_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAW".to_owned(),
                    material: chart(WireOverlapChartV1::Material),
                    current_spatial: chart(WireOverlapChartV1::CurrentSpatial),
                }],
            },
        }
    }

    #[test]
    fn overlap_wire_roundtrip_and_digest_are_frozen() {
        let value = overlap();
        value
            .validate_local(RemeshDecoderLimits::default())
            .unwrap();
        let bytes = value.canonical_json().unwrap();
        assert_eq!(
            MeshRevisionOverlapEnvelopeV1::from_json(&bytes, RemeshDecoderLimits::default())
                .unwrap(),
            value
        );
        assert_eq!(
            value.digest().unwrap().to_string(),
            "3dddd39d50506b9750192c1b50edad999801e17e4a4e2ea5d84fae19544b6af5"
        );
    }

    #[test]
    fn overlap_wire_rejects_substitution_and_aggregate_resource_excess() {
        let value = overlap();
        let mut substituted = value.clone();
        substituted.wire.target.mesh_sha256 = substituted.wire.source.mesh_sha256.clone();
        assert!(
            substituted
                .validate_local(RemeshDecoderLimits::default())
                .is_err()
        );

        let limits = RemeshDecoderLimits {
            max_mesh_overlap_cell_fragments: 1,
            ..RemeshDecoderLimits::default()
        };
        assert!(
            MeshRevisionOverlapEnvelopeV1::from_json(&value.canonical_json().unwrap(), limits,)
                .is_err()
        );
    }
}
