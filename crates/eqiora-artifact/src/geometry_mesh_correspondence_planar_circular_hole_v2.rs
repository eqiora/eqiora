//! Direct source correspondence for planar circular-hole Geometry v2.

use std::collections::BTreeSet;

use eqiora_core::Diagnostic;
use eqiora_geometry::{CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION};
use eqiora_meshing::MeshEntity;
use serde::{Deserialize, Serialize};

use super::{CORRESPONDENCE_SCHEMA, GeometryMeshCorrespondenceEnvelopeV1, WireCorrespondenceV1};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, GeometryDecoderLimits, SimplicialMeshEnvelopeV1,
    invalid_artifact,
};

const PLANAR_CIRCULAR_HOLE_V2_SOURCE: &str = "planar-circular-hole-v2";
const EXACT_BOUNDARY_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceParentOutward {
    LeftOfCanonicalFacet,
    RightOfCanonicalFacet,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceFrontierLineage {
    pub(crate) facet_index: usize,
    pub(crate) parent_outward: SourceParentOutward,
}

/// Private result of one Mesh construction pass with structural source lineage.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlanarCircularHoleV2Assignments {
    pub(crate) face_cells: Vec<usize>,
    pub(crate) frontiers: [Vec<SourceFrontierLineage>; EXACT_BOUNDARY_COUNT],
}

impl PlanarCircularHoleV2Assignments {
    pub(super) fn face_cells(&self) -> &[usize] {
        &self.face_cells
    }

    pub(super) fn frontiers(&self) -> &[Vec<SourceFrontierLineage>; EXACT_BOUNDARY_COUNT] {
        &self.frontiers
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WirePlanarCircularHoleV2CorrespondenceV1 {
    schema: String,
    encoding: String,
    source: String,
    pub(super) geometry_sha256: String,
    pub(super) mesh_sha256: String,
    dimension: u64,
    faces: Vec<WireFaceAssignment>,
    frontiers: Vec<WireFrontierAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFaceAssignment {
    geometry_face: u64,
    cell_indices: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFrontierAssignment {
    parent_face: u64,
    geometry_edge: u64,
    facet_indices: Vec<u64>,
    parent_outward: Vec<WireFacetOutward>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireFacetOutward {
    LeftOfCanonicalFacet,
    RightOfCanonicalFacet,
}

impl GeometryMeshCorrespondenceEnvelopeV1 {
    /// Bind an independently generated planar circular-hole v2 Mesh to its
    /// source edges through producer-emitted facet assignments.
    ///
    /// The five entries are source edges 0..4 in Geometry order.  The
    /// assignments must uniquely partition the complete exposed Mesh
    /// frontier.  Parent incidence and outward orientation are derived from
    /// Mesh topology, never from coordinates or a classification tolerance.
    pub fn from_planar_circular_hole_v2_mesh_assignments(
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        edge_facets: [Vec<usize>; EXACT_BOUNDARY_COUNT],
    ) -> Result<Self, Diagnostic> {
        require_planar_circular_hole_v2(geometry)?;
        let assignments = assigned_mesh_entities(mesh, edge_facets)?;
        let wire = planar_circular_hole_v2_wire_from_assignments(geometry, mesh, &assignments)?;
        let envelope = Self {
            wire: WireCorrespondenceV1::PlanarCircularHoleV2(wire),
        };
        envelope.validate_local(GeometryDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Revalidate producer-emitted source assignments against this persisted
    /// correspondence and the exact Geometry and Mesh identities.
    pub fn validate_against_planar_circular_hole_v2_mesh_assignments(
        &self,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        edge_facets: [Vec<usize>; EXACT_BOUNDARY_COUNT],
    ) -> Result<(), Diagnostic> {
        let expected =
            Self::from_planar_circular_hole_v2_mesh_assignments(geometry, mesh, edge_facets)?;
        if self != &expected {
            return Err(invalid_artifact(
                "planar circular-hole v2 correspondence differs from producer-owned Mesh assignments",
            ));
        }
        Ok(())
    }

    /// Mesh entities that exactly realize one named planar circular-hole v2
    /// source entity set.
    ///
    /// Exact resource replay remains the caller's admission step after
    /// decoding untrusted correspondence bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` when this is not the v2 source variant, the Geometry
    /// identity differs, or the named set is absent or incomplete.
    pub fn planar_circular_hole_v2_entity_set_entities(
        &self,
        geometry: &CanonicalGeometryV1,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        let WireCorrespondenceV1::PlanarCircularHoleV2(wire) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence has no planar circular-hole v2 entity sets",
            ));
        };
        require_planar_circular_hole_v2(geometry)?;
        if wire.geometry_sha256 != ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string()
        {
            return Err(invalid_artifact(
                "correspondence geometry digest differs from planar circular-hole Geometry v2",
            ));
        }
        let set = geometry.entity_set(name).ok_or_else(|| {
            invalid_artifact(format!(
                "planar circular-hole Geometry v2 has no entity set named '{name}'"
            ))
        })?;
        resolve_planar_circular_hole_v2_entity_set(wire, set.dimension(), set.members())
    }

    /// Mesh facets assigned to one exact source-edge index.
    pub fn planar_circular_hole_v2_source_edge_entities(
        &self,
        geometry: &CanonicalGeometryV1,
        source_edge: usize,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        let WireCorrespondenceV1::PlanarCircularHoleV2(wire) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence has no planar circular-hole v2 source edges",
            ));
        };
        require_planar_circular_hole_v2(geometry)?;
        if wire.geometry_sha256 != ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string()
        {
            return Err(invalid_artifact(
                "correspondence geometry digest differs from planar circular-hole Geometry v2",
            ));
        }
        let assignment = wire
            .frontiers
            .get(source_edge)
            .filter(|entry| usize::try_from(entry.geometry_edge) == Ok(source_edge))
            .ok_or_else(|| invalid_artifact("source edge is absent from correspondence"))?;
        assignment
            .facet_indices
            .iter()
            .map(|&facet| Ok(MeshEntity::new(EDGE_DIMENSION, local(facet, "mesh facet")?)))
            .collect()
    }
}

fn assigned_mesh_entities(
    mesh: &SimplicialMeshEnvelopeV1,
    edge_facets: [Vec<usize>; EXACT_BOUNDARY_COUNT],
) -> Result<PlanarCircularHoleV2Assignments, Diagnostic> {
    use eqiora_meshing::{MeshGeometry as _, MeshTopology as _};

    let native = mesh.mesh();
    if native.topological_dimension() != FACE_DIMENSION
        || native.geometric_dimension() != FACE_DIMENSION
    {
        return Err(invalid_artifact(
            "planar circular-hole v2 assignments require a two-dimensional Mesh",
        ));
    }
    let cell_count = native
        .entity_count(FACE_DIMENSION)
        .ok_or_else(|| invalid_artifact("assigned Mesh has no face stratum"))?;
    let facet_count = native
        .entity_count(EDGE_DIMENSION)
        .ok_or_else(|| invalid_artifact("assigned Mesh has no edge stratum"))?;
    let exposed = (0..facet_count)
        .filter(|&index| {
            native.is_boundary_entity(MeshEntity::new(EDGE_DIMENSION, index)) == Some(true)
        })
        .collect::<BTreeSet<_>>();
    let mut assigned = BTreeSet::new();
    let mut frontiers: [Vec<SourceFrontierLineage>; EXACT_BOUNDARY_COUNT] =
        std::array::from_fn(|_| Vec::new());
    for (source_edge, facets) in edge_facets.into_iter().enumerate() {
        if facets.is_empty() {
            return Err(invalid_artifact(format!(
                "source edge {source_edge} has no producer-emitted Mesh frontier"
            )));
        }
        for facet_index in facets {
            if facet_index >= facet_count || !assigned.insert(facet_index) {
                return Err(invalid_artifact(
                    "producer Mesh frontier assignments are duplicate or out of range",
                ));
            }
            if !exposed.contains(&facet_index) {
                return Err(invalid_artifact(
                    "producer Mesh frontier assignment names an interior facet",
                ));
            }
            frontiers[source_edge].push(SourceFrontierLineage {
                facet_index,
                parent_outward: parent_outward_from_topology(native, facet_index)?,
            });
        }
        frontiers[source_edge].sort_by_key(|entry| entry.facet_index);
    }
    if assigned != exposed {
        return Err(invalid_artifact(
            "producer Mesh frontier assignments omit an exposed facet",
        ));
    }
    Ok(PlanarCircularHoleV2Assignments {
        face_cells: (0..cell_count).collect(),
        frontiers,
    })
}

fn parent_outward_from_topology(
    mesh: &eqiora_meshing::SimplicialMesh,
    facet_index: usize,
) -> Result<SourceParentOutward, Diagnostic> {
    use eqiora_meshing::MeshTopology as _;

    let facet = MeshEntity::new(EDGE_DIMENSION, facet_index);
    let vertices = mesh
        .entity_vertices(facet)
        .ok_or_else(|| invalid_artifact("assigned frontier has no vertex closure"))?;
    let [first, second] = vertices.as_slice() else {
        return Err(invalid_artifact("assigned frontier is not a segment"));
    };
    let adjacent = mesh
        .incidence(facet, FACE_DIMENSION)
        .ok_or_else(|| invalid_artifact("assigned frontier has no parent incidence"))?;
    let [parent] = adjacent.as_slice() else {
        return Err(invalid_artifact(
            "assigned frontier must have exactly one parent-cell incidence",
        ));
    };
    let cell = mesh
        .cells()
        .get(parent.entity.index())
        .ok_or_else(|| invalid_artifact("assigned frontier parent cell is unavailable"))?;
    let forward = (0..cell.len()).any(|position| {
        cell[position] == first.index() && cell[(position + 1) % cell.len()] == second.index()
    });
    let reverse = (0..cell.len()).any(|position| {
        cell[position] == second.index() && cell[(position + 1) % cell.len()] == first.index()
    });
    match (forward, reverse) {
        (true, false) => Ok(SourceParentOutward::RightOfCanonicalFacet),
        (false, true) => Ok(SourceParentOutward::LeftOfCanonicalFacet),
        _ => Err(invalid_artifact(
            "assigned frontier orientation is absent or ambiguous in its parent cell",
        )),
    }
}

fn planar_circular_hole_v2_wire_from_assignments(
    geometry: &CanonicalGeometryV1,
    mesh: &SimplicialMeshEnvelopeV1,
    assignments: &PlanarCircularHoleV2Assignments,
) -> Result<WirePlanarCircularHoleV2CorrespondenceV1, Diagnostic> {
    let faces = vec![WireFaceAssignment {
        geometry_face: 0,
        cell_indices: assignments
            .face_cells()
            .iter()
            .map(|&cell| portable(cell, "mesh cell"))
            .collect::<Result<Vec<_>, _>>()?,
    }];
    let frontiers = assignments
        .frontiers()
        .iter()
        .enumerate()
        .map(|(geometry_edge, lineage)| {
            Ok(WireFrontierAssignment {
                parent_face: 0,
                geometry_edge: portable(geometry_edge, "geometry edge")?,
                facet_indices: lineage
                    .iter()
                    .map(|entry| portable(entry.facet_index, "mesh facet"))
                    .collect::<Result<Vec<_>, _>>()?,
                parent_outward: lineage
                    .iter()
                    .map(|entry| match entry.parent_outward {
                        SourceParentOutward::LeftOfCanonicalFacet => {
                            WireFacetOutward::LeftOfCanonicalFacet
                        }
                        SourceParentOutward::RightOfCanonicalFacet => {
                            WireFacetOutward::RightOfCanonicalFacet
                        }
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    Ok(WirePlanarCircularHoleV2CorrespondenceV1 {
        schema: CORRESPONDENCE_SCHEMA.to_owned(),
        encoding: CANONICAL_ENCODING.to_owned(),
        source: PLANAR_CIRCULAR_HOLE_V2_SOURCE.to_owned(),
        geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
        mesh_sha256: mesh.digest()?.to_string(),
        dimension: 2,
        faces,
        frontiers,
    })
}

fn require_planar_circular_hole_v2(geometry: &CanonicalGeometryV1) -> Result<(), Diagnostic> {
    let replayed = CanonicalGeometryV1::decode_circular_hole_canonical(
        geometry.canonical_bytes(),
        Default::default(),
    )
    .map_err(|_| invalid_artifact("correspondence requires planar circular-hole Geometry v2"))?;
    if replayed != *geometry {
        return Err(invalid_artifact(
            "planar circular-hole Geometry v2 differs from canonical replay",
        ));
    }
    Ok(())
}
impl WirePlanarCircularHoleV2CorrespondenceV1 {
    pub(super) fn validate_local(&self, limits: GeometryDecoderLimits) -> Result<(), Diagnostic> {
        if self.schema != CORRESPONDENCE_SCHEMA
            || self.encoding != CANONICAL_ENCODING
            || self.source != PLANAR_CIRCULAR_HOLE_V2_SOURCE
            || self.dimension != 2
        {
            return Err(invalid_artifact(
                "unsupported planar circular-hole v2 correspondence schema, encoding, source, or dimension",
            ));
        }
        ArtifactDigest::from_hex(self.geometry_sha256.clone())?;
        ArtifactDigest::from_hex(self.mesh_sha256.clone())?;
        if self.faces.len() != 1
            || self.faces[0].geometry_face != 0
            || self.faces[0].cell_indices.is_empty()
            || !strictly_sorted_u64(&self.faces[0].cell_indices)
            || self.frontiers.len() != 5
        {
            return Err(invalid_artifact(
                "planar circular-hole v2 correspondence requires one complete face and five source frontiers",
            ));
        }
        let canonical_frontiers = self.frontiers.iter().enumerate().all(|(edge, frontier)| {
            frontier.parent_face == 0
                && usize::try_from(frontier.geometry_edge) == Ok(edge)
                && !frontier.facet_indices.is_empty()
                && strictly_sorted_u64(&frontier.facet_indices)
                && frontier.facet_indices.len() == frontier.parent_outward.len()
        });
        let unique_facets = self
            .frontiers
            .iter()
            .flat_map(|frontier| frontier.facet_indices.iter().copied())
            .collect::<BTreeSet<_>>();
        let frontier_memberships = self
            .frontiers
            .iter()
            .map(|frontier| frontier.facet_indices.len())
            .sum::<usize>();
        if !canonical_frontiers || unique_facets.len() != frontier_memberships {
            return Err(invalid_artifact(
                "planar circular-hole v2 frontiers must be nonempty, canonical, oriented, and disjoint",
            ));
        }
        let assignment_count = self
            .faces
            .len()
            .checked_add(self.frontiers.len())
            .ok_or_else(|| invalid_artifact("source assignment count overflows usize"))?;
        let membership_count = self.faces[0]
            .cell_indices
            .len()
            .checked_add(frontier_memberships)
            .ok_or_else(|| invalid_artifact("source membership count overflows usize"))?;
        if assignment_count > limits.max_geometry_entities
            || membership_count > limits.max_geometry_mesh_memberships
        {
            return Err(invalid_artifact(
                "planar circular-hole v2 assignments or memberships exceed decoder limits",
            ));
        }
        Ok(())
    }
}
fn resolve_planar_circular_hole_v2_entity_set(
    wire: &WirePlanarCircularHoleV2CorrespondenceV1,
    dimension: usize,
    members: &[usize],
) -> Result<Vec<MeshEntity>, Diagnostic> {
    let mut entities = BTreeSet::new();
    match dimension {
        EDGE_DIMENSION => {
            for &member in members {
                let assignment = wire
                    .frontiers
                    .iter()
                    .find(|entry| usize::try_from(entry.geometry_edge) == Ok(member))
                    .ok_or_else(|| {
                        invalid_artifact(
                            "planar circular-hole v2 source edge set is not fully realized",
                        )
                    })?;
                for &facet in &assignment.facet_indices {
                    entities.insert(MeshEntity::new(EDGE_DIMENSION, local(facet, "mesh facet")?));
                }
            }
        }
        FACE_DIMENSION => {
            for &member in members {
                let assignment = wire
                    .faces
                    .iter()
                    .find(|entry| usize::try_from(entry.geometry_face) == Ok(member))
                    .ok_or_else(|| {
                        invalid_artifact(
                            "planar circular-hole v2 source face set is not fully realized",
                        )
                    })?;
                for &cell in &assignment.cell_indices {
                    entities.insert(MeshEntity::new(FACE_DIMENSION, local(cell, "mesh cell")?));
                }
            }
        }
        _ => {
            return Err(invalid_artifact(
                "planar circular-hole v2 entity set dimension is unsupported",
            ));
        }
    }
    Ok(entities.into_iter().collect())
}

fn strictly_sorted_u64(values: &[u64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn portable(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value)
        .map_err(|_| invalid_artifact(format!("{label} index exceeds portable u64")))
}

fn local(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value)
        .map_err(|_| invalid_artifact(format!("{label} index exceeds local usize")))
}
