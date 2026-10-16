//! Geometry-derived sources for the geometry-mesh correspondence artifact.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_geometry::{
    CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION, PlanarFace, PlanarRegion, VERTEX_DIMENSION,
};
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshQualityGate, MeshTopology, SimplicialMesh,
};
use eqiora_schema::kernel::BoundarySide;
use serde::{Deserialize, Serialize};

use super::{CORRESPONDENCE_SCHEMA, GeometryMeshCorrespondenceEnvelopeV1, WireCorrespondenceV1};
use crate::{
    AffineTriangleMeshPolicy, ArtifactDigest, CANONICAL_ENCODING, GeometryDecoderLimits,
    GeometryDefinitionV1, SimplicialMeshEnvelopeV1, invalid_artifact,
};

const AUTHORED_REGION_SOURCE: &str = "authored-planar-region-v1";
const ADJACENT_PARTITION_SOURCE: &str = "adjacent-rectangle-partition-affine-triangle-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireAuthoredRegionCorrespondenceV1 {
    schema: String,
    encoding: String,
    source: String,
    pub(super) geometry_sha256: String,
    pub(super) mesh_sha256: String,
    dimension: u64,
    vertices: Vec<WireVertexAssignment>,
    faces: Vec<WireFaceAssignment>,
    frontiers: Vec<WireFrontierAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireVertexAssignment {
    geometry_vertex: u64,
    mesh_vertex: u64,
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

#[derive(Clone, Copy)]
struct RegionEdge {
    index: usize,
    parent_face: usize,
    start: [f64; 2],
    end: [f64; 2],
}

struct GeneratedAssignments {
    vertices: Vec<WireVertexAssignment>,
    faces: Vec<WireFaceAssignment>,
    frontiers: Vec<WireFrontierAssignment>,
}

impl GeometryMeshCorrespondenceEnvelopeV1 {
    /// Generate the bounded two-region adjacent-rectangle affine-triangle occurrence.
    pub(crate) fn from_adjacent_rectangle_partition_affine_triangles(
        geometry: &CanonicalGeometryV1,
        cells: [usize; 2],
    ) -> Result<(SimplicialMeshEnvelopeV1, Self), Diagnostic> {
        let policy = AffineTriangleMeshPolicy::new(cells)?;
        if policy.cells() != [2, 2] {
            return Err(invalid_artifact(
                "the adjacent-rectangle partition requires exactly a 2 by 2 affine subdivision",
            ));
        }
        let region = require_adjacent_rectangle_partition(geometry)?;
        let vertices = region.vertices();
        let xs = [vertices[0][0], vertices[2][0], vertices[4][0]];
        let ys = [
            vertices[0][1],
            (vertices[0][1] + vertices[1][1]) * 0.5,
            vertices[1][1],
        ];
        if !ys[1].is_finite() {
            return Err(invalid_artifact("partition midpoint is non-finite"));
        }
        let mut mesh_vertices = Vec::with_capacity(9);
        for x in xs {
            for y in ys {
                mesh_vertices.push(vec![x, y]);
            }
        }
        let vertex = |ix: usize, iy: usize| ix * 3 + iy;
        let mut triangles = Vec::with_capacity(8);
        for ix in 0..2 {
            for iy in 0..2 {
                let ll = vertex(ix, iy);
                let lr = vertex(ix + 1, iy);
                let ul = vertex(ix, iy + 1);
                let ur = vertex(ix + 1, iy + 1);
                triangles.push(vec![ll, lr, ur]);
                triangles.push(vec![ll, ur, ul]);
            }
        }
        let native = SimplicialMesh::new(
            2,
            mesh_vertices,
            triangles,
            MeshQualityGate::new(f64::MIN_POSITIVE)
                .expect("minimum positive f64 is an admitted quality threshold"),
        )?;
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&native)?;
        let assignments = adjacent_partition_assignments();
        let correspondence = Self {
            wire: WireCorrespondenceV1::AuthoredRegion(WireAuthoredRegionCorrespondenceV1 {
                schema: CORRESPONDENCE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source: ADJACENT_PARTITION_SOURCE.to_owned(),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                dimension: 2,
                vertices: assignments.vertices,
                faces: assignments.faces,
                frontiers: assignments.frontiers,
            }),
        };
        correspondence.validate_local(GeometryDecoderLimits::default())?;
        Ok((mesh, correspondence))
    }

    /// Mesh entities realizing one exact named set of the adjacent partition.
    pub fn adjacent_rectangle_partition_entity_set_entities(
        &self,
        geometry: &CanonicalGeometryV1,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        require_adjacent_rectangle_partition(geometry)?;
        self.canonical_region_entity_set_entities(geometry, name)
    }

    /// Generate the unique complete correspondence for one authored planar
    /// region and one conforming affine-triangle mesh.
    ///
    /// Mesh labels are not inputs. Face ownership, parent-relative frontier
    /// facets, geometry vertices, and every parent-outward direction are
    /// discovered from coordinates and incidence, then replay-validated.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the mesh is a conforming triangulation of the
    /// exact region, every region entity is realized, and named entity sets
    /// are unambiguous.
    pub fn from_region(
        geometry: &GeometryDefinitionV1,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let region = geometry.region()?;
        let assignments = generate_assignments(&region, mesh.mesh())?;
        let envelope = Self {
            wire: WireCorrespondenceV1::AuthoredRegion(WireAuthoredRegionCorrespondenceV1 {
                schema: CORRESPONDENCE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source: AUTHORED_REGION_SOURCE.to_owned(),
                geometry_sha256: geometry.digest()?.to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                dimension: 2,
                vertices: assignments.vertices,
                faces: assignments.faces,
                frontiers: assignments.frontiers,
            }),
        };
        envelope.validate_against_region(geometry, mesh)?;
        Ok(envelope)
    }

    fn canonical_region_entity_set_entities(
        &self,
        geometry: &CanonicalGeometryV1,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        let WireCorrespondenceV1::AuthoredRegion(wire) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence has no authored region entity sets",
            ));
        };
        if wire.geometry_sha256 != ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string()
        {
            return Err(invalid_artifact(
                "correspondence geometry digest differs from Geometry",
            ));
        }
        let region = geometry
            .region()
            .ok_or_else(|| invalid_artifact("Geometry is not a straight-edged planar region"))?;
        let set = region
            .entity_set(name)
            .ok_or_else(|| invalid_artifact(format!("region has no entity set named '{name}'")))?;
        resolve_entity_set(wire, set.dimension(), set.members())
    }

    /// Replay and validate an authored-region correspondence against its exact
    /// geometry and mesh resources.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale digests, changed membership, missing or
    /// relabelled facets, wrong orientation, or a nonconforming mesh.
    pub fn validate_against_region(
        &self,
        geometry: &GeometryDefinitionV1,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        self.validate_local(GeometryDecoderLimits::default())?;
        let WireCorrespondenceV1::AuthoredRegion(actual) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence was not derived from an authored planar region",
            ));
        };
        let region = geometry.region()?;
        let assignments = generate_assignments(&region, mesh.mesh())?;
        let expected = WireAuthoredRegionCorrespondenceV1 {
            schema: CORRESPONDENCE_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            source: AUTHORED_REGION_SOURCE.to_owned(),
            geometry_sha256: geometry.digest()?.to_string(),
            mesh_sha256: mesh.digest()?.to_string(),
            dimension: 2,
            vertices: assignments.vertices,
            faces: assignments.faces,
            frontiers: assignments.frontiers,
        };
        if actual != &expected {
            return Err(invalid_artifact(
                "authored-region correspondence differs from exact geometry and mesh replay",
            ));
        }
        Ok(())
    }

    /// Mesh entities that exactly realize one named region entity set.
    ///
    /// The result has the set's dimension and canonical mesh-entity order.
    /// Exact resource replay remains the caller's admission step after
    /// decoding untrusted correspondence bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` when this is not an authored-region correspondence,
    /// the geometry digest differs, or the set is absent or incomplete.
    pub fn region_entity_set_entities(
        &self,
        geometry: &GeometryDefinitionV1,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        let WireCorrespondenceV1::AuthoredRegion(wire) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence has no authored region entity sets",
            ));
        };
        if wire.geometry_sha256 != geometry.digest()?.to_string() {
            return Err(invalid_artifact(
                "correspondence geometry digest differs from the region artifact",
            ));
        }
        let region = geometry.region()?;
        let set = region
            .entity_set(name)
            .ok_or_else(|| invalid_artifact(format!("region has no entity set named '{name}'")))?;
        resolve_entity_set(wire, set.dimension(), set.members())
    }
}

fn adjacent_partition_assignments() -> GeneratedAssignments {
    GeneratedAssignments {
        vertices: (0_u64..)
            .zip([0_u64, 2, 3, 5, 6, 8])
            .map(|(geometry_vertex, mesh_vertex)| WireVertexAssignment {
                mesh_vertex,
                geometry_vertex,
            })
            .collect(),
        faces: vec![
            WireFaceAssignment {
                geometry_face: 0,
                cell_indices: vec![0, 1, 2, 3],
            },
            WireFaceAssignment {
                geometry_face: 1,
                cell_indices: vec![4, 5, 6, 7],
            },
        ],
        frontiers: vec![
            frontier(0, 0, &[1], WireFacetOutward::RightOfCanonicalFacet),
            frontier(0, 1, &[7, 10], WireFacetOutward::RightOfCanonicalFacet),
            frontier(0, 2, &[6], WireFacetOutward::LeftOfCanonicalFacet),
            frontier(0, 3, &[0, 3], WireFacetOutward::LeftOfCanonicalFacet),
            frontier(1, 4, &[8], WireFacetOutward::RightOfCanonicalFacet),
            frontier(1, 5, &[14, 15], WireFacetOutward::RightOfCanonicalFacet),
            frontier(1, 6, &[13], WireFacetOutward::LeftOfCanonicalFacet),
            frontier(1, 7, &[7, 10], WireFacetOutward::LeftOfCanonicalFacet),
        ],
    }
}

fn frontier(
    parent_face: u64,
    geometry_edge: u64,
    facet_indices: &[u64],
    parent_outward: WireFacetOutward,
) -> WireFrontierAssignment {
    WireFrontierAssignment {
        parent_face,
        geometry_edge,
        facet_indices: facet_indices.to_vec(),
        parent_outward: vec![parent_outward; facet_indices.len()],
    }
}

fn require_adjacent_rectangle_partition(
    geometry: &CanonicalGeometryV1,
) -> Result<&PlanarRegion, Diagnostic> {
    let (bounds, interface_x) = geometry
        .planar_adjacent_rectangle_partition()
        .ok_or_else(|| invalid_artifact("Geometry is not an adjacent rectangle partition"))?;
    let region = geometry
        .region()
        .ok_or_else(|| invalid_artifact("Geometry is not an adjacent rectangle partition"))?;
    let vertices = region.vertices();
    let faces = region.faces();
    if vertices.len() != 6
        || faces.len() != 2
        || faces.iter().any(|face| !face.holes().is_empty())
        || faces[0].outer() != [0, 2, 3, 1]
        || faces[1].outer() != [2, 4, 5, 3]
        || vertices[0][0] != vertices[1][0]
        || vertices[2][0] != vertices[3][0]
        || vertices[4][0] != vertices[5][0]
        || !(vertices[0][0] < vertices[2][0] && vertices[2][0] < vertices[4][0])
        || vertices[0][1] != vertices[2][1]
        || vertices[2][1] != vertices[4][1]
        || vertices[1][1] != vertices[3][1]
        || vertices[3][1] != vertices[5][1]
        || vertices[0][1] >= vertices[1][1]
        || bounds
            != &[
                [vertices[0][0], vertices[4][0]],
                [vertices[0][1], vertices[1][1]],
            ]
        || interface_x != vertices[2][0]
        || (bounds[0][0] + bounds[0][1]) * 0.5 != interface_x
    {
        return Err(invalid_artifact(
            "Geometry is not the bounded equal-width adjacent two-rectangle partition",
        ));
    }
    let covered = region
        .entity_sets()
        .iter()
        .flat_map(|set| {
            set.members()
                .iter()
                .map(move |&member| (set.dimension(), member))
        })
        .collect::<BTreeSet<_>>();
    let expected = (0..8)
        .map(|edge| (EDGE_DIMENSION, edge))
        .chain((0..2).map(|face| (FACE_DIMENSION, face)))
        .collect::<BTreeSet<_>>();
    let memberships = region
        .entity_sets()
        .iter()
        .map(|set| set.members().len())
        .sum::<usize>();
    if covered != expected || memberships != expected.len() {
        return Err(invalid_artifact(
            "adjacent partition named topology must cover every region and parent-relative boundary exactly once",
        ));
    }
    Ok(region)
}

impl WireAuthoredRegionCorrespondenceV1 {
    pub(super) fn refine(
        &mut self,
        source: &SimplicialMesh,
        target: &SimplicialMesh,
        overlap: &eqiora_meshing::SimplicialRevisionOverlap2d,
    ) -> Result<(), Diagnostic> {
        for face in &mut self.faces {
            let refined = super::affine_triangles::RefinedFace::new(
                source,
                target,
                overlap,
                &face.cell_indices,
                self.frontiers
                    .iter()
                    .filter(|frontier| frontier.parent_face == face.geometry_face)
                    .flat_map(|frontier| frontier.facet_indices.iter().copied()),
            )?;
            face.cell_indices = refined.cells()?;
            for frontier in self
                .frontiers
                .iter_mut()
                .filter(|frontier| frontier.parent_face == face.geometry_face)
            {
                let fragments = refined.facets(&frontier.facet_indices)?;
                let facets = fragments
                    .iter()
                    .map(|fragment| {
                        super::affine_triangles::portable(fragment.target_facet().index())
                    })
                    .collect::<Result<_, _>>()?;
                let orientations = fragments
                    .iter()
                    .map(|fragment| {
                        let vertices = target
                            .entity_vertices(MeshEntity::new(1, fragment.target_facet().index()))
                            .expect("accepted edge vertices");
                        let a = &target.vertices()[vertices[0].index()];
                        let b = &target.vertices()[vertices[1].index()];
                        let normal = fragment.target_outward_normal();
                        // Project the already proved parent-outward normal onto the
                        // left normal of the canonical (vertex-index ordered) edge.
                        let left = (a[1] - b[1]).mul_add(normal[0], (b[0] - a[0]) * normal[1]);
                        if !left.is_finite() || left == 0.0 {
                            return Err(invalid_artifact(
                                "refined facet orientation is not representable",
                            ));
                        }
                        Ok(if left > 0.0 {
                            WireFacetOutward::LeftOfCanonicalFacet
                        } else {
                            WireFacetOutward::RightOfCanonicalFacet
                        })
                    })
                    .collect::<Result<_, Diagnostic>>()?;
                frontier.facet_indices = facets;
                frontier.parent_outward = orientations;
            }
        }
        Ok(())
    }

    pub(super) fn validate_local(&self, limits: GeometryDecoderLimits) -> Result<(), Diagnostic> {
        if self.schema != CORRESPONDENCE_SCHEMA
            || self.encoding != CANONICAL_ENCODING
            || (self.source != AUTHORED_REGION_SOURCE && self.source != ADJACENT_PARTITION_SOURCE)
            || self.dimension != 2
        {
            return Err(invalid_artifact(
                "unsupported authored-region correspondence schema, encoding, source, or dimension",
            ));
        }
        ArtifactDigest::from_hex(self.geometry_sha256.clone())?;
        ArtifactDigest::from_hex(self.mesh_sha256.clone())?;
        if self.vertices.is_empty() || self.faces.is_empty() || self.frontiers.is_empty() {
            return Err(invalid_artifact(
                "authored-region correspondence requires vertices, faces, and frontiers",
            ));
        }
        let assignment_count = self
            .vertices
            .len()
            .checked_add(self.faces.len())
            .and_then(|count| count.checked_add(self.frontiers.len()))
            .ok_or_else(|| invalid_artifact("region assignment count overflows usize"))?;
        if assignment_count > limits.max_geometry_entities {
            return Err(invalid_artifact(
                "authored-region assignment count exceeds decoder limits",
            ));
        }
        validate_canonical_assignments(self)?;
        let memberships = self
            .faces
            .iter()
            .map(|face| face.cell_indices.len())
            .chain(
                self.frontiers
                    .iter()
                    .map(|frontier| frontier.facet_indices.len()),
            )
            .try_fold(self.vertices.len(), |total, count| total.checked_add(count))
            .ok_or_else(|| invalid_artifact("region membership count overflows usize"))?;
        if memberships > limits.max_geometry_mesh_memberships {
            return Err(invalid_artifact(
                "authored-region memberships exceed decoder limits",
            ));
        }
        Ok(())
    }
}

fn validate_canonical_assignments(
    wire: &WireAuthoredRegionCorrespondenceV1,
) -> Result<(), Diagnostic> {
    if !wire
        .vertices
        .windows(2)
        .all(|pair| pair[0].geometry_vertex < pair[1].geometry_vertex)
        || wire
            .vertices
            .iter()
            .map(|entry| entry.mesh_vertex)
            .collect::<BTreeSet<_>>()
            .len()
            != wire.vertices.len()
        || !wire
            .faces
            .windows(2)
            .all(|pair| pair[0].geometry_face < pair[1].geometry_face)
        || !wire.frontiers.windows(2).all(|pair| {
            (pair[0].parent_face, pair[0].geometry_edge)
                < (pair[1].parent_face, pair[1].geometry_edge)
        })
    {
        return Err(invalid_artifact(
            "authored-region assignments must be unique and canonical",
        ));
    }
    if wire
        .faces
        .iter()
        .any(|face| face.cell_indices.is_empty() || !strictly_sorted_u64(&face.cell_indices))
        || wire.frontiers.iter().any(|frontier| {
            frontier.facet_indices.is_empty()
                || !strictly_sorted_u64(&frontier.facet_indices)
                || frontier.facet_indices.len() != frontier.parent_outward.len()
        })
    {
        return Err(invalid_artifact(
            "authored-region memberships must be nonempty, canonical, and oriented",
        ));
    }
    Ok(())
}

fn strictly_sorted_u64(values: &[u64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn generate_assignments(
    region: &PlanarRegion,
    mesh: &SimplicialMesh,
) -> Result<GeneratedAssignments, Diagnostic> {
    if mesh.topological_dimension() != FACE_DIMENSION {
        return Err(invalid_artifact(
            "authored planar region requires a two-dimensional triangle mesh",
        ));
    }
    reject_ambiguous_entity_sets(region)?;
    let edges = region_edges(region);
    let vertices = assign_vertices(region, mesh)?;
    let (faces, cell_owners) = assign_cells(region, mesh)?;
    let frontiers = assign_frontiers(region, mesh, &edges, &cell_owners)?;
    Ok(GeneratedAssignments {
        vertices,
        faces,
        frontiers,
    })
}

#[path = "geometry_mesh_correspondence_sources/geometry.rs"]
mod geometry;

use geometry::{
    assign_cells, assign_frontiers, assign_vertices, region_edges, reject_ambiguous_entity_sets,
    resolve_entity_set,
};
pub(super) use geometry::{
    cartesian_facet_roles, entity_coordinates, point_inside_cartesian, validate_parent_outward_cell,
};

#[cfg(test)]
#[path = "geometry_mesh_correspondence_sources/adjacent_partition_tests.rs"]
mod adjacent_partition_tests;
