//! Direct source correspondence for planar rectangle Geometry v2.

use std::collections::BTreeSet;

use eqiora_core::Diagnostic;
use eqiora_geometry::{CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION};
use eqiora_meshing::{
    CartesianMesh, MeshEntity, MeshQualityGate, MeshTopology, OrientationCode, SimplicialMesh,
};
use serde::{Deserialize, Serialize};

use super::{CORRESPONDENCE_SCHEMA, GeometryMeshCorrespondenceEnvelopeV1, WireCorrespondenceV1};
use crate::{
    AffineTriangleMeshPolicy, ArtifactDigest, CANONICAL_ENCODING, CartesianMeshEnvelopeV1,
    GeometryDecoderLimits, MeshDecoderLimits, SimplicialMeshEnvelopeV1, invalid_artifact,
};

const CARTESIAN_SOURCE: &str = "planar-rectangle-v2-cartesian-v1";
const AFFINE_TRIANGLE_SOURCE: &str = "planar-rectangle-v2-affine-triangle-v1";
const SOURCE_EDGE_COUNT: usize = 4;
const AFFINE_TRIANGLE_MINIMUM_MEAN_RATIO: f64 = f64::MIN_POSITIVE;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WirePlanarRectangleV2CorrespondenceV1 {
    schema: String,
    encoding: String,
    source: String,
    pub(super) geometry_sha256: String,
    pub(super) mesh_sha256: String,
    dimension: u64,
    face: WireFaceAssignment,
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
}

impl GeometryMeshCorrespondenceEnvelopeV1 {
    /// Generate one exact uniform Cartesian Mesh and its direct source correspondence.
    ///
    /// Source edge order is the rectangle-v2 topology order: x-lower,
    /// x-upper, y-lower, y-upper. Facet membership is derived from canonical
    /// topological multi-indices and exact incidence, never from coordinates.
    pub fn from_planar_rectangle_v2_cartesian(
        geometry: &CanonicalGeometryV1,
        cells: [usize; 2],
    ) -> Result<(CartesianMeshEnvelopeV1, Self), Diagnostic> {
        let bounds = require_planar_rectangle_v2(geometry)?;
        if cells.contains(&0) {
            return Err(invalid_artifact(
                "Cartesian rectangle cell counts must be positive",
            ));
        }
        let native = CartesianMesh::uniform(bounds, &cells)?;
        let mesh = CartesianMeshEnvelopeV1::from_mesh(&native)?;
        let wire = wire_from_resources(geometry, &mesh)?;
        let correspondence = Self {
            wire: WireCorrespondenceV1::PlanarRectangleV2(wire),
        };
        correspondence.validate_local(GeometryDecoderLimits::default())?;
        Ok((mesh, correspondence))
    }

    /// Generate one exact affine-triangle Mesh and its direct source correspondence.
    ///
    /// Vertices and structured cells use x-major/y-minor canonical order. Each
    /// cell is split along its lower-left to upper-right diagonal into the
    /// positively oriented triangles `[LL, LR, UR]` then `[LL, UR, UL]`.
    pub(crate) fn from_planar_rectangle_v2_affine_triangles(
        geometry: &CanonicalGeometryV1,
        cells: [usize; 2],
    ) -> Result<(SimplicialMeshEnvelopeV1, Self), Diagnostic> {
        let bounds = require_planar_rectangle_v2(geometry)?;
        let policy = AffineTriangleMeshPolicy::new(cells)?;
        let [nx, ny] = policy.cells();
        let limits = MeshDecoderLimits::default();
        let vertex_count = nx
            .checked_add(1)
            .and_then(|x| ny.checked_add(1).and_then(|y| x.checked_mul(y)))
            .ok_or_else(|| invalid_artifact("affine-triangle vertex count overflows usize"))?;
        let triangle_count = nx
            .checked_mul(ny)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| invalid_artifact("affine-triangle cell count overflows usize"))?;
        if vertex_count > limits.max_mesh_vertices
            || triangle_count > limits.max_mesh_cells
            || vertex_count
                .checked_mul(2)
                .is_none_or(|count| count > limits.max_mesh_coordinate_values)
            || triangle_count
                .checked_mul(3)
                .is_none_or(|count| count > limits.max_mesh_connectivity_indices)
        {
            return Err(invalid_artifact(
                "affine-triangle analytic mesh counts exceed artifact limits",
            ));
        }
        let mut vertices = Vec::with_capacity(vertex_count);
        for ix in 0..=nx {
            let x = subdivided_coordinate(bounds[0], ix, nx)?;
            for iy in 0..=ny {
                vertices.push(vec![x, subdivided_coordinate(bounds[1], iy, ny)?]);
            }
        }
        let vertex = |ix: usize, iy: usize| ix * (ny + 1) + iy;
        let mut triangles = Vec::with_capacity(triangle_count);
        for ix in 0..nx {
            for iy in 0..ny {
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
            vertices,
            triangles,
            MeshQualityGate::new(AFFINE_TRIANGLE_MINIMUM_MEAN_RATIO)
                .expect("minimum positive f64 is an admitted quality threshold"),
        )?;
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&native)?;
        let wire = wire_from_affine_triangle_resources(geometry, &mesh, policy)?;
        let correspondence = Self {
            wire: WireCorrespondenceV1::PlanarRectangleV2(wire),
        };
        correspondence.validate_local(GeometryDecoderLimits::default())?;
        Ok((mesh, correspondence))
    }

    /// Replay the complete rectangle-v2 Cartesian production relation.
    pub fn validate_against_planar_rectangle_v2_cartesian(
        &self,
        geometry: &CanonicalGeometryV1,
        mesh: &CartesianMeshEnvelopeV1,
        cells: [usize; 2],
    ) -> Result<(), Diagnostic> {
        let (expected_mesh, expected) = Self::from_planar_rectangle_v2_cartesian(geometry, cells)?;
        if mesh != &expected_mesh || self != &expected {
            return Err(invalid_artifact(
                "planar rectangle v2 Cartesian resources differ from exact replay",
            ));
        }
        Ok(())
    }

    /// Mesh entities realizing one exact rectangle-v2 entity set.
    pub fn planar_rectangle_v2_entity_set_entities(
        &self,
        geometry: &CanonicalGeometryV1,
        name: &str,
    ) -> Result<Vec<MeshEntity>, Diagnostic> {
        let WireCorrespondenceV1::PlanarRectangleV2(wire) = &self.wire else {
            return Err(invalid_artifact(
                "correspondence has no planar rectangle v2 entity sets",
            ));
        };
        require_planar_rectangle_v2(geometry)?;
        if wire.geometry_sha256 != ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string()
        {
            return Err(invalid_artifact(
                "correspondence geometry digest differs from planar rectangle Geometry v2",
            ));
        }
        let set = geometry.entity_set(name).ok_or_else(|| {
            invalid_artifact(format!(
                "planar rectangle Geometry v2 has no entity set named '{name}'"
            ))
        })?;
        let mut entities = BTreeSet::new();
        match set.dimension() {
            EDGE_DIMENSION => {
                for &member in set.members() {
                    let frontier = wire
                        .frontiers
                        .get(member)
                        .filter(|entry| usize::try_from(entry.geometry_edge) == Ok(member))
                        .ok_or_else(|| {
                            invalid_artifact(
                                "planar rectangle source edge is absent from correspondence",
                            )
                        })?;
                    for &facet in &frontier.facet_indices {
                        entities
                            .insert(MeshEntity::new(EDGE_DIMENSION, local(facet, "mesh facet")?));
                    }
                }
            }
            FACE_DIMENSION => {
                if set.members() != [0] {
                    return Err(invalid_artifact(
                        "planar rectangle face set differs from its exact source face",
                    ));
                }
                for &cell in &wire.face.cell_indices {
                    entities.insert(MeshEntity::new(FACE_DIMENSION, local(cell, "mesh cell")?));
                }
            }
            _ => {
                return Err(invalid_artifact(
                    "planar rectangle entity set dimension is unsupported",
                ));
            }
        }
        Ok(entities.into_iter().collect())
    }
}

fn wire_from_resources(
    geometry: &CanonicalGeometryV1,
    mesh: &CartesianMeshEnvelopeV1,
) -> Result<WirePlanarRectangleV2CorrespondenceV1, Diagnostic> {
    require_planar_rectangle_v2(geometry)?;
    let native = mesh.mesh();
    if native.topological_dimension() != 2 || mesh.dimension() != 2 {
        return Err(invalid_artifact(
            "planar rectangle correspondence requires a two-dimensional Cartesian Mesh",
        ));
    }
    let (cell_indices, frontiers) = topology_assignments(native)?;
    Ok(WirePlanarRectangleV2CorrespondenceV1 {
        schema: CORRESPONDENCE_SCHEMA.to_owned(),
        encoding: CANONICAL_ENCODING.to_owned(),
        source: CARTESIAN_SOURCE.to_owned(),
        geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
        mesh_sha256: mesh.digest()?.to_string(),
        dimension: 2,
        face: WireFaceAssignment {
            geometry_face: 0,
            cell_indices,
        },
        frontiers,
    })
}

fn wire_from_affine_triangle_resources(
    geometry: &CanonicalGeometryV1,
    mesh: &SimplicialMeshEnvelopeV1,
    policy: AffineTriangleMeshPolicy,
) -> Result<WirePlanarRectangleV2CorrespondenceV1, Diagnostic> {
    require_planar_rectangle_v2(geometry)?;
    if mesh.dimension() != 2 || mesh.mesh().topological_dimension() != 2 {
        return Err(invalid_artifact(
            "planar rectangle affine-triangle correspondence requires a two-dimensional Mesh",
        ));
    }
    let (cell_indices, frontiers) =
        affine_triangle_topology_assignments(mesh.mesh(), policy.cells())?;
    Ok(WirePlanarRectangleV2CorrespondenceV1 {
        schema: CORRESPONDENCE_SCHEMA.to_owned(),
        encoding: CANONICAL_ENCODING.to_owned(),
        source: AFFINE_TRIANGLE_SOURCE.to_owned(),
        geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
        mesh_sha256: mesh.digest()?.to_string(),
        dimension: 2,
        face: WireFaceAssignment {
            geometry_face: 0,
            cell_indices,
        },
        frontiers,
    })
}

fn subdivided_coordinate(bounds: [f64; 2], index: usize, cells: usize) -> Result<f64, Diagnostic> {
    if index == 0 {
        return Ok(bounds[0]);
    }
    if index == cells {
        return Ok(bounds[1]);
    }
    let t = index as f64 / cells as f64;
    let value = bounds[0] * (1.0 - t) + bounds[1] * t;
    if !value.is_finite() {
        return Err(invalid_artifact(
            "affine-triangle subdivision produced a non-finite coordinate",
        ));
    }
    Ok(value)
}

fn affine_triangle_topology_assignments(
    mesh: &SimplicialMesh,
    cells: [usize; 2],
) -> Result<(Vec<u64>, Vec<WireFrontierAssignment>), Diagnostic> {
    let [nx, ny] = cells;
    let expected_cells = nx
        .checked_mul(ny)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| invalid_artifact("affine-triangle cell count overflows usize"))?;
    if mesh.entity_count(FACE_DIMENSION) != Some(expected_cells) {
        return Err(invalid_artifact(
            "affine-triangle Mesh cell count differs from its policy",
        ));
    }
    let facet_count = mesh
        .entity_count(EDGE_DIMENSION)
        .ok_or_else(|| invalid_artifact("affine-triangle Mesh has no edge stratum"))?;
    let mut facets_by_vertices = std::collections::BTreeMap::new();
    let mut exposed = BTreeSet::new();
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(EDGE_DIMENSION, facet_index);
        let vertices = mesh
            .entity_vertices(facet)
            .ok_or_else(|| invalid_artifact("affine-triangle facet has no vertex closure"))?
            .into_iter()
            .map(MeshEntity::index)
            .collect::<Vec<_>>();
        if vertices.len() != 2 || facets_by_vertices.insert(vertices, facet_index).is_some() {
            return Err(invalid_artifact(
                "affine-triangle facets require unique canonical segment closures",
            ));
        }
        let adjacent = mesh
            .incidence(facet, FACE_DIMENSION)
            .ok_or_else(|| invalid_artifact("affine-triangle facet incidence is unavailable"))?;
        if adjacent.len() == 1 {
            exposed.insert(facet_index);
        } else if adjacent.len() != 2 {
            return Err(invalid_artifact(
                "affine-triangle facet requires one or two parent cells",
            ));
        }
    }

    let vertex = |ix: usize, iy: usize| ix * (ny + 1) + iy;
    let boundary_pairs: [Vec<[usize; 2]>; SOURCE_EDGE_COUNT] = [
        (0..ny)
            .map(|iy| [vertex(0, iy), vertex(0, iy + 1)])
            .collect(),
        (0..ny)
            .map(|iy| [vertex(nx, iy), vertex(nx, iy + 1)])
            .collect(),
        (0..nx)
            .map(|ix| [vertex(ix, 0), vertex(ix + 1, 0)])
            .collect(),
        (0..nx)
            .map(|ix| [vertex(ix, ny), vertex(ix + 1, ny)])
            .collect(),
    ];
    let mut assigned = BTreeSet::new();
    let mut frontiers = Vec::with_capacity(SOURCE_EDGE_COUNT);
    for (source_edge, pairs) in boundary_pairs.into_iter().enumerate() {
        let mut facet_indices = Vec::with_capacity(pairs.len());
        for mut pair in pairs {
            pair.sort_unstable();
            let facet = *facets_by_vertices.get(pair.as_slice()).ok_or_else(|| {
                invalid_artifact("affine-triangle source boundary segment is absent")
            })?;
            if !exposed.contains(&facet) || !assigned.insert(facet) {
                return Err(invalid_artifact(
                    "affine-triangle source boundary membership is missing or duplicated",
                ));
            }
            validate_affine_boundary_incidence(mesh, facet, source_edge)?;
            facet_indices.push(portable(facet, "mesh facet")?);
        }
        facet_indices.sort_unstable();
        frontiers.push(WireFrontierAssignment {
            parent_face: 0,
            geometry_edge: portable(source_edge, "geometry edge")?,
            facet_indices,
        });
    }
    if assigned != exposed {
        return Err(invalid_artifact(
            "affine-triangle frontiers do not completely and exclusively cover the boundary",
        ));
    }
    let cell_indices = (0..expected_cells)
        .map(|cell| portable(cell, "mesh cell"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((cell_indices, frontiers))
}

fn validate_affine_boundary_incidence(
    mesh: &SimplicialMesh,
    facet_index: usize,
    source_edge: usize,
) -> Result<(), Diagnostic> {
    let facet = MeshEntity::new(EDGE_DIMENSION, facet_index);
    let adjacent = mesh
        .incidence(facet, FACE_DIMENSION)
        .ok_or_else(|| invalid_artifact("affine-triangle boundary incidence is unavailable"))?;
    let [parent] = adjacent.as_slice() else {
        return Err(invalid_artifact(
            "affine-triangle boundary facet requires exactly one parent cell",
        ));
    };
    let expected_ordinal = [1, 2, 0, 2][source_edge];
    if parent.local_ordinal != expected_ordinal {
        return Err(invalid_artifact(
            "affine-triangle boundary local incidence differs from the fixed diagonal convention",
        ));
    }
    let permutation = mesh
        .orientation_permutation(parent.orientation, 2)
        .ok_or_else(|| invalid_artifact("affine-triangle boundary orientation is unavailable"))?;
    let expected = if source_edge == 3 { [1, 0] } else { [0, 1] };
    if permutation.images() != expected {
        return Err(invalid_artifact(
            "affine-triangle boundary orientation differs from canonical source order",
        ));
    }
    Ok(())
}

trait RectangleCartesianTopology {
    fn entity_count(&self, dimension: usize) -> Option<usize>;
    fn axis_cell_count(&self, axis: usize) -> Option<usize>;
    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<eqiora_meshing::EntityIncidence>>;
    fn entity_vertices(&self, entity: MeshEntity) -> Option<Vec<MeshEntity>>;
    fn vertex_multi_index(&self, vertex: MeshEntity) -> Option<&[usize]>;
}

impl RectangleCartesianTopology for CartesianMesh {
    fn entity_count(&self, dimension: usize) -> Option<usize> {
        MeshTopology::entity_count(self, dimension)
    }

    fn axis_cell_count(&self, axis: usize) -> Option<usize> {
        CartesianMesh::axis_cell_count(self, axis)
    }

    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<eqiora_meshing::EntityIncidence>> {
        MeshTopology::incidence(self, entity, target_dimension)
    }

    fn entity_vertices(&self, entity: MeshEntity) -> Option<Vec<MeshEntity>> {
        CartesianMesh::entity_vertices(self, entity)
    }

    fn vertex_multi_index(&self, vertex: MeshEntity) -> Option<&[usize]> {
        CartesianMesh::vertex_multi_index(self, vertex)
    }
}

fn topology_assignments(
    native: &impl RectangleCartesianTopology,
) -> Result<(Vec<u64>, Vec<WireFrontierAssignment>), Diagnostic> {
    let cell_count = native
        .entity_count(FACE_DIMENSION)
        .ok_or_else(|| invalid_artifact("Cartesian Mesh has no face stratum"))?;
    let facet_count = native
        .entity_count(EDGE_DIMENSION)
        .ok_or_else(|| invalid_artifact("Cartesian Mesh has no edge stratum"))?;
    let cells = [
        native
            .axis_cell_count(0)
            .ok_or_else(|| invalid_artifact("Cartesian Mesh omitted x-axis cells"))?,
        native
            .axis_cell_count(1)
            .ok_or_else(|| invalid_artifact("Cartesian Mesh omitted y-axis cells"))?,
    ];
    let mut source_facets: [Vec<usize>; SOURCE_EDGE_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut exposed = BTreeSet::new();
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(EDGE_DIMENSION, facet_index);
        let adjacent = native
            .incidence(facet, FACE_DIMENSION)
            .ok_or_else(|| invalid_artifact("Cartesian frontier incidence is unavailable"))?;
        if adjacent.len() == 2 {
            continue;
        }
        let [parent] = adjacent.as_slice() else {
            return Err(invalid_artifact(
                "Cartesian frontier requires exactly one parent-cell incidence",
            ));
        };
        if parent.orientation != OrientationCode::identity() || parent.local_ordinal >= 4 {
            return Err(invalid_artifact(
                "Cartesian frontier has noncanonical local-side orientation",
            ));
        }
        let vertices = native
            .entity_vertices(facet)
            .ok_or_else(|| invalid_artifact("Cartesian frontier has no vertex closure"))?;
        let source_edge = structural_source_edge(native, &vertices, cells)?;
        let expected_local_side = [2, 3, 0, 1][source_edge];
        if parent.local_ordinal != expected_local_side {
            return Err(invalid_artifact(
                "Cartesian frontier local side differs from its source rectangle edge",
            ));
        }
        source_facets[source_edge].push(facet_index);
        exposed.insert(facet_index);
    }
    let assigned = source_facets
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    if source_facets.iter().any(Vec::is_empty) || assigned != exposed {
        return Err(invalid_artifact(
            "Cartesian rectangle frontiers must completely and exclusively realize four source edges",
        ));
    }
    let frontiers = source_facets
        .into_iter()
        .enumerate()
        .map(|(geometry_edge, facets)| {
            Ok(WireFrontierAssignment {
                parent_face: 0,
                geometry_edge: portable(geometry_edge, "geometry edge")?,
                facet_indices: facets
                    .into_iter()
                    .map(|facet| portable(facet, "mesh facet"))
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let cell_indices = (0..cell_count)
        .map(|cell| portable(cell, "mesh cell"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((cell_indices, frontiers))
}

fn structural_source_edge(
    mesh: &impl RectangleCartesianTopology,
    vertices: &[MeshEntity],
    cells: [usize; 2],
) -> Result<usize, Diagnostic> {
    if vertices.len() != 2 {
        return Err(invalid_artifact(
            "Cartesian rectangle frontier must be a segment",
        ));
    }
    let indices = vertices
        .iter()
        .map(|&vertex| {
            mesh.vertex_multi_index(vertex)
                .map(<[usize]>::to_vec)
                .ok_or_else(|| invalid_artifact("Cartesian frontier vertex index is unavailable"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for axis in 0..2 {
        if indices.iter().all(|index| index[axis] == indices[0][axis]) {
            return match (axis, indices[0][axis]) {
                (0, 0) => Ok(0),
                (0, value) if value == cells[0] => Ok(1),
                (1, 0) => Ok(2),
                (1, value) if value == cells[1] => Ok(3),
                _ => Err(invalid_artifact(
                    "Cartesian exposed facet is not on a canonical rectangle side",
                )),
            };
        }
    }
    Err(invalid_artifact(
        "Cartesian rectangle frontier has no fixed structural axis",
    ))
}

fn require_planar_rectangle_v2(
    geometry: &CanonicalGeometryV1,
) -> Result<&[[f64; 2]; 2], Diagnostic> {
    let replayed = CanonicalGeometryV1::decode_planar_rectangle_v2_canonical(
        geometry.canonical_bytes(),
        Default::default(),
    )
    .map_err(|_| invalid_artifact("CartesianMesher requires planar rectangle Geometry v2"))?;
    if replayed != *geometry {
        return Err(invalid_artifact(
            "planar rectangle Geometry v2 differs from canonical replay",
        ));
    }
    geometry
        .planar_rectangle_bounds()
        .ok_or_else(|| invalid_artifact("CartesianMesher requires planar rectangle Geometry v2"))
}

impl WirePlanarRectangleV2CorrespondenceV1 {
    pub(super) fn refine(
        &mut self,
        source: &SimplicialMesh,
        target: &SimplicialMesh,
        overlap: &eqiora_meshing::SimplicialRevisionOverlap2d,
    ) -> Result<(), Diagnostic> {
        let face = super::affine_triangles::RefinedFace::new(
            source,
            target,
            overlap,
            &self.face.cell_indices,
            self.frontiers
                .iter()
                .flat_map(|frontier| frontier.facet_indices.iter().copied()),
        )?;
        self.face.cell_indices = face.cells()?;
        for frontier in &mut self.frontiers {
            frontier.facet_indices = face
                .facets(&frontier.facet_indices)?
                .iter()
                .map(|fragment| portable(fragment.target_facet().index(), "refined facet"))
                .collect::<Result<_, _>>()?;
        }
        Ok(())
    }

    pub(super) fn validate_local(&self, limits: GeometryDecoderLimits) -> Result<(), Diagnostic> {
        if self.schema != CORRESPONDENCE_SCHEMA
            || self.encoding != CANONICAL_ENCODING
            || !matches!(
                self.source.as_str(),
                CARTESIAN_SOURCE | AFFINE_TRIANGLE_SOURCE
            )
            || self.dimension != 2
        {
            return Err(invalid_artifact(
                "unsupported planar rectangle v2 correspondence schema, encoding, source, or dimension",
            ));
        }
        ArtifactDigest::from_hex(self.geometry_sha256.clone())?;
        ArtifactDigest::from_hex(self.mesh_sha256.clone())?;
        if self.face.geometry_face != 0
            || self.face.cell_indices.is_empty()
            || !strictly_sorted(&self.face.cell_indices)
            || self.frontiers.len() != SOURCE_EDGE_COUNT
        {
            return Err(invalid_artifact(
                "planar rectangle correspondence requires one complete face and four source frontiers",
            ));
        }
        let canonical = self.frontiers.iter().enumerate().all(|(edge, frontier)| {
            frontier.parent_face == 0
                && usize::try_from(frontier.geometry_edge) == Ok(edge)
                && !frontier.facet_indices.is_empty()
                && strictly_sorted(&frontier.facet_indices)
        });
        let unique = self
            .frontiers
            .iter()
            .flat_map(|frontier| frontier.facet_indices.iter().copied())
            .collect::<BTreeSet<_>>();
        let frontier_memberships = self
            .frontiers
            .iter()
            .map(|frontier| frontier.facet_indices.len())
            .sum::<usize>();
        let memberships = self
            .face
            .cell_indices
            .len()
            .checked_add(frontier_memberships)
            .ok_or_else(|| invalid_artifact("rectangle membership count overflows usize"))?;
        if !canonical
            || unique.len() != frontier_memberships
            || SOURCE_EDGE_COUNT + 1 > limits.max_geometry_entities
            || memberships > limits.max_geometry_mesh_memberships
        {
            return Err(invalid_artifact(
                "planar rectangle assignments must be canonical, disjoint, and within limits",
            ));
        }
        Ok(())
    }
}

fn strictly_sorted(values: &[u64]) -> bool {
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

#[cfg(test)]
#[path = "geometry_mesh_correspondence_planar_rectangle_v2/tests.rs"]
mod tests;
