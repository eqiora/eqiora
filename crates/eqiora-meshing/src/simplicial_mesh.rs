//! Fixed-connectivity affine simplex meshes and acceptance-quality evidence.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    AffineGeometryLinearization, AffineGeometryMap, EntityIncidence, MeshEntity, MeshGeometry,
    MeshTopology, OrientationCode, ReferenceCell, ReferenceTopology, VertexPermutation,
};

mod boundary;

const MAX_MESH_ENTITIES: usize = 8_000_000;

/// Fail-closed conditioning policy for accepted affine simplex cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshQualityGate {
    minimum_mean_ratio: f64,
}

impl MeshQualityGate {
    /// Require every cell to have positive orientation and at least this
    /// scale-invariant mean-ratio quality.
    ///
    /// # Errors
    /// Returns `EQ0803` unless the threshold lies in `(0, 1]`.
    pub fn new(minimum_mean_ratio: f64) -> Result<Self, Diagnostic> {
        if !minimum_mean_ratio.is_finite() || minimum_mean_ratio <= 0.0 || minimum_mean_ratio > 1.0
        {
            return Err(invalid_mesh(
                "simplex mesh quality threshold must be finite and lie in (0, 1]",
            ));
        }
        Ok(Self { minimum_mean_ratio })
    }

    /// Minimum accepted scale-invariant cell quality.
    #[must_use]
    pub const fn minimum_mean_ratio(self) -> f64 {
        self.minimum_mean_ratio
    }
}

/// Conditioning evidence accumulated while a mesh revision is accepted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshQualityReport {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

impl MeshQualityReport {
    /// Minimum scale-invariant mean-ratio quality over all cells.
    #[must_use]
    pub const fn minimum_mean_ratio(self) -> f64 {
        self.minimum_mean_ratio
    }

    /// Minimum positive signed cell measure scale over all cells.
    #[must_use]
    pub const fn minimum_signed_measure_scale(self) -> f64 {
        self.minimum_signed_measure_scale
    }
}

/// Runtime-dimensional, fixed-connectivity mesh of affine simplex cells.
///
/// Topology is stored independently from coordinates. Cell vertex order is
/// retained because it defines the affine map orientation; all lower strata
/// are deduplicated by sorted vertex closure. The constructor rejects duplicate
/// cells, isolated vertices, non-manifold facets, inverted cells, and cells
/// below the requested quality gate.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMesh {
    dimension: usize,
    vertices: Vec<Vec<f64>>,
    cells: Vec<Vec<usize>>,
    strata: Vec<Vec<Vec<usize>>>,
    lookups: Vec<BTreeMap<Vec<usize>, usize>>,
    boundary_entities: Vec<Vec<bool>>,
    orientation_codes: BTreeMap<Vec<usize>, u32>,
    quality_gate: MeshQualityGate,
    quality_report: MeshQualityReport,
}

impl SimplicialMesh {
    /// Validate and construct one full-dimensional affine simplex mesh.
    ///
    /// # Errors
    /// Returns `EQ0803` for invalid dimensions/connectivity, duplicate or
    /// non-manifold topology, non-finite coordinates, inverted/degenerate
    /// cells, or a quality-gate violation.
    pub fn new(
        dimension: usize,
        vertices: Vec<Vec<f64>>,
        cells: Vec<Vec<usize>>,
        quality_gate: MeshQualityGate,
    ) -> Result<Self, Diagnostic> {
        if dimension == 0 || vertices.is_empty() || cells.is_empty() {
            return Err(invalid_mesh(
                "simplex mesh requires positive dimension, vertices, and cells",
            ));
        }
        if vertices.iter().any(|vertex| {
            vertex.len() != dimension || vertex.iter().any(|coordinate| !coordinate.is_finite())
        }) {
            return Err(invalid_mesh(format!(
                "simplex mesh vertices must be finite vectors of dimension {dimension}",
            )));
        }
        let reference = ReferenceTopology::new(ReferenceCell::simplex(dimension)?)?;
        let vertex_arity = dimension
            .checked_add(1)
            .ok_or_else(|| invalid_mesh("simplex cell arity overflows usize"))?;
        let mut cell_keys = BTreeSet::new();
        let mut used_vertices = vec![false; vertices.len()];
        let mut minimum_mean_ratio = f64::INFINITY;
        let mut minimum_signed_measure_scale = f64::INFINITY;
        for (cell_index, cell) in cells.iter().enumerate() {
            if cell.len() != vertex_arity
                || cell.iter().any(|&vertex| vertex >= vertices.len())
                || has_duplicate(cell)
            {
                return Err(invalid_mesh(format!(
                    "simplex cell {cell_index} requires {vertex_arity} distinct in-range vertices",
                )));
            }
            let mut key = cell.clone();
            key.sort_unstable();
            if !cell_keys.insert(key) {
                return Err(invalid_mesh("simplex mesh contains a duplicate cell"));
            }
            for &vertex in cell {
                used_vertices[vertex] = true;
            }
            let map = AffineGeometryMap::from_simplex_vertices(
                cell.iter()
                    .map(|&vertex| vertices[vertex].clone())
                    .collect(),
            )?;
            let quality = map.square_quality()?;
            if quality.signed_measure_scale() <= 0.0 {
                return Err(invalid_mesh(format!(
                    "simplex cell {cell_index} is inverted or has non-positive orientation",
                )));
            }
            if quality.mean_ratio() < quality_gate.minimum_mean_ratio {
                return Err(invalid_mesh(format!(
                    "simplex cell {cell_index} mean-ratio quality {} is below the required {}",
                    quality.mean_ratio(),
                    quality_gate.minimum_mean_ratio
                )));
            }
            minimum_mean_ratio = minimum_mean_ratio.min(quality.mean_ratio());
            minimum_signed_measure_scale =
                minimum_signed_measure_scale.min(quality.signed_measure_scale());
        }
        if used_vertices.iter().any(|used| !used) {
            return Err(invalid_mesh("simplex mesh contains an isolated vertex"));
        }

        let (strata, lookups) = build_strata(dimension, vertices.len(), &cells, &reference)?;
        let boundary_entities = classify_boundary_entities(dimension, &cells, &strata, &lookups)?;
        let orientation_codes = build_orientation_codes(dimension, &cells, &strata, &lookups)?;
        Ok(Self {
            dimension,
            vertices,
            cells,
            strata,
            lookups,
            boundary_entities,
            orientation_codes,
            quality_gate,
            quality_report: MeshQualityReport {
                minimum_mean_ratio,
                minimum_signed_measure_scale,
            },
        })
    }

    /// Accepted coordinates in canonical vertex order.
    #[must_use]
    pub fn vertices(&self) -> &[Vec<f64>] {
        &self.vertices
    }

    /// Positively oriented cell closures in cell order.
    #[must_use]
    pub fn cells(&self) -> &[Vec<usize>] {
        &self.cells
    }

    /// Acceptance policy recorded with this mesh revision.
    #[must_use]
    pub const fn quality_gate(&self) -> MeshQualityGate {
        self.quality_gate
    }

    /// Mesh-quality evidence computed before acceptance.
    #[must_use]
    pub const fn quality_report(&self) -> MeshQualityReport {
        self.quality_report
    }

    /// Ordered vertex closure of a valid entity.
    #[must_use]
    pub fn entity_vertices(&self, entity: MeshEntity) -> Option<Vec<MeshEntity>> {
        self.entity_vertex_indices(entity).map(|vertices| {
            vertices
                .iter()
                .map(|&vertex| MeshEntity::new(0, vertex))
                .collect()
        })
    }

    /// Whether a valid entity lies in the boundary closure.
    #[must_use]
    pub fn is_boundary_entity(&self, entity: MeshEntity) -> Option<bool> {
        self.boundary_entities
            .get(entity.dimension())?
            .get(entity.index())
            .copied()
    }

    /// Linearize one entity map under an accepted vertex-velocity field.
    ///
    /// # Errors
    /// Returns `EQ0803` for an invalid entity/action shape or affine-map
    /// failure.
    pub fn linearized_geometry_map(
        &self,
        entity: MeshEntity,
        vertex_velocities: &[Vec<f64>],
    ) -> Result<AffineGeometryLinearization, Diagnostic> {
        if vertex_velocities.len() != self.vertices.len()
            || vertex_velocities.iter().any(|entry| {
                entry.len() != self.dimension || entry.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid_mesh(
                "simplex mesh velocity does not match the accepted mesh revision",
            ));
        }
        let vertices = self
            .entity_vertex_indices(entity)
            .ok_or_else(|| invalid_mesh("simplex entity geometry is unavailable"))?;
        let map = self
            .geometry_map(entity)
            .ok_or_else(|| invalid_mesh("simplex entity geometry is unavailable"))?;
        let origin_tangent = vertex_velocities[vertices[0]].clone();
        let reference_dimension = entity.dimension();
        let mut jacobian_tangent = vec![0.0; self.dimension * reference_dimension];
        for column in 0..reference_dimension {
            for row in 0..self.dimension {
                jacobian_tangent[row * reference_dimension + column] =
                    vertex_velocities[vertices[column + 1]][row] - origin_tangent[row];
            }
        }
        AffineGeometryLinearization::new(map, origin_tangent, jacobian_tangent)
    }

    fn entity_vertex_indices(&self, entity: MeshEntity) -> Option<&[usize]> {
        if entity.dimension() == self.dimension {
            self.cells.get(entity.index()).map(Vec::as_slice)
        } else {
            self.strata
                .get(entity.dimension())?
                .get(entity.index())
                .map(Vec::as_slice)
        }
    }

    fn orientation_code(&self, images: &[usize]) -> OrientationCode {
        if images
            .iter()
            .enumerate()
            .all(|(index, &image)| index == image)
        {
            OrientationCode::identity()
        } else {
            OrientationCode::new(
                *self
                    .orientation_codes
                    .get(images)
                    .expect("all mesh incidence orientations were interned"),
            )
        }
    }

    fn lower_incidence(
        &self,
        source: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        let source_vertices = self.entity_vertex_indices(source)?;
        Some(
            combinations(source_vertices.len(), target_dimension + 1)
                .into_iter()
                .enumerate()
                .map(|(local_ordinal, local_vertices)| {
                    let ordered = local_vertices
                        .iter()
                        .map(|&local| source_vertices[local])
                        .collect::<Vec<_>>();
                    let mut key = ordered.clone();
                    key.sort_unstable();
                    let index = self.lookups[target_dimension][&key];
                    EntityIncidence {
                        entity: MeshEntity::new(target_dimension, index),
                        local_ordinal,
                        orientation: self.orientation_code(&permutation_images(&key, &ordered)),
                    }
                })
                .collect(),
        )
    }
}

impl MeshTopology for SimplicialMesh {
    /// Resolve an orientation code into its exact vertex permutation.
    fn orientation_permutation(
        &self,
        code: OrientationCode,
        arity: usize,
    ) -> Option<VertexPermutation> {
        if code == OrientationCode::identity() {
            return Some(VertexPermutation::identity(arity));
        }
        let images = self
            .orientation_codes
            .iter()
            .find_map(|(images, &candidate)| (candidate == code.code()).then(|| images.clone()))?;
        (images.len() == arity)
            .then(|| VertexPermutation::new(images).expect("interned orientation is a permutation"))
    }

    fn topological_dimension(&self) -> usize {
        self.dimension
    }

    fn entity_count(&self, dimension: usize) -> Option<usize> {
        self.strata.get(dimension).map(Vec::len)
    }

    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        self.entity_vertex_indices(entity)?;
        self.strata.get(target_dimension)?;
        if entity.dimension() == target_dimension {
            return Some(vec![EntityIncidence {
                entity,
                local_ordinal: 0,
                orientation: OrientationCode::identity(),
            }]);
        }
        if target_dimension < entity.dimension() {
            return self.lower_incidence(entity, target_dimension);
        }

        let source_key = {
            let mut key = self.entity_vertex_indices(entity)?.to_vec();
            key.sort_unstable();
            key
        };
        let mut result = Vec::new();
        for target_index in 0..self.strata[target_dimension].len() {
            let target = MeshEntity::new(target_dimension, target_index);
            let target_vertices = self.entity_vertex_indices(target)?;
            if !source_key
                .iter()
                .all(|vertex| target_vertices.contains(vertex))
            {
                continue;
            }
            let incidence = self.lower_incidence(target, entity.dimension())?;
            let candidate = incidence
                .into_iter()
                .find(|candidate| candidate.entity == entity)
                .expect("subset relation has one reverse incidence");
            result.push(EntityIncidence {
                entity: target,
                local_ordinal: candidate.local_ordinal,
                orientation: candidate.orientation,
            });
        }
        Some(result)
    }
}

impl MeshGeometry for SimplicialMesh {
    type Map<'a> = AffineGeometryMap;

    fn geometric_dimension(&self) -> usize {
        self.dimension
    }

    fn geometry_map(&self, entity: MeshEntity) -> Option<Self::Map<'_>> {
        let vertices = self.entity_vertex_indices(entity)?;
        AffineGeometryMap::from_simplex_vertices(
            vertices
                .iter()
                .map(|&vertex| self.vertices[vertex].clone())
                .collect(),
        )
        .ok()
    }
}

fn build_strata(
    dimension: usize,
    vertex_count: usize,
    cells: &[Vec<usize>],
    reference: &ReferenceTopology,
) -> Result<StrataAndLookups, Diagnostic> {
    let mut strata = vec![Vec::new(); dimension + 1];
    strata[0] = (0..vertex_count).map(|vertex| vec![vertex]).collect();
    for (entity_dimension, stratum) in strata.iter_mut().enumerate().take(dimension).skip(1) {
        let mut entities = BTreeSet::new();
        let reference_count = reference
            .entity_count(entity_dimension)
            .expect("simplex reference owns every stratum");
        for cell in cells {
            for reference_index in 0..reference_count {
                let reference_entity = reference
                    .entity(entity_dimension, reference_index)
                    .expect("reference stratum count is valid");
                let mut key = reference_entity
                    .vertex_ordinals()
                    .iter()
                    .map(|&local| cell[local])
                    .collect::<Vec<_>>();
                key.sort_unstable();
                entities.insert(key);
            }
        }
        *stratum = entities.into_iter().collect();
    }
    strata[dimension] = cells
        .iter()
        .map(|cell| {
            let mut key = cell.clone();
            key.sort_unstable();
            key
        })
        .collect();
    let total_entities = strata.iter().map(Vec::len).sum::<usize>();
    if total_entities > MAX_MESH_ENTITIES {
        return Err(invalid_mesh(format!(
            "simplex mesh has {total_entities} entities, exceeding the inspectable artifact limit",
        )));
    }
    let lookups = strata
        .iter()
        .map(|entities| {
            entities
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, vertices)| (vertices, index))
                .collect::<BTreeMap<_, _>>()
        })
        .collect();
    Ok((strata, lookups))
}

type StrataAndLookups = (Vec<Vec<Vec<usize>>>, Vec<BTreeMap<Vec<usize>, usize>>);

fn classify_boundary_entities(
    dimension: usize,
    cells: &[Vec<usize>],
    strata: &[Vec<Vec<usize>>],
    lookups: &[BTreeMap<Vec<usize>, usize>],
) -> Result<Vec<Vec<bool>>, Diagnostic> {
    let facet_dimension = dimension - 1;
    let mut facet_adjacency = vec![0_usize; strata[facet_dimension].len()];
    for cell in cells {
        for local_facet in combinations(cell.len(), dimension) {
            let mut key = local_facet
                .iter()
                .map(|&local| cell[local])
                .collect::<Vec<_>>();
            key.sort_unstable();
            facet_adjacency[lookups[facet_dimension][&key]] += 1;
        }
    }
    if facet_adjacency.iter().any(|&adjacency| adjacency > 2) {
        return Err(invalid_mesh(
            "simplex mesh contains a non-manifold facet with more than two cells",
        ));
    }
    let boundary_facets = strata[facet_dimension]
        .iter()
        .zip(&facet_adjacency)
        .filter_map(|(vertices, &adjacency)| (adjacency == 1).then_some(vertices))
        .collect::<Vec<_>>();
    if boundary_facets.is_empty() {
        return Err(invalid_mesh(
            "simplex mesh requires a nonempty boundary for the current volume realization",
        ));
    }
    Ok(strata
        .iter()
        .enumerate()
        .map(|(entity_dimension, entities)| {
            entities
                .iter()
                .map(|vertices| {
                    entity_dimension < dimension
                        && boundary_facets.iter().any(|facet| {
                            vertices
                                .iter()
                                .all(|vertex| facet.binary_search(vertex).is_ok())
                        })
                })
                .collect()
        })
        .collect())
}

fn build_orientation_codes(
    dimension: usize,
    cells: &[Vec<usize>],
    strata: &[Vec<Vec<usize>>],
    lookups: &[BTreeMap<Vec<usize>, usize>],
) -> Result<BTreeMap<Vec<usize>, u32>, Diagnostic> {
    let mut permutations = BTreeSet::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        for target_dimension in 0..dimension {
            for local_vertices in combinations(cell.len(), target_dimension + 1) {
                let ordered = local_vertices
                    .iter()
                    .map(|&local| cell[local])
                    .collect::<Vec<_>>();
                let mut key = ordered.clone();
                key.sort_unstable();
                let target = lookups[target_dimension]
                    .get(&key)
                    .and_then(|&index| strata[target_dimension].get(index))
                    .ok_or_else(|| {
                        invalid_mesh(format!(
                            "simplex cell {cell_index} has an unresolved incidence",
                        ))
                    })?;
                let images = permutation_images(target, &ordered);
                if images
                    .iter()
                    .enumerate()
                    .any(|(index, &image)| index != image)
                {
                    permutations.insert(images);
                }
            }
        }
    }
    permutations
        .into_iter()
        .enumerate()
        .map(|(index, permutation)| {
            let code = u32::try_from(index + 1)
                .map_err(|_| invalid_mesh("simplex orientation-code table exceeds u32"))?;
            Ok((permutation, code))
        })
        .collect()
}

fn permutation_images(canonical: &[usize], ordered: &[usize]) -> Vec<usize> {
    canonical
        .iter()
        .map(|vertex| {
            ordered
                .iter()
                .position(|candidate| candidate == vertex)
                .expect("incidence vertex sets agree")
        })
        .collect()
}

fn combinations(universe: usize, selection: usize) -> Vec<Vec<usize>> {
    if selection == 0 {
        return vec![Vec::new()];
    }
    if selection > universe {
        return Vec::new();
    }
    let mut indices = (0..selection).collect::<Vec<_>>();
    let mut result = Vec::new();
    loop {
        result.push(indices.clone());
        let Some(position) = (0..selection)
            .rev()
            .find(|&position| indices[position] < universe - selection + position)
        else {
            break;
        };
        indices[position] += 1;
        for next in position + 1..selection {
            indices[next] = indices[next - 1] + 1;
        }
    }
    result
}

fn has_duplicate(values: &[usize]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}

fn invalid_mesh(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> MeshQualityGate {
        MeshQualityGate::new(0.05).unwrap()
    }

    fn two_triangle_mesh() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![2, 3, 0]],
            gate(),
        )
        .unwrap()
    }

    #[test]
    fn builds_oriented_strata_and_boundary_closure() {
        let mesh = two_triangle_mesh();
        assert_eq!(mesh.entity_count(0), Some(4));
        assert_eq!(mesh.entity_count(1), Some(5));
        assert_eq!(mesh.entity_count(2), Some(2));
        let interior_edge = (0..5)
            .map(|index| MeshEntity::new(1, index))
            .find(|&edge| mesh.incidence(edge, 2).unwrap().len() == 2)
            .unwrap();
        assert!(!mesh.is_boundary_entity(interior_edge).unwrap());
        assert_eq!(mesh.incidence(interior_edge, 2).unwrap().len(), 2);

        let reversed_edge = mesh
            .incidence(MeshEntity::new(2, 1), 1)
            .unwrap()
            .into_iter()
            .find(|incidence| incidence.orientation != OrientationCode::identity())
            .unwrap();
        let permutation = mesh
            .orientation_permutation(reversed_edge.orientation, 2)
            .unwrap();
        assert_eq!(permutation.images(), &[1, 0]);
    }

    #[test]
    fn supports_runtime_dimensional_tetrahedra_and_geometry_actions() {
        let mesh = SimplicialMesh::new(
            3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![vec![0, 1, 2, 3]],
            gate(),
        )
        .unwrap();
        assert_eq!(mesh.entity_count(1), Some(6));
        assert_eq!(mesh.entity_count(2), Some(4));
        assert_eq!(mesh.quality_report().minimum_mean_ratio(), 1.0);

        let velocity = mesh
            .vertices()
            .iter()
            .map(|vertex| vec![vertex[0], 0.0, 0.0])
            .collect::<Vec<_>>();
        let linearized = mesh
            .linearized_geometry_map(MeshEntity::new(3, 0), &velocity)
            .unwrap();
        assert_eq!(linearized.jacobian_tangent()[0], 1.0);
        assert_eq!(linearized.measure_scale_tangent(), 1.0);
    }

    #[test]
    fn rejects_inversion_nonmanifold_topology_and_low_quality() {
        let vertices = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            SimplicialMesh::new(2, vertices.clone(), vec![vec![0, 2, 1]], gate())
                .unwrap_err()
                .code(),
            codes::INVALID_MESH
        );
        let sliver = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0e-8, 1.0e-8]];
        assert_eq!(
            SimplicialMesh::new(
                2,
                sliver,
                vec![vec![0, 1, 2]],
                MeshQualityGate::new(0.5).unwrap(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_MESH
        );
        let nonmanifold_vertices = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 1.0],
            vec![0.5, -1.0],
            vec![0.5, 2.0],
        ];
        assert_eq!(
            SimplicialMesh::new(
                2,
                nonmanifold_vertices,
                vec![vec![0, 1, 2], vec![1, 0, 3], vec![0, 1, 4]],
                MeshQualityGate::new(0.01).unwrap(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_MESH
        );
    }
}
