mod incidence;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    AffineGeometryLinearization, AffineGeometryMap, EntityIncidence, MeshEntity, MeshGeometry,
    MeshTopology, OrientationCode, ReferenceCell, ReferenceTopology,
};

const MAX_CARTESIAN_ENTITIES: usize = 8_000_000;
const MAX_CARTESIAN_VERTEX_REFERENCES: usize = 64_000_000;

/// Runtime-dimensional conforming mesh of an axis-aligned Cartesian box.
///
/// Every entity is represented by the axes along which it is free and one
/// anchor index on every axis. The same construction therefore generates
/// vertices, edges, facets, and cells without dimension-specific tables.
/// Geometry is kept separate from topology through affine entity maps.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianMesh {
    axes: Vec<Vec<f64>>,
    strata: Vec<Vec<CartesianEntity>>,
    reference_topologies: Vec<ReferenceTopology>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CartesianEntity {
    free_axes: Vec<usize>,
    anchors: Vec<usize>,
    vertices: Vec<usize>,
}

impl CartesianMesh {
    /// Validate the complete analytic allocation shape of Cartesian cell counts.
    ///
    /// This performs the same entity and closure-reference bounds used by
    /// construction without allocating coordinates or topology.
    pub fn validate_cell_counts(cells_per_axis: &[usize]) -> Result<(), Diagnostic> {
        if cells_per_axis.is_empty() || cells_per_axis.contains(&0) {
            return Err(invalid_mesh(
                "Cartesian cell counts require one positive count per axis",
            ));
        }
        let (entities, vertex_references) = cells_per_axis.iter().try_fold(
            (1_usize, 1_usize),
            |(entities, vertex_references), &cells| {
                let entity_factor = cells
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| invalid_mesh("Cartesian entity shape overflows usize"))?;
                let reference_factor = cells
                    .checked_mul(3)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        invalid_mesh("Cartesian vertex-reference shape overflows usize")
                    })?;
                Ok::<_, Diagnostic>((
                    entities
                        .checked_mul(entity_factor)
                        .ok_or_else(|| invalid_mesh("Cartesian entity count overflows usize"))?,
                    vertex_references
                        .checked_mul(reference_factor)
                        .ok_or_else(|| {
                            invalid_mesh("Cartesian vertex-reference count overflows usize")
                        })?,
                ))
            },
        )?;
        if entities > MAX_CARTESIAN_ENTITIES || vertex_references > MAX_CARTESIAN_VERTEX_REFERENCES
        {
            return Err(invalid_mesh(format!(
                "Cartesian mesh requires {entities} entities and {vertex_references} vertex references, exceeding inspectable artifact limits"
            )));
        }
        Ok(())
    }

    /// Construct a conforming mesh from the vertex coordinates of each axis.
    ///
    /// Axis coordinates must be finite and strictly increasing. At least one
    /// axis and two vertices per axis are required.
    ///
    /// # Errors
    /// Returns `EQ0803` for invalid coordinates, shape/count overflow, or a
    /// mesh exceeding the inspectable in-memory artifact limits.
    pub fn from_axes(axes: Vec<Vec<f64>>) -> Result<Self, Diagnostic> {
        validate_axes(&axes)?;
        let dimension = axes.len();
        let vertex_strides = row_major_strides(
            &axes.iter().map(Vec::len).collect::<Vec<_>>(),
            "Cartesian vertex shape",
        )?;
        let reference_topologies = (0..=dimension)
            .map(|entity_dimension| {
                let cell = reference_cell(entity_dimension)?;
                ReferenceTopology::new(cell)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut entity_count = 0_usize;
        let mut vertex_reference_count = 0_usize;
        let mut strata = Vec::with_capacity(dimension + 1);
        for entity_dimension in 0..=dimension {
            let closure_vertices = checked_power_of_two(entity_dimension)?;
            let mut stratum = Vec::new();
            for free_axes in combinations(dimension, entity_dimension) {
                let shape = axes
                    .iter()
                    .enumerate()
                    .map(|(axis, coordinates)| {
                        if free_axes.binary_search(&axis).is_ok() {
                            coordinates.len() - 1
                        } else {
                            coordinates.len()
                        }
                    })
                    .collect::<Vec<_>>();
                let count = checked_product(&shape, "Cartesian entity shape")?;
                entity_count = entity_count
                    .checked_add(count)
                    .ok_or_else(|| invalid_mesh("Cartesian entity count overflows usize"))?;
                vertex_reference_count = vertex_reference_count
                    .checked_add(count.checked_mul(closure_vertices).ok_or_else(|| {
                        invalid_mesh("Cartesian entity vertex-reference count overflows usize")
                    })?)
                    .ok_or_else(|| {
                        invalid_mesh("Cartesian entity vertex-reference count overflows usize")
                    })?;
                if entity_count > MAX_CARTESIAN_ENTITIES
                    || vertex_reference_count > MAX_CARTESIAN_VERTEX_REFERENCES
                {
                    return Err(invalid_mesh(format!(
                        "Cartesian mesh requires {entity_count} entities and {vertex_reference_count} vertex references, exceeding inspectable artifact limits"
                    )));
                }

                stratum.try_reserve_exact(count).map_err(|_| {
                    invalid_mesh("Cartesian entity allocation exceeds platform capacity")
                })?;
                for linear_anchor in 0..count {
                    let anchors = delinearize(linear_anchor, &shape);
                    let vertices =
                        entity_vertices(&free_axes, &anchors, &vertex_strides, closure_vertices)?;
                    stratum.push(CartesianEntity {
                        free_axes: free_axes.clone(),
                        anchors,
                        vertices,
                    });
                }
            }
            strata.push(stratum);
        }

        Ok(Self {
            axes,
            strata,
            reference_topologies,
        })
    }

    /// Construct an axis-aligned uniform Cartesian mesh.
    ///
    /// `bounds[axis]` and `cells_per_axis[axis]` describe the same physical
    /// axis. Cell counts may differ by axis.
    ///
    /// # Errors
    /// Returns `EQ0803` for incompatible shapes, invalid bounds/counts,
    /// unrepresentable spacing, or artifact resource overflow.
    pub fn uniform(bounds: &[[f64; 2]], cells_per_axis: &[usize]) -> Result<Self, Diagnostic> {
        if bounds.is_empty() || bounds.len() != cells_per_axis.len() {
            return Err(invalid_mesh(
                "uniform Cartesian bounds and cell counts require one common positive dimension",
            ));
        }
        Self::validate_cell_counts(cells_per_axis)?;
        let axes = bounds
            .iter()
            .zip(cells_per_axis)
            .enumerate()
            .map(|(axis, (bounds, &cells))| uniform_axis(*bounds, cells, axis))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_axes(axes)
    }

    /// Vertex coordinates for one physical axis.
    #[must_use]
    pub fn axis_coordinates(&self, axis: usize) -> Option<&[f64]> {
        self.axes.get(axis).map(Vec::as_slice)
    }

    /// Cell count on one physical axis.
    #[must_use]
    pub fn axis_cell_count(&self, axis: usize) -> Option<usize> {
        self.axes.get(axis).map(|coordinates| coordinates.len() - 1)
    }

    /// Mesh vertex entities in the canonical local order of `entity`.
    #[must_use]
    pub fn entity_vertices(&self, entity: MeshEntity) -> Option<Vec<MeshEntity>> {
        self.entity(entity).map(|entity| {
            entity
                .vertices
                .iter()
                .copied()
                .map(|vertex| MeshEntity::new(0, vertex))
                .collect()
        })
    }

    /// Physical coordinates of a valid vertex entity.
    #[must_use]
    pub fn vertex_coordinates(&self, vertex: MeshEntity) -> Option<Vec<f64>> {
        if vertex.dimension() != 0 {
            return None;
        }
        let entity = self.entity(vertex)?;
        Some(
            entity
                .anchors
                .iter()
                .enumerate()
                .map(|(axis, &index)| self.axes[axis][index])
                .collect(),
        )
    }

    /// Axis-local vertex indices of a valid vertex entity.
    #[must_use]
    pub fn vertex_multi_index(&self, vertex: MeshEntity) -> Option<&[usize]> {
        (vertex.dimension() == 0)
            .then(|| self.entity(vertex).map(|entity| entity.anchors.as_slice()))
            .flatten()
    }

    /// Top-dimensional cell at one axis-local cell multi-index.
    #[must_use]
    pub fn cell_at(&self, cell_indices: &[usize]) -> Option<MeshEntity> {
        if cell_indices.len() != self.axes.len()
            || cell_indices
                .iter()
                .enumerate()
                .any(|(axis, &index)| index >= self.axes[axis].len() - 1)
        {
            return None;
        }
        let shape = self
            .axes
            .iter()
            .map(|coordinates| coordinates.len() - 1)
            .collect::<Vec<_>>();
        let strides = row_major_strides(&shape, "Cartesian cell shape").ok()?;
        let index = linearize(cell_indices, &strides).ok()?;
        Some(MeshEntity::new(self.axes.len(), index))
    }

    /// Axis-local cell indices of one valid top-dimensional cell entity.
    #[must_use]
    pub fn cell_multi_index(&self, cell: MeshEntity) -> Option<&[usize]> {
        (cell.dimension() == self.axes.len())
            .then(|| self.entity(cell).map(|entity| entity.anchors.as_slice()))
            .flatten()
    }

    /// Whether an entity lies in the boundary closure of the Cartesian box.
    #[must_use]
    pub fn is_boundary_entity(&self, entity: MeshEntity) -> Option<bool> {
        let entity = self.entity(entity)?;
        Some((0..self.axes.len()).any(|axis| {
            entity.free_axes.binary_search(&axis).is_err()
                && (entity.anchors[axis] == 0 || entity.anchors[axis] + 1 == self.axes[axis].len())
        }))
    }

    /// Physical coordinate of an entity center (`xi = 0`).
    #[must_use]
    pub fn entity_center(&self, entity: MeshEntity) -> Option<Vec<f64>> {
        self.entity_geometry(entity)
            .ok()
            .map(|map| map.origin().to_vec())
    }

    /// Cartesian axes free within an entity, in physical-axis order.
    #[must_use]
    pub fn entity_free_axes(&self, entity: MeshEntity) -> Option<&[usize]> {
        self.entity(entity)
            .map(|entity| entity.free_axes.as_slice())
    }

    /// Largest physical side length among all cells and axes.
    #[must_use]
    pub fn maximum_cell_width(&self) -> f64 {
        self.axes
            .iter()
            .flat_map(|axis| axis.windows(2))
            .map(|pair| pair[1] - pair[0])
            .fold(0.0, f64::max)
    }

    /// Current lower/upper coordinate of one physical axis.
    #[must_use]
    pub fn axis_bounds(&self, axis: usize) -> Option<[f64; 2]> {
        self.axes
            .get(axis)
            .map(|coordinates| [coordinates[0], coordinates[coordinates.len() - 1]])
    }

    /// Linearize one entity map with respect to its axis endpoints.
    ///
    /// `endpoint_tangents` gives the lower and upper coordinate tangent for
    /// `axis`. Every intermediate axis vertex retains its normalized position
    /// between those endpoints. The resulting motion is therefore a global
    /// affine pullback that preserves topology and axis alignment without
    /// importing Semantic Model boundary vocabulary into the mesh layer.
    ///
    /// # Errors
    /// Returns `EQ0803` for an invalid entity or axis.
    pub fn linearize_axis_endpoints(
        &self,
        entity: MeshEntity,
        axis: usize,
        endpoint_tangents: [f64; 2],
    ) -> Result<AffineGeometryLinearization, Diagnostic> {
        if endpoint_tangents.iter().any(|value| !value.is_finite()) {
            return Err(invalid_mesh("Cartesian endpoint tangents must be finite"));
        }
        let entity_data = self
            .entity(entity)
            .ok_or_else(|| invalid_mesh("Cartesian endpoint action requested an invalid entity"))?;
        let coordinates = self.axes.get(axis).ok_or_else(|| {
            invalid_mesh("Cartesian endpoint action requested an invalid physical axis")
        })?;
        let coordinate_tangent = |index: usize| {
            let fraction = (coordinates[index] - coordinates[0])
                / (coordinates[coordinates.len() - 1] - coordinates[0]);
            endpoint_tangents[0] * (1.0 - fraction) + endpoint_tangents[1] * fraction
        };
        let primal = self.entity_geometry(entity)?;
        let reference_dimension = entity_data.free_axes.len();
        let mut origin_tangent = vec![0.0; self.axes.len()];
        let mut jacobian_tangent = vec![0.0; self.axes.len() * reference_dimension];
        let anchor = entity_data.anchors[axis];
        match entity_data.free_axes.binary_search(&axis) {
            Ok(column) => {
                let lower = coordinate_tangent(anchor);
                let upper = coordinate_tangent(anchor + 1);
                origin_tangent[axis] = 0.5 * (lower + upper);
                jacobian_tangent[axis * reference_dimension + column] = 0.5 * (upper - lower);
            }
            Err(_) => origin_tangent[axis] = coordinate_tangent(anchor),
        }
        AffineGeometryLinearization::new(primal, origin_tangent, jacobian_tangent)
    }

    fn entity(&self, entity: MeshEntity) -> Option<&CartesianEntity> {
        self.strata.get(entity.dimension())?.get(entity.index())
    }

    fn entity_geometry(&self, entity: MeshEntity) -> Result<AffineGeometryMap, Diagnostic> {
        let entity = self
            .entity(entity)
            .ok_or_else(|| invalid_mesh("Cartesian geometry requested for an invalid entity"))?;
        let entity_dimension = entity.free_axes.len();
        let mut origin = Vec::with_capacity(self.axes.len());
        let mut jacobian = vec![0.0; self.axes.len() * entity_dimension];
        for (axis, coordinates) in self.axes.iter().enumerate() {
            let anchor = entity.anchors[axis];
            match entity.free_axes.binary_search(&axis) {
                Ok(column) => {
                    let lower = coordinates[anchor];
                    let upper = coordinates[anchor + 1];
                    origin.push(lower + 0.5 * (upper - lower));
                    jacobian[axis * entity_dimension + column] = 0.5 * (upper - lower);
                }
                Err(_) => origin.push(coordinates[anchor]),
            }
        }
        AffineGeometryMap::new(
            reference_cell(entity_dimension)?,
            self.axes.len(),
            origin,
            jacobian,
        )
    }
}

impl MeshTopology for CartesianMesh {
    fn topological_dimension(&self) -> usize {
        self.axes.len()
    }

    fn entity_count(&self, dimension: usize) -> Option<usize> {
        self.strata.get(dimension).map(Vec::len)
    }

    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        self.incident_entities(entity, target_dimension)
    }
}

impl MeshGeometry for CartesianMesh {
    type Map<'a> = AffineGeometryMap;

    fn geometric_dimension(&self) -> usize {
        self.axes.len()
    }

    fn geometry_map(&self, entity: MeshEntity) -> Option<Self::Map<'_>> {
        self.entity_geometry(entity).ok()
    }
}

fn validate_axes(axes: &[Vec<f64>]) -> Result<(), Diagnostic> {
    if axes.is_empty() {
        return Err(invalid_mesh(
            "Cartesian mesh requires at least one physical axis",
        ));
    }
    for (axis, coordinates) in axes.iter().enumerate() {
        if coordinates.len() < 2 {
            return Err(invalid_mesh(format!(
                "Cartesian axis {axis} requires at least two vertices"
            )));
        }
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
            || coordinates
                .windows(2)
                .any(|pair| pair[1] <= pair[0] || !(pair[1] - pair[0]).is_finite())
        {
            return Err(invalid_mesh(format!(
                "Cartesian axis {axis} coordinates must be finite and strictly increasing with representable cells"
            )));
        }
    }
    Ok(())
}

fn uniform_axis(bounds: [f64; 2], cells: usize, axis: usize) -> Result<Vec<f64>, Diagnostic> {
    let [lower, upper] = bounds;
    if !lower.is_finite() || !upper.is_finite() || upper <= lower || cells == 0 {
        return Err(invalid_mesh(format!(
            "uniform Cartesian axis {axis} requires finite increasing bounds and a positive cell count"
        )));
    }
    let vertex_count = cells
        .checked_add(1)
        .ok_or_else(|| invalid_mesh("uniform Cartesian vertex count overflows usize"))?;
    if vertex_count > MAX_CARTESIAN_ENTITIES {
        return Err(invalid_mesh(format!(
            "uniform Cartesian axis {axis} exceeds inspectable artifact limits"
        )));
    }
    let spacing = (upper - lower) / cells as f64;
    if !spacing.is_finite()
        || spacing <= 0.0
        || lower + spacing <= lower
        || upper - spacing >= upper
    {
        return Err(invalid_mesh(format!(
            "uniform Cartesian axis {axis} spacing is not representable at its endpoints"
        )));
    }
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(vertex_count)
        .map_err(|_| invalid_mesh("uniform Cartesian coordinate allocation failed"))?;
    for index in 0..cells {
        coordinates.push(lower + index as f64 * spacing);
    }
    coordinates.push(upper);
    Ok(coordinates)
}

fn reference_cell(dimension: usize) -> Result<ReferenceCell, Diagnostic> {
    if dimension == 0 {
        Ok(ReferenceCell::point())
    } else {
        ReferenceCell::hypercube(dimension)
            .map_err(|_| invalid_mesh("Cartesian entity reference dimension is invalid"))
    }
}

fn checked_product(shape: &[usize], description: &str) -> Result<usize, Diagnostic> {
    shape.iter().try_fold(1_usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| invalid_mesh(format!("{description} overflows usize")))
    })
}

fn checked_power_of_two(exponent: usize) -> Result<usize, Diagnostic> {
    u32::try_from(exponent)
        .ok()
        .and_then(|exponent| 2_usize.checked_pow(exponent))
        .ok_or_else(|| invalid_mesh("Cartesian entity closure size overflows usize"))
}

fn row_major_strides(shape: &[usize], description: &str) -> Result<Vec<usize>, Diagnostic> {
    checked_product(shape, description)?;
    let mut strides = vec![1_usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or_else(|| invalid_mesh(format!("{description} strides overflow usize")))?;
    }
    Ok(strides)
}

fn delinearize(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    indices
}

fn linearize(indices: &[usize], strides: &[usize]) -> Result<usize, Diagnostic> {
    indices
        .iter()
        .zip(strides)
        .try_fold(0_usize, |linear, (&index, &stride)| {
            linear
                .checked_add(index.checked_mul(stride).ok_or_else(|| {
                    invalid_mesh("Cartesian vertex index multiplication overflows usize")
                })?)
                .ok_or_else(|| invalid_mesh("Cartesian vertex index overflows usize"))
        })
}

fn entity_vertices(
    free_axes: &[usize],
    anchors: &[usize],
    vertex_strides: &[usize],
    vertex_count: usize,
) -> Result<Vec<usize>, Diagnostic> {
    let mut vertices = Vec::with_capacity(vertex_count);
    for bits in 0..vertex_count {
        let mut indices = anchors.to_vec();
        for (ordinal, &axis) in free_axes.iter().enumerate() {
            indices[axis] += (bits >> ordinal) & 1;
        }
        vertices.push(linearize(&indices, vertex_strides)?);
    }
    Ok(vertices)
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

fn invalid_mesh(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use crate::{GeometryMap, MeshGeometry};

    use super::*;

    #[test]
    fn generates_all_two_dimensional_strata_and_oriented_incidence() {
        let mesh = CartesianMesh::from_axes(vec![vec![0.0, 1.0, 3.0], vec![-1.0, 2.0]]).unwrap();

        assert_eq!(mesh.topological_dimension(), 2);
        assert_eq!(mesh.entity_count(0), Some(6));
        assert_eq!(mesh.entity_count(1), Some(7));
        assert_eq!(mesh.entity_count(2), Some(2));
        assert_eq!(
            mesh.entity_vertices(MeshEntity::new(2, 0)).unwrap(),
            vec![
                MeshEntity::new(0, 0),
                MeshEntity::new(0, 2),
                MeshEntity::new(0, 1),
                MeshEntity::new(0, 3),
            ]
        );

        let cell_edges = mesh.incidence(MeshEntity::new(2, 0), 1).unwrap();
        assert_eq!(cell_edges.len(), 4);
        assert_eq!(
            cell_edges
                .iter()
                .map(|incidence| incidence.local_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        let shared = cell_edges
            .iter()
            .find(|incidence| {
                mesh.incidence(incidence.entity, 2)
                    .is_some_and(|cells| cells.len() == 2)
            })
            .unwrap();
        assert_eq!(mesh.incidence(shared.entity, 2).unwrap().len(), 2);
    }

    #[test]
    fn maps_cells_facets_and_vertices_through_one_affine_contract() {
        let mesh = CartesianMesh::from_axes(vec![vec![0.0, 2.0], vec![-1.0, 3.0]]).unwrap();
        let cell = mesh.geometry_map(MeshEntity::new(2, 0)).unwrap();
        let mut physical = [0.0; 2];
        cell.map_point(&[0.0, 0.0], &mut physical).unwrap();
        assert_eq!(physical, [1.0, 1.0]);
        assert_eq!(cell.measure_scale(), 2.0);

        let facet = mesh.incidence(MeshEntity::new(2, 0), 1).unwrap()[0].entity;
        let facet_map = mesh.geometry_map(facet).unwrap();
        assert_eq!(facet_map.reference_cell().dimension(), 1);
        assert!(facet_map.measure_scale() == 1.0 || facet_map.measure_scale() == 2.0);

        assert_eq!(
            mesh.vertex_coordinates(MeshEntity::new(0, 3)).unwrap(),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn one_cube_has_dimension_general_entity_counts() {
        let mesh =
            CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], &[1, 1, 1]).unwrap();
        assert_eq!(mesh.entity_count(0), Some(8));
        assert_eq!(mesh.entity_count(1), Some(12));
        assert_eq!(mesh.entity_count(2), Some(6));
        assert_eq!(mesh.entity_count(3), Some(1));
        assert_eq!(mesh.maximum_cell_width(), 1.0);
    }

    #[test]
    fn rejects_invalid_axes_and_unrepresentable_uniform_spacing() {
        assert_eq!(
            CartesianMesh::from_axes(Vec::new()).unwrap_err().code(),
            codes::INVALID_MESH
        );
        assert_eq!(
            CartesianMesh::from_axes(vec![vec![0.0, 0.0]])
                .unwrap_err()
                .code(),
            codes::INVALID_MESH
        );
        assert_eq!(
            CartesianMesh::uniform(&[[1.0, 1.0]], &[2])
                .unwrap_err()
                .code(),
            codes::INVALID_MESH
        );
    }

    #[test]
    fn box_bound_action_moves_nonuniform_mesh_by_one_affine_pullback() {
        let mesh =
            CartesianMesh::from_axes(vec![vec![0.0, 0.25, 1.0], vec![-2.0, -0.5, 2.0]]).unwrap();
        let cell = mesh.cell_at(&[1, 0]).unwrap();
        let linearized = mesh.linearize_axis_endpoints(cell, 0, [0.0, 1.0]).unwrap();
        let mut physical = [0.0; 2];
        let mut tangent = [0.0; 2];
        linearized
            .map_point_jvp(&[-0.4, 0.3], &mut physical, &mut tangent)
            .unwrap();
        assert!((tangent[0] - physical[0]).abs() < 2.0e-15);
        assert_eq!(tangent[1], 0.0);
        assert!(
            (linearized.measure_scale_tangent() - linearized.map().measure_scale()).abs() < 2.0e-15
        );

        let facet = mesh
            .incidence(cell, 1)
            .unwrap()
            .into_iter()
            .map(|incidence| incidence.entity)
            .find(|facet| mesh.entity_free_axes(*facet) == Some(&[1][..]))
            .unwrap();
        let facet_linearized = mesh.linearize_axis_endpoints(facet, 0, [0.0, 1.0]).unwrap();
        assert!(facet_linearized.measure_scale_tangent().abs() < 2.0e-15);
    }
}
