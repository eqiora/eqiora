use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::quadrature::ReferenceCell;

/// Dimension-tagged entity ID used by the arbitrary-dimensional mesh contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshEntity {
    dimension: usize,
    index: usize,
}

impl MeshEntity {
    /// Construct an entity in one mesh-local dimension stratum.
    #[must_use]
    pub const fn new(dimension: usize, index: usize) -> Self {
        Self { dimension, index }
    }

    /// Topological dimension.
    #[must_use]
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Zero-based index within the dimension stratum.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Opaque orientation/permutation code interpreted with a reference cell.
///
/// Code zero is always identity. Higher-dimensional implementations may
/// assign other stable codes to subentity vertex permutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct OrientationCode(u32);

impl OrientationCode {
    /// Identity orientation.
    #[must_use]
    pub const fn identity() -> Self {
        Self(0)
    }

    /// Construct a reference-cell-specific orientation code.
    #[must_use]
    pub const fn new(code: u32) -> Self {
        Self(code)
    }

    /// Raw stable code.
    #[must_use]
    pub const fn code(self) -> u32 {
        self.0
    }
}

/// One oriented incidence relation in a source entity's local topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityIncidence {
    /// Incident entity.
    pub entity: MeshEntity,
    /// Lower-dimensional entity number in the containing reference cell.
    /// This has the same meaning when incidence is queried in either direction.
    pub local_ordinal: usize,
    /// Orientation relative to the incident entity's canonical ordering.
    pub orientation: OrientationCode,
}

/// Arbitrary-dimensional topology contract.
///
/// Entities are stratified by topological dimension. `incidence` works in
/// either direction, so cell closures and facet-to-cell adjacency share one
/// operation instead of separate dimension-specific APIs.
pub trait MeshTopology {
    /// Resolve an incidence orientation in this mesh's own code inventory.
    /// The default admits only the identity code; meshes with nontrivial
    /// orientations must resolve their exact vertex permutations.
    fn orientation_permutation(
        &self,
        code: OrientationCode,
        arity: usize,
    ) -> Option<crate::VertexPermutation> {
        (code == OrientationCode::identity()).then(|| crate::VertexPermutation::identity(arity))
    }

    /// Topological mesh dimension.
    fn topological_dimension(&self) -> usize;

    /// Entity count in a valid dimension stratum.
    fn entity_count(&self, dimension: usize) -> Option<usize>;

    /// Oriented entities of `target_dimension` incident to `entity` in
    /// deterministic reference-cell order. Invalid IDs or dimensions return
    /// `None`.
    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>>;
}

/// Runtime-dimensional physical map for one mesh entity.
///
/// Jacobians are row-major with shape
/// `physical_dimension x reference_dimension`. Callers provide output storage
/// so the contract does not require heap allocation in local kernels.
pub trait GeometryMap {
    /// Reference-cell topology and dimension.
    fn reference_cell(&self) -> ReferenceCell;

    /// Dimension of embedding physical coordinates.
    fn physical_dimension(&self) -> usize;

    /// Map one reference point to physical coordinates.
    ///
    /// # Errors
    /// Returns `EQ0803` for incompatible buffer dimensions or a point outside
    /// the reference cell.
    fn map_point(&self, reference: &[f64], physical: &mut [f64]) -> Result<(), Diagnostic>;

    /// Evaluate the row-major geometry Jacobian.
    ///
    /// # Errors
    /// Returns `EQ0803` for incompatible buffer dimensions or a point outside
    /// the reference cell.
    fn jacobian_at(&self, reference: &[f64], jacobian: &mut [f64]) -> Result<(), Diagnostic>;
}

/// Geometry ownership contract paired with [`MeshTopology`].
pub trait MeshGeometry: MeshTopology {
    /// Concrete map type; implementations may use an enum for mixed cells.
    type Map<'a>: GeometryMap
    where
        Self: 'a;

    /// Dimension of embedding physical coordinates.
    fn geometric_dimension(&self) -> usize;

    /// Geometry map for a valid mesh entity.
    fn geometry_map(&self, entity: MeshEntity) -> Option<Self::Map<'_>>;
}

/// Stable index of a vertex in one mesh revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexId(usize);

impl VertexId {
    /// Construct a mesh-local vertex index. Mesh accessors still validate bounds.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based mesh-local index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Stable index of a cell in one mesh revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellId(usize);

impl CellId {
    /// Construct a mesh-local cell index. Mesh accessors still validate bounds.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based mesh-local index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Stable index of a codimension-one facet in one mesh revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FacetId(usize);

impl FacetId {
    /// Construct a mesh-local facet index. Mesh accessors still validate bounds.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based mesh-local index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Oriented cells adjacent to a line facet.
///
/// `minus` is geometrically left and `plus` is geometrically right. Exactly
/// one side is absent on a boundary facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FacetCells1d {
    /// Cell on the negative-coordinate side.
    pub minus: Option<CellId>,
    /// Cell on the positive-coordinate side.
    pub plus: Option<CellId>,
}

/// Affine geometry map from the reference segment `[-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentGeometry1d {
    left: f64,
    right: f64,
}

impl SegmentGeometry1d {
    /// Construct a finite increasing affine segment map.
    ///
    /// # Errors
    /// Returns `EQ0803` when the endpoints do not define a valid segment.
    pub fn new(left: f64, right: f64) -> Result<Self, Diagnostic> {
        if !left.is_finite() || !right.is_finite() || right <= left || !(right - left).is_finite() {
            return Err(invalid_mesh(
                "segment geometry requires finite increasing endpoints",
            ));
        }
        Ok(Self { left, right })
    }

    /// Left physical coordinate.
    #[must_use]
    pub const fn left(self) -> f64 {
        self.left
    }

    /// Right physical coordinate.
    #[must_use]
    pub const fn right(self) -> f64 {
        self.right
    }

    /// Cell length.
    #[must_use]
    pub fn measure(self) -> f64 {
        self.right - self.left
    }

    /// Cell center.
    #[must_use]
    pub fn center(self) -> f64 {
        self.left + 0.5 * self.measure()
    }

    /// Absolute Jacobian of the reference-to-physical map.
    #[must_use]
    pub fn jacobian(self) -> f64 {
        0.5 * self.measure()
    }

    /// Map a reference coordinate in `[-1, 1]` to physical space.
    #[must_use]
    pub fn map(self, reference: f64) -> f64 {
        self.center() + self.jacobian() * reference
    }
}

/// Point geometry of a line facet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointGeometry1d {
    /// Physical coordinate.
    pub coordinate: f64,
}

/// Geometry-map variant returned by [`LineMesh`] through the generic contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineGeometryMap {
    /// Vertex/facet point geometry.
    Point(PointGeometry1d),
    /// Cell segment geometry.
    Segment(SegmentGeometry1d),
}

/// Validated one-dimensional mesh with explicit topology and geometry APIs.
///
/// Vertices are strictly increasing. Cell `i` is incident to vertices `i`
/// and `i + 1`; facet `i` is geometrically coincident with vertex `i` but has
/// a distinct ID type because vertices and facets play different roles.
#[derive(Debug, Clone, PartialEq)]
pub struct LineMesh {
    coordinates: Vec<f64>,
}

impl LineMesh {
    /// Construct a mesh from strictly increasing finite vertex coordinates.
    ///
    /// # Errors
    /// Returns `EQ0803` if there are fewer than two vertices or if adjacent
    /// coordinates do not define a finite, representable positive cell.
    pub fn from_vertices(coordinates: Vec<f64>) -> Result<Self, Diagnostic> {
        if coordinates.len() < 2 {
            return Err(invalid_mesh("a line mesh requires at least two vertices"));
        }
        if coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(invalid_mesh("line-mesh coordinates must be finite"));
        }
        if coordinates
            .windows(2)
            .any(|pair| pair[1] <= pair[0] || !(pair[1] - pair[0]).is_finite())
        {
            return Err(invalid_mesh(
                "line-mesh coordinates must be strictly increasing with finite cells",
            ));
        }
        Ok(Self { coordinates })
    }

    /// Construct an equal-cell mesh on a finite increasing interval.
    ///
    /// # Errors
    /// Returns `EQ0803` for invalid endpoints, an empty mesh, overflow, or a
    /// spacing too small to distinguish adjacent coordinates.
    pub fn uniform(start: f64, end: f64, cells: usize) -> Result<Self, Diagnostic> {
        if !start.is_finite() || !end.is_finite() || end <= start {
            return Err(invalid_mesh(
                "uniform line endpoints must be finite and strictly increasing",
            ));
        }
        if cells == 0 || cells == usize::MAX {
            return Err(invalid_mesh("uniform line requires at least one cell"));
        }
        let spacing = (end - start) / cells as f64;
        if !spacing.is_finite()
            || spacing <= 0.0
            || start + spacing <= start
            || end - spacing >= end
        {
            return Err(invalid_mesh(
                "uniform line spacing is not representable at its endpoints",
            ));
        }
        let mut coordinates = Vec::new();
        coordinates.try_reserve_exact(cells + 1).map_err(|_| {
            invalid_mesh("uniform line coordinate allocation exceeds platform capacity")
        })?;
        for index in 0..cells {
            coordinates.push(start + index as f64 * spacing);
        }
        coordinates.push(end);
        Self::from_vertices(coordinates)
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.coordinates.len()
    }

    /// Number of cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.coordinates.len() - 1
    }

    /// Number of codimension-one facets.
    #[must_use]
    pub fn facet_count(&self) -> usize {
        self.coordinates.len()
    }

    /// Vertex IDs in deterministic topology order.
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = VertexId> + '_ {
        (0..self.vertex_count()).map(VertexId)
    }

    /// Cell IDs in deterministic topology order.
    pub fn cells(&self) -> impl ExactSizeIterator<Item = CellId> + '_ {
        (0..self.cell_count()).map(CellId)
    }

    /// Facet IDs in deterministic topology order.
    pub fn facets(&self) -> impl ExactSizeIterator<Item = FacetId> + '_ {
        (0..self.facet_count()).map(FacetId)
    }

    /// Physical coordinate attached to a vertex.
    #[must_use]
    pub fn vertex_coordinate(&self, vertex: VertexId) -> Option<f64> {
        self.coordinates.get(vertex.0).copied()
    }

    /// Oriented vertices incident to a cell.
    #[must_use]
    pub fn cell_vertices(&self, cell: CellId) -> Option<[VertexId; 2]> {
        (cell.0 < self.cell_count()).then(|| [VertexId(cell.0), VertexId(cell.0 + 1)])
    }

    /// Physical geometry map for a cell.
    #[must_use]
    pub fn cell_geometry(&self, cell: CellId) -> Option<SegmentGeometry1d> {
        let [left, right] = self.cell_vertices(cell)?;
        Some(SegmentGeometry1d {
            left: self.coordinates[left.0],
            right: self.coordinates[right.0],
        })
    }

    /// Point geometry for a facet.
    #[must_use]
    pub fn facet_geometry(&self, facet: FacetId) -> Option<PointGeometry1d> {
        self.coordinates
            .get(facet.0)
            .copied()
            .map(|coordinate| PointGeometry1d { coordinate })
    }

    /// Oriented cell incidence for a facet.
    #[must_use]
    pub fn facet_cells(&self, facet: FacetId) -> Option<FacetCells1d> {
        if facet.0 >= self.facet_count() {
            return None;
        }
        Some(FacetCells1d {
            minus: (facet.0 > 0).then(|| CellId(facet.0 - 1)),
            plus: (facet.0 < self.cell_count()).then_some(CellId(facet.0)),
        })
    }

    /// Largest cell measure.
    #[must_use]
    pub fn max_cell_measure(&self) -> f64 {
        self.coordinates
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(0.0, f64::max)
    }

    /// Common spacing when all cells are equal to floating-point tolerance.
    #[must_use]
    pub fn uniform_spacing(&self) -> Option<f64> {
        let spacing = self.coordinates[1] - self.coordinates[0];
        let scale = spacing
            .abs()
            .max(self.coordinates[0].abs())
            .max(self.coordinates[self.coordinates.len() - 1].abs());
        let tolerance = 64.0 * f64::EPSILON * scale;
        self.coordinates
            .windows(2)
            .all(|pair| ((pair[1] - pair[0]) - spacing).abs() <= tolerance)
            .then_some(spacing)
    }
}

impl MeshTopology for LineMesh {
    fn topological_dimension(&self) -> usize {
        1
    }

    fn entity_count(&self, dimension: usize) -> Option<usize> {
        match dimension {
            0 => Some(self.vertex_count()),
            1 => Some(self.cell_count()),
            _ => None,
        }
    }

    fn incidence(
        &self,
        entity: MeshEntity,
        target_dimension: usize,
    ) -> Option<Vec<EntityIncidence>> {
        if entity.index >= self.entity_count(entity.dimension)?
            || target_dimension > self.topological_dimension()
        {
            return None;
        }
        match (entity.dimension, target_dimension) {
            (source, target) if source == target => Some(vec![EntityIncidence {
                entity,
                local_ordinal: 0,
                orientation: OrientationCode::identity(),
            }]),
            (1, 0) => Some(vec![
                EntityIncidence {
                    entity: MeshEntity::new(0, entity.index),
                    local_ordinal: 0,
                    orientation: OrientationCode::identity(),
                },
                EntityIncidence {
                    entity: MeshEntity::new(0, entity.index + 1),
                    local_ordinal: 1,
                    orientation: OrientationCode::identity(),
                },
            ]),
            (0, 1) => {
                let mut cells = Vec::with_capacity(2);
                if entity.index > 0 {
                    cells.push(EntityIncidence {
                        entity: MeshEntity::new(1, entity.index - 1),
                        local_ordinal: 1,
                        orientation: OrientationCode::identity(),
                    });
                }
                if entity.index < self.cell_count() {
                    cells.push(EntityIncidence {
                        entity: MeshEntity::new(1, entity.index),
                        local_ordinal: 0,
                        orientation: OrientationCode::identity(),
                    });
                }
                Some(cells)
            }
            _ => Some(Vec::new()),
        }
    }
}

impl MeshGeometry for LineMesh {
    type Map<'a> = LineGeometryMap;

    fn geometric_dimension(&self) -> usize {
        1
    }

    fn geometry_map(&self, entity: MeshEntity) -> Option<Self::Map<'_>> {
        match entity.dimension {
            0 => self
                .facet_geometry(FacetId::new(entity.index))
                .map(LineGeometryMap::Point),
            1 => self
                .cell_geometry(CellId::new(entity.index))
                .map(LineGeometryMap::Segment),
            _ => None,
        }
    }
}

impl GeometryMap for SegmentGeometry1d {
    fn reference_cell(&self) -> ReferenceCell {
        ReferenceCell::segment()
    }

    fn physical_dimension(&self) -> usize {
        1
    }

    fn map_point(&self, reference: &[f64], physical: &mut [f64]) -> Result<(), Diagnostic> {
        check_geometry_buffers(reference, physical, 1, 1)?;
        if !(-1.0..=1.0).contains(&reference[0]) {
            return Err(invalid_mesh(
                "segment reference coordinate lies outside [-1, 1]",
            ));
        }
        physical[0] = self.map(reference[0]);
        Ok(())
    }

    fn jacobian_at(&self, reference: &[f64], jacobian: &mut [f64]) -> Result<(), Diagnostic> {
        check_geometry_buffers(reference, jacobian, 1, 1)?;
        if !(-1.0..=1.0).contains(&reference[0]) {
            return Err(invalid_mesh(
                "segment reference coordinate lies outside [-1, 1]",
            ));
        }
        jacobian[0] = self.jacobian();
        Ok(())
    }
}

impl GeometryMap for PointGeometry1d {
    fn reference_cell(&self) -> ReferenceCell {
        ReferenceCell::point()
    }

    fn physical_dimension(&self) -> usize {
        1
    }

    fn map_point(&self, reference: &[f64], physical: &mut [f64]) -> Result<(), Diagnostic> {
        check_geometry_buffers(reference, physical, 0, 1)?;
        physical[0] = self.coordinate;
        Ok(())
    }

    fn jacobian_at(&self, reference: &[f64], jacobian: &mut [f64]) -> Result<(), Diagnostic> {
        check_geometry_buffers(reference, jacobian, 0, 0)
    }
}

impl GeometryMap for LineGeometryMap {
    fn reference_cell(&self) -> ReferenceCell {
        match self {
            Self::Point(map) => map.reference_cell(),
            Self::Segment(map) => map.reference_cell(),
        }
    }

    fn physical_dimension(&self) -> usize {
        1
    }

    fn map_point(&self, reference: &[f64], physical: &mut [f64]) -> Result<(), Diagnostic> {
        match self {
            Self::Point(map) => map.map_point(reference, physical),
            Self::Segment(map) => map.map_point(reference, physical),
        }
    }

    fn jacobian_at(&self, reference: &[f64], jacobian: &mut [f64]) -> Result<(), Diagnostic> {
        match self {
            Self::Point(map) => map.jacobian_at(reference, jacobian),
            Self::Segment(map) => map.jacobian_at(reference, jacobian),
        }
    }
}

fn check_geometry_buffers(
    reference: &[f64],
    output: &[f64],
    reference_dimension: usize,
    output_length: usize,
) -> Result<(), Diagnostic> {
    if reference.len() != reference_dimension || output.len() != output_length {
        return Err(invalid_mesh(format!(
            "geometry map expected reference/output lengths {reference_dimension}/{output_length}, received {}/{}",
            reference.len(),
            output.len()
        )));
    }
    if reference.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(invalid_mesh(
            "geometry-map reference coordinates must be finite",
        ));
    }
    Ok(())
}

fn invalid_mesh(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_and_geometry_are_oriented_and_distinct() {
        let mesh = LineMesh::from_vertices(vec![-1.0, -0.25, 0.5, 2.0]).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.cell_count(), 3);
        assert_eq!(mesh.facet_count(), 4);
        assert_eq!(
            mesh.cell_vertices(CellId(1)),
            Some([VertexId(1), VertexId(2)])
        );
        assert_eq!(mesh.cell_geometry(CellId(1)).unwrap().center(), 0.125);
        assert_eq!(
            mesh.facet_cells(FacetId(2)),
            Some(FacetCells1d {
                minus: Some(CellId(1)),
                plus: Some(CellId(2)),
            })
        );
        assert_eq!(
            mesh.facet_cells(FacetId(0)),
            Some(FacetCells1d {
                minus: None,
                plus: Some(CellId(0)),
            })
        );

        let closure = mesh.incidence(MeshEntity::new(1, 1), 0).unwrap();
        assert_eq!(closure[0].entity, MeshEntity::new(0, 1));
        assert_eq!(closure[0].local_ordinal, 0);
        assert_eq!(closure[1].entity, MeshEntity::new(0, 2));
        assert_eq!(closure[1].local_ordinal, 1);

        let map = mesh.geometry_map(MeshEntity::new(1, 1)).unwrap();
        let mut physical = [0.0];
        map.map_point(&[0.0], &mut physical).unwrap();
        assert_eq!(physical, [0.125]);
    }

    #[test]
    fn rejects_nonfinite_and_unordered_geometry() {
        assert_eq!(
            LineMesh::from_vertices(vec![0.0, 0.0]).unwrap_err().code(),
            codes::INVALID_MESH
        );
        assert_eq!(
            LineMesh::uniform(0.0, f64::INFINITY, 4).unwrap_err().code(),
            codes::INVALID_MESH
        );
    }
}
