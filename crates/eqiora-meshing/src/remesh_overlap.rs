//! Deterministic common refinements between validated affine triangle meshes.
//!
//! Mesh-local indices remain revision-local.  This module derives positive-
//! measure geometric fragments from coordinates and validates complete
//! coverage in both directions; it never treats equal indices or equal entity
//! counts as correspondence evidence.  Adaptive Shewchuk predicates decide
//! topology, while finite intersection coordinates, measures, and moments are
//! checked independently before admission.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use robust::{Coord, orient2d};

use crate::{CellId, FacetId, MeshEntity, MeshTopology, SimplicialMesh};

mod coverage;
mod intersection;
mod refinement;
mod selection;
use coverage::{validate_cell_coverage, validate_facet_coverage};
use intersection::{collinear_segment_overlap, intersect_triangles};
mod rounding;
use rounding::{certified_line_point, line_parameter_through_rounding_cell};

const DIMENSION: usize = 2;
// This is a deliberately bounded quadratic CPU reference. Production-scale
// construction belongs behind a spatial-search realization, not a larger
// accidental allocation budget in the semantic contract.
const MAX_PAIR_TESTS: usize = 262_144;
const MAX_CELL_FRAGMENTS: usize = 1_048_576;
const MAX_FACET_FRAGMENTS: usize = MAX_PAIR_TESTS;

type Point2d = [f64; DIMENSION];
type Triangle2d = [Point2d; 3];
type Segment2d = [Point2d; 2];

/// Coordinate chart in which a common refinement was constructed.
///
/// The chart is an exact identity-bearing role for later artifact and
/// numerical consumers.  This type does not transform coordinates or infer a
/// chart from which Field happens to consume the overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OverlapCoordinateChart2d {
    /// Material/reference coordinates, used by absolute solid state.
    Material,
    /// Current spatial coordinates, used by the deformed fluid state.
    CurrentSpatial,
}

/// One retained boundary facet together with its exact parent cell.
///
/// No orientation or normal is accepted from the caller.  The parent must be
/// the unique selected cell incident to the facet; its geometry derives the
/// parent-outward unit normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RetainedFacetSide2d {
    facet: FacetId,
    parent: CellId,
}

impl RetainedFacetSide2d {
    /// Name one mesh-local facet and its selected parent cell.
    #[must_use]
    pub const fn new(facet: FacetId, parent: CellId) -> Self {
        Self { facet, parent }
    }

    /// Mesh-local retained facet.
    #[must_use]
    pub const fn facet(self) -> FacetId {
        self.facet
    }

    /// Exact selected parent cell from which outward orientation is derived.
    #[must_use]
    pub const fn parent(self) -> CellId {
        self.parent
    }
}

/// One canonical positive triangle in a cell-pair common refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct RevisionCellFragment2d {
    source_cell: CellId,
    target_cell: CellId,
    triangle: Triangle2d,
    area: f64,
    first_moment: Point2d,
}

impl RevisionCellFragment2d {
    /// Exact source cell owning this fragment.
    #[must_use]
    pub const fn source_cell(&self) -> CellId {
        self.source_cell
    }

    /// Exact target cell owning this fragment.
    #[must_use]
    pub const fn target_cell(&self) -> CellId {
        self.target_cell
    }

    /// Positively oriented canonical triangle coordinates.
    #[must_use]
    pub const fn triangle(&self) -> &Triangle2d {
        &self.triangle
    }

    /// Strictly positive finite triangle area.
    #[must_use]
    pub const fn area(&self) -> f64 {
        self.area
    }

    /// Exact-form first coordinate moment `integral [x, y] dA` evaluated in
    /// binary64 over this affine triangle.
    #[must_use]
    pub const fn first_moment(&self) -> Point2d {
        self.first_moment
    }
}

/// One canonical positive segment in a retained-facet common refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct RevisionFacetFragment2d {
    source: RetainedFacetSide2d,
    target: RetainedFacetSide2d,
    segment: Segment2d,
    length: f64,
    first_moment: Point2d,
    source_outward_normal: Point2d,
    target_outward_normal: Point2d,
}

impl RevisionFacetFragment2d {
    /// Exact source facet.
    #[must_use]
    pub const fn source_facet(&self) -> FacetId {
        self.source.facet()
    }

    /// Exact source parent cell.
    #[must_use]
    pub const fn source_parent(&self) -> CellId {
        self.source.parent()
    }

    /// Exact target facet.
    #[must_use]
    pub const fn target_facet(&self) -> FacetId {
        self.target.facet()
    }

    /// Exact target parent cell.
    #[must_use]
    pub const fn target_parent(&self) -> CellId {
        self.target.parent()
    }

    /// Canonical lexicographically increasing intersection segment.
    #[must_use]
    pub const fn segment(&self) -> &Segment2d {
        &self.segment
    }

    /// Strictly positive finite segment length.
    #[must_use]
    pub const fn length(&self) -> f64 {
        self.length
    }

    /// First coordinate moment `integral [x, y] ds`.
    #[must_use]
    pub const fn first_moment(&self) -> Point2d {
        self.first_moment
    }

    /// Source parent-outward unit normal, derived from incidence and geometry.
    #[must_use]
    pub const fn source_outward_normal(&self) -> Point2d {
        self.source_outward_normal
    }

    /// Target parent-outward unit normal, derived from incidence and geometry.
    #[must_use]
    pub const fn target_outward_normal(&self) -> Point2d {
        self.target_outward_normal
    }
}

/// Accepted many-to-many common refinement of two selected triangle regions.
///
/// Source and target inventories are canonicalized but remain distinct.  Cell
/// fragments prove complete area and first-moment coverage for every selected
/// entity in both revisions.  Retained facets are attached through
/// [`Self::with_retained_facets`], which additionally proves true subset-
/// frontier incidence, derived parent-outward orientation, and bidirectional
/// length/moment coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialRevisionOverlap2d {
    chart: OverlapCoordinateChart2d,
    source_cells: Vec<CellId>,
    target_cells: Vec<CellId>,
    source_triangles: Vec<Triangle2d>,
    target_triangles: Vec<Triangle2d>,
    cell_fragments: Vec<RevisionCellFragment2d>,
    source_retained_facets: Vec<RetainedFacetSide2d>,
    target_retained_facets: Vec<RetainedFacetSide2d>,
    retained_facet_fragments: Vec<RevisionFacetFragment2d>,
}

impl SimplicialRevisionOverlap2d {
    /// Derive and validate one selected-region common refinement.
    ///
    /// Cell declaration order is non-semantic and is canonicalized. Duplicate
    /// or out-of-range cells fail rather than disappearing.  Every selected
    /// source and target cell must be covered completely by positive fragments
    /// in the other revision.
    ///
    /// # Errors
    /// Returns `EQ0803` for a non-2D mesh, invalid/duplicate/empty inventory,
    /// excessive quadratic work, robustly degenerate input, uncertifiable
    /// intersection construction, or incomplete bidirectional coverage.
    pub fn new(
        chart: OverlapCoordinateChart2d,
        source: &SimplicialMesh,
        source_cells: &[CellId],
        target: &SimplicialMesh,
        target_cells: &[CellId],
    ) -> Result<Self, Diagnostic> {
        require_triangle_mesh(source, "source")?;
        require_triangle_mesh(target, "target")?;
        let source_cells = canonical_cells(source, source_cells, "source")?;
        let target_cells = canonical_cells(target, target_cells, "target")?;
        require_pair_budget(source_cells.len(), target_cells.len(), "cell")?;

        let source_triangles = selected_triangles(source, &source_cells, "source")?;
        let target_triangles = selected_triangles(target, &target_cells, "target")?;
        let mut fragments = Vec::new();
        for (source_cell, source_triangle) in source_cells.iter().zip(&source_triangles) {
            let source_bounds = Bounds2d::new(source_triangle);
            for (target_cell, target_triangle) in target_cells.iter().zip(&target_triangles) {
                if !source_bounds.intersects(Bounds2d::new(target_triangle)) {
                    continue;
                }
                for triangle in intersect_triangles(source_triangle, target_triangle)? {
                    let (area, first_moment) = triangle_measure(&triangle)?;
                    push_bounded(
                        &mut fragments,
                        RevisionCellFragment2d {
                            source_cell: *source_cell,
                            target_cell: *target_cell,
                            triangle,
                            area,
                            first_moment,
                        },
                        MAX_CELL_FRAGMENTS,
                        "cell fragment",
                    )?;
                }
            }
        }
        fragments.sort_by(compare_cell_fragments);
        if fragments.is_empty() {
            return Err(invalid_overlap(
                "selected source and target regions have no positive-area common refinement",
            ));
        }
        require_unique_cell_fragments(&fragments)?;
        validate_cell_coverage(
            &source_cells,
            &source_triangles,
            &target_cells,
            &target_triangles,
            &fragments,
        )?;

        Ok(Self {
            chart,
            source_cells,
            target_cells,
            source_triangles,
            target_triangles,
            cell_fragments: fragments,
            source_retained_facets: Vec::new(),
            target_retained_facets: Vec::new(),
            retained_facet_fragments: Vec::new(),
        })
    }

    /// Attach one retained semantic-boundary facet relation atomically.
    ///
    /// Side declaration order is non-semantic and is canonicalized. Duplicate
    /// facets fail. Each side must be a true frontier facet of this overlap's
    /// already admitted cell subset, with the supplied parent as its unique
    /// selected incident cell. Outward normals are derived, never accepted.
    /// Every supplied facet must then have complete positive-length coverage
    /// in both revisions.
    ///
    /// # Errors
    /// Returns `EQ0803` for stale meshes, duplicate/invalid/non-frontier sides,
    /// excessive work, inconsistent parent-outward orientation, degenerate
    /// segments, or incomplete bidirectional retained-facet coverage.
    pub fn with_retained_facets(
        mut self,
        source: &SimplicialMesh,
        source_sides: &[RetainedFacetSide2d],
        target: &SimplicialMesh,
        target_sides: &[RetainedFacetSide2d],
    ) -> Result<Self, Diagnostic> {
        require_triangle_mesh(source, "source")?;
        require_triangle_mesh(target, "target")?;
        // Replaying exact normalized coordinates binds facet derivation to the
        // admitted revisions. Area and first moment alone cannot distinguish a
        // sheared triangle with the same measure and centroid.
        let source_triangles = selected_triangles(source, &self.source_cells, "source")?;
        let target_triangles = selected_triangles(target, &self.target_cells, "target")?;
        require_same_triangles(&source_triangles, &self.source_triangles, "source")?;
        require_same_triangles(&target_triangles, &self.target_triangles, "target")?;
        validate_cell_coverage(
            &self.source_cells,
            &self.source_triangles,
            &self.target_cells,
            &self.target_triangles,
            &self.cell_fragments,
        )?;

        let source_sides = canonical_sides(source, &self.source_cells, source_sides, "source")?;
        let target_sides = canonical_sides(target, &self.target_cells, target_sides, "target")?;
        require_pair_budget(source_sides.len(), target_sides.len(), "retained facet")?;
        let source_geometry = side_geometry(source, &source_sides, "source")?;
        let target_geometry = side_geometry(target, &target_sides, "target")?;

        let mut fragments = Vec::new();
        for (source_side, source_geometry) in source_sides.iter().zip(&source_geometry) {
            for (target_side, target_geometry) in target_sides.iter().zip(&target_geometry) {
                let Some(segment) =
                    collinear_segment_overlap(&source_geometry.segment, &target_geometry.segment)?
                else {
                    continue;
                };
                let normal_dot = dot(
                    source_geometry.outward_normal,
                    target_geometry.outward_normal,
                );
                if !normal_dot.is_finite() || normal_dot <= 0.0 {
                    return Err(invalid_overlap(
                        "retained source and target facet parents derive incompatible outward orientations",
                    ));
                }
                let (length, first_moment) = segment_measure(&segment)?;
                push_bounded(
                    &mut fragments,
                    RevisionFacetFragment2d {
                        source: *source_side,
                        target: *target_side,
                        segment,
                        length,
                        first_moment,
                        source_outward_normal: source_geometry.outward_normal,
                        target_outward_normal: target_geometry.outward_normal,
                    },
                    MAX_FACET_FRAGMENTS,
                    "retained-facet fragment",
                )?;
            }
        }
        fragments.sort_by(compare_facet_fragments);
        if fragments.is_empty() {
            return Err(invalid_overlap(
                "retained source and target facets have no positive-length common refinement",
            ));
        }
        require_unique_facet_fragments(&fragments)?;
        validate_facet_coverage(
            &source_sides,
            &source_geometry,
            &target_sides,
            &target_geometry,
            &fragments,
        )?;

        self.source_retained_facets = source_sides;
        self.target_retained_facets = target_sides;
        self.retained_facet_fragments = fragments;
        Ok(self)
    }

    /// Exact coordinate chart of every fragment.
    #[must_use]
    pub const fn chart(&self) -> OverlapCoordinateChart2d {
        self.chart
    }

    /// Canonically ordered selected source cells.
    #[must_use]
    pub fn source_cells(&self) -> &[CellId] {
        &self.source_cells
    }

    /// Canonically ordered selected target cells.
    #[must_use]
    pub fn target_cells(&self) -> &[CellId] {
        &self.target_cells
    }

    /// Canonical `(source cell, target cell, fragment)` common refinement.
    #[must_use]
    pub fn cell_fragments(&self) -> &[RevisionCellFragment2d] {
        &self.cell_fragments
    }

    /// Canonically ordered retained source facets and derived-parent inputs.
    #[must_use]
    pub fn source_retained_facets(&self) -> &[RetainedFacetSide2d] {
        &self.source_retained_facets
    }

    /// Canonically ordered retained target facets and derived-parent inputs.
    #[must_use]
    pub fn target_retained_facets(&self) -> &[RetainedFacetSide2d] {
        &self.target_retained_facets
    }

    /// Canonical retained-facet segment common refinement.
    #[must_use]
    pub fn retained_facet_fragments(&self) -> &[RevisionFacetFragment2d] {
        &self.retained_facet_fragments
    }
}

#[derive(Debug, Clone, Copy)]
struct Bounds2d {
    minimum: Point2d,
    maximum: Point2d,
}

impl Bounds2d {
    fn new(points: &[Point2d]) -> Self {
        let mut minimum = [f64::INFINITY; DIMENSION];
        let mut maximum = [f64::NEG_INFINITY; DIMENSION];
        for point in points {
            for axis in 0..DIMENSION {
                minimum[axis] = minimum[axis].min(point[axis]);
                maximum[axis] = maximum[axis].max(point[axis]);
            }
        }
        Self { minimum, maximum }
    }

    fn intersects(self, other: Self) -> bool {
        (0..DIMENSION).all(|axis| {
            self.minimum[axis] <= other.maximum[axis] && other.minimum[axis] <= self.maximum[axis]
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SideGeometry2d {
    segment: Segment2d,
    outward_normal: Point2d,
    length: f64,
    first_moment: Point2d,
}

fn require_triangle_mesh(mesh: &SimplicialMesh, label: &str) -> Result<(), Diagnostic> {
    if mesh.topological_dimension() != DIMENSION
        || mesh.vertices().iter().any(|vertex| {
            vertex.len() != DIMENSION || vertex.iter().any(|value| !value.is_finite())
        })
        || mesh.cells().iter().any(|cell| cell.len() != 3)
    {
        return Err(invalid_overlap(format!(
            "{label} common-refinement mesh must be one finite intrinsic-2D affine-triangle revision",
        )));
    }
    Ok(())
}

fn canonical_cells(
    mesh: &SimplicialMesh,
    cells: &[CellId],
    label: &str,
) -> Result<Vec<CellId>, Diagnostic> {
    if cells.is_empty() {
        return Err(invalid_overlap(format!(
            "{label} common-refinement cell inventory must be nonempty",
        )));
    }
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("accepted 2D mesh owns cells");
    let mut canonical = cells.to_vec();
    canonical.sort_unstable();
    if canonical.iter().any(|cell| cell.index() >= cell_count) {
        return Err(invalid_overlap(format!(
            "{label} common-refinement cell is outside its mesh revision",
        )));
    }
    if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid_overlap(format!(
            "{label} common-refinement cell inventory contains a duplicate",
        )));
    }
    Ok(canonical)
}

fn canonical_sides(
    mesh: &SimplicialMesh,
    selected_cells: &[CellId],
    sides: &[RetainedFacetSide2d],
    label: &str,
) -> Result<Vec<RetainedFacetSide2d>, Diagnostic> {
    if sides.is_empty() {
        return Err(invalid_overlap(format!(
            "{label} retained-facet inventory must be nonempty",
        )));
    }
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .expect("accepted 2D mesh owns facets");
    let mut canonical = sides.to_vec();
    canonical.sort_unstable();
    if canonical
        .iter()
        .any(|side| side.facet().index() >= facet_count)
    {
        return Err(invalid_overlap(format!(
            "{label} retained facet is outside its mesh revision",
        )));
    }
    if canonical
        .windows(2)
        .any(|pair| pair[0].facet() == pair[1].facet())
    {
        return Err(invalid_overlap(format!(
            "{label} retained-facet inventory contains a duplicate facet",
        )));
    }
    for side in &canonical {
        if selected_cells.binary_search(&side.parent()).is_err() {
            return Err(invalid_overlap(format!(
                "{label} retained-facet parent is absent from the admitted cell subset",
            )));
        }
        let adjacent = mesh
            .incidence(
                MeshEntity::new(DIMENSION - 1, side.facet().index()),
                DIMENSION,
            )
            .ok_or_else(|| {
                invalid_overlap(format!("{label} retained-facet incidence is absent"))
            })?;
        let selected = adjacent
            .iter()
            .filter(|entry| {
                selected_cells
                    .binary_search(&CellId::new(entry.entity.index()))
                    .is_ok()
            })
            .map(|entry| CellId::new(entry.entity.index()))
            .collect::<Vec<_>>();
        if selected.as_slice() != [side.parent()] {
            return Err(invalid_overlap(format!(
                "{label} retained facet must be a true frontier with its supplied parent as the unique selected incident cell",
            )));
        }
    }
    Ok(canonical)
}

fn require_pair_budget(source: usize, target: usize, label: &str) -> Result<(), Diagnostic> {
    if source
        .checked_mul(target)
        .is_none_or(|pairs| pairs > MAX_PAIR_TESTS)
    {
        return Err(invalid_overlap(format!(
            "quadratic {label} common-refinement work exceeds the bounded reference limit",
        )));
    }
    Ok(())
}

fn push_bounded<T>(
    values: &mut Vec<T>,
    value: T,
    limit: usize,
    label: &str,
) -> Result<(), Diagnostic> {
    if values.len() >= limit {
        return Err(invalid_overlap(format!(
            "{label} count exceeds the bounded reference limit",
        )));
    }
    if values.len() == values.capacity() {
        values.try_reserve(1).map_err(|_| {
            invalid_overlap(format!(
                "memory for the bounded {label} inventory is unavailable",
            ))
        })?;
    }
    values.push(value);
    Ok(())
}

fn selected_triangles(
    mesh: &SimplicialMesh,
    cells: &[CellId],
    label: &str,
) -> Result<Vec<Triangle2d>, Diagnostic> {
    cells
        .iter()
        .map(|cell| {
            let connectivity = mesh.cells().get(cell.index()).ok_or_else(|| {
                invalid_overlap(format!("{label} common-refinement cell is unavailable"))
            })?;
            let triangle = std::array::from_fn(|local| {
                let vertex = &mesh.vertices()[connectivity[local]];
                [normalize_zero(vertex[0]), normalize_zero(vertex[1])]
            });
            require_positive_triangle(&triangle, label)?;
            Ok(triangle)
        })
        .collect()
}

fn require_same_triangles(
    actual: &[Triangle2d],
    admitted: &[Triangle2d],
    label: &str,
) -> Result<(), Diagnostic> {
    let exact = actual.len() == admitted.len()
        && actual.iter().zip(admitted).all(|(left, right)| {
            left.iter().flatten().zip(right.iter().flatten()).all(
                |(left_coordinate, right_coordinate)| {
                    left_coordinate.to_bits() == right_coordinate.to_bits()
                },
            )
        });
    if !exact {
        return Err(invalid_overlap(format!(
            "{label} selected triangle geometry does not exactly match the admitted mesh revision",
        )));
    }
    Ok(())
}

fn side_geometry(
    mesh: &SimplicialMesh,
    sides: &[RetainedFacetSide2d],
    label: &str,
) -> Result<Vec<SideGeometry2d>, Diagnostic> {
    sides
        .iter()
        .map(|side| {
            let facet = MeshEntity::new(DIMENSION - 1, side.facet().index());
            let mut facet_vertices = mesh.entity_vertices(facet).ok_or_else(|| {
                invalid_overlap(format!("{label} retained facet has no vertices"))
            })?;
            if facet_vertices.len() != 2 {
                return Err(invalid_overlap(format!(
                    "{label} retained facet is not a segment",
                )));
            }
            facet_vertices.sort_by_key(|vertex| vertex.index());
            let mut segment = [
                point(mesh, facet_vertices[0].index())?,
                point(mesh, facet_vertices[1].index())?,
            ];
            canonicalize_segment(&mut segment);
            let (length, first_moment) = segment_measure(&segment)?;

            let parent_vertices = mesh.cells().get(side.parent().index()).ok_or_else(|| {
                invalid_overlap(format!("{label} retained parent is unavailable"))
            })?;
            let opposite = parent_vertices
                .iter()
                .copied()
                .find(|vertex| !facet_vertices.iter().any(|entry| entry.index() == *vertex))
                .ok_or_else(|| {
                    invalid_overlap(format!(
                        "{label} retained facet has no unique opposite parent vertex",
                    ))
                })?;
            let opposite = point(mesh, opposite)?;
            let side_sign = orientation(segment[0], segment[1], opposite);
            if side_sign == 0.0 || !side_sign.is_finite() {
                return Err(invalid_overlap(format!(
                    "{label} retained facet parent orientation is degenerate",
                )));
            }
            let tangent = [segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]];
            let outward_normal = if side_sign > 0.0 {
                [tangent[1] / length, -tangent[0] / length]
            } else {
                [-tangent[1] / length, tangent[0] / length]
            }
            .map(normalize_zero);
            if outward_normal.iter().any(|value| !value.is_finite()) {
                return Err(invalid_overlap(format!(
                    "{label} retained-facet outward normal is non-finite",
                )));
            }
            Ok(SideGeometry2d {
                segment,
                outward_normal,
                length,
                first_moment,
            })
        })
        .collect()
}

fn point(mesh: &SimplicialMesh, vertex: usize) -> Result<Point2d, Diagnostic> {
    let value = mesh
        .vertices()
        .get(vertex)
        .ok_or_else(|| invalid_overlap("common-refinement vertex is unavailable"))?;
    if value.len() != DIMENSION || value.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(invalid_overlap(
            "common-refinement vertex must be one finite 2D point",
        ));
    }
    Ok([normalize_zero(value[0]), normalize_zero(value[1])])
}

fn require_positive_triangle(triangle: &Triangle2d, label: &str) -> Result<(), Diagnostic> {
    if triangle.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid_overlap(format!(
            "{label} common-refinement triangle contains a non-finite coordinate",
        )));
    }
    let sign = orientation(triangle[0], triangle[1], triangle[2]);
    if !sign.is_finite() || sign <= 0.0 {
        return Err(invalid_overlap(format!(
            "{label} common-refinement triangle is robustly degenerate or inverted",
        )));
    }
    Ok(())
}

fn triangle_measure(triangle: &Triangle2d) -> Result<(f64, Point2d), Diagnostic> {
    let area = 0.5 * orientation(triangle[0], triangle[1], triangle[2]);
    let first_moment = std::array::from_fn(|axis| {
        area * (triangle[0][axis] + triangle[1][axis] + triangle[2][axis]) / 3.0
    });
    if !area.is_finite() || area <= 0.0 || first_moment.iter().any(|value| !value.is_finite()) {
        return Err(invalid_overlap(
            "common-refinement triangle measure or first moment is non-finite or non-positive",
        ));
    }
    Ok((area, first_moment))
}

fn segment_measure(segment: &Segment2d) -> Result<(f64, Point2d), Diagnostic> {
    let dx = segment[1][0] - segment[0][0];
    let dy = segment[1][1] - segment[0][1];
    let length = dx.hypot(dy);
    let first_moment = [
        length * 0.5 * (segment[0][0] + segment[1][0]),
        length * 0.5 * (segment[0][1] + segment[1][1]),
    ];
    if !length.is_finite() || length <= 0.0 || first_moment.iter().any(|value| !value.is_finite()) {
        return Err(invalid_overlap(
            "common-refinement retained segment is degenerate or non-finite",
        ));
    }
    Ok((length, first_moment))
}

fn require_unique_cell_fragments(fragments: &[RevisionCellFragment2d]) -> Result<(), Diagnostic> {
    if fragments.windows(2).any(|pair| {
        pair[0].source_cell == pair[1].source_cell
            && pair[0].target_cell == pair[1].target_cell
            && pair[0].triangle == pair[1].triangle
    }) {
        Err(invalid_overlap(
            "common refinement contains a duplicate cell fragment",
        ))
    } else {
        Ok(())
    }
}

fn require_unique_facet_fragments(fragments: &[RevisionFacetFragment2d]) -> Result<(), Diagnostic> {
    if fragments.windows(2).any(|pair| {
        pair[0].source == pair[1].source
            && pair[0].target == pair[1].target
            && pair[0].segment == pair[1].segment
    }) {
        Err(invalid_overlap(
            "common refinement contains a duplicate retained-facet fragment",
        ))
    } else {
        Ok(())
    }
}

fn compare_cell_fragments(
    left: &RevisionCellFragment2d,
    right: &RevisionCellFragment2d,
) -> Ordering {
    left.source_cell
        .cmp(&right.source_cell)
        .then_with(|| left.target_cell.cmp(&right.target_cell))
        .then_with(|| compare_point_arrays(&left.triangle, &right.triangle))
}

fn compare_facet_fragments(
    left: &RevisionFacetFragment2d,
    right: &RevisionFacetFragment2d,
) -> Ordering {
    left.source
        .cmp(&right.source)
        .then_with(|| left.target.cmp(&right.target))
        .then_with(|| compare_point_arrays(&left.segment, &right.segment))
}

fn compare_point_arrays<const N: usize>(left: &[Point2d; N], right: &[Point2d; N]) -> Ordering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let order = compare_points(left, right);
            (order != Ordering::Equal).then_some(order)
        })
        .unwrap_or(Ordering::Equal)
}

fn compare_points(left: &Point2d, right: &Point2d) -> Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
}

fn canonicalize_segment(segment: &mut Segment2d) {
    segment.iter_mut().flatten().for_each(|value| {
        *value = normalize_zero(*value);
    });
    if compare_points(&segment[1], &segment[0]) == Ordering::Less {
        segment.swap(0, 1);
    }
}

fn orientation(left: Point2d, right: Point2d, query: Point2d) -> f64 {
    orient2d(
        Coord {
            x: left[0],
            y: left[1],
        },
        Coord {
            x: right[0],
            y: right[1],
        },
        Coord {
            x: query[0],
            y: query[1],
        },
    )
}

fn dot(left: Point2d, right: Point2d) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

const fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn invalid_overlap(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests;
