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
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use robust::{Coord, orient2d};

use crate::{CellId, FacetId, MeshEntity, MeshTopology, SimplicialMesh};

mod rounding;
use rounding::{certified_line_point, line_parameter_through_rounding_cell};

const DIMENSION: usize = 2;
// This is a deliberately bounded quadratic CPU reference. Production-scale
// construction belongs behind a spatial-search realization, not a larger
// accidental allocation budget in the semantic contract.
const MAX_PAIR_TESTS: usize = 262_144;
const MAX_CELL_FRAGMENTS: usize = 1_048_576;
const MAX_FACET_FRAGMENTS: usize = MAX_PAIR_TESTS;
const COVERAGE_ULPS: f64 = 16_384.0;

type Point2d = [f64; DIMENSION];
type Triangle2d = [Point2d; 3];
type Segment2d = [Point2d; 2];

#[derive(Debug, Clone, PartialEq)]
struct ConstructedPoint2d {
    exact: [BigRational; DIMENSION],
    rounded: Point2d,
}

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

#[derive(Debug, Clone, Copy, Default)]
struct CoverageAccumulator {
    area_or_length: CompensatedSum,
    first_moment: [CompensatedSum; DIMENSION],
    term_count: usize,
}

impl CoverageAccumulator {
    fn add(&mut self, measure: f64, first_moment: Point2d) -> Result<(), Diagnostic> {
        self.area_or_length.add(measure)?;
        for (sum, value) in self.first_moment.iter_mut().zip(first_moment) {
            sum.add(value)?;
        }
        self.term_count = self
            .term_count
            .checked_add(1)
            .ok_or_else(|| invalid_overlap("common-refinement coverage count overflows usize"))?;
        Ok(())
    }

    fn require(self, measure: f64, first_moment: Point2d, label: &str) -> Result<(), Diagnostic> {
        require_close(
            self.area_or_length,
            measure,
            self.term_count,
            &format!("{label} measure"),
        )?;
        for (axis, (sum, expected)) in self.first_moment.into_iter().zip(first_moment).enumerate() {
            require_close(
                sum,
                expected,
                self.term_count,
                &format!("{label} first moment axis {axis}"),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
    absolute_sum: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), Diagnostic> {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        self.absolute_sum += value.abs();
        if !self.sum.is_finite() || !self.correction.is_finite() || !self.absolute_sum.is_finite() {
            return Err(invalid_overlap(
                "common-refinement coverage accumulation overflowed",
            ));
        }
        Ok(())
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
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

fn intersect_triangles(
    source: &Triangle2d,
    target: &Triangle2d,
) -> Result<Vec<Triangle2d>, Diagnostic> {
    require_positive_triangle(source, "source")?;
    require_positive_triangle(target, "target")?;
    let mut candidates = Vec::with_capacity(12);
    for point in source
        .iter()
        .copied()
        .filter(|point| point_in_closed_triangle(*point, target))
    {
        candidates.push(ConstructedPoint2d::from_binary64(point)?);
    }
    for point in target
        .iter()
        .copied()
        .filter(|point| point_in_closed_triangle(*point, source))
    {
        candidates.push(ConstructedPoint2d::from_binary64(point)?);
    }
    for source_edge in triangle_edges(source) {
        for target_edge in triangle_edges(target) {
            if let Some(point) = proper_segment_intersection(source_edge, target_edge)? {
                candidates.push(point);
            }
        }
    }
    let polygon = convex_hull(candidates);
    if polygon.len() < 3 {
        return Ok(Vec::new());
    }
    let mut triangles = Vec::with_capacity(polygon.len() - 2);
    for index in 1..polygon.len() - 1 {
        let exact_sign = exact_orientation(
            &polygon[0].exact,
            &polygon[index].exact,
            &polygon[index + 1].exact,
        );
        if exact_sign.is_zero() {
            return Err(invalid_overlap(
                "exact common-refinement polygon triangulation is degenerate",
            ));
        }
        if exact_sign < BigRational::zero() {
            return Err(invalid_overlap(
                "canonical common-refinement polygon lost positive orientation",
            ));
        }
        let triangle = [
            polygon[0].rounded,
            polygon[index].rounded,
            polygon[index + 1].rounded,
        ];
        triangle_measure(&triangle).map_err(|_| {
            invalid_overlap(
                "exact positive common-refinement fragment collapses under canonical nearest-binary64 coordinates",
            )
        })?;
        triangles.push(triangle);
    }
    Ok(triangles)
}

fn point_in_closed_triangle(point: Point2d, triangle: &Triangle2d) -> bool {
    triangle_edges(triangle)
        .into_iter()
        .all(|edge| orientation(edge[0], edge[1], point) >= 0.0)
}

fn triangle_edges(triangle: &Triangle2d) -> [Segment2d; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

fn proper_segment_intersection(
    source: Segment2d,
    target: Segment2d,
) -> Result<Option<ConstructedPoint2d>, Diagnostic> {
    let source_start = orientation(target[0], target[1], source[0]);
    let source_end = orientation(target[0], target[1], source[1]);
    let target_start = orientation(source[0], source[1], target[0]);
    let target_end = orientation(source[0], source[1], target[1]);
    if !opposite_sign(source_start, source_end) || !opposite_sign(target_start, target_end) {
        return Ok(None);
    }
    let source_start = exact_point(source[0])?;
    let source_direction = exact_direction(source[0], source[1])?;
    let target_start = exact_point(target[0])?;
    let target_direction = exact_direction(target[0], target[1])?;
    let offset = [
        &target_start[0] - &source_start[0],
        &target_start[1] - &source_start[1],
    ];
    let denominator = exact_cross(&source_direction, &target_direction);
    if denominator.is_zero() {
        return Err(invalid_overlap(
            "robust proper crossing has an exactly parallel construction denominator",
        ));
    }
    let source_parameter = exact_cross(&offset, &target_direction) / &denominator;
    let target_parameter = exact_cross(&offset, &source_direction) / denominator;
    if source_parameter <= BigRational::zero()
        || source_parameter >= BigRational::one()
        || target_parameter <= BigRational::zero()
        || target_parameter >= BigRational::one()
    {
        return Err(invalid_overlap(
            "robust proper crossing disagrees with exact binary-rational segment parameters",
        ));
    }
    let exact = [
        &source_start[0] + &source_parameter * &source_direction[0],
        &source_start[1] + source_parameter * &source_direction[1],
    ];
    let point = [
        round_exact_coordinate(&exact[0])?,
        round_exact_coordinate(&exact[1])?,
    ];
    Ok(Some(ConstructedPoint2d {
        exact,
        rounded: point.map(normalize_zero),
    }))
}

fn exact_point(point: Point2d) -> Result<[BigRational; DIMENSION], Diagnostic> {
    point
        .map(|coordinate| {
            BigRational::from_float(coordinate).ok_or_else(|| {
                invalid_overlap("finite mesh coordinate has no exact binary-rational image")
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid_overlap("exact point construction lost its fixed dimension"))
}

impl ConstructedPoint2d {
    fn from_binary64(point: Point2d) -> Result<Self, Diagnostic> {
        Ok(Self {
            exact: exact_point(point)?,
            rounded: point.map(normalize_zero),
        })
    }
}

fn exact_direction(start: Point2d, end: Point2d) -> Result<[BigRational; DIMENSION], Diagnostic> {
    let start = exact_point(start)?;
    let end = exact_point(end)?;
    Ok([&end[0] - &start[0], &end[1] - &start[1]])
}

fn exact_cross(left: &[BigRational; DIMENSION], right: &[BigRational; DIMENSION]) -> BigRational {
    &left[0] * &right[1] - &left[1] * &right[0]
}

fn exact_orientation(
    left: &[BigRational; DIMENSION],
    right: &[BigRational; DIMENSION],
    query: &[BigRational; DIMENSION],
) -> BigRational {
    let first = [&right[0] - &left[0], &right[1] - &left[1]];
    let second = [&query[0] - &left[0], &query[1] - &left[1]];
    exact_cross(&first, &second)
}

fn round_exact_coordinate(value: &BigRational) -> Result<f64, Diagnostic> {
    let rounded = value.to_f64().ok_or_else(|| {
        invalid_overlap("exact proper-intersection coordinate is outside binary64 range")
    })?;
    if !rounded.is_finite() {
        return Err(invalid_overlap(
            "exact proper-intersection coordinate rounded to a non-finite binary64 value",
        ));
    }
    let represented = BigRational::from_float(rounded)
        .expect("a finite binary64 coordinate always has an exact rational image");
    let certified = match value.cmp(&represented) {
        Ordering::Equal => true,
        Ordering::Greater => {
            BigRational::from_float(next_up(rounded)).is_some_and(|upper| value <= &upper)
        }
        Ordering::Less => {
            BigRational::from_float(next_down(rounded)).is_some_and(|lower| value >= &lower)
        }
    };
    if !certified {
        return Err(invalid_overlap(
            "exact proper-intersection coordinate did not round within one binary64 step",
        ));
    }
    Ok(rounded)
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits((1_u64 << 63) | 1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn opposite_sign(left: f64, right: f64) -> bool {
    (left < 0.0 && right > 0.0) || (left > 0.0 && right < 0.0)
}

fn convex_hull(mut points: Vec<ConstructedPoint2d>) -> Vec<ConstructedPoint2d> {
    points.sort_by(compare_constructed_points);
    points.dedup_by(|left, right| left.exact == right.exact);
    if points.len() < 3 {
        return points;
    }
    let mut lower: Vec<ConstructedPoint2d> = Vec::with_capacity(points.len());
    for point in &points {
        while lower.len() >= 2
            && exact_orientation(
                &lower[lower.len() - 2].exact,
                &lower[lower.len() - 1].exact,
                &point.exact,
            ) <= BigRational::zero()
        {
            lower.pop();
        }
        lower.push(point.clone());
    }
    let mut upper: Vec<ConstructedPoint2d> = Vec::with_capacity(points.len());
    for point in points.iter().rev() {
        while upper.len() >= 2
            && exact_orientation(
                &upper[upper.len() - 2].exact,
                &upper[upper.len() - 1].exact,
                &point.exact,
            ) <= BigRational::zero()
        {
            upper.pop();
        }
        upper.push(point.clone());
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn compare_constructed_points(left: &ConstructedPoint2d, right: &ConstructedPoint2d) -> Ordering {
    left.exact[0]
        .cmp(&right.exact[0])
        .then_with(|| left.exact[1].cmp(&right.exact[1]))
}

fn collinear_segment_overlap(
    source: &Segment2d,
    target: &Segment2d,
) -> Result<Option<Segment2d>, Diagnostic> {
    let source_signs = [
        orientation(source[0], source[1], target[0]),
        orientation(source[0], source[1], target[1]),
    ];
    let target_signs = [
        orientation(target[0], target[1], source[0]),
        orientation(target[0], target[1], source[1]),
    ];
    if source_signs.iter().any(|sign| !sign.is_finite())
        || target_signs.iter().any(|sign| !sign.is_finite())
    {
        return Err(invalid_overlap(
            "retained-facet collinearity predicate is non-finite",
        ));
    }
    let exactly_collinear = source_signs == [0.0, 0.0] && target_signs == [0.0, 0.0];
    let representative = if compare_point_arrays(source, target).is_le() {
        source
    } else {
        target
    };
    let axis = dominant_axis(representative);
    let direction = (representative[1][axis] - representative[0][axis]).signum();
    let source_ordered = ordered_along(source, axis, direction);
    let target_ordered = ordered_along(target, axis, direction);
    let lower = if projected_coordinate(source_ordered[0], axis, direction)
        >= projected_coordinate(target_ordered[0], axis, direction)
    {
        source_ordered[0]
    } else {
        target_ordered[0]
    };
    let upper = if projected_coordinate(source_ordered[1], axis, direction)
        <= projected_coordinate(target_ordered[1], axis, direction)
    {
        source_ordered[1]
    } else {
        target_ordered[1]
    };
    if projected_coordinate(upper, axis, direction) <= projected_coordinate(lower, axis, direction)
    {
        return Ok(None);
    }
    if !exactly_collinear
        && ![lower, upper]
            .into_iter()
            .map(|point| {
                Ok(
                    line_parameter_through_rounding_cell(source, point)?.is_some()
                        && line_parameter_through_rounding_cell(target, point)?.is_some(),
                )
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?
            .into_iter()
            .all(|certified| certified)
    {
        return Ok(None);
    }
    let mut segment = [
        certified_line_point(representative, lower)?,
        certified_line_point(representative, upper)?,
    ];
    canonicalize_segment(&mut segment);
    let _ = segment_measure(&segment)?;
    Ok(Some(segment))
}

fn dominant_axis(segment: &Segment2d) -> usize {
    usize::from((segment[1][1] - segment[0][1]).abs() > (segment[1][0] - segment[0][0]).abs())
}

fn ordered_along(segment: &Segment2d, axis: usize, direction: f64) -> Segment2d {
    if projected_coordinate(segment[0], axis, direction)
        <= projected_coordinate(segment[1], axis, direction)
    {
        *segment
    } else {
        [segment[1], segment[0]]
    }
}

fn projected_coordinate(point: Point2d, axis: usize, direction: f64) -> f64 {
    direction * point[axis]
}

fn validate_cell_coverage(
    source_cells: &[CellId],
    source_triangles: &[Triangle2d],
    target_cells: &[CellId],
    target_triangles: &[Triangle2d],
    fragments: &[RevisionCellFragment2d],
) -> Result<(), Diagnostic> {
    let mut source = source_cells
        .iter()
        .copied()
        .map(|cell| (cell, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut target = target_cells
        .iter()
        .copied()
        .map(|cell| (cell, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    for fragment in fragments {
        source
            .get_mut(&fragment.source_cell)
            .ok_or_else(|| invalid_overlap("cell fragment references an unselected source cell"))?
            .add(fragment.area, fragment.first_moment)?;
        target
            .get_mut(&fragment.target_cell)
            .ok_or_else(|| invalid_overlap("cell fragment references an unselected target cell"))?
            .add(fragment.area, fragment.first_moment)?;
    }
    for (cell, triangle) in source_cells.iter().zip(source_triangles) {
        let (area, moment) = triangle_measure(triangle)?;
        source[cell].require(area, moment, &format!("source cell {}", cell.index()))?;
    }
    for (cell, triangle) in target_cells.iter().zip(target_triangles) {
        let (area, moment) = triangle_measure(triangle)?;
        target[cell].require(area, moment, &format!("target cell {}", cell.index()))?;
    }
    Ok(())
}

fn validate_facet_coverage(
    source_sides: &[RetainedFacetSide2d],
    source_geometry: &[SideGeometry2d],
    target_sides: &[RetainedFacetSide2d],
    target_geometry: &[SideGeometry2d],
    fragments: &[RevisionFacetFragment2d],
) -> Result<(), Diagnostic> {
    let mut source = source_sides
        .iter()
        .copied()
        .map(|side| (side, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut target = target_sides
        .iter()
        .copied()
        .map(|side| (side, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    for fragment in fragments {
        source
            .get_mut(&fragment.source)
            .ok_or_else(|| invalid_overlap("facet fragment references an unselected source side"))?
            .add(fragment.length, fragment.first_moment)?;
        target
            .get_mut(&fragment.target)
            .ok_or_else(|| invalid_overlap("facet fragment references an unselected target side"))?
            .add(fragment.length, fragment.first_moment)?;
    }
    for (side, geometry) in source_sides.iter().zip(source_geometry) {
        source[side].require(
            geometry.length,
            geometry.first_moment,
            &format!("source facet {}", side.facet().index()),
        )?;
    }
    for (side, geometry) in target_sides.iter().zip(target_geometry) {
        target[side].require(
            geometry.length,
            geometry.first_moment,
            &format!("target facet {}", side.facet().index()),
        )?;
    }
    Ok(())
}

fn require_close(
    actual: CompensatedSum,
    expected: f64,
    term_count: usize,
    label: &str,
) -> Result<(), Diagnostic> {
    let actual_value = actual.value();
    let scale = expected
        .abs()
        .max(actual.absolute_sum)
        .max(f64::MIN_POSITIVE);
    let count = term_count.max(1) as f64;
    let tolerance = COVERAGE_ULPS * f64::EPSILON * count * scale;
    if !actual_value.is_finite()
        || !expected.is_finite()
        || !tolerance.is_finite()
        || (actual_value - expected).abs() > tolerance
    {
        return Err(invalid_overlap(format!(
            "common-refinement {label} is incomplete or multiply covered: expected {expected}, got {actual_value}",
        )));
    }
    Ok(())
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
mod tests {
    use super::*;
    use crate::{MeshQualityGate, MeshTopology};

    fn quality() -> MeshQualityGate {
        MeshQualityGate::new(1.0e-8).unwrap()
    }

    fn source_square() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
            quality(),
        )
        .unwrap()
    }

    fn crossed_square() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 3], vec![1, 2, 3]],
            quality(),
        )
        .unwrap()
    }

    fn split_boundary_square() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![0.5, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
            ],
            vec![
                vec![0, 1, 5],
                vec![1, 2, 5],
                vec![2, 3, 5],
                vec![3, 4, 5],
                vec![4, 0, 5],
            ],
            quality(),
        )
        .unwrap()
    }

    fn unit_triangle() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0, 1, 2]],
            quality(),
        )
        .unwrap()
    }

    fn equal_measure_centroid_shear() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![-1.0 / 12.0, 0.0],
                vec![11.0 / 12.0, 0.0],
                vec![1.0 / 6.0, 1.0],
            ],
            vec![vec![0, 1, 2]],
            quality(),
        )
        .unwrap()
    }

    fn all_cells(mesh: &SimplicialMesh) -> Vec<CellId> {
        (0..mesh.entity_count(2).unwrap())
            .map(CellId::new)
            .collect()
    }

    fn facet_with_vertices(mesh: &SimplicialMesh, vertices: [usize; 2]) -> FacetId {
        let expected = vertices
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        (0..mesh.entity_count(1).unwrap())
            .map(FacetId::new)
            .find(|facet| {
                mesh.entity_vertices(MeshEntity::new(1, facet.index()))
                    .unwrap()
                    .into_iter()
                    .map(MeshEntity::index)
                    .collect::<std::collections::BTreeSet<_>>()
                    == expected
            })
            .unwrap()
    }

    #[test]
    fn crossed_diagonals_produce_many_to_many_fragments_without_index_inference() {
        let source = source_square();
        let target = crossed_square();
        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap();

        assert_eq!(overlap.cell_fragments().len(), 4);
        let pairs = overlap
            .cell_fragments()
            .iter()
            .map(|fragment| (fragment.source_cell(), fragment.target_cell()))
            .collect::<Vec<_>>();
        assert_eq!(
            pairs,
            vec![
                (CellId::new(0), CellId::new(0)),
                (CellId::new(0), CellId::new(1)),
                (CellId::new(1), CellId::new(0)),
                (CellId::new(1), CellId::new(1)),
            ]
        );
        assert!(overlap.cell_fragments().iter().all(|fragment| {
            (fragment.area() - 0.25).abs() <= 16.0 * f64::EPSILON
                && orientation(
                    fragment.triangle()[0],
                    fragment.triangle()[1],
                    fragment.triangle()[2],
                ) > 0.0
        }));
    }

    #[test]
    fn retained_boundary_split_has_complete_orientation_aware_coverage() {
        let source = source_square();
        let target = split_boundary_square();
        let source_bottom =
            RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 1]), CellId::new(0));
        let target_left =
            RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));
        let target_right =
            RetainedFacetSide2d::new(facet_with_vertices(&target, [1, 2]), CellId::new(1));
        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::CurrentSpatial,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap()
        .with_retained_facets(
            &source,
            &[source_bottom],
            &target,
            &[target_right, target_left],
        )
        .unwrap();

        assert_eq!(overlap.retained_facet_fragments().len(), 2);
        assert_eq!(
            overlap.target_retained_facets(),
            &[target_left, target_right]
        );
        for fragment in overlap.retained_facet_fragments() {
            assert!((fragment.length() - 0.5).abs() <= 8.0 * f64::EPSILON);
            assert_eq!(fragment.source_outward_normal(), [0.0, -1.0]);
            assert_eq!(fragment.target_outward_normal(), [0.0, -1.0]);
            assert_eq!(
                fragment.source_outward_normal()[0].to_bits(),
                0.0_f64.to_bits()
            );
            assert_eq!(
                fragment.target_outward_normal()[0].to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn selected_region_requires_bidirectional_coverage() {
        let source = source_square();
        let target = crossed_square();
        let error = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &[CellId::new(0)],
        )
        .unwrap_err();
        assert!(error.message().contains("incomplete or multiply covered"));
    }

    #[test]
    fn robust_predicate_rejects_a_degenerate_triangle() {
        let degenerate = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let valid = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let error = intersect_triangles(&degenerate, &valid).unwrap_err();
        assert!(error.message().contains("robustly degenerate"));
    }

    #[test]
    fn exact_positive_fragment_that_collapses_in_binary64_fails_closed() {
        let source = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let target = [[0.0, -1.0], [2.0, -1.0], [1.0, f64::from_bits(1)]];
        let error = intersect_triangles(&source, &target).unwrap_err();
        assert!(
            error
                .message()
                .contains("collapses under canonical nearest-binary64")
        );
    }

    #[test]
    fn exact_intersection_construction_is_canonical_near_an_axis() {
        let source = [[0.5, 0.0], [0.500_375_000_000_000_8, 0.500_124_791_077_715]];
        let target = [
            [0.500_198_529_411_765_1, 0.250_066_065_864_672_67],
            [0.0, 0.25],
        ];
        let expected = proper_segment_intersection(source, target)
            .unwrap()
            .expect("fixture is one robust proper crossing");
        for (source, target) in [
            ([source[1], source[0]], target),
            (source, [target[1], target[0]]),
            (target, source),
            ([target[1], target[0]], [source[1], source[0]]),
        ] {
            let actual = proper_segment_intersection(source, target)
                .unwrap()
                .expect("orientation changes preserve the proper crossing");
            assert_eq!(
                actual.rounded.map(f64::to_bits),
                expected.rounded.map(f64::to_bits)
            );
            assert_eq!(actual.exact, expected.exact);
        }
    }

    #[test]
    fn rounding_cells_admit_one_trace_split_but_reject_larger_drift() {
        let source = [[1.0, 0.0], [1.001_500_000_000_003, 0.500_499_164_310_860_2]];
        let rounded_midpoint = [1.000_750_000_000_001_6, 0.250_249_582_155_430_1];
        let target = [source[0], rounded_midpoint];
        let forward = collinear_segment_overlap(&source, &target)
            .unwrap()
            .expect("one-step-rounded trace split is certified");
        let reverse = collinear_segment_overlap(&target, &source)
            .unwrap()
            .expect("source-target exchange preserves the certificate");
        assert_eq!(
            forward.map(|point| point.map(f64::to_bits)),
            reverse.map(|point| point.map(f64::to_bits))
        );

        let mut drifted = target;
        drifted[1][0] = next_up(next_up(rounded_midpoint[0]));
        assert!(
            collinear_segment_overlap(&source, &drifted)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn unrelated_adjacent_binary64_crossings_remain_distinct() {
        let adjacent = next_up(1.0);
        let first =
            proper_segment_intersection([[1.0, -1.0], [1.0, 2.0]], [[0.0, 1.0], [2.0, 1.0]])
                .unwrap()
                .unwrap();
        let second = proper_segment_intersection(
            [[adjacent, -1.0], [adjacent, 2.0]],
            [[0.0, adjacent], [2.0, adjacent]],
        )
        .unwrap()
        .unwrap();
        assert_ne!(first.exact, second.exact);
        assert_ne!(first.rounded, second.rounded);

        let third = ConstructedPoint2d::from_binary64([0.0, 2.0]).unwrap();
        let hull = convex_hull(vec![first, second, third]);
        assert_eq!(hull.len(), 3);
        let skinny = [hull[0].rounded, hull[1].rounded, hull[2].rounded];
        assert!(triangle_measure(&skinny).unwrap().0 > 0.0);
    }

    #[test]
    fn input_order_is_normalized_but_duplicates_fail() {
        let source = source_square();
        let target = crossed_square();
        let forward = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &[CellId::new(0), CellId::new(1)],
            &target,
            &[CellId::new(0), CellId::new(1)],
        )
        .unwrap();
        let reordered = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &[CellId::new(1), CellId::new(0)],
            &target,
            &[CellId::new(1), CellId::new(0)],
        )
        .unwrap();
        assert_eq!(forward, reordered);

        let duplicate = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &[CellId::new(0), CellId::new(0)],
            &target,
            &all_cells(&target),
        )
        .unwrap_err();
        assert!(duplicate.message().contains("duplicate"));
    }

    #[test]
    fn retained_facets_reject_same_measure_same_centroid_stale_geometry() {
        let source = unit_triangle();
        let target = unit_triangle();
        let stale_source = equal_measure_centroid_shear();
        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap();
        let stale_bottom =
            RetainedFacetSide2d::new(facet_with_vertices(&stale_source, [0, 1]), CellId::new(0));
        let target_bottom =
            RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));

        let error = overlap
            .with_retained_facets(&stale_source, &[stale_bottom], &target, &[target_bottom])
            .unwrap_err();
        assert!(error.message().contains("exactly match"));
    }

    #[test]
    fn retained_sides_must_be_true_frontiers_and_cover_both_revisions() {
        let source = source_square();
        let target = split_boundary_square();
        let interior =
            RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 2]), CellId::new(0));
        let target_left =
            RetainedFacetSide2d::new(facet_with_vertices(&target, [0, 1]), CellId::new(0));
        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap();
        let error = overlap
            .with_retained_facets(&source, &[interior], &target, &[target_left])
            .unwrap_err();
        assert!(error.message().contains("true frontier"));

        let source_bottom =
            RetainedFacetSide2d::new(facet_with_vertices(&source, [0, 1]), CellId::new(0));
        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap();
        let error = overlap
            .with_retained_facets(&source, &[source_bottom], &target, &[target_left])
            .unwrap_err();
        assert!(error.message().contains("incomplete or multiply covered"));

        let overlap = SimplicialRevisionOverlap2d::new(
            OverlapCoordinateChart2d::Material,
            &source,
            &all_cells(&source),
            &target,
            &all_cells(&target),
        )
        .unwrap();
        let error = overlap
            .with_retained_facets(
                &source,
                &[source_bottom, source_bottom],
                &target,
                &[target_left],
            )
            .unwrap_err();
        assert!(error.message().contains("duplicate facet"));
    }
}
