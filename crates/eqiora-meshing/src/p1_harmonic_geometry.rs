//! Solver-free acceptance relation for fixed-topology P1 harmonic coordinates.
//!
//! This module owns the geometry-level statement that a candidate moving mesh
//! is the unique discrete harmonic extension of one mesh-wide solid
//! displacement.  It deliberately owns neither a linear solver nor a physics
//! partition type: exact cell coverage, the complete cross-region interface,
//! Dirichlet ownership, and affine-simplex Laplace coefficients are all
//! reconstructed from the immutable reference mesh.

use std::collections::{BTreeSet, VecDeque};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    AffineGeometryMap, CellId, FacetId, MeshEntity, MeshGeometry, MeshTopology, SimplicialMesh,
    VertexId,
};

const RESIDUAL_ULPS: f64 = 16_384.0;
const MAX_DENSE_RELATION_COEFFICIENTS: usize = 8_000_000;

/// The P1 Laplace relation defining one fixed-topology harmonic coordinate field.
///
/// The relation is sealed to an immutable reference mesh and an exact
/// two-region cell partition.  Solid/interface vertices are driven by the
/// supplied solid displacement, fluid-only exterior vertices are fixed, and
/// the remaining fluid vertices satisfy `K_II u_I + K_ID u_D = 0` component by
/// component.  Construction assembles that relation but performs no solve.
#[derive(Debug, Clone, PartialEq)]
pub struct P1HarmonicCoordinateRelation<const D: usize> {
    reference_mesh: SimplicialMesh,
    fluid_cells: Vec<CellId>,
    solid_cells: Vec<CellId>,
    interface_facets: Vec<FacetId>,
    solid_mask: Vec<bool>,
    solid_vertices: Vec<VertexId>,
    driver_vertices: Vec<VertexId>,
    fixed_exterior_vertices: Vec<VertexId>,
    fluid_interior_vertices: Vec<VertexId>,
    interior_stiffness: Vec<f64>,
    driver_stiffness: Vec<f64>,
    driver_rhs_norms: Vec<f64>,
}

impl<const D: usize> P1HarmonicCoordinateRelation<D> {
    /// Seal the solver-independent coordinate relation on one reference mesh.
    ///
    /// Cell and facet inventories must be strictly increasing.  Fluid and
    /// solid cells must cover every cell exactly once, and `interface_facets`
    /// must equal the complete set of cross-region facets.  Solver tolerances
    /// and admission policy are intentionally absent from this geometry-level
    /// relation.
    ///
    /// # Errors
    /// Returns `EQ0803` unless the mesh is intrinsic 2D/3D, the partition and
    /// complete Dirichlet closure are exact and the bounded dense relation can
    /// be assembled from positive affine simplices.
    pub fn new(
        reference_mesh: &SimplicialMesh,
        fluid_cells: Vec<CellId>,
        solid_cells: Vec<CellId>,
        interface_facets: Vec<FacetId>,
    ) -> Result<Self, Diagnostic> {
        require_supported_mesh::<D>(reference_mesh)?;
        require_strict_ids(
            fluid_cells.iter().map(|cell| cell.index()),
            "fluid cell inventory",
        )?;
        require_strict_ids(
            solid_cells.iter().map(|cell| cell.index()),
            "solid cell inventory",
        )?;
        require_strict_ids(
            interface_facets.iter().map(|facet| facet.index()),
            "interface facet inventory",
        )?;
        if fluid_cells.is_empty() || solid_cells.is_empty() || interface_facets.is_empty() {
            return Err(invalid(
                "P1 harmonic coordinates require non-empty fluid, solid, and interface inventories",
            ));
        }

        let cell_count = reference_mesh
            .entity_count(D)
            .expect("accepted simplex mesh owns a cell stratum");
        let facet_count = reference_mesh
            .entity_count(D - 1)
            .expect("accepted simplex mesh owns a facet stratum");
        let mut material = vec![CellMaterial::Unassigned; cell_count];
        assign_cells(&mut material, &fluid_cells, CellMaterial::Fluid)?;
        assign_cells(&mut material, &solid_cells, CellMaterial::Solid)?;
        if material.contains(&CellMaterial::Unassigned) {
            return Err(invalid(
                "P1 harmonic coordinate regions must cover every mesh cell exactly once",
            ));
        }
        if interface_facets
            .iter()
            .any(|facet| facet.index() >= facet_count)
        {
            return Err(invalid(
                "P1 harmonic coordinate interface facet is outside the reference mesh",
            ));
        }

        let supplied_interface = interface_facets
            .iter()
            .map(|facet| facet.index())
            .collect::<BTreeSet<_>>();
        let exact_interface = exact_interface::<D>(reference_mesh, &material)?;
        if supplied_interface != exact_interface {
            return Err(invalid(
                "P1 harmonic coordinate interface facets must equal the complete cross-region facet set",
            ));
        }

        let fluid_vertices = cell_vertices::<D>(reference_mesh, &fluid_cells)?;
        let solid_vertices = cell_vertices::<D>(reference_mesh, &solid_cells)?;
        let driver_vertices = facet_vertices::<D>(reference_mesh, &interface_facets)?;
        let shared_vertices = fluid_vertices
            .iter()
            .copied()
            .filter(|vertex| solid_vertices.binary_search(vertex).is_ok())
            .collect::<Vec<_>>();
        if shared_vertices != driver_vertices {
            return Err(invalid(
                "P1 harmonic coordinate regions may share vertices only through complete interface facets",
            ));
        }

        let vertex_count = reference_mesh.vertices().len();
        let mut solid_mask = vec![false; vertex_count];
        for vertex in &solid_vertices {
            solid_mask[vertex.index()] = true;
        }
        let (fluid_boundary, exterior_fluid_boundary) =
            fluid_boundaries::<D>(reference_mesh, &fluid_cells, &supplied_interface)?;
        let driver_set = driver_vertices
            .iter()
            .map(|vertex| vertex.index())
            .collect::<BTreeSet<_>>();
        if driver_set
            .iter()
            .any(|vertex| !fluid_boundary.contains(vertex) || !solid_mask[*vertex])
            || fluid_boundary.iter().any(|vertex| {
                !driver_set.contains(vertex) && !exterior_fluid_boundary.contains(vertex)
            })
        {
            return Err(invalid(
                "P1 harmonic coordinate fluid boundary lacks a complete interface/fixed-exterior Dirichlet closure",
            ));
        }
        let fixed_exterior_vertices = exterior_fluid_boundary
            .iter()
            .copied()
            .filter(|vertex| !solid_mask[*vertex])
            .map(VertexId::new)
            .collect::<Vec<_>>();
        let fluid_interior_vertices = fluid_vertices
            .iter()
            .copied()
            .filter(|vertex| !fluid_boundary.contains(&vertex.index()))
            .collect::<Vec<_>>();

        let adjacency = region_vertex_adjacency::<D>(reference_mesh, &fluid_cells)?;
        let mut prescribed = vec![false; vertex_count];
        for vertex in driver_vertices.iter().chain(&fixed_exterior_vertices) {
            prescribed[vertex.index()] = true;
        }
        require_boundary_reachability(&fluid_vertices, &adjacency, &prescribed)?;

        let interior_count = fluid_interior_vertices.len();
        let driver_count = driver_vertices.len();
        let square = interior_count
            .checked_mul(interior_count)
            .ok_or_else(|| invalid("P1 harmonic coordinate relation width overflows usize"))?;
        let coupling = interior_count
            .checked_mul(driver_count)
            .ok_or_else(|| invalid("P1 harmonic coordinate coupling width overflows usize"))?;
        let coefficient_count = square
            .checked_add(coupling)
            .ok_or_else(|| invalid("P1 harmonic coordinate storage overflows usize"))?;
        if coefficient_count > MAX_DENSE_RELATION_COEFFICIENTS {
            return Err(invalid(format!(
                "P1 harmonic coordinate reference relation admits at most {MAX_DENSE_RELATION_COEFFICIENTS} dense coefficients, got {coefficient_count}",
            )));
        }
        let (interior_stiffness, driver_stiffness) = assemble_dirichlet_laplacian::<D>(
            reference_mesh,
            &fluid_cells,
            &fluid_interior_vertices,
            &driver_vertices,
        )?;
        validate_interior_stiffness(&interior_stiffness, interior_count)?;
        let driver_rhs_norms = driver_rhs_norms(&driver_stiffness, interior_count, driver_count)?;

        Ok(Self {
            reference_mesh: reference_mesh.clone(),
            fluid_cells,
            solid_cells,
            interface_facets,
            solid_mask,
            solid_vertices,
            driver_vertices,
            fixed_exterior_vertices,
            fluid_interior_vertices,
            interior_stiffness,
            driver_stiffness,
            driver_rhs_norms,
        })
    }

    /// Immutable mesh revision defining topology, reference coordinates, and coefficients.
    #[must_use]
    pub const fn reference_mesh(&self) -> &SimplicialMesh {
        &self.reference_mesh
    }

    /// Fluid cells in canonical mesh order.
    #[must_use]
    pub fn fluid_cells(&self) -> &[CellId] {
        &self.fluid_cells
    }

    /// Solid cells in canonical mesh order.
    #[must_use]
    pub fn solid_cells(&self) -> &[CellId] {
        &self.solid_cells
    }

    /// Complete cross-region interface in canonical mesh-facet order.
    #[must_use]
    pub fn interface_facets(&self) -> &[FacetId] {
        &self.interface_facets
    }

    /// Vertices driven exactly by the mesh-wide solid-displacement field.
    #[must_use]
    pub fn solid_vertices(&self) -> &[VertexId] {
        &self.solid_vertices
    }

    /// Interface vertices supplying the fluid harmonic Dirichlet trace.
    #[must_use]
    pub fn driver_vertices(&self) -> &[VertexId] {
        &self.driver_vertices
    }

    /// Fluid-only physical-exterior vertices fixed to reference coordinates.
    #[must_use]
    pub fn fixed_exterior_vertices(&self) -> &[VertexId] {
        &self.fixed_exterior_vertices
    }

    /// Fluid vertices governed by the assembled interior Laplace relation.
    #[must_use]
    pub fn fluid_interior_vertices(&self) -> &[VertexId] {
        &self.fluid_interior_vertices
    }

    /// Row-major `K_II` in [`Self::fluid_interior_vertices`] order.
    ///
    /// This solver-neutral lowered operator is exposed read-only so numerical
    /// realizations can solve the same relation later certified here.
    #[must_use]
    pub fn interior_stiffness(&self) -> &[f64] {
        &self.interior_stiffness
    }

    /// Row-major `K_ID` with interior rows and [`Self::driver_vertices`] columns.
    #[must_use]
    pub fn driver_stiffness(&self) -> &[f64] {
        &self.driver_stiffness
    }

    /// Euclidean norm of each unit-driver right-hand side in driver-vertex order.
    ///
    /// A solver-owning layer can combine these geometry-derived norms with its
    /// exact linear plan to derive the residual targets later supplied to
    /// [`Self::validate_current_coordinates`].
    #[must_use]
    pub fn driver_rhs_norms(&self) -> &[f64] {
        &self.driver_rhs_norms
    }

    /// Certify candidate coordinates as the admitted harmonic extension.
    ///
    /// `solid_displacement` is mesh-wide and must be exact zero outside the
    /// solid closure.  Candidate solid coordinates must equal `reference +
    /// displacement` exactly, fixed fluid-exterior coordinates must equal the
    /// reference exactly, and fluid-interior coordinates must satisfy the P1
    /// Laplace relation.  The residual allowance is the triangle-inequality
    /// combination of the supplied per-driver solver targets plus a binary64
    /// reduction roundoff term; no solver-produced coordinate or
    /// caller-authored residual is trusted.  Targets are an external admission
    /// policy, not geometry state: artifact/numerics callers must derive and
    /// bind them from the exact realization plan or accepted solve reports.
    ///
    /// # Errors
    /// Returns `EQ0803` for incompatible/non-finite fields or targets, altered
    /// Dirichlet coordinates, overflow, or a harmonic residual outside the
    /// supplied solver bound.
    pub fn validate_current_coordinates(
        &self,
        solid_displacement: &[[f64; D]],
        current_coordinates: &[Vec<f64>],
        driver_residual_targets: &[f64],
    ) -> Result<(), Diagnostic> {
        if driver_residual_targets.len() != self.driver_vertices.len()
            || driver_residual_targets
                .iter()
                .any(|target| !target.is_finite() || *target < 0.0)
        {
            return Err(invalid(
                "P1 harmonic coordinate residual targets must contain one finite non-negative value per driver vertex",
            ));
        }
        let vertex_count = self.reference_mesh.vertices().len();
        if solid_displacement.len() != vertex_count
            || solid_displacement
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(format!(
                "P1 harmonic solid displacement must be one finite {D}-vector per reference vertex",
            )));
        }
        if solid_displacement
            .iter()
            .enumerate()
            .any(|(vertex, value)| !self.solid_mask[vertex] && *value != [0.0; D])
        {
            return Err(invalid(
                "P1 harmonic solid displacement must be exact zero outside the solid closure",
            ));
        }
        if current_coordinates.len() != vertex_count
            || current_coordinates.iter().any(|coordinates| {
                coordinates.len() != D || coordinates.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(format!(
                "P1 harmonic current coordinates must be one finite {D}-vector per reference vertex",
            )));
        }

        for vertex in &self.solid_vertices {
            for component in 0..D {
                let expected = self.reference_mesh.vertices()[vertex.index()][component]
                    + solid_displacement[vertex.index()][component];
                if !expected.is_finite()
                    || current_coordinates[vertex.index()][component] != expected
                {
                    return Err(invalid(
                        "P1 harmonic solid coordinates must equal reference plus the exact solid displacement",
                    ));
                }
            }
        }
        for vertex in &self.fixed_exterior_vertices {
            if current_coordinates[vertex.index()] != self.reference_mesh.vertices()[vertex.index()]
            {
                return Err(invalid(
                    "P1 harmonic fixed fluid-exterior coordinates must equal the reference exactly",
                ));
            }
        }

        self.validate_harmonic_residual(
            solid_displacement,
            current_coordinates,
            driver_residual_targets,
        )
    }

    fn validate_harmonic_residual(
        &self,
        solid_displacement: &[[f64; D]],
        current_coordinates: &[Vec<f64>],
        driver_residual_targets: &[f64],
    ) -> Result<(), Diagnostic> {
        let interior_count = self.fluid_interior_vertices.len();
        let driver_count = self.driver_vertices.len();
        for component in 0..D {
            let mut residual_norm = 0.0_f64;
            let mut rounding_norm = 0.0_f64;
            for row in 0..interior_count {
                let mut residual = 0.0;
                let mut scale = 0.0;
                let mut coordinate_roundoff = 0.0;
                for (column, vertex) in self.fluid_interior_vertices.iter().enumerate() {
                    let coefficient = self.interior_stiffness[row * interior_count + column];
                    let current = current_coordinates[vertex.index()][component];
                    let reference = self.reference_mesh.vertices()[vertex.index()][component];
                    let displacement = current - reference;
                    let term = coefficient * displacement;
                    residual += term;
                    scale += term.abs();
                    coordinate_roundoff +=
                        coefficient.abs() * f64::EPSILON * (current.abs() + reference.abs());
                }
                for (column, vertex) in self.driver_vertices.iter().enumerate() {
                    let term = self.driver_stiffness[row * driver_count + column]
                        * solid_displacement[vertex.index()][component];
                    residual += term;
                    scale += term.abs();
                }
                residual_norm = residual_norm.hypot(residual);
                let row_roundoff = coordinate_roundoff
                    + RESIDUAL_ULPS * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
                rounding_norm = rounding_norm.hypot(row_roundoff);
            }
            let solver_bound = self
                .driver_vertices
                .iter()
                .zip(driver_residual_targets)
                .map(|(vertex, target)| {
                    solid_displacement[vertex.index()][component].abs() * target
                })
                .sum::<f64>();
            let tolerance = solver_bound + rounding_norm;
            if !residual_norm.is_finite()
                || !rounding_norm.is_finite()
                || !solver_bound.is_finite()
                || !tolerance.is_finite()
                || residual_norm > tolerance
            {
                return Err(invalid(format!(
                    "P1 harmonic coordinate component {component} violates its solver-bounded Laplace residual: {residual_norm:e} > {tolerance:e}",
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellMaterial {
    Unassigned,
    Fluid,
    Solid,
}

fn require_supported_mesh<const D: usize>(mesh: &SimplicialMesh) -> Result<(), Diagnostic> {
    if !matches!(D, 2 | 3)
        || mesh.topological_dimension() != D
        || mesh.geometric_dimension() != D
        || mesh
            .vertices()
            .iter()
            .any(|coordinates| coordinates.len() != D)
    {
        return Err(invalid(format!(
            "P1 harmonic coordinates require one intrinsic {D}D affine-simplex mesh with D equal to 2 or 3",
        )));
    }
    Ok(())
}

fn require_strict_ids(
    ids: impl Iterator<Item = usize>,
    label: &'static str,
) -> Result<(), Diagnostic> {
    let mut previous = None;
    for id in ids {
        if previous.is_some_and(|previous| id <= previous) {
            return Err(invalid(format!(
                "P1 harmonic coordinate {label} must be strictly increasing",
            )));
        }
        previous = Some(id);
    }
    Ok(())
}

fn assign_cells(
    material: &mut [CellMaterial],
    cells: &[CellId],
    assignment: CellMaterial,
) -> Result<(), Diagnostic> {
    for cell in cells {
        let entry = material.get_mut(cell.index()).ok_or_else(|| {
            invalid("P1 harmonic coordinate region cell is outside the reference mesh")
        })?;
        if *entry != CellMaterial::Unassigned {
            return Err(invalid(
                "P1 harmonic coordinate cell appears in more than one region inventory",
            ));
        }
        *entry = assignment;
    }
    Ok(())
}

fn exact_interface<const D: usize>(
    mesh: &SimplicialMesh,
    material: &[CellMaterial],
) -> Result<BTreeSet<usize>, Diagnostic> {
    let facet_count = mesh
        .entity_count(D - 1)
        .expect("accepted simplex mesh owns a facet stratum");
    let mut interface = BTreeSet::new();
    for facet_index in 0..facet_count {
        let adjacent = mesh
            .incidence(MeshEntity::new(D - 1, facet_index), D)
            .ok_or_else(|| invalid("P1 harmonic coordinate facet incidence is unavailable"))?;
        if adjacent.len() == 2
            && material[adjacent[0].entity.index()] != material[adjacent[1].entity.index()]
        {
            interface.insert(facet_index);
        }
    }
    Ok(interface)
}

fn cell_vertices<const D: usize>(
    mesh: &SimplicialMesh,
    cells: &[CellId],
) -> Result<Vec<VertexId>, Diagnostic> {
    let mut vertices = BTreeSet::new();
    for cell in cells {
        let closure = mesh
            .entity_vertices(MeshEntity::new(D, cell.index()))
            .ok_or_else(|| invalid("P1 harmonic coordinate cell closure is unavailable"))?;
        if closure.len() != D + 1 {
            return Err(invalid(format!(
                "P1 harmonic coordinate cell is not an affine {D}-simplex",
            )));
        }
        vertices.extend(
            closure
                .into_iter()
                .map(|vertex| VertexId::new(vertex.index())),
        );
    }
    Ok(vertices.into_iter().collect())
}

fn facet_vertices<const D: usize>(
    mesh: &SimplicialMesh,
    facets: &[FacetId],
) -> Result<Vec<VertexId>, Diagnostic> {
    let mut vertices = BTreeSet::new();
    for facet in facets {
        let closure = mesh
            .entity_vertices(MeshEntity::new(D - 1, facet.index()))
            .ok_or_else(|| invalid("P1 harmonic coordinate interface closure is unavailable"))?;
        if closure.len() != D {
            return Err(invalid(format!(
                "P1 harmonic coordinate interface is not an affine {}-simplex",
                D - 1,
            )));
        }
        vertices.extend(
            closure
                .into_iter()
                .map(|vertex| VertexId::new(vertex.index())),
        );
    }
    Ok(vertices.into_iter().collect())
}

fn fluid_boundaries<const D: usize>(
    mesh: &SimplicialMesh,
    fluid_cells: &[CellId],
    interface_facets: &BTreeSet<usize>,
) -> Result<(BTreeSet<usize>, BTreeSet<usize>), Diagnostic> {
    let fluid = fluid_cells
        .iter()
        .map(|cell| cell.index())
        .collect::<BTreeSet<_>>();
    let facet_count = mesh
        .entity_count(D - 1)
        .expect("accepted simplex mesh owns a facet stratum");
    let mut boundary = BTreeSet::new();
    let mut exterior = BTreeSet::new();
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(D - 1, facet_index);
        let adjacent = mesh
            .incidence(facet, D)
            .ok_or_else(|| invalid("P1 harmonic coordinate facet incidence is unavailable"))?;
        if adjacent
            .iter()
            .filter(|cell| fluid.contains(&cell.entity.index()))
            .count()
            != 1
        {
            continue;
        }
        let closure = mesh
            .entity_vertices(facet)
            .ok_or_else(|| invalid("P1 harmonic coordinate boundary closure is unavailable"))?;
        boundary.extend(closure.iter().map(|vertex| vertex.index()));
        match adjacent.len() {
            1 => exterior.extend(closure.iter().map(|vertex| vertex.index())),
            2 if interface_facets.contains(&facet_index) => {}
            _ => {
                return Err(invalid(
                    "P1 harmonic fluid boundary is neither physical exterior nor exact region interface",
                ));
            }
        }
    }
    Ok((boundary, exterior))
}

fn region_vertex_adjacency<const D: usize>(
    mesh: &SimplicialMesh,
    cells: &[CellId],
) -> Result<Vec<Vec<usize>>, Diagnostic> {
    let mut adjacency = vec![BTreeSet::new(); mesh.vertices().len()];
    for cell in cells {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(D, cell.index()))
            .ok_or_else(|| invalid("P1 harmonic coordinate fluid cell is outside the mesh"))?;
        for left in 0..vertices.len() {
            for right in left + 1..vertices.len() {
                adjacency[vertices[left].index()].insert(vertices[right].index());
                adjacency[vertices[right].index()].insert(vertices[left].index());
            }
        }
    }
    Ok(adjacency
        .into_iter()
        .map(|neighbors| neighbors.into_iter().collect())
        .collect())
}

fn require_boundary_reachability(
    fluid_vertices: &[VertexId],
    adjacency: &[Vec<usize>],
    prescribed: &[bool],
) -> Result<(), Diagnostic> {
    if adjacency.len() != prescribed.len() {
        return Err(invalid(
            "P1 harmonic coordinate boundary reachability has incompatible topology",
        ));
    }
    let fluid = fluid_vertices
        .iter()
        .map(|vertex| vertex.index())
        .collect::<BTreeSet<_>>();
    let mut visited = vec![false; adjacency.len()];
    let mut pending = VecDeque::new();
    for vertex in fluid_vertices {
        if prescribed[vertex.index()] {
            visited[vertex.index()] = true;
            pending.push_back(vertex.index());
        }
    }
    if pending.is_empty() {
        return Err(invalid(
            "P1 harmonic coordinate fluid region has no prescribed boundary vertex",
        ));
    }
    while let Some(vertex) = pending.pop_front() {
        for &neighbor in &adjacency[vertex] {
            if fluid.contains(&neighbor) && !visited[neighbor] {
                visited[neighbor] = true;
                pending.push_back(neighbor);
            }
        }
    }
    if fluid_vertices.iter().any(|vertex| !visited[vertex.index()]) {
        return Err(invalid(
            "P1 harmonic coordinate fluid component is not anchored to the complete Dirichlet closure",
        ));
    }
    Ok(())
}

fn assemble_dirichlet_laplacian<const D: usize>(
    mesh: &SimplicialMesh,
    fluid_cells: &[CellId],
    interior_vertices: &[VertexId],
    driver_vertices: &[VertexId],
) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
    let mut interior_position = vec![None; mesh.vertices().len()];
    let mut driver_position = vec![None; mesh.vertices().len()];
    for (position, vertex) in interior_vertices.iter().enumerate() {
        interior_position[vertex.index()] = Some(position);
    }
    for (position, vertex) in driver_vertices.iter().enumerate() {
        driver_position[vertex.index()] = Some(position);
    }
    let interior_count = interior_vertices.len();
    let driver_count = driver_vertices.len();
    let mut interior = vec![0.0; interior_count * interior_count];
    let mut driver = vec![0.0; interior_count * driver_count];
    for cell in fluid_cells {
        let vertices = mesh
            .cells()
            .get(cell.index())
            .ok_or_else(|| invalid("P1 harmonic coordinate fluid cell is outside the mesh"))?;
        let local = affine_simplex_laplacian::<D>(mesh, vertices)?;
        for local_row in 0..D + 1 {
            let Some(row) = interior_position[vertices[local_row]] else {
                continue;
            };
            for local_column in 0..D + 1 {
                let vertex = vertices[local_column];
                if let Some(column) = interior_position[vertex] {
                    interior[row * interior_count + column] +=
                        local[local_row * (D + 1) + local_column];
                } else if let Some(column) = driver_position[vertex] {
                    driver[row * driver_count + column] +=
                        local[local_row * (D + 1) + local_column];
                }
            }
        }
    }
    if interior
        .iter()
        .chain(&driver)
        .any(|coefficient| !coefficient.is_finite())
    {
        return Err(invalid(
            "P1 harmonic coordinate Laplace assembly produced a non-finite coefficient",
        ));
    }
    Ok((interior, driver))
}

fn affine_simplex_laplacian<const D: usize>(
    mesh: &SimplicialMesh,
    vertices: &[usize],
) -> Result<Vec<f64>, Diagnostic> {
    if !matches!(D, 2 | 3) || vertices.len() != D + 1 {
        return Err(invalid(format!(
            "P1 harmonic coordinate cell is not an affine {D}-simplex in supported dimension 2 or 3",
        )));
    }
    let points = vertices
        .iter()
        .map(|&vertex| {
            let point = mesh
                .vertices()
                .get(vertex)
                .ok_or_else(|| invalid("P1 harmonic coordinate cell has an invalid vertex"))?;
            Ok(point.clone())
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let geometry = AffineGeometryMap::from_simplex_vertices(points)?;
    let quality = geometry.square_quality()?;
    if quality.signed_measure_scale() <= 0.0 {
        return Err(invalid(
            "P1 harmonic coordinates require positively oriented reference simplices",
        ));
    }
    let inverse = geometry.inverse_jacobian()?;
    let mut gradients = vec![vec![0.0; D]; D + 1];
    for axis in 0..D {
        gradients[0][axis] = -(0..D)
            .map(|reference_axis| inverse[reference_axis * D + axis])
            .sum::<f64>();
        for local in 1..D + 1 {
            gradients[local][axis] = inverse[(local - 1) * D + axis];
        }
    }
    let measure = geometry.measure_scale() / simplex_factorial(D) as f64;
    let mut local = vec![0.0; (D + 1) * (D + 1)];
    for row in 0..D + 1 {
        for column in 0..D + 1 {
            local[row * (D + 1) + column] = measure
                * gradients[row]
                    .iter()
                    .zip(&gradients[column])
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
        }
    }
    if !measure.is_finite()
        || measure <= 0.0
        || local.iter().any(|coefficient| !coefficient.is_finite())
    {
        return Err(invalid(
            "P1 harmonic coordinate affine-simplex Laplacian is not finite and positive-measure",
        ));
    }
    Ok(local)
}

fn simplex_factorial(dimension: usize) -> usize {
    (1..=dimension).product()
}

fn validate_interior_stiffness(matrix: &[f64], size: usize) -> Result<(), Diagnostic> {
    if size == 0 {
        return Ok(());
    }
    if matrix.len() != size * size {
        return Err(invalid(
            "P1 harmonic coordinate interior stiffness has an invalid layout",
        ));
    }
    for row in 0..size {
        if matrix[row * size + row] <= 0.0 {
            return Err(invalid(
                "P1 harmonic coordinate interior stiffness has a non-positive diagonal",
            ));
        }
        for column in 0..row {
            if matrix[row * size + column] != matrix[column * size + row] {
                return Err(invalid(
                    "P1 harmonic coordinate interior stiffness is not exactly symmetric",
                ));
            }
        }
    }
    Ok(())
}

fn driver_rhs_norms(
    driver_stiffness: &[f64],
    interior_count: usize,
    driver_count: usize,
) -> Result<Vec<f64>, Diagnostic> {
    let mut norms = Vec::with_capacity(driver_count);
    for column in 0..driver_count {
        let rhs_norm = (0..interior_count).fold(0.0_f64, |norm, row| {
            norm.hypot(driver_stiffness[row * driver_count + column])
        });
        if !rhs_norm.is_finite() {
            return Err(invalid(
                "P1 harmonic coordinate driver right-hand-side norm overflowed",
            ));
        }
        norms.push(rhs_norm);
    }
    Ok(norms)
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshQualityGate;

    const RELATIVE_TOLERANCE: f64 = 1.0e-13;
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-15;

    fn partitioned_strip() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
        let x_coordinates = [0.0, 0.5, 1.0, 2.0];
        let y_coordinates = [0.0, 0.5, 1.0];
        let nx = x_coordinates.len();
        let vertices = y_coordinates
            .iter()
            .flat_map(|&y| x_coordinates.iter().map(move |&x| vec![x, y]))
            .collect::<Vec<_>>();
        let mut cells = Vec::new();
        let mut fluid_cells = Vec::new();
        let mut solid_cells = Vec::new();
        for x in 0..nx - 1 {
            for y in 0..y_coordinates.len() - 1 {
                let lower_left = y * nx + x;
                let lower_right = lower_left + 1;
                let upper_left = (y + 1) * nx + x;
                let upper_right = upper_left + 1;
                for triangle in [
                    vec![lower_left, lower_right, upper_right],
                    vec![lower_left, upper_right, upper_left],
                ] {
                    let id = CellId::new(cells.len());
                    if x_coordinates[x + 1] <= 1.0 {
                        fluid_cells.push(id);
                    } else {
                        solid_cells.push(id);
                    }
                    cells.push(triangle);
                }
            }
        }
        let mesh = SimplicialMesh::new(
            2,
            vertices,
            cells,
            MeshQualityGate::new(0.1).expect("valid quality gate"),
        )
        .expect("valid strip mesh");
        let interface_facets = interface_facets_at_x::<2>(&mesh, 1.0);
        (mesh, fluid_cells, solid_cells, interface_facets)
    }

    fn partitioned_block() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
        let x_coordinates = [0.0, 0.5, 1.0, 2.0];
        let y_coordinates = [0.0, 0.5, 1.0];
        let z_coordinates = [0.0, 0.5, 1.0];
        let nx = x_coordinates.len();
        let ny = y_coordinates.len();
        let vertex = |x: usize, y: usize, z: usize| z * ny * nx + y * nx + x;
        let mut vertices = Vec::new();
        for &z in &z_coordinates {
            for &y in &y_coordinates {
                for &x in &x_coordinates {
                    vertices.push(vec![x, y, z]);
                }
            }
        }
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut cells = Vec::new();
        let mut fluid_cells = Vec::new();
        let mut solid_cells = Vec::new();
        for x in 0..nx - 1 {
            for y in 0..ny - 1 {
                for z in 0..z_coordinates.len() - 1 {
                    for permutation in permutations {
                        let mut offset = [0, 0, 0];
                        let mut tetrahedron = vec![vertex(x, y, z)];
                        for axis in permutation {
                            offset[axis] = 1;
                            tetrahedron.push(vertex(x + offset[0], y + offset[1], z + offset[2]));
                        }
                        if signed_tetrahedron_measure(&vertices, &tetrahedron) < 0.0 {
                            tetrahedron.swap(1, 2);
                        }
                        let id = CellId::new(cells.len());
                        if x_coordinates[x + 1] <= 1.0 {
                            fluid_cells.push(id);
                        } else {
                            solid_cells.push(id);
                        }
                        cells.push(tetrahedron);
                    }
                }
            }
        }
        let mesh = SimplicialMesh::new(
            3,
            vertices,
            cells,
            MeshQualityGate::new(0.02).expect("valid quality gate"),
        )
        .expect("valid tetrahedral block");
        let interface_facets = interface_facets_at_x::<3>(&mesh, 1.0);
        (mesh, fluid_cells, solid_cells, interface_facets)
    }

    fn interface_facets_at_x<const D: usize>(mesh: &SimplicialMesh, x: f64) -> Vec<FacetId> {
        (0..mesh
            .entity_count(D - 1)
            .expect("test mesh owns a facet stratum"))
            .filter_map(|facet| {
                let vertices = mesh
                    .entity_vertices(MeshEntity::new(D - 1, facet))
                    .expect("test facet owns vertices");
                vertices
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][0] == x)
                    .then_some(FacetId::new(facet))
            })
            .collect()
    }

    fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
        let origin = &vertices[cell[0]];
        let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
        column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
            - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
            + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
    }

    fn relation_2d() -> P1HarmonicCoordinateRelation<2> {
        let (mesh, fluid, solid, interface) = partitioned_strip();
        P1HarmonicCoordinateRelation::<2>::new(&mesh, fluid, solid, interface)
            .expect("valid 2D harmonic relation")
    }

    fn residual_targets<const D: usize>(relation: &P1HarmonicCoordinateRelation<D>) -> Vec<f64> {
        relation
            .driver_rhs_norms()
            .iter()
            .map(|rhs_norm| ABSOLUTE_TOLERANCE.max(RELATIVE_TOLERANCE * rhs_norm))
            .collect()
    }

    fn solid_displacement_2d(relation: &P1HarmonicCoordinateRelation<2>) -> Vec<[f64; 2]> {
        let mut displacement = vec![[0.0; 2]; relation.reference_mesh.vertices().len()];
        for vertex in relation.solid_vertices() {
            let point = &relation.reference_mesh.vertices()[vertex.index()];
            displacement[vertex.index()] = [
                0.01 + 0.02 * point[0] - 0.03 * point[1],
                -0.02 + 0.01 * point[0] + 0.04 * point[1],
            ];
        }
        displacement
    }

    fn solid_displacement_3d(relation: &P1HarmonicCoordinateRelation<3>) -> Vec<[f64; 3]> {
        let mut displacement = vec![[0.0; 3]; relation.reference_mesh.vertices().len()];
        for vertex in relation.solid_vertices() {
            let point = &relation.reference_mesh.vertices()[vertex.index()];
            displacement[vertex.index()] = [
                0.01 + 0.02 * point[0] - 0.01 * point[1] + 0.03 * point[2],
                -0.02 + 0.01 * point[0] + 0.04 * point[1] - 0.02 * point[2],
                0.03 - 0.02 * point[0] + 0.01 * point[1] + 0.02 * point[2],
            ];
        }
        displacement
    }

    fn one_interior_solution<const D: usize>(
        relation: &P1HarmonicCoordinateRelation<D>,
        solid: &[[f64; D]],
    ) -> Vec<Vec<f64>> {
        assert_eq!(relation.fluid_interior_vertices.len(), 1);
        let mut displacement = vec![[0.0; D]; relation.reference_mesh.vertices().len()];
        for vertex in &relation.solid_vertices {
            displacement[vertex.index()] = solid[vertex.index()];
        }
        let interior = relation.fluid_interior_vertices[0].index();
        for component in 0..D {
            let prescribed = relation
                .driver_vertices
                .iter()
                .enumerate()
                .map(|(column, vertex)| {
                    relation.driver_stiffness[column] * solid[vertex.index()][component]
                })
                .sum::<f64>();
            displacement[interior][component] = -prescribed / relation.interior_stiffness[0];
        }
        relation
            .reference_mesh
            .vertices()
            .iter()
            .zip(displacement)
            .map(|(reference, displacement)| {
                reference
                    .iter()
                    .zip(displacement)
                    .map(|(reference, displacement)| reference + displacement)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn accepted_relations_certify_two_and_three_dimensional_harmonic_coordinates() {
        let relation_2d = relation_2d();
        let solid_2d = solid_displacement_2d(&relation_2d);
        let coordinates_2d = one_interior_solution(&relation_2d, &solid_2d);
        let targets_2d = residual_targets(&relation_2d);
        relation_2d
            .validate_current_coordinates(&solid_2d, &coordinates_2d, &targets_2d)
            .expect("2D harmonic coordinates certify");
        assert_eq!(relation_2d.fluid_interior_vertices().len(), 1);
        assert!(!relation_2d.fixed_exterior_vertices().is_empty());

        let (mesh, fluid, solid_cells, interface) = partitioned_block();
        let relation_3d =
            P1HarmonicCoordinateRelation::<3>::new(&mesh, fluid, solid_cells, interface)
                .expect("valid 3D harmonic relation");
        let solid_3d = solid_displacement_3d(&relation_3d);
        let coordinates_3d = one_interior_solution(&relation_3d, &solid_3d);
        let targets_3d = residual_targets(&relation_3d);
        relation_3d
            .validate_current_coordinates(&solid_3d, &coordinates_3d, &targets_3d)
            .expect("3D harmonic coordinates certify");
        assert_eq!(relation_3d.fluid_interior_vertices().len(), 1);
    }

    #[test]
    fn positive_but_nonharmonic_interior_coordinates_are_rejected() {
        let relation = relation_2d();
        let solid = solid_displacement_2d(&relation);
        let mut coordinates = one_interior_solution(&relation, &solid);
        let targets = residual_targets(&relation);
        let interior = relation.fluid_interior_vertices()[0].index();
        coordinates[interior][0] += 1.0e-3;

        SimplicialMesh::new(
            2,
            coordinates.clone(),
            relation.reference_mesh().cells().to_vec(),
            relation.reference_mesh().quality_gate(),
        )
        .expect("perturbation remains a positive accepted mesh");
        assert!(
            relation
                .validate_current_coordinates(&solid, &coordinates, &targets)
                .is_err()
        );
    }

    #[test]
    fn independently_authored_solid_or_fixed_coordinates_are_rejected() {
        let relation = relation_2d();
        let solid = solid_displacement_2d(&relation);
        let coordinates = one_interior_solution(&relation, &solid);
        let targets = residual_targets(&relation);

        let mut altered_solid = coordinates.clone();
        let solid_vertex = relation.solid_vertices()[0].index();
        altered_solid[solid_vertex][0] += 1.0e-12;
        assert!(
            relation
                .validate_current_coordinates(&solid, &altered_solid, &targets)
                .is_err()
        );

        let mut altered_fixed = coordinates;
        let fixed_vertex = relation.fixed_exterior_vertices()[0].index();
        altered_fixed[fixed_vertex][1] += 1.0e-12;
        assert!(
            relation
                .validate_current_coordinates(&solid, &altered_fixed, &targets)
                .is_err()
        );
    }

    #[test]
    fn external_residual_targets_must_exactly_cover_the_driver_inventory() {
        let relation = relation_2d();
        let solid = solid_displacement_2d(&relation);
        let coordinates = one_interior_solution(&relation, &solid);
        let mut targets = residual_targets(&relation);

        assert!(
            relation
                .validate_current_coordinates(&solid, &coordinates, &targets[1..])
                .is_err()
        );
        targets[0] = f64::NAN;
        assert!(
            relation
                .validate_current_coordinates(&solid, &coordinates, &targets)
                .is_err()
        );
        targets[0] = -1.0;
        assert!(
            relation
                .validate_current_coordinates(&solid, &coordinates, &targets)
                .is_err()
        );
    }

    #[test]
    fn incomplete_region_coverage_and_interface_are_rejected() {
        let (mesh, fluid, mut solid, interface) = partitioned_strip();
        solid.pop();
        assert!(
            P1HarmonicCoordinateRelation::<2>::new(&mesh, fluid.clone(), solid, interface.clone())
                .is_err()
        );

        let (_, _, complete_solid, _) = partitioned_strip();
        assert!(
            P1HarmonicCoordinateRelation::<2>::new(
                &mesh,
                fluid,
                complete_solid,
                interface.into_iter().skip(1).collect(),
            )
            .is_err()
        );
    }
}
