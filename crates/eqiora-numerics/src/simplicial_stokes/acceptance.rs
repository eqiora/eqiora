use eqiora_assembly::LinearSystem;
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    AffineGeometryMap, GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule,
    SimplicialMesh,
};

use super::layout::MixedLayout;
use super::{
    COMPONENTS, DIMENSION, P1_BASIS_COUNT, REQUIRED_ERROR_QUADRATURE_EXACTNESS,
    REQUIRED_QUADRATURE_EXACTNESS, invalid,
};
use crate::discrete_space::{DiscreteSpace, SimplexP1Space};

pub(super) fn require_compatible_boundary_flux(
    mesh: &SimplicialMesh,
    fixed_velocity: &[Option<[f64; COMPONENTS]>],
) -> Result<(), Diagnostic> {
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .expect("2D simplex mesh owns edges");
    let mut net_flux = CompensatedSum::default();
    let mut flux_scale = CompensatedSum::default();
    let mut boundary_facet_count = 0_usize;
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(DIMENSION - 1, facet_index);
        if !mesh
            .is_boundary_entity(facet)
            .expect("mesh owns every edge boundary classification")
        {
            continue;
        }
        let vertices = mesh
            .entity_vertices(facet)
            .expect("accepted boundary edge owns two vertices");
        let [left, right]: [MeshEntity; 2] = vertices
            .try_into()
            .expect("2D simplex edge has two vertices");
        let left_coordinates = &mesh.vertices()[left.index()];
        let right_coordinates = &mesh.vertices()[right.index()];
        let tangent = [
            right_coordinates[0] - left_coordinates[0],
            right_coordinates[1] - left_coordinates[1],
        ];
        let length = tangent[0].hypot(tangent[1]);
        let midpoint = [
            0.5 * (left_coordinates[0] + right_coordinates[0]),
            0.5 * (left_coordinates[1] + right_coordinates[1]),
        ];
        let adjacent = mesh
            .incidence(facet, DIMENSION)
            .expect("accepted boundary edge owns cell incidence");
        let [cell] = adjacent.as_slice() else {
            return Err(invalid(
                "MINI Stokes boundary edge must have exactly one adjacent fluid cell",
            ));
        };
        let cell_vertices = mesh
            .entity_vertices(cell.entity)
            .expect("accepted triangle owns three vertices");
        let mut centroid = [0.0; DIMENSION];
        for vertex in &cell_vertices {
            for (axis, coordinate) in centroid.iter_mut().enumerate() {
                *coordinate += mesh.vertices()[vertex.index()][axis] / 3.0;
            }
        }
        let mut normal = [tangent[1] / length, -tangent[0] / length];
        if normal[0] * (midpoint[0] - centroid[0]) + normal[1] * (midpoint[1] - centroid[1]) < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let left_velocity =
            fixed_velocity[left.index()].expect("every boundary vertex owns prescribed velocity");
        let right_velocity =
            fixed_velocity[right.index()].expect("every boundary vertex owns prescribed velocity");
        let contribution = 0.5
            * length
            * ((left_velocity[0] + right_velocity[0]) * normal[0]
                + (left_velocity[1] + right_velocity[1]) * normal[1]);
        net_flux.add(contribution);
        flux_scale.add(contribution.abs());
        boundary_facet_count += 1;
    }
    let net_flux = net_flux.total();
    let flux_scale = flux_scale.total();
    let accumulated_roundoff = boundary_facet_count as f64 * f64::EPSILON;
    if accumulated_roundoff >= 1.0 {
        return Err(invalid(
            "MINI Stokes boundary inventory exceeds the floating-point validation limit",
        ));
    }
    let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
    let tolerance = 128.0 * gamma * flux_scale.max(1.0);
    if net_flux.abs() > tolerance {
        return Err(invalid(format!(
            "MINI Stokes prescribed P1 boundary trace has incompatible net outward flux {net_flux:e} (tolerance {tolerance:e})"
        )));
    }
    Ok(())
}

pub(crate) fn require_weak_incompressibility(
    full_system: &LinearSystem,
    residual: &[f64],
    layout: &MixedLayout,
    gauge_multiplier: Option<f64>,
    residual_target: f64,
) -> Result<f64, Diagnostic> {
    let gauge_weights = (0..layout.vertex_count)
        .map(|pressure| {
            let row = layout.full_pressure_offset + pressure;
            layout
                .full_gauge()
                .map(|gauge| full_system.matrix().entry(row, gauge).unwrap_or(0.0))
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    require_weak_incompressibility_from_gauge_weights(
        residual,
        layout,
        &gauge_weights,
        gauge_multiplier,
        residual_target,
    )
}

pub(crate) fn require_weak_incompressibility_from_gauge_weights(
    residual: &[f64],
    layout: &MixedLayout,
    gauge_weights: &[f64],
    gauge_multiplier: Option<f64>,
    residual_target: f64,
) -> Result<f64, Diagnostic> {
    if gauge_weights.len() != layout.vertex_count || residual.len() != layout.full_size {
        return Err(invalid(
            "MINI Stokes acceptance data differs from the exact full layout",
        ));
    }
    let mut squared_norm = 0.0;
    let mut squared_gauge_norm = 0.0;
    for (pressure, &gauge_weight) in gauge_weights.iter().enumerate() {
        let row = layout.full_pressure_offset + pressure;
        let multiplier = gauge_multiplier.unwrap_or(0.0);
        let continuity_residual = residual[row] - gauge_weight * multiplier;
        squared_norm += continuity_residual * continuity_residual;
        squared_gauge_norm += gauge_weight * gauge_weight;
    }
    let norm = squared_norm.sqrt();
    let gauge_norm = squared_gauge_norm.sqrt();
    let multiplier = gauge_multiplier.unwrap_or(0.0);
    let roundoff =
        4096.0 * f64::EPSILON * (1.0 + norm + multiplier.abs() * gauge_norm + residual_target);
    let tolerance = residual_target + multiplier.abs() * gauge_norm + roundoff;
    if !norm.is_finite() || norm > tolerance {
        return Err(invalid(format!(
            "MINI Stokes weak continuity residual {norm:e} exceeds the accepted bound {tolerance:e}"
        )));
    }
    Ok(norm)
}

pub(super) fn integrate_body_force<F>(
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    body_force: &F,
) -> Result<[f64; COMPONENTS], Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let mut result = [0.0; COMPONENTS];
    for cell_index in 0..mesh.entity_count(DIMENSION).expect("mesh owns cells") {
        let geometry = mesh
            .geometry_map(MeshEntity::new(DIMENSION, cell_index))
            .expect("accepted simplex cell owns geometry");
        for point in quadrature.points() {
            let mut coordinates = [0.0; DIMENSION];
            geometry.map_point(&point.coordinates, &mut coordinates)?;
            let force = body_force(coordinates)?;
            if force.iter().any(|value| !value.is_finite()) {
                return Err(invalid("MINI Stokes body force is non-finite"));
            }
            let scale = point.weight * geometry.measure_scale();
            for component in 0..COMPONENTS {
                result[component] += scale * force[component];
            }
        }
    }
    Ok(result)
}

pub(crate) fn integrate_pressure(
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    pressure: &[f64],
) -> Result<f64, Diagnostic> {
    let mut result = 0.0;
    let pressure_space = SimplexP1Space::new(DIMENSION)?;
    for cell_index in 0..mesh.entity_count(DIMENSION).expect("mesh owns cells") {
        let cell = MeshEntity::new(DIMENSION, cell_index);
        let geometry = mesh
            .geometry_map(cell)
            .expect("accepted simplex cell owns geometry");
        let vertices = mesh
            .entity_vertices(cell)
            .expect("accepted simplex cell owns vertices");
        for point in quadrature.points() {
            let basis = pressure_space.tabulate(&point.coordinates)?;
            let value = (0..P1_BASIS_COUNT)
                .map(|local| basis.values()[local] * pressure[vertices[local].index()])
                .sum::<f64>();
            result += point.weight * geometry.measure_scale() * value;
        }
    }
    Ok(result)
}

pub(crate) fn require_zero_gauge_multiplier(
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    gauge_multiplier: f64,
    residual_target: f64,
) -> Result<(), Diagnostic> {
    let mut measure = 0.0;
    for cell_index in 0..mesh.entity_count(DIMENSION).expect("mesh owns cells") {
        let geometry = mesh
            .geometry_map(MeshEntity::new(DIMENSION, cell_index))
            .expect("accepted simplex cell owns geometry");
        measure += geometry.measure_scale()
            * quadrature
                .points()
                .iter()
                .map(|point| point.weight)
                .sum::<f64>();
    }
    let pressure_rows = mesh.vertices().len() as f64;
    let tolerance = pressure_rows.sqrt() * residual_target / measure
        + 4096.0 * f64::EPSILON * (1.0 + gauge_multiplier.abs());
    if gauge_multiplier.abs() > tolerance {
        return Err(invalid(format!(
            "MINI Stokes gauge multiplier {gauge_multiplier:e} exceeds the compatible-flow bound {tolerance:e}"
        )));
    }
    Ok(())
}

pub(super) fn validate_problem(
    mesh: &SimplicialMesh,
    viscosity: f64,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if mesh.topological_dimension() != DIMENSION
        || mesh
            .vertices()
            .iter()
            .any(|coordinates| coordinates.len() != DIMENSION)
    {
        return Err(invalid(
            "MINI Stokes v0 requires an intrinsic 2D triangular mesh",
        ));
    }
    if !viscosity.is_finite() || viscosity <= 0.0 {
        return Err(invalid("MINI Stokes viscosity must be finite and positive"));
    }
    require_connected_fluid_mesh(mesh)?;
    require_quadrature(quadrature)
}

fn require_connected_fluid_mesh(mesh: &SimplicialMesh) -> Result<(), Diagnostic> {
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D simplex mesh owns cells");
    let mut visited = vec![false; cell_count];
    let mut pending = vec![0_usize];
    visited[0] = true;
    while let Some(cell_index) = pending.pop() {
        let cell = MeshEntity::new(DIMENSION, cell_index);
        for facet in mesh
            .incidence(cell, DIMENSION - 1)
            .expect("accepted triangle owns edge incidence")
        {
            for adjacent in mesh
                .incidence(facet.entity, DIMENSION)
                .expect("accepted edge owns cell incidence")
            {
                if !visited[adjacent.entity.index()] {
                    visited[adjacent.entity.index()] = true;
                    pending.push(adjacent.entity.index());
                }
            }
        }
    }
    if visited.iter().any(|visited| !visited) {
        return Err(invalid(
            "MINI Stokes bounded realization requires one connected fluid mesh",
        ));
    }
    Ok(())
}

pub(crate) fn require_local_geometry(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if geometry.reference_cell() != quadrature.reference_cell()
        || geometry.reference_cell().dimension() != DIMENSION
        || geometry.physical_dimension() != DIMENSION
    {
        return Err(invalid(
            "MINI Stokes geometry and triangle quadrature differ",
        ));
    }
    require_quadrature(quadrature)
}

pub(crate) fn require_facet_geometry(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if geometry.reference_cell() != quadrature.reference_cell()
        || geometry.reference_cell().dimension() != DIMENSION - 1
        || geometry.physical_dimension() != DIMENSION
        || quadrature.polynomial_exactness().unwrap_or(0)
            < super::REQUIRED_FACET_QUADRATURE_EXACTNESS
    {
        return Err(invalid(
            "MINI Stokes traction facet requires a 1D-in-2D affine edge and quadrature exact through degree one",
        ));
    }
    Ok(())
}

fn require_quadrature(quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
    if quadrature.reference_cell() != eqiora_meshing::ReferenceCell::simplex(DIMENSION)?
        || quadrature.polynomial_exactness().unwrap_or(0) < REQUIRED_QUADRATURE_EXACTNESS
    {
        return Err(invalid(
            "MINI Stokes requires triangle quadrature exact through total degree four",
        ));
    }
    Ok(())
}

pub(super) fn require_error_quadrature(quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
    if quadrature.reference_cell() != eqiora_meshing::ReferenceCell::simplex(DIMENSION)?
        || quadrature.polynomial_exactness().unwrap_or(0) < REQUIRED_ERROR_QUADRATURE_EXACTNESS
    {
        return Err(invalid(
            "MINI Stokes error evidence requires triangle quadrature exact through total degree six",
        ));
    }
    Ok(())
}

#[derive(Debug, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum + self.correction
    }
}
