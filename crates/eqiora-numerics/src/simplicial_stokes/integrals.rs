//! Independent physical integrals, shared by explicit queries and required constraints.
use super::{COMPONENTS, DIMENSION, P1_BASIS_COUNT, invalid};
use crate::discrete_space::DiscreteSpace;
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};

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
    let pressure_space = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::simplex(DIMENSION)?,
    )?;
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

pub(super) fn integrated_traction(
    mesh: &SimplicialMesh,
    facets: &[super::boundary::PreparedTractionFacet2d],
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let mut integrated = [0.0; COMPONENTS];
    for facet in facets {
        let geometry = mesh
            .geometry_map(facet.facet)
            .expect("validated boundary facet owns affine geometry");
        let length = geometry.measure_scale();
        for (result, value) in integrated.iter_mut().zip(facet.value) {
            *result += length * value;
        }
    }
    if integrated.iter().any(|component| !component.is_finite()) {
        return Err(invalid(
            "MINI Stokes integrated boundary traction is non-finite",
        ));
    }
    Ok(integrated)
}
