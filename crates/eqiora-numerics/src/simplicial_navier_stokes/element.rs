use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshGeometry, MeshTopology, QuadratureRule, ReferenceCellFamily};

use super::{
    REQUIRED_CONVECTIVE_FACET_QUADRATURE_EXACTNESS, REQUIRED_CONVECTIVE_QUADRATURE_EXACTNESS,
    invalid,
};
use crate::simplicial_stokes::SimplicialMiniVelocityField2d;
use crate::simplicial_stokes::element::{MiniSpaces, physical_gradients};
use crate::simplicial_stokes::{COMPONENTS, DIMENSION, P1_BASIS_COUNT, VELOCITY_BASIS_COUNT};

pub(super) struct ConvectiveRealizationEvidence {
    pub(super) skew_residual_norm: f64,
    pub(super) skew_power: f64,
    pub(super) conservative_defect_norm: f64,
    pub(super) defect_identity_error: f64,
}

pub(super) fn require_convective_evidence_quadrature(
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if cell_quadrature.polynomial_exactness().unwrap_or(0)
        < REQUIRED_CONVECTIVE_QUADRATURE_EXACTNESS
    {
        return Err(invalid(
            "convective evidence requires degree-eight cell quadrature exactness",
        ));
    }
    let facet_cell = facet_quadrature.reference_cell();
    if facet_cell.family() != ReferenceCellFamily::Simplex
        || facet_cell.dimension() != DIMENSION - 1
        || facet_quadrature.polynomial_exactness().unwrap_or(0)
            < REQUIRED_CONVECTIVE_FACET_QUADRATURE_EXACTNESS
    {
        return Err(invalid(
            "convective boundary-flux evidence requires degree-three quadrature exactness on the one-dimensional unit simplex",
        ));
    }
    Ok(())
}

pub(super) fn integrate_convective_evidence(
    mesh: &eqiora_meshing::SimplicialMesh,
    velocity: &SimplicialMiniVelocityField2d,
    density: f64,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
) -> Result<ConvectiveRealizationEvidence, Diagnostic> {
    require_convective_evidence_quadrature(cell_quadrature, facet_quadrature)?;
    let spaces = MiniSpaces::new()?;
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D simplex mesh owns cells");
    let vertex_count = mesh.vertices().len();
    let width = COMPONENTS * (vertex_count + cell_count);
    let mut skew_global = vec![0.0; width];
    let mut conservative_global = vec![0.0; width];
    let mut expected_defect_global = vec![0.0; width];
    for cell in 0..cell_count {
        let entity = eqiora_meshing::MeshEntity::new(DIMENSION, cell);
        let geometry = mesh
            .geometry_map(entity)
            .expect("accepted simplex cell owns geometry");
        let vertices = mesh
            .entity_vertices(entity)
            .expect("accepted simplex cell owns vertices");
        let coefficients = local_velocity_coefficients(velocity, cell, &vertices);
        let inverse = geometry.inverse_jacobian()?;
        let mut local_residual = [0.0; crate::simplicial_stokes::LOCAL_VELOCITY_DOF_COUNT];
        let mut conservative = [0.0; crate::simplicial_stokes::LOCAL_VELOCITY_DOF_COUNT];
        let mut expected_defect = [0.0; crate::simplicial_stokes::LOCAL_VELOCITY_DOF_COUNT];
        for point in cell_quadrature.points() {
            let basis = spaces.tabulate(&point.coordinates)?;
            let gradients = physical_gradients(&basis, &inverse);
            let (value, gradient) = evaluate_velocity(&coefficients, &basis.values, &gradients);
            let divergence = (0..DIMENSION).map(|axis| gradient[axis][axis]).sum::<f64>();
            let scale = point.weight * geometry.measure_scale();
            for (row_basis, test_gradient) in gradients.iter().enumerate() {
                let velocity_dot_test_gradient = dot(&value, test_gradient);
                for component in 0..COMPONENTS {
                    let row = local_velocity(row_basis, component);
                    local_residual[row] += 0.5
                        * scale
                        * density
                        * (dot(&value, &gradient[component]) * basis.values[row_basis]
                            - velocity_dot_test_gradient * value[component]);
                    conservative[row] -=
                        scale * density * velocity_dot_test_gradient * value[component];
                    expected_defect[row] -= 0.5
                        * scale
                        * density
                        * divergence
                        * value[component]
                        * basis.values[row_basis];
                }
            }
        }
        for (basis, vertex) in vertices.iter().enumerate() {
            for component in 0..COMPONENTS {
                let local = local_velocity(basis, component);
                let global = COMPONENTS * vertex.index() + component;
                skew_global[global] += local_residual[local];
                conservative_global[global] += conservative[local];
                expected_defect_global[global] += expected_defect[local];
            }
        }
        for component in 0..COMPONENTS {
            let local = local_velocity(P1_BASIS_COUNT, component);
            let global = COMPONENTS * (vertex_count + cell) + component;
            skew_global[global] += local_residual[local];
            conservative_global[global] += conservative[local];
            expected_defect_global[global] += expected_defect[local];
        }
    }
    add_boundary_flux_defect(
        mesh,
        velocity,
        density,
        facet_quadrature,
        &mut expected_defect_global,
    )?;
    let coefficients = velocity
        .vertex_values()
        .iter()
        .chain(velocity.cell_bubble_values())
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let skew_residual_norm = super::norm(&skew_global)?;
    let energy = skew_global
        .iter()
        .zip(coefficients)
        .map(|(residual, coefficient)| residual * coefficient)
        .sum::<f64>();
    let mut defect = conservative_global;
    let mut identity_error = expected_defect_global;
    for ((skew, defect), identity_error) in
        skew_global.iter().zip(&mut defect).zip(&mut identity_error)
    {
        *defect = skew - *defect;
        *identity_error = *defect - *identity_error;
    }
    if !energy.is_finite() {
        return Err(invalid("convective evidence is non-finite"));
    }
    Ok(ConvectiveRealizationEvidence {
        skew_residual_norm,
        skew_power: energy,
        conservative_defect_norm: super::norm(&defect)?,
        defect_identity_error: super::norm(&identity_error)?,
    })
}

fn add_boundary_flux_defect(
    mesh: &eqiora_meshing::SimplicialMesh,
    velocity: &SimplicialMiniVelocityField2d,
    density: f64,
    quadrature: &QuadratureRule,
    expected_defect: &mut [f64],
) -> Result<(), Diagnostic> {
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .expect("2D simplex mesh owns edge entities");
    for facet_index in 0..facet_count {
        let facet = eqiora_meshing::MeshEntity::new(DIMENSION - 1, facet_index);
        if !mesh
            .is_boundary_entity(facet)
            .expect("accepted mesh classifies every edge")
        {
            continue;
        }
        let vertices = mesh
            .entity_vertices(facet)
            .expect("accepted boundary edge owns vertices");
        let adjacent = mesh
            .incidence(facet, DIMENSION)
            .expect("accepted boundary edge owns cell incidence");
        if vertices.len() != 2 || adjacent.len() != 1 {
            return Err(invalid(
                "convective boundary-flux evidence requires one segment and one incident parent cell",
            ));
        }
        let parent = mesh
            .cells()
            .get(adjacent[0].entity.index())
            .expect("accepted boundary incidence names a cell");
        let opposite = parent
            .iter()
            .copied()
            .find(|candidate| !vertices.iter().any(|vertex| vertex.index() == *candidate))
            .ok_or_else(|| invalid("boundary edge has no unique opposite parent vertex"))?;
        let first = &mesh.vertices()[vertices[0].index()];
        let second = &mesh.vertices()[vertices[1].index()];
        let opposite = &mesh.vertices()[opposite];
        let tangent = [second[0] - first[0], second[1] - first[1]];
        let parent_side =
            tangent[0] * (opposite[1] - first[1]) - tangent[1] * (opposite[0] - first[0]);
        if !parent_side.is_finite() || parent_side == 0.0 {
            return Err(invalid(
                "convective boundary-flux evidence found a degenerate parent orientation",
            ));
        }
        // This is the parent-outward unit normal multiplied by edge measure.
        // Keeping the product avoids a normalization and its cancelling
        // multiplication in the boundary integral.
        let outward_normal_measure = if parent_side > 0.0 {
            [tangent[1], -tangent[0]]
        } else {
            [-tangent[1], tangent[0]]
        };
        for point in quadrature.points() {
            let coordinate = point.coordinates[0];
            let basis = [1.0 - coordinate, coordinate];
            // The MINI bubble has exactly zero trace, so only the two P1
            // endpoint coefficients contribute on this edge.
            let value = std::array::from_fn::<_, COMPONENTS, _>(|component| {
                basis[0] * velocity.vertex_values()[vertices[0].index()][component]
                    + basis[1] * velocity.vertex_values()[vertices[1].index()][component]
            });
            let scale = 0.5 * point.weight * density * dot(&value, &outward_normal_measure);
            for local in 0..2 {
                for component in 0..COMPONENTS {
                    expected_defect[COMPONENTS * vertices[local].index() + component] +=
                        scale * value[component] * basis[local];
                }
            }
        }
    }
    Ok(())
}

pub(super) fn local_velocity_coefficients(
    velocity: &SimplicialMiniVelocityField2d,
    cell: usize,
    vertices: &[eqiora_meshing::MeshEntity],
) -> [[f64; COMPONENTS]; VELOCITY_BASIS_COUNT] {
    std::array::from_fn(|basis| {
        if basis < P1_BASIS_COUNT {
            velocity.vertex_values()[vertices[basis].index()]
        } else {
            velocity.cell_bubble_values()[cell]
        }
    })
}

pub(super) fn evaluate_velocity(
    coefficients: &[[f64; COMPONENTS]; VELOCITY_BASIS_COUNT],
    basis: &[f64; VELOCITY_BASIS_COUNT],
    gradients: &[[f64; DIMENSION]; VELOCITY_BASIS_COUNT],
) -> ([f64; COMPONENTS], [[f64; DIMENSION]; COMPONENTS]) {
    let mut value = [0.0; COMPONENTS];
    let mut gradient = [[0.0; DIMENSION]; COMPONENTS];
    for local in 0..VELOCITY_BASIS_COUNT {
        for component in 0..COMPONENTS {
            value[component] += coefficients[local][component] * basis[local];
            for axis in 0..DIMENSION {
                gradient[component][axis] +=
                    coefficients[local][component] * gradients[local][axis];
            }
        }
    }
    (value, gradient)
}

fn dot(left: &[f64; DIMENSION], right: &[f64; DIMENSION]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

const fn local_velocity(basis: usize, component: usize) -> usize {
    basis * COMPONENTS + component
}
