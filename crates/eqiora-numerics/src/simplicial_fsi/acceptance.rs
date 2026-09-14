//! Independent residual, interface-action, pressure, and energy acceptance.

use eqiora_assembly::{CsrMatrix, LinearSystem};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, QuadratureRule, SimplicialMesh};
use eqiora_solver::CanonicalCsrSystemView;

use super::api::FixedReferenceFsiEnergyBalance;
use super::contract::{
    FixedReferenceFsiMaterial, FixedReferenceFsiState, FixedReferenceFsiStepConfig,
};
use super::element::dot;
use super::layout::FsiLayout;
use super::partition::FixedReferenceFsiPartition;
use super::{invalid, mini_count, p1_count};
use crate::affine_fem::physical_gradient;
use crate::continuum_kinematics::{symmetric_gradient, twice_symmetric_gradient_squared_norm};
use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace, SimplexP1Space};

pub(super) struct EnergyEvaluation<'a, const D: usize = 2> {
    pub(super) mesh: &'a SimplicialMesh,
    pub(super) partition: &'a FixedReferenceFsiPartition<D>,
    pub(super) previous: &'a FixedReferenceFsiState<D>,
    pub(super) next_vertex_velocity: &'a [[f64; D]],
    pub(super) next_bubbles: &'a std::collections::BTreeMap<eqiora_meshing::CellId, [f64; D]>,
    pub(super) next_displacement: &'a [[f64; D]],
    pub(super) config: FixedReferenceFsiStepConfig<D>,
    pub(super) quadrature: &'a QuadratureRule,
}

pub(super) fn energy_balance<const D: usize>(
    evaluation: EnergyEvaluation<'_, D>,
) -> Result<FixedReferenceFsiEnergyBalance, Diagnostic> {
    let EnergyEvaluation {
        mesh,
        partition,
        previous,
        next_vertex_velocity,
        next_bubbles,
        next_displacement,
        config,
        quadrature,
    } = evaluation;
    let material = config.material();
    let mut previous_kinetic = 0.0;
    let mut next_kinetic = 0.0;
    let mut previous_elastic = 0.0;
    let mut next_elastic = 0.0;
    let mut kinetic_increment = 0.0;
    let mut elastic_increment = 0.0;
    let mut viscous_dissipation = 0.0;
    let p1_count = p1_count::<D>();
    let mini_count = mini_count::<D>();
    let mini = SimplexP1BubbleSpace::new(D)?;
    let p1 = SimplexP1Space::new(D)?;

    for cell in partition.fluid_cells().iter().copied() {
        let entity = MeshEntity::new(D, cell.index());
        let geometry = mesh
            .geometry_map(entity)
            .expect("accepted fluid cell owns geometry");
        let inverse = geometry.inverse_jacobian()?;
        let vertices = mesh
            .entity_vertices(entity)
            .expect("accepted cell owns vertices");
        for point in quadrature.points() {
            let basis = mini.tabulate(&point.coordinates)?;
            let gradients = (0..mini_count)
                .map(|index| {
                    physical_gradient(
                        basis.gradient(index).expect("accepted MINI basis"),
                        &inverse,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let mut old = [0.0; D];
            let mut new = [0.0; D];
            let mut new_gradient = [[0.0; D]; D];
            for local in 0..p1_count {
                for component in 0..D {
                    old[component] += basis.values()[local]
                        * previous.vertex_velocity()[vertices[local].index()][component];
                    new[component] += basis.values()[local]
                        * next_vertex_velocity[vertices[local].index()][component];
                    for axis in 0..D {
                        new_gradient[component][axis] += gradients[local][axis]
                            * next_vertex_velocity[vertices[local].index()][component];
                    }
                }
            }
            for component in 0..D {
                old[component] += basis.values()[p1_count]
                    * previous.fluid_cell_bubble_velocity()[&cell][component];
                new[component] += basis.values()[p1_count] * next_bubbles[&cell][component];
                for axis in 0..D {
                    new_gradient[component][axis] +=
                        gradients[p1_count][axis] * next_bubbles[&cell][component];
                }
            }
            let weight = point.weight * geometry.measure_scale();
            previous_kinetic += 0.5 * weight * material.fluid_density() * dot(&old, &old);
            next_kinetic += 0.5 * weight * material.fluid_density() * dot(&new, &new);
            let difference: [f64; D] =
                std::array::from_fn(|component| new[component] - old[component]);
            kinetic_increment +=
                0.5 * weight * material.fluid_density() * dot(&difference, &difference);
            viscous_dissipation += weight
                * material.fluid_dynamic_viscosity()
                * twice_symmetric_gradient_squared_norm(&new_gradient);
        }
    }

    for cell in partition.solid_cells() {
        let entity = MeshEntity::new(D, cell.index());
        let geometry = mesh
            .geometry_map(entity)
            .expect("accepted solid cell owns geometry");
        let inverse = geometry.inverse_jacobian()?;
        let vertices = mesh
            .entity_vertices(entity)
            .expect("accepted cell owns vertices");
        for point in quadrature.points() {
            let basis = p1.tabulate(&point.coordinates)?;
            let gradients = (0..p1_count)
                .map(|index| {
                    physical_gradient(
                        basis.gradient(index).expect("accepted P1 basis"),
                        &inverse,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let mut old_velocity = [0.0; D];
            let mut new_velocity = [0.0; D];
            let mut old_displacement_gradient = [[0.0; D]; D];
            let mut new_displacement_gradient = [[0.0; D]; D];
            let mut increment_gradient = [[0.0; D]; D];
            for local in 0..p1_count {
                let vertex = vertices[local].index();
                for component in 0..D {
                    old_velocity[component] +=
                        basis.values()[local] * previous.vertex_velocity()[vertex][component];
                    new_velocity[component] +=
                        basis.values()[local] * next_vertex_velocity[vertex][component];
                    for axis in 0..D {
                        old_displacement_gradient[component][axis] += gradients[local][axis]
                            * previous.solid_displacement()[vertex][component];
                        new_displacement_gradient[component][axis] +=
                            gradients[local][axis] * next_displacement[vertex][component];
                        increment_gradient[component][axis] += gradients[local][axis]
                            * (next_displacement[vertex][component]
                                - previous.solid_displacement()[vertex][component]);
                    }
                }
            }
            let weight = point.weight * geometry.measure_scale();
            previous_kinetic +=
                0.5 * weight * material.solid_density() * dot(&old_velocity, &old_velocity);
            next_kinetic +=
                0.5 * weight * material.solid_density() * dot(&new_velocity, &new_velocity);
            let difference: [f64; D] =
                std::array::from_fn(|component| new_velocity[component] - old_velocity[component]);
            kinetic_increment +=
                0.5 * weight * material.solid_density() * dot(&difference, &difference);
            previous_elastic +=
                weight * elastic_energy_density(&old_displacement_gradient, material);
            next_elastic += weight * elastic_energy_density(&new_displacement_gradient, material);
            elastic_increment += weight * elastic_energy_density(&increment_gradient, material);
        }
    }
    let viscous_dissipation = config.time_step() * viscous_dissipation;
    let defect = next_kinetic - previous_kinetic + next_elastic - previous_elastic
        + kinetic_increment
        + elastic_increment
        + viscous_dissipation;
    let values = [
        previous_kinetic,
        next_kinetic,
        previous_elastic,
        next_elastic,
        kinetic_increment,
        elastic_increment,
        viscous_dissipation,
        defect,
    ];
    if values.into_iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "fixed-reference FSI energy evidence must be finite",
        ));
    }
    Ok(FixedReferenceFsiEnergyBalance {
        previous_kinetic,
        next_kinetic,
        previous_elastic,
        next_elastic,
        kinetic_increment,
        elastic_increment,
        viscous_dissipation,
        defect,
    })
}

fn elastic_energy_density<const D: usize>(
    gradient: &[[f64; D]; D],
    material: FixedReferenceFsiMaterial<D>,
) -> f64 {
    material
        .solid_material()
        .strain_energy_density(&symmetric_gradient(gradient))
}

pub(super) fn require_pressure_closed_by_complete_operator<const D: usize>(
    system: &LinearSystem,
    layout: &FsiLayout<D>,
) -> Result<f64, Diagnostic> {
    let mut constant_pressure = vec![0.0; layout.reduced_size()];
    for dof in layout.reduced_pressure_dofs() {
        constant_pressure[dof] = 1.0;
    }
    let action = system.matrix().multiply(&constant_pressure)?;
    let action_norm = norm(&action);
    let matrix_scale = system
        .matrix()
        .values()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * (layout.reduced_size() as f64).sqrt();
    let tolerance = 8192.0 * f64::EPSILON * matrix_scale;
    if !action_norm.is_finite() || action_norm <= tolerance {
        return Err(invalid(format!(
            "fixed-reference FSI complete operator leaves constant pressure unclosed: action {action_norm:e}, threshold {tolerance:e}"
        )));
    }
    Ok(action_norm)
}

pub(super) fn require_symmetric(matrix: &CsrMatrix) -> Result<(), Diagnostic> {
    let scale = matrix
        .values()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 4096.0 * f64::EPSILON * scale.max(1.0);
    for row in 0..matrix.rows() {
        for column in 0..matrix.columns() {
            let left = matrix.entry(row, column).expect("indices are in range");
            let right = matrix.entry(column, row).expect("indices are in range");
            if (left - right).abs() > tolerance {
                return Err(invalid(format!(
                    "fixed-reference FSI reduced operator is not symmetric at ({row}, {column})"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn apply_canonical(
    system: &CanonicalCsrSystemView,
    values: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let mut output = vec![0.0; system.rows()];
    let problem = system.linear_problem()?;
    eqiora_solver::LinearOperator::apply(problem.operator(), values, &mut output)?;
    Ok(output)
}

pub(super) fn kinematic_residual_norm<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    previous: &FixedReferenceFsiState<D>,
    velocity: &[[f64; D]],
    displacement: &[[f64; D]],
    time_step: f64,
) -> f64 {
    partition
        .solid_vertices()
        .iter()
        .flat_map(|vertex| {
            (0..D).map(move |component| {
                displacement[vertex.index()][component]
                    - previous.solid_displacement()[vertex.index()][component]
                    - time_step * velocity[vertex.index()][component]
            })
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

pub(super) fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
