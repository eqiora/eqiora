//! Independent residual, interface-action, pressure, and energy acceptance.

use eqiora_assembly::{CsrMatrix, LinearSystem};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, QuadratureRule, SimplicialMesh};
use eqiora_solver::CanonicalCsrSystemView;

use super::api::FixedReferenceFsiEnergyBalance;
use super::contract::{FixedReferenceFsiState, FixedReferenceFsiStepConfig};
use super::element::dot;
use super::invalid;
use super::layout::FsiLayout;
use super::partition::FixedReferenceFsiPartition;
use crate::affine_fem::physical_gradient;
use crate::continuum_kinematics::{symmetric_gradient, twice_symmetric_gradient_squared_norm};

pub(super) struct EnergyEvaluation<'a, const D: usize = 2> {
    pub(super) mesh: &'a SimplicialMesh,
    pub(super) partition: &'a FixedReferenceFsiPartition<D>,
    pub(super) layout: &'a FsiLayout<D>,
    pub(super) previous: &'a FixedReferenceFsiState<D>,
    pub(super) next: &'a FixedReferenceFsiState<D>,
    pub(super) config: &'a FixedReferenceFsiStepConfig<D>,
    pub(super) quadrature: &'a QuadratureRule,
}

pub(super) fn energy_balance<const D: usize>(
    evaluation: EnergyEvaluation<'_, D>,
) -> Result<FixedReferenceFsiEnergyBalance, Diagnostic> {
    let EnergyEvaluation {
        mesh,
        partition,
        layout,
        previous,
        next,
        config,
        quadrature,
    } = evaluation;
    layout.require_material(config)?;
    let mut previous_kinetic = 0.0;
    let mut next_kinetic = 0.0;
    let mut previous_elastic = 0.0;
    let mut next_elastic = 0.0;
    let mut kinetic_increment = 0.0;
    let mut elastic_increment = 0.0;
    let mut viscous_dissipation = 0.0;
    for (&field, &(domain, density)) in &config.material().densities {
        let (_, field_layout) = layout
            .mapping()
            .field_layout(field)
            .expect("validated kinetic Field");
        let basis = crate::form_compiler::region::basis(
            field_layout.space,
            eqiora_meshing::ReferenceCell::simplex(D)?,
        )?;
        for cell in partition
            .domain_cells(domain.downcast().expect("Domain"))
            .expect("validated Domain")
        {
            let entity = MeshEntity::new(D, cell.index());
            let geometry = mesh
                .geometry_map(entity)
                .ok_or_else(|| invalid("kinetic cell has no exact geometry"))?;
            let inverse = geometry.inverse_jacobian()?;
            let keys = layout.mapping().cell_field_keys(cell.index(), field)?;
            for point in quadrature.points() {
                let tabulation = basis.tabulate(&point.coordinates)?;
                let gradients = (0..tabulation.values().len())
                    .map(|i| {
                        physical_gradient(
                            tabulation.gradient(i).expect("basis gradient"),
                            &inverse,
                            D,
                        )
                    })
                    .collect::<Vec<_>>();
                let (old, _) = sample::<D>(previous, &keys, tabulation.values(), &gradients)?;
                let (new, gradient) = sample::<D>(next, &keys, tabulation.values(), &gradients)?;
                let difference: [f64; D] = std::array::from_fn(|i| new[i] - old[i]);
                let weight = point.weight * geometry.measure_scale();
                previous_kinetic += 0.5 * weight * density * dot(&old, &old);
                next_kinetic += 0.5 * weight * density * dot(&new, &new);
                kinetic_increment += 0.5 * weight * density * dot(&difference, &difference);
                if let Some(&(_, viscosity)) = config.material().viscosities.get(&field) {
                    viscous_dissipation +=
                        weight * viscosity * twice_symmetric_gradient_squared_norm(&gradient);
                }
            }
        }
    }
    for (&field, &(domain, material)) in &config.material().elasticities {
        let rate = layout.state_rate(field)?;
        let (_, field_layout) = layout
            .mapping()
            .field_layout(rate)
            .expect("validated state-rate Field");
        let basis = crate::form_compiler::region::basis(
            field_layout.space,
            eqiora_meshing::ReferenceCell::simplex(D)?,
        )?;
        for cell in partition
            .domain_cells(domain.downcast().expect("Domain"))
            .expect("validated Domain")
        {
            let entity = MeshEntity::new(D, cell.index());
            let geometry = mesh
                .geometry_map(entity)
                .ok_or_else(|| invalid("elastic cell has no exact geometry"))?;
            let inverse = geometry.inverse_jacobian()?;
            let keys = layout
                .mapping()
                .cell_field_keys(cell.index(), rate)?
                .into_iter()
                .map(|key| crate::region_assembly::mapping::FieldDof { field, ..key })
                .collect::<Vec<_>>();
            for point in quadrature.points() {
                let tabulation = basis.tabulate(&point.coordinates)?;
                let gradients = (0..tabulation.values().len())
                    .map(|i| {
                        physical_gradient(
                            tabulation.gradient(i).expect("basis gradient"),
                            &inverse,
                            D,
                        )
                    })
                    .collect::<Vec<_>>();
                let (_, old) = sample::<D>(previous, &keys, tabulation.values(), &gradients)?;
                let (_, new) = sample::<D>(next, &keys, tabulation.values(), &gradients)?;
                let difference =
                    std::array::from_fn(|i| std::array::from_fn(|j| new[i][j] - old[i][j]));
                let weight = point.weight * geometry.measure_scale();
                previous_elastic +=
                    weight * material.strain_energy_density(&symmetric_gradient(&old));
                next_elastic += weight * material.strain_energy_density(&symmetric_gradient(&new));
                elastic_increment +=
                    weight * material.strain_energy_density(&symmetric_gradient(&difference));
            }
        }
    }
    let viscous_dissipation = config.time_step() * viscous_dissipation;
    let defect = next_kinetic - previous_kinetic + next_elastic - previous_elastic
        + kinetic_increment
        + elastic_increment
        + viscous_dissipation;
    if [
        previous_kinetic,
        next_kinetic,
        previous_elastic,
        next_elastic,
        kinetic_increment,
        elastic_increment,
        viscous_dissipation,
        defect,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err(invalid("transient energy evidence must be finite"));
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

fn sample<const D: usize>(
    state: &FixedReferenceFsiState<D>,
    keys: &[crate::region_assembly::mapping::FieldDof],
    values: &[f64],
    gradients: &[Vec<f64>],
) -> Result<([f64; D], [[f64; D]; D]), Diagnostic> {
    if keys.len() != values.len() * D || gradients.len() != values.len() {
        return Err(invalid(
            "energy Field coordinates differ from exact vector basis",
        ));
    }
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..values.len() {
        for component in 0..D {
            let key = keys[local * D + component];
            let coefficient = state
                .fields
                .get(&key.field)
                .and_then(|field| field.coefficients.get(&key))
                .copied()
                .ok_or_else(|| invalid("energy history omits an exact Field coordinate"))?;
            value[component] += values[local] * coefficient;
            for axis in 0..D {
                gradient[component][axis] += gradients[local][axis] * coefficient;
            }
        }
    }
    Ok((value, gradient))
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
    previous: &FixedReferenceFsiState<D>,
    next: &FixedReferenceFsiState<D>,
    step: &eqiora_realization::BackwardEulerStep,
) -> Result<f64, Diagnostic> {
    let mut squared = 0.0;
    for binding in step.eliminated_states() {
        let pair = binding.pair();
        let old = previous
            .fields
            .get(&pair.state().erase())
            .ok_or_else(|| invalid("kinematic history omits exact state"))?;
        let state = next
            .fields
            .get(&pair.state().erase())
            .ok_or_else(|| invalid("kinematic recovery omits exact state"))?;
        let rate = next
            .fields
            .get(&pair.rate().erase())
            .ok_or_else(|| invalid("kinematic recovery omits exact rate"))?;
        for (&key, value) in &state.coefficients {
            let old = old
                .coefficients
                .get(&key)
                .ok_or_else(|| invalid("kinematic history omits exact coordinate"))?;
            let rate_key = crate::region_assembly::mapping::FieldDof {
                field: pair.rate().erase(),
                ..key
            };
            let rate = rate
                .coefficients
                .get(&rate_key)
                .ok_or_else(|| invalid("kinematic rate omits exact coordinate"))?;
            squared += (value - old - step.duration().value() * rate).powi(2);
        }
    }
    let norm = squared.sqrt();
    if !norm.is_finite() {
        return Err(invalid("kinematic residual is nonfinite"));
    }
    Ok(norm)
}

pub(super) fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
