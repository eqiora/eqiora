//! One anonymous mathematical quadrature kernel for typed and callback-owned forms.

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_meshing::{
    AffineGeometryLinearization, AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCell,
};
use eqiora_realization::Space;
use num_complex::ComplexFloat;
use std::ops::{AddAssign, SubAssign};

use crate::discrete_space::DiscreteSpace;
use crate::form_compiler::bilinear::{Basis, Pairing};

use super::invalid;

pub(in crate::form_compiler) type IntegralActions<S> =
    (LocalContribution<S>, Option<LocalContribution<S>>);

/// Physical basis components and scalar coordinates are independent axes.
/// A complex edge coefficient has two real coordinates, but its basis remains
/// a three-dimensional vector with the Space-owned Piola transformation.
pub(in crate::form_compiler) struct IntegralField {
    pub space: Space,
    pub components: usize,
    pub scalar_coordinates: usize,
}

pub(in crate::form_compiler) struct IntegralTerm {
    pub row: usize,
    pub column: usize,
    pub pairing: Pairing,
    pub trial_scale: f64,
    pub test_coordinate: usize,
    pub trial_coordinate: usize,
}

pub(in crate::form_compiler) fn integrate<
    S: Scalar + ComplexFloat<Real = f64> + From<f64> + AddAssign + SubAssign,
>(
    reference: ReferenceCell,
    fields: &[IntegralField],
    terms: &[IntegralTerm],
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    values: impl Fn(&[f64], &mut [S], &mut [S], &mut [S]) -> Result<(), Diagnostic>,
) -> Result<LocalContribution<S>, Diagnostic> {
    integrate_with_geometry_direction(
        reference,
        fields,
        terms,
        geometry,
        None,
        quadrature,
        |_, point, _, coefficients, forcing, flux, _, _, _| {
            values(point, coefficients, forcing, flux)
        },
    )
    .map(|(primal, _)| primal)
}

/// Integrate primal and geometry-direction actions through the same pairings.
/// The caller differentiates spatial data at the supplied physical point;
/// this kernel owns basis maps, physical gradients and measure derivatives.
pub(in crate::form_compiler) fn integrate_with_geometry_direction<
    S: Scalar + ComplexFloat<Real = f64> + From<f64> + AddAssign + SubAssign,
>(
    reference: ReferenceCell,
    fields: &[IntegralField],
    terms: &[IntegralTerm],
    geometry: &AffineGeometryMap,
    direction: Option<&AffineGeometryLinearization>,
    quadrature: &QuadratureRule,
    values: impl Fn(
        &[f64],
        &[f64],
        &[f64],
        &mut [S],
        &mut [S],
        &mut [S],
        &mut [S],
        &mut [S],
        &mut [S],
    ) -> Result<(), Diagnostic>,
) -> Result<IntegralActions<S>, Diagnostic> {
    if direction.is_some_and(|direction| direction.map() != geometry) {
        return Err(invalid(
            "quadrature geometry direction has a different primal map",
        ));
    }
    let scalar = <S as From<f64>>::from;
    let zero = scalar(0.0);
    let dimension = reference.dimension();
    if fields.is_empty()
        || fields
            .iter()
            .any(|field| !(1..=2).contains(&field.scalar_coordinates))
        || geometry.reference_cell() != reference
        || quadrature.reference_cell() != reference
        || geometry.physical_dimension() != dimension
    {
        return Err(invalid(
            "region geometry, quadrature or Field count mismatch",
        ));
    }
    let spaces = fields
        .iter()
        .map(|field| DiscreteSpace::new(field.space, reference))
        .collect::<Result<Vec<_>, _>>()?;
    let mut offsets = vec![0usize];
    for (space, field) in spaces.iter().zip(fields) {
        let local = space
            .field_dof_count(field.components)?
            .checked_mul(field.scalar_coordinates)
            .ok_or_else(|| invalid("region scalar coordinate count overflow"))?;
        offsets.push(
            offsets
                .last()
                .unwrap()
                .checked_add(local)
                .ok_or_else(|| invalid("region local DOF count overflow"))?,
        );
    }
    if terms.iter().any(|term| {
        fields
            .get(term.row)
            .zip(fields.get(term.column))
            .is_none_or(|(test, trial)| {
                !term
                    .pairing
                    .accepts(dimension, test.components, trial.components)
                    || term.test_coordinate >= test.scalar_coordinates
                    || term.trial_coordinate >= trial.scalar_coordinates
            })
            || !term.trial_scale.is_finite()
    }) {
        return Err(invalid(
            "local pairing is incompatible with its Field shapes or scale",
        ));
    }
    let count = *offsets.last().unwrap();
    let entries = count
        .checked_mul(count)
        .ok_or_else(|| invalid("region matrix size overflow"))?;
    let mut matrix = vec![zero; entries];
    let mut rhs = vec![zero; count];
    let mut matrix_tangent = direction.map(|_| vec![zero; entries]);
    let mut rhs_tangent = direction.map(|_| vec![zero; count]);
    let mut coefficients = vec![zero; terms.len()];
    let mut forcing_offsets = vec![0usize];
    for field in fields {
        forcing_offsets
            .push(forcing_offsets.last().unwrap() + field.components * field.scalar_coordinates);
    }
    let mut forcing = vec![zero; *forcing_offsets.last().unwrap()];
    let mut flux_load = vec![zero; forcing.len() * dimension];
    let mut physical = vec![0.0; dimension];
    let mut physical_tangent = vec![0.0; dimension];
    let mut coefficient_tangents =
        direction.map_or_else(Vec::new, |_| vec![zero; coefficients.len()]);
    let mut forcing_tangents = direction.map_or_else(Vec::new, |_| vec![zero; forcing.len()]);
    let mut flux_tangents = direction.map_or_else(Vec::new, |_| vec![zero; flux_load.len()]);
    for point in quadrature.points() {
        match direction {
            Some(direction) => {
                direction.map_point_jvp(&point.coordinates, &mut physical, &mut physical_tangent)?
            }
            None => geometry.map_point(&point.coordinates, &mut physical)?,
        }
        let tabulations = spaces
            .iter()
            .zip(fields)
            .map(|(space, field)| match direction {
                Some(direction) => space
                    .tabulate_field_linearized(direction, &point.coordinates, field.components)
                    .map(|(primal, tangent)| (primal, Some(tangent))),
                None => space
                    .tabulate_field_on(geometry, &point.coordinates, field.components)
                    .map(|primal| (primal, None)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        coefficients.fill(zero);
        forcing.fill(zero);
        flux_load.fill(zero);
        coefficient_tangents.fill(zero);
        forcing_tangents.fill(zero);
        flux_tangents.fill(zero);
        values(
            &point.coordinates,
            &physical,
            &physical_tangent,
            &mut coefficients,
            &mut forcing,
            &mut flux_load,
            &mut coefficient_tangents,
            &mut forcing_tangents,
            &mut flux_tangents,
        )?;
        if coefficients
            .iter()
            .chain(&forcing)
            .chain(&flux_load)
            .chain(&coefficient_tangents)
            .chain(&forcing_tangents)
            .chain(&flux_tangents)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("region coefficient or forcing is non-finite"));
        }
        let weight = point.weight * geometry.measure_scale();
        let weight_tangent = direction.map_or(0.0, |direction| {
            point.weight * direction.measure_scale_tangent()
        });
        for (row, field) in fields.iter().enumerate() {
            let table = &tabulations[row].0;
            for test in 0..offsets[row + 1] - offsets[row] {
                let coordinate = test % field.scalar_coordinates;
                let basis = test / field.scalar_coordinates;
                let component_slot = |component| {
                    forcing_offsets[row] + component * field.scalar_coordinates + coordinate
                };
                let flux_slot = |component| {
                    forcing_offsets[row] * dimension
                        + component * field.scalar_coordinates
                        + coordinate
                };
                let value = table.value(basis).expect("validated Field basis");
                let load = value
                    .iter()
                    .enumerate()
                    .fold(zero, |sum, (component, value)| {
                        sum + forcing[component_slot(component)] * scalar(*value)
                    });
                let gradient = table.gradient(basis).expect("validated Field gradient");
                let flux = gradient
                    .iter()
                    .enumerate()
                    .fold(zero, |sum, (component, value)| {
                        sum + flux_load[flux_slot(component)] * scalar(*value)
                    });
                rhs[offsets[row] + test] += scalar(weight) * (load - flux);
                if let Some(tangent_rhs) = &mut rhs_tangent {
                    let tangent_table =
                        tabulations[row].1.as_ref().expect("bound geometry tangent");
                    let tangent_value =
                        tangent_table.value(basis).expect("validated tangent basis");
                    let load_tangent = value.iter().zip(tangent_value).enumerate().fold(
                        zero,
                        |sum, (component, (value, tangent))| {
                            sum + forcing_tangents[component_slot(component)] * scalar(*value)
                                + forcing[component_slot(component)] * scalar(*tangent)
                        },
                    );
                    let tangent_gradient = tangent_table
                        .gradient(basis)
                        .expect("validated gradient tangent");
                    let flux_tangent = gradient.iter().zip(tangent_gradient).enumerate().fold(
                        zero,
                        |sum, (component, (value, tangent))| {
                            sum + flux_tangents[flux_slot(component)] * scalar(*value)
                                + flux_load[flux_slot(component)] * scalar(*tangent)
                        },
                    );
                    tangent_rhs[offsets[row] + test] += scalar(weight_tangent) * (load - flux)
                        + scalar(weight) * (load_tangent - flux_tangent);
                }
            }
        }
        for (term_index, (term, coefficient)) in terms.iter().zip(&coefficients).enumerate() {
            let row = term.row;
            let column = term.column;
            let test_table = &tabulations[row].0;
            let trial_table = &tabulations[column].0;
            let test_width = fields[row].scalar_coordinates;
            let trial_width = fields[column].scalar_coordinates;
            for test in 0..(offsets[row + 1] - offsets[row]) / test_width {
                for trial in 0..(offsets[column + 1] - offsets[column]) / trial_width {
                    let test_basis = Basis {
                        value: test_table.value(test).expect("validated test basis"),
                        gradient: test_table.gradient(test).expect("validated test gradient"),
                    };
                    let trial_basis = Basis {
                        value: trial_table.value(trial).expect("validated trial basis"),
                        gradient: trial_table
                            .gradient(trial)
                            .expect("validated trial gradient"),
                    };
                    let pairing = term.pairing.entry(test_basis, trial_basis);
                    let slot = (offsets[row] + test * test_width + term.test_coordinate) * count
                        + offsets[column]
                        + trial * trial_width
                        + term.trial_coordinate;
                    matrix[slot] +=
                        scalar(weight) * *coefficient * scalar(pairing) * scalar(term.trial_scale);
                    if let Some(tangent_matrix) = &mut matrix_tangent {
                        let test_tangent =
                            tabulations[row].1.as_ref().expect("bound geometry tangent");
                        let trial_tangent = tabulations[column]
                            .1
                            .as_ref()
                            .expect("bound geometry tangent");
                        let pairing_tangent = term.pairing.entry(
                            Basis {
                                value: test_tangent.value(test).expect("validated tangent basis"),
                                gradient: test_tangent
                                    .gradient(test)
                                    .expect("validated tangent gradient"),
                            },
                            trial_basis,
                        ) + term.pairing.entry(
                            test_basis,
                            Basis {
                                value: trial_tangent.value(trial).expect("validated tangent basis"),
                                gradient: trial_tangent
                                    .gradient(trial)
                                    .expect("validated tangent gradient"),
                            },
                        );
                        tangent_matrix[slot] += (scalar(weight_tangent)
                            * *coefficient
                            * scalar(pairing)
                            + scalar(weight) * coefficient_tangents[term_index] * scalar(pairing)
                            + scalar(weight) * *coefficient * scalar(pairing_tangent))
                            * scalar(term.trial_scale);
                    }
                }
            }
        }
    }
    let tangent = matrix_tangent
        .zip(rhs_tangent)
        .map(|(matrix, rhs)| LocalContribution::new(count, count, matrix, rhs))
        .transpose()?;
    Ok((LocalContribution::new(count, count, matrix, rhs)?, tangent))
}

pub(in crate::form_compiler) fn integrate_scalar<
    S: Scalar + ComplexFloat<Real = f64> + From<f64> + AddAssign + SubAssign,
>(
    dimension: usize,
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    values: impl Fn(&[f64]) -> Result<(S, S), Diagnostic>,
) -> Result<LocalContribution<S>, Diagnostic> {
    integrate(
        ReferenceCell::hypercube(dimension)?,
        &[IntegralField {
            space: Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            components: 1,
            scalar_coordinates: 1,
        }],
        &[IntegralTerm {
            test_coordinate: 0,
            trial_coordinate: 0,
            row: 0,
            column: 0,
            pairing: Pairing::Gradient,
            trial_scale: 1.0,
        }],
        geometry,
        quadrature,
        |point, coefficient, forcing, _| {
            (coefficient[0], forcing[0]) = values(point)?;
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests;
