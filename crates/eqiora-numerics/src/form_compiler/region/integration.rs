//! One anonymous mathematical quadrature kernel for typed and callback-owned forms.

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCell};
use eqiora_realization::Space;
use num_complex::ComplexFloat;
use std::ops::{AddAssign, SubAssign};

use crate::discrete_space::DiscreteSpace;
use crate::form_compiler::bilinear::{Basis, Pairing};

use super::invalid;

pub(in crate::form_compiler) struct IntegralTerm {
    pub row: usize,
    pub column: usize,
    pub pairing: Pairing,
    pub trial_scale: f64,
}

pub(in crate::form_compiler) fn integrate<
    S: Scalar + ComplexFloat<Real = f64> + From<f64> + AddAssign + SubAssign,
>(
    reference: ReferenceCell,
    fields: &[(Space, usize)],
    terms: &[IntegralTerm],
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    values: impl Fn(&[f64], &mut [S], &mut [S], &mut [S]) -> Result<(), Diagnostic>,
) -> Result<LocalContribution<S>, Diagnostic> {
    let scalar = <S as From<f64>>::from;
    let zero = scalar(0.0);
    let dimension = reference.dimension();
    if fields.is_empty()
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
        .map(|(space, _)| DiscreteSpace::new(*space, reference))
        .collect::<Result<Vec<_>, _>>()?;
    let mut offsets = vec![0usize];
    for (space, (_, components)) in spaces.iter().zip(fields) {
        let local = space.field_dof_count(*components)?;
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
            .is_none_or(|(test, trial)| !term.pairing.accepts(dimension, test.1, trial.1))
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
    let mut coefficients = vec![zero; terms.len()];
    let mut forcing_offsets = vec![0usize];
    for (_, components) in fields {
        forcing_offsets.push(forcing_offsets.last().unwrap() + components);
    }
    let mut forcing = vec![zero; *forcing_offsets.last().unwrap()];
    let mut isotropic_flux = vec![zero; fields.len()];
    let mut physical = vec![0.0; dimension];
    for point in quadrature.points() {
        geometry.map_point(&point.coordinates, &mut physical)?;
        let tabulations = spaces
            .iter()
            .zip(fields)
            .map(|(space, (_, components))| {
                space.tabulate_field_on(geometry, &point.coordinates, *components)
            })
            .collect::<Result<Vec<_>, _>>()?;
        coefficients.fill(zero);
        forcing.fill(zero);
        isotropic_flux.fill(zero);
        values(
            &physical,
            &mut coefficients,
            &mut forcing,
            &mut isotropic_flux,
        )?;
        if coefficients
            .iter()
            .chain(&forcing)
            .chain(&isotropic_flux)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("region coefficient or forcing is non-finite"));
        }
        let weight = point.weight * geometry.measure_scale();
        for (row, (_, components)) in fields.iter().enumerate() {
            let table = &tabulations[row];
            for test in 0..offsets[row + 1] - offsets[row] {
                let value = table.value(test).expect("validated Field basis");
                let load = value
                    .iter()
                    .enumerate()
                    .fold(zero, |sum, (component, value)| {
                        sum + forcing[forcing_offsets[row] + component] * scalar(*value)
                    });
                let flux = if *components == dimension {
                    isotropic_flux[row]
                        * scalar(table.divergence(test).expect("vector Field basis"))
                } else if isotropic_flux[row] == zero {
                    zero
                } else {
                    return Err(invalid(
                        "isotropic flux requires a physical vector test Field",
                    ));
                };
                rhs[offsets[row] + test] += scalar(weight) * (load - flux);
            }
        }
        for (term, coefficient) in terms.iter().zip(&coefficients) {
            let row = term.row;
            let column = term.column;
            let test_table = &tabulations[row];
            let trial_table = &tabulations[column];
            for test in 0..offsets[row + 1] - offsets[row] {
                for trial in 0..offsets[column + 1] - offsets[column] {
                    let entry = scalar(weight)
                        * *coefficient
                        * scalar(
                            term.pairing.entry(
                                Basis {
                                    value: test_table.value(test).expect("validated test basis"),
                                    gradient: test_table
                                        .gradient(test)
                                        .expect("validated test gradient"),
                                },
                                Basis {
                                    value: trial_table.value(trial).expect("validated trial basis"),
                                    gradient: trial_table
                                        .gradient(trial)
                                        .expect("validated trial gradient"),
                                },
                            ),
                        );
                    matrix[(offsets[row] + test) * count + offsets[column] + trial] +=
                        entry * scalar(term.trial_scale);
                }
            }
        }
    }
    LocalContribution::new(count, count, matrix, rhs)
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
        &[(Space::continuous_lagrange(std::num::NonZeroU16::MIN), 1)],
        &[IntegralTerm {
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
