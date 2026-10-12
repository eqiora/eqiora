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
    fields: &[(Space, usize)],
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
    let mut matrix_tangent = direction.map(|_| vec![zero; entries]);
    let mut rhs_tangent = direction.map(|_| vec![zero; count]);
    let mut coefficients = vec![zero; terms.len()];
    let mut forcing_offsets = vec![0usize];
    for (_, components) in fields {
        forcing_offsets.push(forcing_offsets.last().unwrap() + components);
    }
    let mut forcing = vec![zero; *forcing_offsets.last().unwrap()];
    let mut isotropic_flux = vec![zero; fields.len()];
    let mut physical = vec![0.0; dimension];
    let mut physical_tangent = vec![0.0; dimension];
    let mut coefficient_tangents =
        direction.map_or_else(Vec::new, |_| vec![zero; coefficients.len()]);
    let mut forcing_tangents = direction.map_or_else(Vec::new, |_| vec![zero; forcing.len()]);
    let mut flux_tangents = direction.map_or_else(Vec::new, |_| vec![zero; isotropic_flux.len()]);
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
            .map(|(space, (_, components))| match direction {
                Some(direction) => space
                    .tabulate_field_linearized(direction, &point.coordinates, *components)
                    .map(|(primal, tangent)| (primal, Some(tangent))),
                None => space
                    .tabulate_field_on(geometry, &point.coordinates, *components)
                    .map(|primal| (primal, None)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        coefficients.fill(zero);
        forcing.fill(zero);
        isotropic_flux.fill(zero);
        coefficient_tangents.fill(zero);
        forcing_tangents.fill(zero);
        flux_tangents.fill(zero);
        values(
            &point.coordinates,
            &physical,
            &physical_tangent,
            &mut coefficients,
            &mut forcing,
            &mut isotropic_flux,
            &mut coefficient_tangents,
            &mut forcing_tangents,
            &mut flux_tangents,
        )?;
        if coefficients
            .iter()
            .chain(&forcing)
            .chain(&isotropic_flux)
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
        for (row, (_, components)) in fields.iter().enumerate() {
            let table = &tabulations[row].0;
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
                if let Some(tangent_rhs) = &mut rhs_tangent {
                    let tangent_table =
                        tabulations[row].1.as_ref().expect("bound geometry tangent");
                    let tangent_value = tangent_table.value(test).expect("validated tangent basis");
                    let load_tangent = value.iter().zip(tangent_value).enumerate().fold(
                        zero,
                        |sum, (component, (value, tangent))| {
                            sum + forcing_tangents[forcing_offsets[row] + component]
                                * scalar(*value)
                                + forcing[forcing_offsets[row] + component] * scalar(*tangent)
                        },
                    );
                    let flux_tangent = if *components == dimension {
                        flux_tangents[row]
                            * scalar(table.divergence(test).expect("vector Field basis"))
                            + isotropic_flux[row]
                                * scalar(
                                    tangent_table
                                        .divergence(test)
                                        .expect("vector tangent basis"),
                                )
                    } else if flux_tangents[row] == zero {
                        zero
                    } else {
                        return Err(invalid(
                            "isotropic flux tangent requires a physical vector test Field",
                        ));
                    };
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
            for test in 0..offsets[row + 1] - offsets[row] {
                for trial in 0..offsets[column + 1] - offsets[column] {
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
                    let slot = (offsets[row] + test) * count + offsets[column] + trial;
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
