//! One anonymous mathematical quadrature kernel for typed and callback-owned forms.

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCell};
use eqiora_realization::Space;
use num_complex::ComplexFloat;
use std::ops::{AddAssign, SubAssign};

use crate::affine_fem::physical_gradient;
use crate::form_compiler::bilinear::{Basis, Pairing};

use super::{binding::basis, invalid};

pub(super) struct IntegralTerm {
    pub row: usize,
    pub column: usize,
    pub pairing: Pairing,
    pub trial_scale: f64,
}

pub(super) fn integrate<
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
        .map(|(space, _)| basis(*space, reference))
        .collect::<Result<Vec<_>, _>>()?;
    let mut offsets = vec![0usize];
    for (space, (_, components)) in spaces.iter().zip(fields) {
        let local = space
            .local_dofs()
            .len()
            .checked_mul(*components)
            .filter(|count| *count > 0)
            .ok_or_else(|| invalid("region local DOF count overflow"))?;
        offsets.push(
            offsets
                .last()
                .unwrap()
                .checked_add(local)
                .ok_or_else(|| invalid("region local DOF count overflow"))?,
        );
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
    let inverse = geometry.inverse_jacobian()?;
    let mut physical = vec![0.0; dimension];
    for point in quadrature.points() {
        geometry.map_point(&point.coordinates, &mut physical)?;
        let tabulations = spaces
            .iter()
            .map(|space| space.tabulate(&point.coordinates))
            .collect::<Result<Vec<_>, _>>()?;
        let gradients = tabulations
            .iter()
            .map(|tabulation| {
                (0..tabulation.values().len())
                    .map(|dof| {
                        physical_gradient(
                            tabulation.gradient(dof).expect("supported basis gradient"),
                            &inverse,
                            dimension,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
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
            for (test, value) in tabulations[row].values().iter().enumerate() {
                for component in 0..*components {
                    rhs[offsets[row] + test * components + component] +=
                        scalar(weight) * forcing[forcing_offsets[row] + component] * scalar(*value);
                    rhs[offsets[row] + test * components + component] -= scalar(weight)
                        * isotropic_flux[row]
                        * scalar(gradients[row][test][component]);
                }
            }
        }
        for (term, coefficient) in terms.iter().zip(&coefficients) {
            let row = term.row;
            let column = term.column;
            let test_components = fields[row].1;
            let trial_components = fields[column].1;
            for test in 0..tabulations[row].values().len() {
                for test_component in 0..test_components {
                    let global_test = offsets[row] + test * test_components + test_component;
                    for trial in 0..tabulations[column].values().len() {
                        for trial_component in 0..trial_components {
                            let local_trial = trial * trial_components + trial_component;
                            let entry = scalar(weight)
                                * *coefficient
                                * scalar(term.pairing.entry(
                                    Basis {
                                        value: tabulations[row].values()[test],
                                        gradient: &gradients[row][test],
                                        component: test_component,
                                    },
                                    Basis {
                                        value: tabulations[column].values()[trial],
                                        gradient: &gradients[column][trial],
                                        component: trial_component,
                                    },
                                ));
                            matrix[global_test * count + offsets[column] + local_trial] +=
                                entry * scalar(term.trial_scale);
                        }
                    }
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
mod tests {
    use super::*;
    use num_complex::Complex64 as C;

    #[test]
    fn complex_diffusion_mass_and_load_share_real_basis_quadrature() {
        let reference = ReferenceCell::hypercube(1).unwrap();
        let geometry = AffineGeometryMap::new(reference, 1, vec![3.0], vec![3.0]).unwrap();
        let quadrature = QuadratureRule::gauss_legendre(2).unwrap();
        let local = integrate(
            reference,
            &[(Space::continuous_lagrange(std::num::NonZeroU16::MIN), 1)],
            &[
                IntegralTerm {
                    row: 0,
                    column: 0,
                    pairing: Pairing::Gradient,
                    trial_scale: 1.0,
                },
                IntegralTerm {
                    row: 0,
                    column: 0,
                    pairing: Pairing::Value,
                    trial_scale: 1.0,
                },
            ],
            &geometry,
            &quadrature,
            |_, coefficients, forcing, _| {
                coefficients[0] = C::new(6.0, 6.0);
                coefficients[1] = C::new(3.0, -1.0);
                forcing[0] = C::new(1.0, 3.0);
                Ok(())
            },
        )
        .unwrap();
        // On [0,6], K = a/6 [[1,-1],[-1,1]],
        // M = q [[2,1],[1,2]], and each load is 3f.
        let expected = [
            C::new(7.0, -1.0),
            C::new(2.0, -2.0),
            C::new(2.0, -2.0),
            C::new(7.0, -1.0),
        ];
        for (actual, expected) in local.matrix().iter().zip(expected) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        for actual in local.rhs() {
            assert!((*actual - C::new(3.0, 9.0)).norm() < 1e-12);
        }
        // Symmetric real bases do not imply a Hermitian coefficient operator.
        assert_ne!(local.matrix()[1], local.matrix()[2].conj());
        assert!(
            integrate_scalar(1, &geometry, &quadrature, |_| Ok((
                C::new(1.0, f64::NAN),
                C::new(0.0, 0.0)
            )))
            .is_err()
        );
    }
}
