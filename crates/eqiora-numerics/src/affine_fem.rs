//! Method-neutral algebra shared by affine finite-element local operators.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{AffineGeometryMap, GeometryMap};

use crate::discrete_space::DiscreteSpace;

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn physical_gradient(
    reference: &[f64],
    inverse_jacobian: &[f64],
    dimension: usize,
) -> Vec<f64> {
    (0..dimension)
        .map(|physical| {
            (0..dimension)
                .map(|reference_axis| {
                    inverse_jacobian[reference_axis * dimension + physical]
                        * reference[reference_axis]
                })
                .sum()
        })
        .collect()
}

pub(crate) fn simplex_p1_physical_gradients<const D: usize>(
    geometry: &AffineGeometryMap,
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    let space = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::simplex(D)?,
    )?;
    let centroid_coordinate = 1.0 / (D as f64 + 1.0);
    if geometry.reference_cell() != space.reference_cell() || geometry.physical_dimension() != D {
        return Err(Diagnostic::error(
            codes::INVALID_DISCRETIZATION,
            "simplex P1 gradients require matching intrinsic affine-simplex geometry",
        ));
    }
    let inverse = geometry.inverse_jacobian()?;
    let basis = space.tabulate(&[centroid_coordinate; D])?;
    Ok((0..space.local_dofs().len())
        .map(|index| {
            physical_gradient(
                basis.gradient(index).expect("accepted P1 basis index"),
                &inverse,
                D,
            )
        })
        .collect())
}

pub(crate) fn weighted_gradient(values: &[f64], gradients: &[Vec<f64>]) -> Vec<f64> {
    let dimension = gradients.first().map_or(0, Vec::len);
    let mut result = vec![0.0; dimension];
    for (value, gradient) in values.iter().zip(gradients) {
        for (entry, gradient) in result.iter_mut().zip(gradient) {
            *entry += value * gradient;
        }
    }
    result
}

pub(crate) fn weighted_gradient_tangent(
    values: &[f64],
    value_tangents: &[f64],
    gradients: &[Vec<f64>],
    gradient_tangents: &[Vec<f64>],
) -> Vec<f64> {
    let dimension = gradients.first().map_or(0, Vec::len);
    let mut result = vec![0.0; dimension];
    for (((value, value_tangent), gradient), gradient_tangent) in values
        .iter()
        .zip(value_tangents)
        .zip(gradients)
        .zip(gradient_tangents)
    {
        for ((entry, gradient), gradient_tangent) in
            result.iter_mut().zip(gradient).zip(gradient_tangent)
        {
            *entry += value_tangent * gradient + value * gradient_tangent;
        }
    }
    result
}
