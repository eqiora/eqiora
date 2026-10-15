//! Method-neutral algebra shared by affine finite-element local operators.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, GeometryMap, ReferenceCellFamily, ReferenceTopology,
};

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

/// Physical unit normal from the exact parent-cell facet incidence.
pub(crate) fn parent_outward_normal(
    cell: &AffineGeometryMap,
    incidence: EntityIncidence,
) -> Result<Vec<f64>, Diagnostic> {
    let invalid = |message: &str| Diagnostic::error(codes::INVALID_DISCRETIZATION, message);
    let dimension = cell.physical_dimension();
    if dimension == 0 || incidence.entity.dimension() != dimension {
        return Err(invalid(
            "natural flux requires an exact positive-dimensional parent incidence",
        ));
    }
    let topology = ReferenceTopology::new(cell.reference_cell())?;
    let facet = topology
        .entity(dimension - 1, incidence.local_ordinal)
        .ok_or_else(|| invalid("natural flux has a foreign parent facet ordinal"))?;
    let vertices = facet.vertex_ordinals();
    let mut reference_normal = vec![0.0; dimension];
    match cell.reference_cell().family() {
        ReferenceCellFamily::Simplex => {
            let missing = (0..=dimension)
                .find(|vertex| !vertices.contains(vertex))
                .ok_or_else(|| invalid("simplex facet has no opposite parent vertex"))?;
            if missing == 0 {
                reference_normal.fill(1.0);
            } else {
                reference_normal[missing - 1] = -1.0;
            }
        }
        ReferenceCellFamily::Hypercube => {
            let axis = (0..dimension)
                .find(|axis| {
                    vertices
                        .iter()
                        .all(|vertex| (vertex >> axis) & 1 == (vertices[0] >> axis) & 1)
                })
                .ok_or_else(|| invalid("hypercube facet has no fixed parent axis"))?;
            reference_normal[axis] = if (vertices[0] >> axis) & 1 == 0 {
                -1.0
            } else {
                1.0
            };
        }
        ReferenceCellFamily::Point => {
            return Err(invalid(
                "natural flux requires a positive-dimensional parent",
            ));
        }
    }
    // Covectors transform by J^{-T}; the exact reference facet owns the sign.
    let inverse = cell.inverse_jacobian()?;
    let mut normal = (0..dimension)
        .map(|physical| {
            (0..dimension)
                .map(|reference| {
                    inverse[reference * dimension + physical] * reference_normal[reference]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let norm = normal
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid("natural flux facet has no finite nonzero normal"));
    }
    for component in &mut normal {
        *component /= norm;
    }
    Ok(normal)
}

/// Embed intrinsic facet basis weights in the exact parent reference cell.
pub(crate) fn facet_reference_point(
    parent: eqiora_meshing::ReferenceCell,
    parent_vertices: &[usize],
    weights: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let invalid = |message: &str| Diagnostic::error(codes::INVALID_DISCRETIZATION, message);
    let dimension = parent.dimension();
    let topology = ReferenceTopology::new(parent)?;
    if parent_vertices.len() != weights.len()
        || parent_vertices
            .iter()
            .any(|vertex| *vertex >= topology.entity_count(0).unwrap_or(0))
        || weights.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "facet basis differs from its exact parent vertex embedding",
        ));
    }
    let mut reference = vec![0.0; dimension];
    for (&vertex, &weight) in parent_vertices.iter().zip(weights) {
        for (axis, coordinate) in reference.iter_mut().enumerate() {
            let value = match parent.family() {
                ReferenceCellFamily::Simplex => f64::from(vertex == axis + 1),
                ReferenceCellFamily::Hypercube => 2.0 * ((vertex >> axis) & 1) as f64 - 1.0,
                _ => {
                    return Err(invalid(
                        "facet embedding requires a positive-dimensional parent",
                    ));
                }
            };
            *coordinate += weight * value;
        }
    }
    Ok(reference)
}
