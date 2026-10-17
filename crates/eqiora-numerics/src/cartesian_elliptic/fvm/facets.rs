//! Shared cell-face packets for primal and differentiated conservative systems.
use super::flux::FluxData;
use super::*;

pub(super) fn prepare<S>(
    mesh: &CartesianMesh,
    centers: &[Vec<f64>],
    components: usize,
    flux: FvmFlux<'_, S>,
    quadrature: &QuadratureRule,
    properties: LinearOperatorProperties,
) -> Result<Vec<CartesianFacetPacket<S>>, Diagnostic>
where
    S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64> + Send + Sync,
{
    let dimension = mesh.topological_dimension();
    let facet_dimension = dimension - 1;
    let facet_count = mesh
        .entity_count(facet_dimension)
        .expect("mesh owns facets");
    let count = facet_count
        .checked_mul(components)
        .and_then(|n| n.checked_mul(components))
        .ok_or_else(|| invalid("Cartesian FVM component facet count overflow"))?;
    let mut packets = Vec::with_capacity(count);
    for index in 0..facet_count {
        let facet = MeshEntity::new(facet_dimension, index);
        let geometry = mesh
            .geometry_map(facet)
            .expect("mesh facet has affine geometry");
        let free_axes = mesh
            .entity_free_axes(facet)
            .expect("mesh facet exposes tangent axes");
        let normal_axis = (0..dimension)
            .find(|axis| free_axes.binary_search(axis).is_err())
            .ok_or_else(|| invalid("Cartesian facet has no normal axis"))?;
        let cells = mesh
            .incidence(facet, dimension)
            .ok_or_else(|| invalid("Cartesian facet adjacency is unavailable"))?;
        let mut point = vec![0.; dimension];
        geometry.map_point(&vec![0.; facet_dimension], &mut point)?;
        match cells.as_slice() {
            [left, right] => {
                let left = left.entity.index();
                let right = right.entity.index();
                let distance = (centers[right][normal_axis] - centers[left][normal_axis]).abs();
                require_positive_distance(distance)?;
                for test in 0..components {
                    for trial in 0..components {
                        let coefficient = (flux.selected().coefficient)(test, trial, &point)?;
                        let transmissibility = facet_transmissibility(
                            &geometry,
                            distance,
                            coefficient,
                            quadrature,
                            if test == trial {
                                properties
                            } else {
                                LinearOperatorProperties::General
                            },
                        )?;
                        packets.push(CartesianFacetPacket {
                            test,
                            trial,
                            transmissibility,
                            kind: CartesianFacetKind::Interior { left, right },
                        });
                    }
                }
            }
            [cell] => {
                let cell = cell.entity.index();
                let distance = (point[normal_axis] - centers[cell][normal_axis]).abs();
                require_positive_distance(distance)?;
                let (axis, side) = cartesian_boundary_facet_side(mesh, facet)?
                    .expect("one-cell facet is a boundary");
                let coefficients =
                    |data: FluxData<'_, S>, properties| -> Result<Vec<Vec<S>>, Diagnostic> {
                        (0..components)
                            .map(|test| {
                                (0..components)
                                    .map(|trial| {
                                        facet_transmissibility(
                                            &geometry,
                                            distance,
                                            (data.coefficient)(test, trial, &point)?,
                                            quadrature,
                                            if test == trial {
                                                properties
                                            } else {
                                                LinearOperatorProperties::General
                                            },
                                        )
                                    })
                                    .collect()
                            })
                            .collect()
                    };
                let values =
                    |data: FluxData<'_, S>| -> Result<Vec<CartesianBoundaryValue<S>>, Diagnostic> {
                        (0..components).map(|component| match (data.boundary)(component, axis, side, &point)? {
                        CartesianBoundaryValue::Essential(value) => Ok(CartesianBoundaryValue::Essential(value)),
                        CartesianBoundaryValue::Natural(_) => Ok(CartesianBoundaryValue::Natural(integrate_boundary_flux(
                            &geometry, quadrature, &|point| match (data.boundary)(component, axis, side, point)? {
                                CartesianBoundaryValue::Natural(value) => Ok(value),
                                _ => Err(invalid("Cartesian boundary changes kind within a facet")),
                            },
                        )?)),
                    }).collect()
                    };
                let matrix = coefficients(flux.value, properties)?;
                let data = values(flux.value)?;
                let direction = flux
                    .direction
                    .map(|direction| -> Result<_, Diagnostic> {
                        Ok((
                            coefficients(direction, LinearOperatorProperties::General)?,
                            values(direction)?,
                        ))
                    })
                    .transpose()?;
                let (effective, rhs) = boundary::response(
                    &matrix,
                    &data,
                    direction.as_ref().map(|(a, b)| (a, b.as_slice())),
                )?;
                let selected = direction.as_ref().map_or(&matrix, |(matrix, _)| matrix);
                for test in 0..components {
                    for trial in 0..components {
                        packets.push(CartesianFacetPacket {
                            test,
                            trial,
                            transmissibility: selected[test][trial],
                            kind: CartesianFacetKind::Boundary {
                                axis,
                                side,
                                cell,
                                natural: matches!(data[test], CartesianBoundaryValue::Natural(_)),
                                coefficient: effective[test][trial],
                                load: if test == trial { rhs[test] } else { S::zero() },
                            },
                        });
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "Cartesian facet requires exactly one or two adjacent cells",
                ));
            }
        }
    }
    Ok(packets)
}
