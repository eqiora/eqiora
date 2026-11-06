use super::*;

pub(super) fn tabulate(
    space: &DiscreteSpace,
    reference: &[f64],
) -> Result<BasisTabulation, Diagnostic> {
    if matches!(space.space.family(), SpaceFamily::ContinuousLagrange { order } if order.get() == 2)
    {
        return quadratic(space, reference);
    }
    match space.space.family() {
        SpaceFamily::CellConstant => Ok(BasisTabulation {
            value_dimension: 1,
            reference_dimension: space.cell.dimension(),
            values: vec![1.0],
            reference_gradients: vec![0.0; space.cell.dimension()],
        }),
        SpaceFamily::SimplexP1Bubble => {
            let dimension = space.cell.dimension();
            let vertex_count = dimension + 1;
            let mut values = Vec::with_capacity(vertex_count + 1);
            values.push(1.0 - reference.iter().sum::<f64>());
            values.extend_from_slice(reference);

            let mut gradients = vec![0.0; (vertex_count + 1) * dimension];
            gradients[..dimension].fill(-1.0);
            for axis in 0..dimension {
                gradients[(axis + 1) * dimension + axis] = 1.0;
            }

            let mut prefix = vec![1.0; dimension + 1];
            for axis in 0..dimension {
                prefix[axis + 1] = prefix[axis] * reference[axis];
            }
            let mut suffix = vec![1.0; dimension + 1];
            for axis in (0..dimension).rev() {
                suffix[axis] = suffix[axis + 1] * reference[axis];
            }
            let lambda_zero = values[0];
            values.push(space.bubble_normalization * lambda_zero * prefix[dimension]);
            for axis in 0..dimension {
                let product_without_axis = prefix[axis] * suffix[axis + 1];
                gradients[vertex_count * dimension + axis] = space.bubble_normalization
                    * (-prefix[dimension] + lambda_zero * product_without_axis);
            }

            Ok(BasisTabulation {
                value_dimension: 1,
                reference_dimension: dimension,
                values,
                reference_gradients: gradients,
            })
        }
        SpaceFamily::ContinuousLagrange { .. }
            if space.cell.family() == ReferenceCellFamily::Simplex =>
        {
            let dimension = space.cell.dimension();
            let mut values = Vec::with_capacity(dimension + 1);
            values.push(1.0 - reference.iter().sum::<f64>());
            values.extend_from_slice(reference);

            let mut gradients = vec![0.0; (dimension + 1) * dimension];
            gradients[..dimension].fill(-1.0);
            for axis in 0..dimension {
                gradients[(axis + 1) * dimension + axis] = 1.0;
            }
            Ok(BasisTabulation {
                value_dimension: 1,
                reference_dimension: dimension,
                values,
                reference_gradients: gradients,
            })
        }
        SpaceFamily::ContinuousLagrange { .. } => {
            let dimension = space.cell.dimension();
            let dof_count = space.local_dofs.len();
            let mut values = vec![0.0; dof_count];
            let mut gradients = vec![0.0; dof_count * dimension];

            for vertex in 0..dof_count {
                let factors = (0..dimension)
                    .map(|axis| {
                        let sign = if (vertex >> axis) & 1 == 0 { -1.0 } else { 1.0 };
                        (sign, 0.5 * (1.0 + sign * reference[axis]))
                    })
                    .collect::<Vec<_>>();
                values[vertex] = factors.iter().map(|(_, factor)| factor).product();
                for axis in 0..dimension {
                    gradients[vertex * dimension + axis] = 0.5
                        * factors[axis].0
                        * factors
                            .iter()
                            .enumerate()
                            .filter(|(other_axis, _)| *other_axis != axis)
                            .map(|(_, (_, factor))| factor)
                            .product::<f64>();
                }
            }
            Ok(BasisTabulation {
                value_dimension: 1,
                reference_dimension: dimension,
                values,
                reference_gradients: gradients,
            })
        }
        SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace => {
            unreachable!("vector basis dispatched separately")
        }
    }
}

fn quadratic(space: &DiscreteSpace, reference: &[f64]) -> Result<BasisTabulation, Diagnostic> {
    let dimension = space.cell.dimension();
    let count = space.local_dofs.len();
    let mut values = Vec::with_capacity(count);
    let mut gradients = Vec::with_capacity(count * dimension);
    if space.cell.family() == ReferenceCellFamily::Simplex {
        let mut lambda = vec![1.0 - reference.iter().sum::<f64>()];
        lambda.extend_from_slice(reference);
        let derivative = |vertex: usize, axis: usize| {
            if vertex == 0 {
                -1.0
            } else if vertex == axis + 1 {
                1.0
            } else {
                0.0
            }
        };
        let topology = ReferenceTopology::new(space.cell)?;
        for dof in &space.local_dofs {
            let vertices = topology
                .entity(dof.entity_dimension, dof.entity_ordinal)
                .expect("quadratic support")
                .vertex_ordinals();
            if vertices.len() == 1 {
                let vertex = vertices[0];
                values.push(lambda[vertex] * (2.0 * lambda[vertex] - 1.0));
                gradients.extend(
                    (0..dimension)
                        .map(|axis| (4.0 * lambda[vertex] - 1.0) * derivative(vertex, axis)),
                );
            } else {
                let (a, b) = (vertices[0], vertices[1]);
                values.push(4.0 * lambda[a] * lambda[b]);
                gradients.extend((0..dimension).map(|axis| {
                    4.0 * (lambda[a] * derivative(b, axis) + lambda[b] * derivative(a, axis))
                }));
            }
        }
    } else {
        for local in 0..count {
            let point = space.interpolation_point(local)?;
            let factors = reference
                .iter()
                .zip(point)
                .map(|(&x, node)| {
                    if node == -1.0 {
                        (x * (x - 1.0) / 2.0, x - 0.5)
                    } else if node == 1.0 {
                        (x * (x + 1.0) / 2.0, x + 0.5)
                    } else {
                        (1.0 - x * x, -2.0 * x)
                    }
                })
                .collect::<Vec<_>>();
            values.push(factors.iter().map(|factor| factor.0).product());
            gradients.extend((0..dimension).map(|axis| {
                factors[axis].1
                    * factors
                        .iter()
                        .enumerate()
                        .filter(|(other, _)| *other != axis)
                        .map(|(_, factor)| factor.0)
                        .product::<f64>()
            }));
        }
    }
    Ok(BasisTabulation {
        reference_dimension: dimension,
        value_dimension: 1,
        values,
        reference_gradients: gradients,
    })
}
