use super::{BasisTabulation, Diagnostic, DiscreteSpace, ReferenceTopology, SpaceFamily};

pub(super) fn orientation_sign(images: &[usize]) -> i8 {
    let inversions = images
        .iter()
        .enumerate()
        .map(|(i, a)| images[i + 1..].iter().filter(|b| a > *b).count())
        .sum::<usize>();
    if inversions % 2 == 0 { 1 } else { -1 }
}

pub(super) fn tabulate(
    space: &DiscreteSpace,
    point: &[f64],
) -> Result<BasisTabulation, Diagnostic> {
    let topology = ReferenceTopology::new(space.cell)?;
    let lambda = [
        1.0 - point.iter().sum::<f64>(),
        point[0],
        point[1],
        point[2],
    ];
    let gradients = [[-1.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut values = Vec::with_capacity(space.local_dofs.len() * 3);
    let mut reference_gradients = Vec::with_capacity(space.local_dofs.len() * 9);
    for dof in &space.local_dofs {
        let vertices = topology
            .entity(dof.entity_dimension, dof.entity_ordinal)
            .expect("validated tetrahedral entity")
            .vertex_ordinals();
        if space.space.family() == SpaceFamily::TetrahedralEdge {
            let (i, j) = (vertices[0], vertices[1]);
            for component in 0..3 {
                values.push(
                    lambda[i] * gradients[j][component] - lambda[j] * gradients[i][component],
                );
                for axis in 0..3 {
                    reference_gradients.push(
                        gradients[i][axis] * gradients[j][component]
                            - gradients[j][axis] * gradients[i][component],
                    );
                }
            }
        } else {
            let opposite = (0..4)
                .find(|vertex| !vertices.contains(vertex))
                .expect("tetrahedral face has one opposite vertex");
            // The simplex boundary coefficient (-1)^opposite compares the sorted
            // face orientation with the outward normal. 1/(3 V_ref) = 2.
            let factor = if opposite % 2 == 0 { 2.0 } else { -2.0 };
            for (component, coordinate) in point.iter().enumerate() {
                let vertex_coordinate = f64::from(opposite == component + 1);
                values.push(factor * (coordinate - vertex_coordinate));
                for axis in 0..3 {
                    reference_gradients.push(if component == axis { factor } else { 0.0 });
                }
            }
        }
    }
    Ok(BasisTabulation {
        reference_dimension: 3,
        value_dimension: 3,
        values,
        reference_gradients,
    })
}
