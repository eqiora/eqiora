use super::*;

// At most 27 terms and three derivative factors: 512 eps bounds the direct
// polynomial evaluations and sums at reference coordinates of magnitude <= 1.
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 512.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn element(cell: ReferenceCell) -> DiscreteSpace {
    DiscreteSpace::new(
        Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap()),
        cell,
    )
    .unwrap()
}

#[test]
fn quadratic_nodes_partition_and_all_space_monomials_and_gradients() {
    for dimension in [2, 3] {
        for cell in [
            ReferenceCell::simplex(dimension).unwrap(),
            ReferenceCell::hypercube(dimension).unwrap(),
        ] {
            let space = element(cell);
            let expected_count = if cell.family() == ReferenceCellFamily::Simplex {
                (dimension + 1) * (dimension + 2) / 2
            } else {
                3usize.pow(dimension as u32)
            };
            assert_eq!(space.local_dofs.len(), expected_count);
            let points = (0..expected_count)
                .map(|dof| space.interpolation_point(dof).unwrap())
                .collect::<Vec<_>>();
            for (node, point) in points.iter().enumerate() {
                for (basis, value) in space.tabulate(point).unwrap().values().iter().enumerate() {
                    close(*value, f64::from(node == basis));
                }
            }
            let point = vec![0.13; dimension];
            let table = space.tabulate(&point).unwrap();
            close(table.values().iter().sum(), 1.0);
            for code in 0..3usize.pow(dimension as u32) {
                let exponents = (0..dimension)
                    .map(|axis| code / 3usize.pow(axis as u32) % 3)
                    .collect::<Vec<_>>();
                if cell.family() == ReferenceCellFamily::Simplex
                    && exponents.iter().sum::<usize>() > 2
                {
                    continue;
                }
                let monomial = |x: &[f64]| {
                    x.iter()
                        .zip(&exponents)
                        .map(|(x, &p)| x.powi(p as i32))
                        .product::<f64>()
                };
                let coefficients = points.iter().map(|x| monomial(x)).collect::<Vec<_>>();
                close(
                    coefficients
                        .iter()
                        .zip(table.values())
                        .map(|(c, n)| c * n)
                        .sum(),
                    monomial(&point),
                );
                for axis in 0..dimension {
                    let derivative = if exponents[axis] == 0 {
                        0.0
                    } else {
                        exponents[axis] as f64
                            * point
                                .iter()
                                .enumerate()
                                .map(|(j, x)| {
                                    x.powi((exponents[j] - usize::from(j == axis)) as i32)
                                })
                                .product::<f64>()
                    };
                    close(
                        coefficients
                            .iter()
                            .enumerate()
                            .map(|(i, c)| c * table.gradient(i).unwrap()[axis])
                            .sum(),
                        derivative,
                    );
                }
            }
        }
    }
}

#[test]
fn quadratic_trace_is_parent_restriction_with_complete_facet_closure() {
    for dimension in [2, 3] {
        for cell in [
            ReferenceCell::simplex(dimension).unwrap(),
            ReferenceCell::hypercube(dimension).unwrap(),
        ] {
            let space = element(cell);
            let topology = ReferenceTopology::new(cell).unwrap();
            for facet in 0..topology.entity_count(dimension - 1).unwrap() {
                let vertices = topology
                    .entity(dimension - 1, facet)
                    .unwrap()
                    .vertex_ordinals();
                let reference = crate::affine_fem::facet_reference_point(
                    cell,
                    vertices,
                    &vec![1.0 / vertices.len() as f64; vertices.len()],
                )
                .unwrap();
                let trace = space.trace_dofs(facet).unwrap();
                assert_eq!(
                    trace.len(),
                    if dimension == 2 {
                        3
                    } else if cell.family() == ReferenceCellFamily::Simplex {
                        6
                    } else {
                        9
                    }
                );
                let table = space.tabulate(&reference).unwrap();
                for (i, value) in table.values().iter().enumerate() {
                    if !trace.contains(&i) {
                        close(*value, 0.0);
                    }
                }
                close(trace.iter().map(|&i| table.values()[i]).sum(), 1.0);
                assert!(
                    trace
                        .iter()
                        .all(|&i| space.local_dofs[i].entity_dimension < dimension)
                );
            }
        }
    }
}

#[test]
fn quadratic_permutations_keep_entity_support_and_reject_non_cube_maps() {
    for dimension in [2, 3] {
        for cell in [
            ReferenceCell::simplex(dimension).unwrap(),
            ReferenceCell::hypercube(dimension).unwrap(),
        ] {
            let space = element(cell);
            let arity = reference_vertex_count(cell).unwrap();
            let permutation = VertexPermutation::new((0..arity).rev().collect()).unwrap();
            let order = space.oriented_dofs(&permutation).unwrap();
            assert_eq!(order.len(), space.local_dofs.len());
            for (i, (other, sign)) in order.iter().enumerate() {
                assert_eq!(*sign, 1);
                assert_eq!(
                    space.local_dofs[i].entity_dimension,
                    space.local_dofs[*other].entity_dimension
                );
            }
            assert_eq!(
                space
                    .oriented_dofs(&permutation.inverse())
                    .unwrap()
                    .iter()
                    .map(|(i, _)| order[*i].0)
                    .collect::<Vec<_>>(),
                (0..order.len()).collect::<Vec<_>>()
            );
        }
        let space = element(ReferenceCell::hypercube(dimension).unwrap());
        let mut bad = (0..reference_vertex_count(space.cell).unwrap()).collect::<Vec<_>>();
        bad.swap(0, 1);
        assert!(
            space
                .oriented_dofs(&VertexPermutation::new(bad).unwrap())
                .is_err()
        );
    }
}
