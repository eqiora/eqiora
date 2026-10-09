use super::*;

fn assert_partition_and_gradient_sum(space: &DiscreteSpace, point: &[f64]) {
    let tabulation = space.tabulate(point).unwrap();
    assert!((tabulation.values().iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
    for axis in 0..tabulation.reference_dimension() {
        let sum = (0..tabulation.values().len())
            .map(|dof| tabulation.gradient(dof).unwrap()[axis])
            .sum::<f64>();
        assert!(sum.abs() < 1.0e-13);
    }
}

#[test]
fn p0_is_the_same_contract_on_all_cell_families() {
    for cell in [
        ReferenceCell::point(),
        ReferenceCell::simplex(4).unwrap(),
        ReferenceCell::hypercube(5).unwrap(),
    ] {
        let space = DiscreteSpace::new(Space::cell_constant(), cell).unwrap();
        let point = match cell.family() {
            eqiora_meshing::ReferenceCellFamily::Point => Vec::new(),
            eqiora_meshing::ReferenceCellFamily::Simplex => vec![0.0; cell.dimension()],
            eqiora_meshing::ReferenceCellFamily::Hypercube => vec![0.0; cell.dimension()],
        };
        assert_partition_and_gradient_sum(&space, &point);
        assert_eq!(space.local_dofs()[0].entity_dimension(), cell.dimension());
    }
}

#[test]
fn simplex_p1_is_nodal_and_dimension_general() {
    for dimension in 1..=5 {
        let space = DiscreteSpace::new(
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            ReferenceCell::simplex(dimension).unwrap(),
        )
        .unwrap();
        assert_partition_and_gradient_sum(&space, &vec![0.2 / dimension as f64; dimension]);
        for vertex in 0..=dimension {
            let mut point = vec![0.0; dimension];
            if vertex > 0 {
                point[vertex - 1] = 1.0;
            }
            let values = space.tabulate(&point).unwrap().values;
            for (dof, value) in values.into_iter().enumerate() {
                assert_eq!(value, if dof == vertex { 1.0 } else { 0.0 });
            }
        }
    }
}

#[test]
fn simplex_p1_bubble_has_a_p1_trace_and_invariant_cell_coefficient() {
    for dimension in 1..=5 {
        let space = DiscreteSpace::new(
            Space::simplex_p1_bubble(),
            ReferenceCell::simplex(dimension).unwrap(),
        )
        .unwrap();
        let vertex_count = dimension + 1;
        let barycenter = vec![1.0 / vertex_count as f64; dimension];
        let centered = space.tabulate(&barycenter).unwrap();
        assert!((centered.values()[vertex_count] - 1.0).abs() < 2.0e-13);
        assert_eq!(
            space.local_dofs()[vertex_count].entity_dimension(),
            dimension
        );

        for vertex in 0..vertex_count {
            let mut point = vec![0.0; dimension];
            if vertex > 0 {
                point[vertex - 1] = 1.0;
            }
            let values = space.tabulate(&point).unwrap().values;
            assert_eq!(values[vertex_count], 0.0);
            for (dof, value) in values[..vertex_count].iter().enumerate() {
                assert_eq!(*value, if dof == vertex { 1.0 } else { 0.0 });
            }
        }

        let permutation = VertexPermutation::new((0..vertex_count).rev().collect()).unwrap();
        let mut expected = permutation
            .images()
            .iter()
            .map(|&index| (index, 1))
            .collect::<Vec<_>>();
        expected.push((vertex_count, 1));
        assert_eq!(space.oriented_dofs(&permutation).unwrap(), expected);
    }
}

#[test]
fn simplex_p1_bubble_gradient_matches_its_normalized_barycentric_definition() {
    let space = DiscreteSpace::new(
        Space::simplex_p1_bubble(),
        ReferenceCell::simplex(2).unwrap(),
    )
    .unwrap();
    let centered = space.tabulate(&[1.0 / 3.0, 1.0 / 3.0]).unwrap();
    for derivative in centered.gradient(3).unwrap() {
        assert!(derivative.abs() < 2.0e-15);
    }

    let boundary_midpoint = space.tabulate(&[0.5, 0.0]).unwrap();
    assert_eq!(boundary_midpoint.values()[3], 0.0);
    assert_eq!(boundary_midpoint.gradient(3).unwrap(), &[0.0, 6.75]);
}

#[test]
fn hypercube_q1_is_nodal_and_dimension_general() {
    for dimension in 1..=5 {
        let space = DiscreteSpace::new(
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            ReferenceCell::hypercube(dimension).unwrap(),
        )
        .unwrap();
        assert_partition_and_gradient_sum(&space, &vec![0.125; dimension]);
        for vertex in 0..space.local_dofs().len() {
            let point = (0..dimension)
                .map(|axis| if (vertex >> axis) & 1 == 0 { -1.0 } else { 1.0 })
                .collect::<Vec<_>>();
            let values = space.tabulate(&point).unwrap().values;
            for (dof, value) in values.into_iter().enumerate() {
                assert_eq!(value, if dof == vertex { 1.0 } else { 0.0 });
            }
        }
    }
}

#[test]
fn orientation_is_explicit_and_validated() {
    let space = DiscreteSpace::new(
        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        ReferenceCell::simplex(2).unwrap(),
    )
    .unwrap();
    assert_eq!(
        space
            .oriented_dofs(&VertexPermutation::new(vec![2, 0, 1]).unwrap())
            .unwrap(),
        vec![(2, 1), (0, 1), (1, 1)]
    );
    assert_eq!(
        VertexPermutation::new(vec![0, 0, 2]).unwrap_err().code(),
        codes::INVALID_MESH
    );
    assert_eq!(
        space
            .oriented_dofs(&VertexPermutation::identity(2))
            .unwrap_err()
            .code(),
        codes::INVALID_DISCRETIZATION
    );
}

#[test]
fn rejects_unsupported_or_excessive_spaces_and_points() {
    assert_eq!(
        DiscreteSpace::new(
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            ReferenceCell::simplex(usize::MAX).unwrap()
        )
        .unwrap_err()
        .code(),
        codes::INVALID_DISCRETIZATION
    );
    assert_eq!(
        DiscreteSpace::new(
            Space::simplex_p1_bubble(),
            ReferenceCell::simplex(usize::MAX).unwrap()
        )
        .unwrap_err()
        .code(),
        codes::INVALID_DISCRETIZATION
    );
    assert_eq!(
        DiscreteSpace::new(
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            ReferenceCell::hypercube(64).unwrap()
        )
        .unwrap_err()
        .code(),
        codes::INVALID_DISCRETIZATION
    );
    let space = DiscreteSpace::new(
        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        ReferenceCell::hypercube(2).unwrap(),
    )
    .unwrap();
    assert_eq!(
        space.tabulate(&[0.0, f64::NAN]).unwrap_err().code(),
        codes::INVALID_DISCRETIZATION
    );
}
