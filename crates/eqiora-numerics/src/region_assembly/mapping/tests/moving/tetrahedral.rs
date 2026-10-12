//! Closed material inventory on an independently prescribed affine tetrahedral path.
use super::*;

#[test]
fn shared_solve_preserves_tetrahedral_material_inventory() {
    let cells = vec![
        vec![0, 1, 2, 4],
        vec![0, 3, 1, 4],
        vec![0, 2, 3, 4],
        vec![1, 3, 2, 4],
    ];
    let reference = SimplicialMesh::new(
        3,
        vec![
            vec![0., 0., 0.],
            vec![1., 0., 0.],
            vec![0., 1., 0.],
            vec![0., 0., 1.],
            vec![0.25, 0.25, 0.25],
        ],
        cells.clone(),
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let form = inventory_form(3);
    let original = FixedTopologyGeometryState::<3>::reference(&reference).unwrap();
    let mut previous = original.clone();
    let mut values = vec![2.; 5];
    // Upper-triangular affine maps: det(F) = a*b*c, independent of shear
    // and translation. The reference tetrahedron has volume 1/6; its four
    // centroid subdivisions each have volume 1/24. A closed material volume
    // initially holds 2/6 kg, so its spatially uniform density is 2/det(F).
    for (scales, shear, shift, determinant) in [
        ([1.5, 2., 0.5], 0.25, [0.25, -0.5, 0.125], 1.5),
        ([2., 1.5, 2.], -0.25, [-0.5, 0.25, 1.], 6.),
        ([1., 1., 1.], 0., [0., 0., 0.], 1.),
    ] {
        let current = FixedTopologyGeometryState::new(
            &reference,
            reference
                .vertices()
                .iter()
                .map(|p| {
                    vec![
                        scales[0] * p[0] + shear * p[1] + shift[0],
                        scales[1] * p[1] + shift[1],
                        scales[2] * p[2] + shift[2],
                    ]
                })
                .collect(),
        )
        .unwrap();
        let action = FixedTopologyGeometryAction::new(&reference, &previous, &current, 1.).unwrap();
        let mesh = action.current_mesh().clone();
        if determinant != 1. {
            let error = execute(&form, &reference, Some(action.clone()), &values).unwrap_err();
            assert!(
                error
                    .message()
                    .contains("current cell topology or coordinates"),
                "{error:?}"
            );
        }
        let wrong_step =
            FixedTopologyGeometryAction::new(&reference, &previous, &current, 0.5).unwrap();
        let error = execute(&form, &mesh, Some(wrong_step), &values).unwrap_err();
        assert!(
            error.message().contains("exact mesh or time step"),
            "{error:?}"
        );
        values = execute(&form, &mesh, Some(action), &values).unwrap();
        for value in &values {
            assert!((value - 2. / determinant).abs() < 1e-11, "{values:?}");
        }
        let inventory: f64 = cells
            .iter()
            .map(|cell| determinant / 96. * cell.iter().map(|&i| values[i]).sum::<f64>())
            .sum();
        assert!((inventory - 1. / 3.).abs() < 1e-11, "{inventory}");
        previous = current;
    }
    let fixed = FixedTopologyGeometryAction::new(&reference, &original, &original, 1.).unwrap();
    let initial = [1., 2., 3., 4., 5.];
    assert_eq!(
        execute(&form, &reference, Some(fixed), &initial).unwrap(),
        execute::<3>(&form, &reference, None, &initial).unwrap(),
    );
}
