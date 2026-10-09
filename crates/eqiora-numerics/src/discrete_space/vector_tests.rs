use eqiora_meshing::{AffineGeometryMap, GeometryMap};

use super::*;

// These are geometric moment oracles, independent of basis formulas. Midpoints
// and centroids integrate affine fields exactly. The roundoff allowance is 256
// eps times scale, covering the small dense maps, products and six-term sums.
const VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 4.0],
];
const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 256.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}
fn subtract(a: &[f64], b: &[f64]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn element(space: Space) -> DiscreteSpace {
    DiscreteSpace::new(space, ReferenceCell::simplex(3).unwrap()).unwrap()
}
fn map(vertices: &[[f64; 3]; 4]) -> AffineGeometryMap {
    AffineGeometryMap::from_simplex_vertices(vertices.iter().map(|p| p.to_vec()).collect()).unwrap()
}
fn value(table: &PhysicalBasisTabulation, coefficients: &[f64]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        coefficients
            .iter()
            .enumerate()
            .map(|(dof, c)| c * table.value(dof).unwrap()[axis])
            .sum()
    })
}

#[test]
fn physical_edge_and_face_moments_are_integrals_on_nonunit_sheared_cells() {
    for vertices in [
        VERTICES,
        [
            [1.0, -2.0, 3.0],
            [3.0, -1.0, 3.0],
            [2.0, 1.0, 4.0],
            [2.0, -1.0, 7.0],
        ],
    ] {
        let map = map(&vertices);
        let edge = element(Space::tetrahedral_edge());
        let face = element(Space::tetrahedral_face());
        for (row, [a, b]) in EDGES.into_iter().enumerate() {
            let mut reference = [0.0; 3];
            for v in [a, b] {
                if v > 0 {
                    reference[v - 1] += 0.5;
                }
            }
            let table = edge.tabulate_on(&map, &reference).unwrap();
            let tangent = subtract(&vertices[b], &vertices[a]);
            for column in 0..6 {
                close(
                    dot(table.value(column).unwrap(), &tangent),
                    f64::from(row == column),
                );
            }
        }
        for (row, [a, b, c]) in FACES.into_iter().enumerate() {
            let mut reference = [0.0; 3];
            for v in [a, b, c] {
                if v > 0 {
                    reference[v - 1] += 1.0 / 3.0;
                }
            }
            let normal = cross(
                subtract(&vertices[b], &vertices[a]),
                subtract(&vertices[c], &vertices[a]),
            )
            .map(|v| v * 0.5);
            let table = face.tabulate_on(&map, &reference).unwrap();
            for column in 0..4 {
                close(
                    dot(table.value(column).unwrap(), &normal),
                    f64::from(row == column),
                );
            }
        }
    }
}

#[test]
fn reproduces_rotation_and_radial_flux_with_their_physical_derivatives() {
    let map = map(&VERTICES);
    let edge = element(Space::tetrahedral_edge());
    let face = element(Space::tetrahedral_face());
    // Integral of (-y,x,0) on edge 12 is 6; all others vanish.
    // Integral of (x,y,z) through face 123 is 3 V = 12.
    for point in [[0.0; 3], [0.125, 0.25, 0.5], [0.25; 3]] {
        let edge_table = edge.tabulate_on(&map, &point).unwrap();
        let face_table = face.tabulate_on(&map, &point).unwrap();
        let actual = value(&edge_table, &[0.0, 0.0, 0.0, 6.0, 0.0, 0.0]);
        for (a, e) in actual
            .into_iter()
            .zip([-3.0 * point[1], 2.0 * point[0], 0.0])
        {
            close(a, e);
        }
        for (a, e) in edge_table
            .curl(3)
            .unwrap()
            .map(|v| v * 6.0)
            .into_iter()
            .zip([0.0, 0.0, 2.0])
        {
            close(a, e);
        }
        for (a, e) in value(&face_table, &[0.0, 0.0, 0.0, 12.0]).into_iter().zip([
            2.0 * point[0],
            3.0 * point[1],
            4.0 * point[2],
        ]) {
            close(a, e);
        }
        close(face_table.divergence(3).unwrap() * 12.0, 3.0);
    }
}

#[test]
fn local_curl_gram_action_and_gradient_kernel_have_independent_values() {
    let table = element(Space::tetrahedral_edge())
        .tabulate_on(&map(&VERTICES), &[0.25; 3])
        .unwrap();
    let gradient = [4.0, 9.0, 16.0, 5.0, 12.0, 7.0]; // differences of (0,4,9,16)
    let expected = [8.0 / 3.0, -8.0 / 3.0, 0.0, 8.0 / 3.0, 0.0, 0.0];
    let rotation = [0.0, 0.0, 0.0, 6.0, 0.0, 0.0];
    let mut energy = 0.0;
    for row in 0..6 {
        let mut action = 0.0;
        let mut null_action = 0.0;
        for column in 0..6 {
            let entry = 4.0 * dot(&table.curl(row).unwrap(), &table.curl(column).unwrap());
            action += entry * rotation[column];
            null_action += entry * gradient[column];
        }
        close(action, expected[row]);
        close(null_action, 0.0);
        energy += rotation[row] * action;
    }
    close(energy, 16.0); // integral of |curl(-y,x,0)|^2 = 4 V
}

#[test]
fn every_positive_vertex_permutation_preserves_physical_fields() {
    for (a, vertex_a) in VERTICES.iter().enumerate() {
        for (b, vertex_b) in VERTICES.iter().enumerate() {
            for (c, vertex_c) in VERTICES.iter().enumerate() {
                for (d, vertex_d) in VERTICES.iter().enumerate() {
                    let Ok(permutation) = VertexPermutation::new(vec![a, b, c, d]) else {
                        continue;
                    };
                    if compatible::orientation_sign(permutation.images()) < 0 {
                        continue;
                    }
                    let vertices = [*vertex_a, *vertex_b, *vertex_c, *vertex_d];
                    let transformed = map(&vertices);
                    let reference = [0.125, 0.25, 0.5];
                    let mut physical = [0.0; 3];
                    transformed.map_point(&reference, &mut physical).unwrap();
                    let original_reference =
                        [physical[0] / 2.0, physical[1] / 3.0, physical[2] / 4.0];
                    for (space, coefficients) in [
                        (
                            Space::tetrahedral_edge(),
                            vec![2.0, -3.0, 5.0, 7.0, -11.0, 13.0],
                        ),
                        (Space::tetrahedral_face(), vec![2.0, -3.0, 5.0, 7.0]),
                    ] {
                        let element = element(space);
                        let permuted = element
                            .oriented_dofs(&permutation)
                            .unwrap()
                            .iter()
                            .map(|&(index, sign)| f64::from(sign) * coefficients[index])
                            .collect::<Vec<_>>();
                        let original = value(
                            &element
                                .tabulate_on(&map(&VERTICES), &original_reference)
                                .unwrap(),
                            &coefficients,
                        );
                        let reordered = value(
                            &element.tabulate_on(&transformed, &reference).unwrap(),
                            &permuted,
                        );
                        for (actual, expected) in reordered.into_iter().zip(original) {
                            close(actual, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sheared_polynomial_derivatives_follow_geometric_moments() {
    let vertices = [
        [1.0, -2.0, 3.0],
        [3.0, -1.0, 3.0],
        [2.0, 1.0, 4.0],
        [2.0, -1.0, 7.0],
    ];
    let geometry = map(&vertices);
    let edge = element(Space::tetrahedral_edge())
        .tabulate_on(&geometry, &[0.25; 3])
        .unwrap();
    let face = element(Space::tetrahedral_face())
        .tabulate_on(&geometry, &[0.25; 3])
        .unwrap();
    let center: [f64; 3] =
        std::array::from_fn(|axis| vertices.iter().map(|p| p[axis]).sum::<f64>() / 4.0);
    let edges = EDGES.map(|[a, b]| {
        let midpoint: [f64; 3] =
            std::array::from_fn(|axis| (vertices[a][axis] + vertices[b][axis]) / 2.0);
        dot(
            &[-midpoint[1], midpoint[0], 0.0],
            &subtract(&vertices[b], &vertices[a]),
        )
    });
    let faces = FACES.map(|[a, b, c]| {
        let center: [f64; 3] = std::array::from_fn(|axis| {
            (vertices[a][axis] + vertices[b][axis] + vertices[c][axis]) / 3.0
        });
        dot(
            &center,
            &cross(
                subtract(&vertices[b], &vertices[a]),
                subtract(&vertices[c], &vertices[a]),
            ),
        ) / 2.0
    });
    for (actual, expected) in value(&edge, &edges)
        .into_iter()
        .zip([-center[1], center[0], 0.0])
    {
        close(actual, expected);
    }
    for (actual, expected) in value(&face, &faces).into_iter().zip(center) {
        close(actual, expected);
    }
    for (axis, expected) in [0.0, 0.0, 2.0].into_iter().enumerate() {
        close(
            edges
                .iter()
                .enumerate()
                .map(|(dof, c)| c * edge.curl(dof).unwrap()[axis])
                .sum(),
            expected,
        );
    }
    close(
        faces
            .iter()
            .enumerate()
            .map(|(dof, c)| c * face.divergence(dof).unwrap())
            .sum(),
        3.0,
    );
}

#[test]
fn rejects_wrong_cells_orders_embedded_and_inverted_maps() {
    for space in [Space::tetrahedral_edge(), Space::tetrahedral_face()] {
        for cell in [
            ReferenceCell::point(),
            ReferenceCell::simplex(2).unwrap(),
            ReferenceCell::hypercube(3).unwrap(),
        ] {
            assert!(DiscreteSpace::new(space, cell).is_err());
        }
        let element = element(space);
        let mut inverted = VERTICES;
        inverted.swap(0, 1);
        assert!(element.tabulate_on(&map(&inverted), &[0.25; 3]).is_err());
        let embedded = AffineGeometryMap::from_simplex_vertices(
            VERTICES
                .iter()
                .map(|p| p.iter().copied().chain([0.0]).collect())
                .collect(),
        )
        .unwrap();
        assert!(element.tabulate_on(&embedded, &[0.25; 3]).is_err());
        assert!(element.tabulate(&[f64::NAN, 0.0, 0.0]).is_err());
        assert!(element.tabulate(&[1.0; 3]).is_err());
    }
    assert!(
        DiscreteSpace::new(
            Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap()),
            ReferenceCell::simplex(3).unwrap()
        )
        .is_err()
    );
}
