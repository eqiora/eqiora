use super::*;
use eqiora_meshing::{ReferenceCell, simplex_duffy_gauss_legendre};

fn geometry(reference: ReferenceCell) -> AffineGeometryMap {
    let d = reference.dimension();
    let mut jacobian = vec![0.0; d * d];
    for axis in 0..d {
        jacobian[axis * d + axis] = (axis + 2) as f64;
    }
    AffineGeometryMap::new(reference, d, vec![7.0; d], jacobian).unwrap()
}

fn assert_weights(local: &LocalContribution<f64>, expected: &[f64]) {
    let multiplier = expected.len();
    assert_eq!(local.rows(), multiplier + 1);
    assert_eq!(local.columns(), multiplier + 1);
    assert!(local.rhs().iter().all(|value| *value == 0.0));
    for row in 0..=multiplier {
        for column in 0..=multiplier {
            let value = local.entry(row, column).unwrap();
            assert_eq!(Some(value), local.entry(column, row));
            let weight = match (row == multiplier, column == multiplier) {
                (true, false) => expected[column],
                (false, true) => expected[row],
                _ => 0.0,
            };
            assert!((value - weight).abs() <= 256.0 * f64::EPSILON * weight.abs().max(1.0));
            if row != multiplier && column != multiplier || row == column {
                assert_eq!(value, 0.0);
            }
        }
    }
}

#[test]
fn scalar_simplex_barycentric_and_constant_integrals_use_physical_measure() {
    // J = diag(2, ..., d+1): volume det(J)/d! = d+1. Each
    // barycentric function integrates to volume/(d+1) = 1.
    for d in 1..=3 {
        let map = geometry(ReferenceCell::simplex(d).unwrap());
        let quadrature = simplex_duffy_gauss_legendre(d, 3).unwrap();
        let local = zero_integral_constraint(
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            &map,
            &quadrature,
        )
        .unwrap();
        assert_weights(&local, &vec![1.0; d + 1]);
        let constant = zero_integral_constraint(Space::cell_constant(), &map, &quadrature).unwrap();
        assert_weights(&constant, &[(d + 1) as f64]);
    }
}

#[test]
fn bubble_and_tensor_product_spaces_keep_their_own_basis_integrals() {
    // On the reference triangle integral(lambda0 lambda1 lambda2) = 1/120.
    // The normalized bubble is 27 times this product; det(J) = 6.
    let triangle = geometry(ReferenceCell::simplex(2).unwrap());
    let bubble = zero_integral_constraint(
        Space::simplex_p1_bubble(),
        &triangle,
        &simplex_duffy_gauss_legendre(2, 3).unwrap(),
    )
    .unwrap();
    assert_weights(&bubble, &[1.0, 1.0, 1.0, 27.0 / 20.0]);

    // Each tensor-product Q1 basis integrates to one on [-1,1]^2,
    // then acquires the physical determinant 2*3.
    let square = geometry(ReferenceCell::hypercube(2).unwrap());
    let q1 = zero_integral_constraint(
        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        &square,
        &QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap(),
    )
    .unwrap();
    assert_weights(&q1, &[6.0; 4]);
}

#[test]
fn foreign_quadrature_and_vector_spaces_cannot_define_a_scalar_constraint() {
    let tetrahedron = geometry(ReferenceCell::simplex(3).unwrap());
    let triangle_rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    assert!(
        zero_integral_constraint(Space::cell_constant(), &tetrahedron, &triangle_rule).is_err()
    );
    let rule = simplex_duffy_gauss_legendre(3, 3).unwrap();
    for space in [Space::tetrahedral_edge(), Space::tetrahedral_face()] {
        assert!(zero_integral_constraint(space, &tetrahedron, &rule).is_err());
    }
}
