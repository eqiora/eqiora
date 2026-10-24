use super::*;
use eqiora_meshing::AffineGeometryLinearization;
use std::num::NonZeroU16;

fn close(actual: f64, expected: f64) {
    // Bounded quadrature and small dense contractions: a roundoff allowance,
    // independent of the observed implementation difference.
    assert!(
        (actual - expected).abs() <= 512.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn tetrahedron() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::simplex(3).unwrap(),
        3,
        vec![0.0; 3],
        vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    )
    .unwrap()
}

#[test]
fn dilation_differentiates_nodal_and_piola_mass_and_stiffness() {
    let map = tetrahedron();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![0.0; 3], map.jacobian().to_vec())
            .unwrap();
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(3, 5).unwrap();
    for (space, components, value_degree) in [
        (Space::continuous_lagrange(NonZeroU16::MIN), 1, 0.0),
        (Space::simplex_p1_bubble(), 3, 0.0),
        (Space::tetrahedral_edge(), 3, -1.0),
        (Space::tetrahedral_face(), 3, -2.0),
    ] {
        for (pairing, exponent) in [
            (Pairing::Value, 3.0 + 2.0 * value_degree),
            (Pairing::Gradient, 1.0 + 2.0 * value_degree),
        ] {
            // Under x -> s x, the measure contributes s^3; each derivative
            // contributes s^-1. Edge/face moment values contribute s^-1/s^-2.
            let (primal, tangent) = integrate_with_geometry_direction(
                None,
                map.reference_cell(),
                &[IntegralField {
                    space,
                    components,
                    scalar_coordinates: 1,
                }],
                &[IntegralTerm {
                    test_coordinate: 0,
                    trial_coordinate: 0,
                    row: 0,
                    column: 0,
                    pairing,
                    trial_scale: 1.0,
                }],
                (&map, Some(&direction)),
                &rule,
                |_, _, _, coefficients, _, _, _, _, _| {
                    coefficients[0] = 1.0;
                    Ok(())
                },
            )
            .unwrap();
            let tangent = tangent.unwrap();
            // A nonzero basis (or basis gradient) has strictly positive Gram
            // trace; individual entry magnitudes depend on its Piola scale.
            assert!(
                (0..primal.rows())
                    .map(|i| primal.entry(i, i).unwrap())
                    .sum::<f64>()
                    > 0.0
            );
            for (actual, original) in tangent.matrix().iter().zip(primal.matrix()) {
                close(*actual, exponent * original);
            }
        }
    }
}

#[test]
fn translated_coordinate_data_has_exact_p1_mass_and_load_derivatives() {
    let map = tetrahedron();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![2.0, 0.0, 0.0], vec![0.0; 9]).unwrap();
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(3, 3).unwrap();
    let (_, tangent) = integrate_with_geometry_direction(
        None,
        map.reference_cell(),
        &[IntegralField {
            space: Space::continuous_lagrange(NonZeroU16::MIN),
            components: 1,
            scalar_coordinates: 1,
        }],
        &[IntegralTerm {
            test_coordinate: 0,
            trial_coordinate: 0,
            row: 0,
            column: 0,
            pairing: Pairing::Value,
            trial_scale: 1.0,
        }],
        (&map, Some(&direction)),
        &rule,
        |_, point, delta, coefficients, forcing, _, coefficient_tangents, forcing_tangents, _| {
            coefficients[0] = point[0];
            forcing[0] = point[0];
            coefficient_tangents[0] = delta[0];
            forcing_tangents[0] = delta[0];
            Ok(())
        },
    )
    .unwrap();
    let tangent = tangent.unwrap();
    // V = 2*3*4/6 = 4. Integral(lambda_i lambda_j) = V*(1+delta_ij)/20,
    // integral(lambda_i) = V/4. The prescribed x coefficient has derivative 2.
    for row in 0..4 {
        close(tangent.rhs()[row], 2.0);
        for column in 0..4 {
            close(
                tangent.entry(row, column).unwrap(),
                if row == column { 0.8 } else { 0.4 },
            );
        }
    }
}
