//! Scalar coordinates never change a Field's physical basis or component axes.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;

fn close(actual: f64, expected: f64) {
    // At most 27 positive quadrature points and bounded affine contractions.
    assert!(
        (actual - expected).abs() <= 4096.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn triangle() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::simplex(2).unwrap(),
        2,
        vec![0., 0.],
        vec![2., 0., 0., 3.],
    )
    .unwrap()
}

#[test]
fn mixed_real_and_complex_coordinates_preserve_component_maps() {
    let map = triangle();
    let fields = [
        IntegralField {
            space: Space::cell_constant(),
            components: 1,
            scalar_coordinates: 1,
        },
        IntegralField {
            space: Space::cell_constant(),
            components: 2,
            scalar_coordinates: 2,
        },
    ];
    // One real scalar and a complex two-channel value. The latter's action is
    // (2+3i)*z + (-1+4i)*conj(swap(z)); additional real-linear couplings connect
    // both Fields. This is not representable by a complex coefficient matrix.
    let expected = [
        [2., 3., 0., 0., -1.],
        [5., 2., -3., -1., 4.],
        [0., 3., 2., 4., 1.],
        [0., -1., 4., 2., -3.],
        [7., 4., 1., 3., 2.],
    ];
    let layout = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)];
    let mut terms = Vec::new();
    for &(row, component, test_coordinate) in &layout {
        for &(column, trial_component, trial_coordinate) in &layout {
            terms.push(IntegralTerm {
                row,
                column,
                test_coordinate,
                trial_coordinate,
                pairing: Pairing::Components {
                    test: BasisComponent::Value(component),
                    trial: BasisComponent::Value(trial_component),
                },
                trial_scale: 1.,
            });
        }
    }
    let load = [11., 13., 17., 19., 23.];
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(2, 2).unwrap();
    let result = integrate(
        map.reference_cell(),
        &fields,
        &terms,
        &map,
        &rule,
        |_, coefficients, forcing, _| {
            for (value, expected) in coefficients.iter_mut().zip(expected.iter().flatten()) {
                *value = *expected;
            }
            forcing.copy_from_slice(&load);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(result.rows(), 5);
    for (i, row) in expected.iter().enumerate() {
        close(result.rhs()[i], 3. * load[i]);
        for (j, value) in row.iter().enumerate() {
            close(result.entry(i, j).unwrap(), 3. * value);
        }
    }
    // A coordinate outside the actual scalar representation rejects before
    // evaluating coefficient data, even when its physical component is valid.
    terms[0].trial_coordinate = 1;
    assert!(
        integrate::<f64>(
            map.reference_cell(),
            &fields,
            &terms,
            &map,
            &rule,
            |_, _, _, _| panic!("invalid coordinate reached coefficient evaluation")
        )
        .is_err()
    );
}

#[test]
fn complex_load_and_flux_keep_derivative_directions_separate() {
    let map = triangle();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![0.; 2], map.jacobian().to_vec())
            .unwrap();
    let field = IntegralField {
        space: Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        components: 2,
        scalar_coordinates: 2,
    };
    let load = [1., 2., 3., 5.];
    // Physical component, derivative axis, scalar coordinate; the y direction
    // must not be mistaken for the imaginary coordinate.
    let flux = [7., 11., 13., 17., 19., 23., 29., 31.];
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(2, 2).unwrap();
    let (result, tangent) = integrate_with_geometry_direction::<f64>(
        &mut Default::default(),
        map.reference_cell(),
        &[field],
        &[],
        (&map, Some(&direction)),
        &rule,
        |_, _, _, _, forcing, flux_load, _, _, _| {
            forcing.copy_from_slice(&load);
            flux_load.copy_from_slice(&flux);
            Ok(())
        },
    )
    .unwrap();
    let tangent = tangent.unwrap();
    assert_eq!(result.rows(), 12);
    for (node, gradient) in [[-0.5, -1. / 3.], [0.5, 0.], [0., 1. / 3.]]
        .iter()
        .enumerate()
    {
        for component in 0..2 {
            for coordinate in 0..2 {
                let value = load[2 * component + coordinate];
                let flux_integral = 3.
                    * (gradient[0] * flux[4 * component + coordinate]
                        + gradient[1] * flux[4 * component + 2 + coordinate]);
                let index = 4 * node + 2 * component + coordinate;
                close(result.rhs()[index], value - flux_integral);
                // Under dilation in 2D: value loads scale as s^2, flux loads as s.
                close(tangent.rhs()[index], 2. * value - flux_integral);
            }
        }
    }
}

#[test]
fn real_linear_curl_action_preserves_complex_edge_moments() {
    let map = AffineGeometryMap::new(
        ReferenceCell::simplex(3).unwrap(),
        3,
        vec![0.; 3],
        vec![2., 0., 0., 0., 3., 0., 0., 0., 4.],
    )
    .unwrap();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![0.; 3], map.jacobian().to_vec())
            .unwrap();
    let field = IntegralField {
        space: Space::tetrahedral_edge(),
        components: 3,
        scalar_coordinates: 2,
    };
    let block = [[2., 5.], [7., -3.]];
    let mut terms = Vec::new();
    for test_coordinate in 0..2 {
        for trial_coordinate in 0..2 {
            terms.push(IntegralTerm {
                row: 0,
                column: 0,
                pairing: Pairing::Curl,
                trial_scale: 1.,
                test_coordinate,
                trial_coordinate,
            });
        }
    }
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(3, 3).unwrap();
    let (result, tangent) = integrate_with_geometry_direction(
        &mut Default::default(),
        map.reference_cell(),
        &[field],
        &terms,
        (&map, Some(&direction)),
        &rule,
        |_, _, _, coefficients, _, _, _, _, _| {
            for (value, expected) in coefficients.iter_mut().zip(block.iter().flatten()) {
                *value = *expected;
            }
            Ok(())
        },
    )
    .unwrap();
    let tangent = tangent.unwrap();
    // N_ij=lambda_i*grad(lambda_j)-lambda_j*grad(lambda_i), so
    // curl(N_ij)=2*grad(lambda_i) x grad(lambda_j). The volume is four.
    let gradients = [
        [-0.5, -1. / 3., -0.25],
        [0.5, 0., 0.],
        [0., 1. / 3., 0.],
        [0., 0., 0.25],
    ];
    let curls = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].map(|(i, j)| {
        let a = gradients[i];
        let b = gradients[j];
        [
            2. * (a[1] * b[2] - a[2] * b[1]),
            2. * (a[2] * b[0] - a[0] * b[2]),
            2. * (a[0] * b[1] - a[1] * b[0]),
        ]
    });
    assert_eq!(result.rows(), 12); // six complex edge moments, not nodal vectors
    for (i, a) in curls.iter().enumerate() {
        for (j, b) in curls.iter().enumerate() {
            let gram = 4. * a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            for (r, row) in block.iter().enumerate() {
                for (c, coefficient) in row.iter().enumerate() {
                    close(
                        result.entry(2 * i + r, 2 * j + c).unwrap(),
                        gram * coefficient,
                    );
                    // Two curls contribute s^-4, the volume contributes s^3.
                    close(
                        tangent.entry(2 * i + r, 2 * j + c).unwrap(),
                        -gram * coefficient,
                    );
                }
            }
        }
    }
}
