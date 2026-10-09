use super::*;
use num_complex::Complex64 as C;

fn vector_local<S: Scalar + ComplexFloat<Real = f64> + From<f64> + AddAssign + SubAssign>(
    space: Space,
    pairings: &[Pairing],
    weights: &[S],
) -> LocalContribution<S> {
    let reference = ReferenceCell::simplex(3).unwrap();
    let geometry = AffineGeometryMap::new(
        reference,
        3,
        vec![0.0; 3],
        vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(3, 3).unwrap();
    let terms = pairings
        .iter()
        .map(|&pairing| IntegralTerm {
            row: 0,
            column: 0,
            pairing,
            trial_scale: 1.0,
        })
        .collect::<Vec<_>>();
    integrate(
        reference,
        &[(space, 3)],
        &terms,
        &geometry,
        &rule,
        |_, coefficients, _, _| {
            coefficients.copy_from_slice(weights);
            Ok(())
        },
    )
    .unwrap()
}

fn close(actual: C, expected: C) {
    // A fixed 27-point positive rule, products of affine bases and at most six
    // contractions. 4096 eps covers quadrature, mapping and contraction roundoff.
    assert!(
        (actual - expected).norm() <= 4096.0 * f64::EPSILON * expected.norm().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn vector_moments_use_the_same_real_and_complex_local_operator() {
    // curl(v).curl(u) = 2 grad(v):grad(u) - 2 sym(grad(v)):sym(grad(u)).
    // This is an element-local algebraic contraction, not an H1 trace assertion.
    let pairings = [Pairing::Gradient, Pairing::SymmetricGradient];
    let real = vector_local(Space::tetrahedral_edge(), &pairings, &[2.0, -2.0]);
    let complex = vector_local(
        Space::tetrahedral_edge(),
        &pairings,
        &[C::new(2.0, 0.0), C::new(-2.0, 0.0)],
    );
    let expected = [
        25.0 / 36.0,
        -4.0 / 9.0,
        -1.0 / 4.0,
        4.0 / 9.0,
        1.0 / 4.0,
        0.0,
        -4.0 / 9.0,
        5.0 / 9.0,
        -1.0 / 9.0,
        -4.0 / 9.0,
        0.0,
        1.0 / 9.0,
        -1.0 / 4.0,
        -1.0 / 9.0,
        13.0 / 36.0,
        0.0,
        -1.0 / 4.0,
        -1.0 / 9.0,
        4.0 / 9.0,
        -4.0 / 9.0,
        0.0,
        4.0 / 9.0,
        0.0,
        0.0,
        1.0 / 4.0,
        0.0,
        -1.0 / 4.0,
        0.0,
        1.0 / 4.0,
        0.0,
        0.0,
        1.0 / 9.0,
        -1.0 / 9.0,
        0.0,
        0.0,
        1.0 / 9.0,
    ];
    assert_eq!(real.matrix().len(), 36); // six moments, not six nodal vectors
    assert_eq!(complex.matrix().len(), 36);
    for (index, expected) in expected.into_iter().enumerate() {
        close(C::new(real.matrix()[index], 0.0), C::new(expected, 0.0));
        close(complex.matrix()[index], C::new(expected, 0.0));
    }
    let rotation = [0.0, 0.0, 0.0, 6.0, 0.0, 0.0];
    let gradient = [4.0, 9.0, 16.0, 5.0, 12.0, 7.0];
    let z: [C; 6] =
        std::array::from_fn(|i| C::new(1.0, 1.0) * rotation[i] + C::new(2.0, -1.0) * gradient[i]);
    let action: [C; 6] = std::array::from_fn(|row| {
        (0..6)
            .map(|column| complex.matrix()[row * 6 + column] * z[column])
            .sum()
    });
    let expected_action = [8.0 / 3.0, -8.0 / 3.0, 0.0, 8.0 / 3.0, 0.0, 0.0];
    for (actual, expected) in action.into_iter().zip(expected_action) {
        close(actual, C::new(1.0, 1.0) * expected);
    }
    close(
        z.iter()
            .zip(action)
            .map(|(z, action)| z.conj() * action)
            .sum(),
        C::new(32.0, 0.0),
    );
    close(
        z.iter().zip(action).map(|(z, action)| z * action).sum(),
        C::new(0.0, 32.0),
    );
}

#[test]
fn face_mass_preserves_flux_measure_and_complex_constitutive_phase() {
    let phase = C::new(2.0, -1.0);
    let local = vector_local(Space::tetrahedral_face(), &[Pairing::Value], &[phase]);
    // Independent simplex moments: V=4, integral(x)=(2,3,4),
    // integral(|x|^2)=58/5; phi_face = sign*(x-opposite_vertex)/12.
    let expected = [
        109.0 / 360.0,
        67.0 / 720.0,
        -7.0 / 120.0,
        11.0 / 360.0,
        67.0 / 720.0,
        37.0 / 180.0,
        7.0 / 720.0,
        13.0 / 720.0,
        -7.0 / 120.0,
        7.0 / 720.0,
        49.0 / 360.0,
        -19.0 / 360.0,
        11.0 / 360.0,
        13.0 / 720.0,
        -19.0 / 360.0,
        29.0 / 360.0,
    ];
    assert_eq!(local.matrix().len(), 16);
    for (actual, expected) in local.matrix().iter().zip(expected) {
        close(*actual, phase * expected);
    }
    assert_ne!(local.matrix()[1], local.matrix()[4].conj());
}

#[test]
fn local_vector_shape_and_pairing_reject_before_coefficient_evaluation() {
    let reference = ReferenceCell::simplex(3).unwrap();
    let geometry = AffineGeometryMap::new(
        reference,
        3,
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let rule = eqiora_meshing::simplex_centroid_rule(3).unwrap();
    for (space, components, pairing) in [
        (Space::tetrahedral_edge(), 1, Pairing::Value),
        (Space::tetrahedral_face(), 2, Pairing::Divergence),
        (
            Space::tetrahedral_edge(),
            3,
            Pairing::TestValueTrialDivergence,
        ),
    ] {
        assert!(
            integrate::<f64>(
                reference,
                &[(space, components)],
                &[IntegralTerm {
                    row: 0,
                    column: 0,
                    pairing,
                    trial_scale: 1.0
                }],
                &geometry,
                &rule,
                |_, _, _, _| panic!("invalid shape reached coefficient evaluation")
            )
            .is_err()
        );
    }
}

#[test]
fn complex_diffusion_mass_and_load_share_real_basis_quadrature() {
    let reference = ReferenceCell::hypercube(1).unwrap();
    let geometry = AffineGeometryMap::new(reference, 1, vec![3.0], vec![3.0]).unwrap();
    let quadrature = QuadratureRule::gauss_legendre(2).unwrap();
    let local = integrate(
        reference,
        &[(Space::continuous_lagrange(std::num::NonZeroU16::MIN), 1)],
        &[
            IntegralTerm {
                row: 0,
                column: 0,
                pairing: Pairing::Gradient,
                trial_scale: 1.0,
            },
            IntegralTerm {
                row: 0,
                column: 0,
                pairing: Pairing::Value,
                trial_scale: 1.0,
            },
        ],
        &geometry,
        &quadrature,
        |_, coefficients, forcing, _| {
            coefficients[0] = C::new(6.0, 6.0);
            coefficients[1] = C::new(3.0, -1.0);
            forcing[0] = C::new(1.0, 3.0);
            Ok(())
        },
    )
    .unwrap();
    // On [0,6], K = a/6 [[1,-1],[-1,1]],
    // M = q [[2,1],[1,2]], and each load is 3f.
    let expected = [
        C::new(7.0, -1.0),
        C::new(2.0, -2.0),
        C::new(2.0, -2.0),
        C::new(7.0, -1.0),
    ];
    for (actual, expected) in local.matrix().iter().zip(expected) {
        assert!((*actual - expected).norm() < 1e-12);
    }
    for actual in local.rhs() {
        assert!((*actual - C::new(3.0, 9.0)).norm() < 1e-12);
    }
    // Symmetric real bases do not imply a Hermitian coefficient operator.
    assert_ne!(local.matrix()[1], local.matrix()[2].conj());
    assert!(
        integrate_scalar(1, &geometry, &quadrature, |_| Ok((
            C::new(1.0, f64::NAN),
            C::new(0.0, 0.0)
        )))
        .is_err()
    );
}
