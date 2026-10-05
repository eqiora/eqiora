use super::*;
use eqiora_core::{DimExponents, FiniteBasis, Id, ScalarDomain};

fn matrix(basis: FiniteBasis, entries: &[(f64, f64)]) -> ValueLiteral {
    ValueLiteral::new(
        ValueType::linear_map(
            basis,
            basis,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        entries.iter().copied(),
    )
    .unwrap()
}

#[test]
fn metric_projector_retains_the_physical_metric_instead_of_euclidean_projection() {
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let a = matrix(basis, &[(1., 0.), (0., -1.), (0., 1.), (13., 0.)]);
    let b = matrix(basis, &[(1., 0.), (0., -1.), (0., 1.), (5., 0.)]);
    let problem = HermitianEigenproblem::new(&a, &b).unwrap();
    let mode = ValueLiteral::new(problem.mode_type().clone(), [(1., 0.), (0., 0.)]).unwrap();
    let projection = problem
        .metric_projector(std::slice::from_ref(&mode), 1e-12)
        .unwrap();
    // e1 e1ᴴ B = [[1,-i],[0,0]], not the Euclidean projector diag(1,0).
    assert_eq!(
        projection.components().unwrap().collect::<Vec<_>>(),
        [(1., 0.), (0., -1.), (0., 0.), (0., 0.)]
    );
    assert_eq!(
        projection.value_type().dimension(),
        DimExponents::DIMENSIONLESS
    );
    assert_eq!(projection.value_type().map_bases(), Some((basis, basis)));
    assert!(
        problem
            .metric_projector(&[mode.clone(), mode.clone()], 1e-12)
            .is_err()
    );
    for tolerance in [0., -1., 1., f64::NAN, f64::INFINITY] {
        assert!(
            problem
                .metric_projector(std::slice::from_ref(&mode), tolerance)
                .is_err()
        );
    }
}

#[test]
fn repeated_eigenspace_projector_ignores_basis_rotation_phase_and_permutation() {
    let basis = FiniteBasis::new(Id::new(), 3).unwrap();
    let a = matrix(
        basis,
        &[
            (2., 0.),
            (0., 0.),
            (0., 0.),
            (0., 0.),
            (8., 0.),
            (0., 0.),
            (0., 0.),
            (0., 0.),
            (27., 0.),
        ],
    );
    let b = matrix(
        basis,
        &[
            (1., 0.),
            (0., 0.),
            (0., 0.),
            (0., 0.),
            (4., 0.),
            (0., 0.),
            (0., 0.),
            (0., 0.),
            (9., 0.),
        ],
    );
    let problem = HermitianEigenproblem::new(&a, &b).unwrap();
    let mode = |values| ValueLiteral::new(problem.mode_type().clone(), values).unwrap();
    let original = [
        mode([(1., 0.), (0., 0.), (0., 0.)]),
        mode([(0., 0.), (0.5, 0.), (0., 0.)]),
    ];
    // Apply the orthogonal rotation with cos=3/5, sin=4/5, then phase i
    // to one vector and permute the columns. Both span the lambda=2 space.
    let rotated = [
        mode([(0., -0.8), (0., 0.3), (0., 0.)]),
        mode([(0.6, 0.), (0.4, 0.), (0., 0.)]),
    ];
    let value = ValueLiteral::from_real(problem.eigenvalue_type().clone(), 2.).unwrap();
    for candidate in &rotated {
        let (residual, normalization) = problem.eigenpair_defects(&value, candidate).unwrap();
        assert!(residual < 1e-14 && normalization < 1e-14);
    }
    for modes in [&original, &rotated] {
        let projector = problem.metric_projector(modes, 1e-14).unwrap();
        for (i, (real, imag)) in projector.components().unwrap().enumerate() {
            let expected = if i == 0 || i == 4 { 1. } else { 0. };
            assert!((real - expected).abs() < 1e-14 && imag.abs() < 1e-14);
        }
    }
    assert!(
        problem
            .metric_projector(
                &[original[0].clone(), mode([(0., 0.), (1., 0.), (0., 0.)])],
                1e-12
            )
            .is_err()
    );
    assert!(problem.metric_projector(&[], 1e-12).is_err());
}

#[test]
fn complex_pencil_checks_analytic_modes_beyond_four_dimensions() {
    // Three independent copies of A=[[4,4i],[-4i,16]], B=diag(2,8).
    // Each block has eigenvalues 1 and 3, with B-unit modes (1/2, ±i/4).
    let basis = FiniteBasis::new(Id::new(), 6).unwrap();
    let mut a = vec![(0., 0.); 36];
    let mut b = a.clone();
    for block in 0..3 {
        let i = 2 * block;
        a[i * 6 + i] = (4., 0.);
        a[i * 6 + i + 1] = (0., 4.);
        a[(i + 1) * 6 + i] = (0., -4.);
        a[(i + 1) * 6 + i + 1] = (16., 0.);
        b[i * 6 + i] = (2., 0.);
        b[(i + 1) * 6 + i + 1] = (8., 0.);
    }
    let a = matrix(basis, &a);
    let b = matrix(basis, &b);
    let pencil = HermitianEigenproblem::new(&a, &b).unwrap();
    assert_eq!(pencil.dimension(), 6);
    // Permuted blocks and phase i change neither the eigenvalue nor defects.
    for block in [2, 0, 1] {
        for (lambda, sign) in [(1., 1.), (3., -1.)] {
            let mut components = vec![(0., 0.); 6];
            components[2 * block] = (0., 0.5);
            components[2 * block + 1] = (-sign * 0.25, 0.);
            let mode = ValueLiteral::new(pencil.mode_type().clone(), components).unwrap();
            let value = ValueLiteral::from_real(pencil.eigenvalue_type().clone(), lambda).unwrap();
            assert_eq!(pencil.eigenpair_defects(&value, &mode).unwrap(), (0., 0.));
            let wrong =
                ValueLiteral::from_real(pencil.eigenvalue_type().clone(), lambda + 1.).unwrap();
            assert!(pencil.eigenpair_defects(&wrong, &mode).unwrap().0 > 0.1);
        }
    }
}

#[test]
fn complete_matrix_admission_rejects_false_hermitian_and_singular_metrics() {
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let identity = matrix(basis, &[(1., 0.), (0., 0.), (0., 0.), (1., 0.)]);
    for entries in [
        [(1., 0.), (0., 1.), (0., 1.), (1., 0.)], // complex symmetric ≠ Hermitian
        [(1., 1.), (0., 0.), (0., 0.), (1., 0.)], // imaginary diagonal
        [(1., 0.), (1., 0.), (0., 0.), (1., 0.)], // upper-triangle corruption
    ] {
        let bad = matrix(basis, &entries);
        assert!(HermitianEigenproblem::new(&bad, &identity).is_err());
        assert!(HermitianEigenproblem::new(&identity, &bad).is_err());
    }
    for entries in [
        [(-1., 0.), (0., 0.), (0., 0.), (-1., 0.)], // explicit negative metric is not reoriented
        [(1., 0.), (0., 0.), (0., 0.), (0., 0.)],   // unhandled nullspace
        [(1., 0.), (0., 0.), (0., 0.), (-1., 0.)],
        [(1., 0.), (2., 0.), (2., 0.), (1., 0.)], // positive diagonal alone is insufficient
    ] {
        assert!(HermitianEigenproblem::new(&identity, &matrix(basis, &entries)).is_err());
    }
}

#[test]
fn candidate_validation_keeps_types_nonzero_condition_and_normalization() {
    let basis = FiniteBasis::new(Id::new(), 1).unwrap();
    let a = matrix(basis, &[(0., 0.)]);
    let b = matrix(basis, &[(4., 0.)]);
    let pencil = HermitianEigenproblem::new(&a, &b).unwrap();
    let value = ValueLiteral::from_real(pencil.eigenvalue_type().clone(), 0.).unwrap();
    let mode = |x| ValueLiteral::new(pencil.mode_type().clone(), [(x, 0.)]).unwrap();
    assert_eq!(
        pencil.eigenpair_defects(&value, &mode(0.5)).unwrap(),
        (0., 0.)
    );
    assert_eq!(
        pencil.eigenpair_defects(&value, &mode(1.)).unwrap(),
        (0., 3.)
    );
    assert!(pencil.eigenpair_defects(&value, &mode(0.)).is_err());
    assert!(pencil.eigenpair_defects(&value, &mode(f64::MAX)).is_err());
    let wrong_dimension = ValueLiteral::from_real(
        pencil
            .eigenvalue_type()
            .clone()
            .with_dimension(DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap())
            .unwrap(),
        0.,
    )
    .unwrap();
    assert!(
        pencil
            .eigenpair_defects(&wrong_dimension, &mode(0.5))
            .is_err()
    );
    let foreign = ValueLiteral::new(
        ValueType::coordinates(
            FiniteBasis::new(Id::new(), 1).unwrap(),
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        [(0.5, 0.)],
    )
    .unwrap();
    assert!(pencil.eigenpair_defects(&value, &foreign).is_err());
}
