//! Exact typed coefficient capture, independent of an external operator action.
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperator,
    LinearOperatorOrientation as Orientation, LinearOperatorProperties as Properties, Oriented,
};
use num_complex::Complex64 as C;

struct Storage<S> {
    values: [S; 4],
    rhs: [S; 2],
}
impl<S> CompleteCsrStorage<S> for Storage<S> {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }
    fn values(&self) -> &[S] {
        &self.values
    }
    fn right_hand_side(&self) -> &[S] {
        &self.rhs
    }
}

#[test]
fn captured_complex_coefficients_drive_all_three_actions_without_erasing_parts() {
    let storage = Storage {
        values: [
            C::new(1., 1.),
            C::new(2., 0.),
            C::new(0., 3.),
            C::new(4., -1.),
        ],
        rhs: [C::new(-5., 5.), C::new(-13., 9.)],
    };
    let system = CanonicalCsrSystemView::new(&storage, Properties::General).unwrap();
    let x = [C::new(1., 2.), C::new(-2., 1.)];
    let problem = system.linear_problem().unwrap();
    assert_eq!(problem.scalar_domain(), eqiora_core::ScalarDomain::Complex);
    assert_eq!(problem.scalar_type(), eqiora_core::ScalarType::F64);
    assert_eq!(problem.right_hand_side(), storage.rhs);
    assert_eq!(
        problem.canonical_csr_system().unwrap().values(),
        storage.values
    );
    for (orientation, expected) in [
        (Orientation::Normal, [C::new(-5., 5.), C::new(-13., 9.)]),
        (
            Orientation::Transposed,
            [C::new(-4., -3.), C::new(-5., 10.)],
        ),
        (
            Orientation::ConjugateTransposed,
            [C::new(6., 7.), C::new(-7., 6.)],
        ),
    ] {
        let mut actual = [C::new(0., 0.); 2];
        Oriented::new(&system, orientation)
            .unwrap()
            .apply(&x, &mut actual)
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn identity_binds_the_domain_and_every_part_while_normalizing_signed_zero() {
    let real = Storage {
        values: [1., 0., 0., 2.],
        rhs: [3., 4.],
    };
    let complex = Storage {
        values: real.values.map(|x| C::new(x, 0.)),
        rhs: real.rhs.map(|x| C::new(x, 0.)),
    };
    let a = CanonicalCsrSystemView::new(&real, Properties::General).unwrap();
    let b = CanonicalCsrSystemView::new(&complex, Properties::General).unwrap();
    assert_ne!(a.agreement_fingerprint(), b.agreement_fingerprint());
    let mut changed = complex;
    changed.values[1] = C::new(-0., -0.);
    let normalized = CanonicalCsrSystemView::new(&changed, Properties::General).unwrap();
    assert_eq!(
        normalized.agreement_fingerprint(),
        b.agreement_fingerprint()
    );
    assert_eq!(normalized.values()[1].re.to_bits(), 0);
    assert_eq!(normalized.values()[1].im.to_bits(), 0);
    changed.values[1].im = 1.;
    let c = CanonicalCsrSystemView::new(&changed, Properties::General).unwrap();
    assert_ne!(c.agreement_fingerprint(), b.agreement_fingerprint());
    changed.values[1].im = 0.;
    changed.rhs[1].im = 1.;
    let d = CanonicalCsrSystemView::new(&changed, Properties::General).unwrap();
    assert_ne!(d.agreement_fingerprint(), b.agreement_fingerprint());
}

#[test]
fn imaginary_nonfinite_values_cannot_enter_a_captured_problem_or_guess() {
    let mut storage = Storage {
        values: [C::new(1., 0.); 4],
        rhs: [C::new(1., 0.); 2],
    };
    storage.values[1].im = f64::INFINITY;
    assert!(CanonicalCsrSystemView::new(&storage, Properties::General).is_err());
    storage.values[1].im = 0.;
    storage.rhs[1].im = f64::NAN;
    assert!(CanonicalCsrSystemView::new(&storage, Properties::General).is_err());
    storage.rhs[1].im = 0.;
    let system = CanonicalCsrSystemView::new(&storage, Properties::General).unwrap();
    assert!(
        system
            .linear_problem()
            .unwrap()
            .with_initial_guess(&[C::new(0., f64::INFINITY); 2])
            .is_err()
    );
}

#[test]
fn scalar_domain_admission_is_independent_of_binary64_precision() {
    use eqiora_core::{ScalarDomain as Domain, ScalarType, diagnostic::codes};
    use eqiora_solver::{
        LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
        SolverPlan,
    };
    let capability = SolverCapability {
        scalar_domain: Domain::Real,
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: Properties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    };
    let real = SolverCapabilities::exact([capability]).unwrap();
    let plan = SolverPlan::new(
        capability.algorithm,
        1e-12,
        1e-14,
        std::num::NonZeroUsize::new(8).unwrap(),
    )
    .unwrap();
    real.require_problem(plan, Domain::Real, ScalarType::F64, Properties::General)
        .unwrap();
    assert!(real.supports_scalar(Domain::Real, ScalarType::F64));
    assert!(!real.supports_scalar(Domain::Complex, ScalarType::F64));
    assert_eq!(
        real.require_problem(plan, Domain::Complex, ScalarType::F64, Properties::General)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    assert!(
        real.require_problem(plan, Domain::Real, ScalarType::F32, Properties::General)
            .is_err()
    );
    let complex = SolverCapabilities::exact([SolverCapability {
        scalar_domain: Domain::Complex,
        ..capability
    }])
    .unwrap();
    complex
        .require_problem(plan, Domain::Complex, ScalarType::F64, Properties::General)
        .unwrap();
    assert!(
        complex
            .require_problem(plan, Domain::Real, ScalarType::F64, Properties::General)
            .is_err()
    );
    assert!(
        SolverCapabilities::exact([SolverCapability {
            scalar_domain: Domain::Integer,
            ..capability
        }])
        .is_err()
    );
    assert!(
        SolverCapabilities::exact([SolverCapability {
            scalar_domain: Domain::Complex,
            operator_properties: Properties::SymmetricPositiveDefinite,
            ..capability
        }])
        .is_err()
    );
    let storage = Storage {
        values: [C::new(1., 0.); 4],
        rhs: [C::new(1., 0.); 2],
    };
    assert!(CanonicalCsrSystemView::new(&storage, Properties::SymmetricPositiveDefinite).is_err());
}

#[test]
fn complex_symmetric_and_hermitian_assertions_are_not_interchangeable() {
    let symmetric = Storage {
        values: [
            C::new(4., 0.),
            C::new(1., 1.),
            C::new(1., 1.),
            C::new(3., 0.),
        ],
        rhs: [C::new(0., 0.); 2],
    };
    CanonicalCsrSystemView::new(&symmetric, Properties::ComplexSymmetric).unwrap();
    for property in [Properties::Hermitian, Properties::HermitianPositiveDefinite] {
        let error = CanonicalCsrSystemView::new(&symmetric, property).unwrap_err();
        assert!(error.message().contains("violate declared"));
    }
    let hermitian = Storage {
        values: [
            C::new(4., 0.),
            C::new(1., 1.),
            C::new(1., -1.),
            C::new(3., 0.),
        ],
        rhs: [C::new(1., 7.), C::new(-3., 4.)],
    };
    let h = CanonicalCsrSystemView::new(&hermitian, Properties::Hermitian).unwrap();
    // Leading principal minors 4 and 4*3 - |1+i|^2 = 10 prove positivity here.
    let hpd =
        CanonicalCsrSystemView::new(&hermitian, Properties::HermitianPositiveDefinite).unwrap();
    assert_ne!(h.agreement_fingerprint(), hpd.agreement_fingerprint());
    assert!(CanonicalCsrSystemView::new(&hermitian, Properties::ComplexSymmetric).is_err());
    let imaginary_diagonal = Storage {
        values: [
            C::new(4., 1.),
            C::new(1., 1.),
            C::new(1., -1.),
            C::new(3., 0.),
        ],
        rhs: hermitian.rhs,
    };
    assert!(CanonicalCsrSystemView::new(&imaginary_diagonal, Properties::Hermitian).is_err());
}

#[test]
fn complex_property_admission_requires_the_exact_assertion_and_implemented_tuple() {
    use eqiora_core::{ScalarDomain, ScalarType};
    use eqiora_solver::{
        LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
    };

    assert!(LinearSolver::ConjugateGradient.accepts(Properties::HermitianPositiveDefinite));
    for property in [
        Properties::Hermitian,
        Properties::ComplexSymmetric,
        Properties::General,
    ] {
        assert!(!LinearSolver::ConjugateGradient.accepts(property));
    }
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        let tuple = SolverCapability {
            scalar_domain: domain,
            scalar_type: ScalarType::F64,
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: Properties::HermitianPositiveDefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        };
        assert_eq!(
            SolverCapabilities::exact([tuple]).is_ok(),
            domain == ScalarDomain::Complex
        );
    }
    // Only the implemented complex tuple is advertised; binary32 remains unsupported.
    assert!(
        !SolverCapabilities::reference().supports_scalar(ScalarDomain::Complex, ScalarType::F32)
    );
}

#[test]
fn one_request_surface_solves_general_and_hpd_complex_systems_and_replays_original_residuals() {
    use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    let expected = [C::new(1., 2.), C::new(-2., 1.)];
    for (values, rhs, properties, algorithm) in [
        (
            [
                C::new(1., 1.),
                C::new(2., 0.),
                C::new(0., 3.),
                C::new(4., -1.),
            ],
            [C::new(-5., 5.), C::new(-13., 9.)],
            Properties::General,
            LinearSolver::BiConjugateGradientStabilized,
        ),
        (
            [
                C::new(4., 0.),
                C::new(1., 1.),
                C::new(1., -1.),
                C::new(3., 0.),
            ],
            [C::new(1., 7.), C::new(-3., 4.)],
            Properties::HermitianPositiveDefinite,
            LinearSolver::ConjugateGradient,
        ),
    ] {
        let storage = Storage { values, rhs };
        let system = CanonicalCsrSystemView::new(&storage, properties).unwrap();
        let plan = SolverPlan::new(algorithm, 1e-12, 1e-12, 32.try_into().unwrap()).unwrap();
        let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
        let solution = request.solve(&system.linear_problem().unwrap()).unwrap();
        for (actual, expected) in solution.values().iter().zip(expected) {
            assert!((*actual - expected).norm() <= 1e-10);
        }
        // Independently multiply the authored 2x2 coefficients, without CSR or lowering.
        let x = solution.values();
        let residual = [
            rhs[0] - (values[0] * x[0] + values[1] * x[1]),
            rhs[1] - (values[2] * x[0] + values[3] * x[1]),
        ];
        let norm = residual[0].norm().hypot(residual[1].norm());
        assert!(norm <= solution.report().residual_target());
        assert!((norm - solution.report().true_residual_norm()).abs() <= 1e-14);
    }
    let real = Storage {
        values: [4., 1., 1., 3.],
        rhs: [2., -5.],
    };
    let system = CanonicalCsrSystemView::new(&real, Properties::SymmetricPositiveDefinite).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    let solution = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
        .solve(&system.linear_problem().unwrap())
        .unwrap();
    for (actual, expected) in solution.values().iter().zip([1., -2.]) {
        assert!((actual - expected).abs() <= 1e-10);
    }
}

#[test]
fn oriented_complex_solves_keep_transpose_and_adjoint_distinct() {
    use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    let storage = Storage {
        values: [
            C::new(1., 1.),
            C::new(2., 0.),
            C::new(0., 3.),
            C::new(4., -1.),
        ],
        rhs: [C::new(-5., 5.), C::new(-13., 9.)],
    };
    let system = CanonicalCsrSystemView::new(&storage, Properties::General).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
    for (orientation, rhs) in [
        (
            Orientation::Transposed,
            [C::new(-4., -3.), C::new(-5., 10.)],
        ),
        (
            Orientation::ConjugateTransposed,
            [C::new(6., 7.), C::new(-7., 6.)],
        ),
    ] {
        let solution = request
            .solve_canonical_oriented(&system, &rhs, orientation)
            .unwrap();
        for (actual, expected) in solution
            .values()
            .iter()
            .zip([C::new(1., 2.), C::new(-2., 1.)])
        {
            assert!((*actual - expected).norm() <= 1e-10);
        }
        assert_eq!(solution.report().orientation(), orientation);
    }
    assert_eq!(system.right_hand_side(), storage.rhs);
}

#[test]
fn hpd_admission_rejects_indefiniteness_even_when_the_rhs_avoids_the_negative_eigenspace() {
    use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
    for negative in [-1., 0.] {
        let storage = Storage {
            values: [
                C::new(1., 0.),
                C::new(0., 0.),
                C::new(0., 0.),
                C::new(negative, 0.),
            ],
            // Ordinary CG sees only the positive eigenvector and could accept.
            rhs: [C::new(1., 1.), C::new(0., 0.)],
        };
        let system =
            CanonicalCsrSystemView::new(&storage, Properties::HermitianPositiveDefinite).unwrap();
        let error = request
            .solve(&system.linear_problem().unwrap())
            .unwrap_err();
        assert!(error.message().contains("Cholesky pivot"));
    }
}

#[test]
fn repeating_a_complex_request_uses_changed_coefficients_instead_of_stale_numeric_state() {
    use eqiora_solver::{
        LinearSolveRequest, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, SolverPlan,
    };
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-12,
        32.try_into().unwrap(),
    )
    .unwrap();
    // The reference provider exposes no retained factors. Callers must execute
    // each candidate freshly when preparation returns None.
    assert!(
        <_ as LinearSolverBackend<C>>::prepare_linear(&REFERENCE_LINEAR_SOLVER, plan)
            .unwrap()
            .is_none()
    );
    let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
    let original = [
        C::new(1., 1.),
        C::new(2., 0.),
        C::new(0., 3.),
        C::new(4., -1.),
    ];
    for scale in [1., 2., 1.] {
        let storage = Storage {
            values: original.map(|value| scale * value),
            rhs: [C::new(-5., 5.), C::new(-13., 9.)],
        };
        let system = CanonicalCsrSystemView::new(&storage, Properties::General).unwrap();
        let solution = request.solve(&system.linear_problem().unwrap()).unwrap();
        for (actual, expected) in solution
            .values()
            .iter()
            .zip([C::new(1., 2.) / scale, C::new(-2., 1.) / scale])
        {
            assert!((*actual - expected).norm() <= 1e-10);
        }
    }
}

#[test]
fn pure_imaginary_coefficient_does_not_create_an_artificial_real_pairing_breakdown() {
    use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    // A = i I, x = [1+2i, -2+i], b = [-2+i, -1-2i].
    let storage = Storage {
        values: [
            C::new(0., 1.),
            C::new(0., 0.),
            C::new(0., 0.),
            C::new(0., 1.),
        ],
        rhs: [C::new(-2., 1.), C::new(-1., -2.)],
    };
    let system = CanonicalCsrSystemView::new(&storage, Properties::General).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-12,
        8.try_into().unwrap(),
    )
    .unwrap();
    let solution = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
        .solve(&system.linear_problem().unwrap())
        .unwrap();
    for (actual, expected) in solution
        .values()
        .iter()
        .zip([C::new(1., 2.), C::new(-2., 1.)])
    {
        assert!((*actual - expected).norm() <= 1e-10);
    }
}
