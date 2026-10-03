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
