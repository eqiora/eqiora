//! Hand-derived complex actions distinguish transpose from conjugate transpose.
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_solver::{LinearOperator, LinearOperatorOrientation, Oriented, OrientedLinearOperator};
use num_complex::Complex64 as C;

#[derive(Debug)]
struct Matrix([[C; 2]; 2]);
impl Matrix {
    fn action(
        &self,
        input: &[C],
        output: &mut [C],
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), Diagnostic> {
        if input.len() != 2
            || output.len() != 2
            || input.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "invalid action buffer",
            ));
        }
        for (i, value) in output.iter_mut().enumerate() {
            *value = C::new(0., 0.);
            for (j, input) in input.iter().enumerate() {
                let coefficient = if transpose {
                    self.0[j][i]
                } else {
                    self.0[i][j]
                };
                *value += if conjugate {
                    coefficient.conj()
                } else {
                    coefficient
                } * input;
            }
        }
        Ok(())
    }
}
impl LinearOperator for Matrix {
    type Scalar = C;
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn apply(&self, input: &[C], output: &mut [C]) -> Result<(), Diagnostic> {
        self.action(input, output, false, false)
    }
}
impl OrientedLinearOperator for Matrix {
    fn supports_orientation(&self, _orientation: LinearOperatorOrientation) -> bool {
        true
    }
    fn apply_oriented(
        &self,
        orientation: LinearOperatorOrientation,
        input: &[C],
        output: &mut [C],
    ) -> Result<(), Diagnostic> {
        self.action(
            input,
            output,
            orientation != LinearOperatorOrientation::Normal,
            orientation == LinearOperatorOrientation::ConjugateTransposed,
        )
    }
}

#[test]
fn complex_transpose_and_conjugate_transpose_have_distinct_actions_and_identity() {
    let a = Matrix([
        [C::new(1., 1.), C::new(2., 0.)],
        [C::new(0., 3.), C::new(4., -1.)],
    ]);
    let x = [C::new(1., 2.), C::new(-2., 1.)];
    let t = Oriented::new(&a, LinearOperatorOrientation::Transposed).unwrap();
    let h = Oriented::new(&a, LinearOperatorOrientation::ConjugateTransposed).unwrap();
    for (action, expected, orientation) in [
        (
            &a as &dyn LinearOperator<Scalar = C>,
            [C::new(-5., 5.), C::new(-13., 9.)],
            LinearOperatorOrientation::Normal,
        ),
        (
            &t,
            [C::new(-4., -3.), C::new(-5., 10.)],
            LinearOperatorOrientation::Transposed,
        ),
        (
            &h,
            [C::new(6., 7.), C::new(-7., 6.)],
            LinearOperatorOrientation::ConjugateTransposed,
        ),
    ] {
        let mut actual = [C::new(0., 0.); 2];
        action.apply(&x, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(action.orientation(), orientation);
    }
    let y = [C::new(2., -1.), C::new(-1., 3.)];
    let mut ax = [C::new(0., 0.); 2];
    let mut ahy = ax;
    a.apply(&x, &mut ax).unwrap();
    h.apply(&y, &mut ahy).unwrap();
    let inner = |a: [C; 2], b: [C; 2]| a[0].conj() * b[0] + a[1].conj() * b[1];
    assert_eq!(inner(y, ax), inner(ahy, x));
}

#[test]
fn hermitian_action_is_its_conjugate_transpose_but_not_its_transpose() {
    // Leading principal minors 4 and 4*3-|1+i|^2=10 establish positive definiteness.
    let a = Matrix([
        [C::new(4., 0.), C::new(1., 1.)],
        [C::new(1., -1.), C::new(3., 0.)],
    ]);
    let x = [C::new(1., 2.), C::new(-2., 1.)];
    let expected = [C::new(1., 7.), C::new(-3., 4.)];
    let mut actual = [C::new(0., 0.); 2];
    a.apply(&x, &mut actual).unwrap();
    assert_eq!(actual, expected);
    Oriented::new(&a, LinearOperatorOrientation::ConjugateTransposed)
        .unwrap()
        .apply(&x, &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
    Oriented::new(&a, LinearOperatorOrientation::Transposed)
        .unwrap()
        .apply(&x, &mut actual)
        .unwrap();
    assert_ne!(actual, expected);
}

#[test]
fn unavailable_conjugate_transpose_rejects_before_invoking_an_action() {
    #[derive(Debug)]
    struct TransposeOnly;
    impl LinearOperator for TransposeOnly {
        type Scalar = C;
        fn rows(&self) -> usize {
            2
        }
        fn columns(&self) -> usize {
            2
        }
        fn apply(&self, _: &[C], _: &mut [C]) -> Result<(), Diagnostic> {
            panic!("admission must precede numerical work")
        }
    }
    impl OrientedLinearOperator for TransposeOnly {
        fn supports_orientation(&self, orientation: LinearOperatorOrientation) -> bool {
            orientation != LinearOperatorOrientation::ConjugateTransposed
        }
        fn apply_oriented(
            &self,
            _: LinearOperatorOrientation,
            _: &[C],
            _: &mut [C],
        ) -> Result<(), Diagnostic> {
            panic!("admission must precede numerical work")
        }
    }
    assert!(Oriented::new(&TransposeOnly, LinearOperatorOrientation::Transposed).is_ok());
    let error = Oriented::new(
        &TransposeOnly,
        LinearOperatorOrientation::ConjugateTransposed,
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}
