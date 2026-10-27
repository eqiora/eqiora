use super::*;

#[test]
fn complex_jacobi_is_the_typed_inverse_and_conjugates_for_faer() {
    let source = Matrix([
        [C::new(2., 1.), C::new(0., 0.)],
        [C::new(0., 0.), C::new(3., -1.)],
    ]);
    let rhs = [C::new(0., 5.), C::new(-5., 5.)];
    let request = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-13,
        1.try_into().unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast)
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    for orientation in [Orientation::Normal, Orientation::ConjugateTransposed] {
        let op = Oriented::new(&source, orientation).unwrap();
        let b = if orientation == Orientation::Normal {
            rhs
        } else {
            [C::new(4., 3.), C::new(-7., 1.)]
        };
        let problem = LinearProblem::new(&op, &b, LinearOperatorProperties::General).unwrap();
        let solution = invoke(&problem, request).unwrap();
        assert_values(solution.values(), &X);
        assert_eq!(solution.report().completed_iterations(), 1);
    }
    let problem = LinearProblem::new(&source, &rhs, LinearOperatorProperties::General).unwrap();
    let inverse = inverse_diagonal(&problem, request).unwrap().unwrap();
    assert_values(&inverse, &[C::new(0.4, -0.2), C::new(0.3, 0.1)]);
    let adapter = FaerPreconditioner::new(2, Some(inverse));
    let input = Mat::from_fn(2, 1, |i, _| rhs[i]);
    let mut output = Mat::zeros(2, 1);
    adapter.conj_apply(
        output.as_mut(),
        input.as_ref(),
        Par::Seq,
        MemStack::new(&mut MemBuffer::new(StackReq::EMPTY)),
    );
    assert_values(output.col_as_slice(0), &[C::new(-1., 2.), C::new(-1., 2.)]);
}

#[test]
fn jacobi_rejects_unavailable_or_invalid_typed_diagonals_and_keeps_scaled_reciprocals() {
    let unit_rhs = [C::new(1., 0.)];
    let request = plan(
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    );
    for diagonal in [
        C::new(0., 0.),
        C::new(-1., 0.),
        C::new(1., 0.25),
        C::new(f64::INFINITY, 0.),
        C::new(f64::NAN, 0.),
        C::new(1e-320, 0.),
    ] {
        let source = Matrix([[diagonal]]);
        let problem = LinearProblem::new(
            &source,
            &unit_rhs,
            LinearOperatorProperties::HermitianPositiveDefinite,
        )
        .unwrap();
        assert!(inverse_diagonal(&problem, request).is_err());
    }
    let source = Matrix([
        [C::new(1e200, 1e200), C::new(0., 0.)],
        [C::new(0., 0.), C::new(1e-200, -1e-200)],
    ]);
    let rhs = [C::new(1., 0.); 2];
    let problem = LinearProblem::new(&source, &rhs, LinearOperatorProperties::General).unwrap();
    let inverse = inverse_diagonal(
        &problem,
        plan(
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Jacobi,
        ),
    )
    .unwrap()
    .unwrap();
    for (value, diagonal) in inverse.iter().zip([source.0[0][0], source.0[1][1]]) {
        assert!((*value * diagonal - C::new(1., 0.)).norm() < 1e-14);
    }
    #[derive(Debug)]
    struct Unavailable;
    impl LinearOperator for Unavailable {
        type Scalar = C;
        fn rows(&self) -> usize {
            1
        }
        fn columns(&self) -> usize {
            1
        }
        fn apply(&self, x: &[C], y: &mut [C]) -> Result<(), Diagnostic> {
            y.copy_from_slice(x);
            Ok(())
        }
    }
    let problem = LinearProblem::new(
        &Unavailable,
        &unit_rhs,
        LinearOperatorProperties::HermitianPositiveDefinite,
    )
    .unwrap();
    assert_eq!(
        inverse_diagonal(&problem, request).unwrap_err().code(),
        codes::INVALID_REALIZATION
    );
}
