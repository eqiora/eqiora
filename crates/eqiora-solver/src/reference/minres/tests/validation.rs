use super::*;

#[test]
fn invalid_or_unavailable_jacobi_diagonals_reject_before_operator_work() {
    #[derive(Debug)]
    struct BeforeWork(Option<f64>);
    impl LinearOperator for BeforeWork {
        type Scalar = f64;
        fn rows(&self) -> usize {
            1
        }
        fn columns(&self) -> usize {
            1
        }
        fn apply(&self, _: &[f64], _: &mut [f64]) -> Result<(), Diagnostic> {
            panic!("invalid Jacobi diagonal must reject before work")
        }
        fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
            if let Some(value) = self.0 {
                output[0] = value;
                Ok(DiagonalAvailability::Available)
            } else {
                Ok(DiagonalAvailability::Unavailable)
            }
        }
    }
    for diagonal in [
        None,
        Some(0.),
        Some(-1.),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        let operator = BeforeWork(diagonal);
        let problem =
            LinearProblem::new(&operator, &[0.], LinearOperatorProperties::Symmetric).unwrap();
        let error = invoke(&problem, plan(1, 1e-12, 1e-12)).unwrap_err();
        assert_eq!(
            error.code(),
            if diagonal.is_none() {
                codes::INVALID_REALIZATION
            } else {
                codes::NUMERICAL_SOLVE_FAILED
            }
        );
    }
}

#[test]
fn square_root_precedes_reciprocal_and_nonfinite_scaled_values_fail_honestly() {
    // 1/D overflows, whereas 1/sqrt(D)=2^537 and the congruence are finite.
    let tiny = f64::from_bits(1);
    let operator = Dense([[tiny]]);
    let rhs = [tiny];
    let problem = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let solution = invoke(&problem, plan(1, 0.5, 0.)).unwrap();
    assert_eq!(solution.values(), &[1.]);
    assert_eq!(solution.report().completed_iterations(), 1);
    assert_eq!(solution.report().true_residual_norm(), 0.);

    let operator = Dense([[1e-300]]);
    let rhs = [1e200];
    let problem = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let error = invoke(&problem, plan(1, 1e-12, 1e-12)).unwrap_err();
    assert!(error.message().contains("non-finite"));
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
}
