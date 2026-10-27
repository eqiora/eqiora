//! Independent two-coordinate complex systems; public tuples remain unchanged.
use crate::*;
use eqiora_solver::{LinearOperatorOrientation as Orientation, Oriented, OrientedLinearOperator};
use faer::complex::Complex64 as C;
use faer::dyn_stack::StackReq;
use faer::matrix_free::LinOp;

mod acceptance;
mod diagonal;

// Before execution: ||A^-1||_2 <= ||adj(A)||_F/|det(A)| < 0.76 for
// GENERAL, and <0.54 for HPD. The fixed residual policy below bounds solution
// error by 1.4e-12 on the listed RHS. 1e-11 allows binary64 reduction roundoff.
const TOLERANCE: f64 = 1e-11;
const X: [C; 2] = [C::new(1., 2.), C::new(-2., 1.)];
const GENERAL: [[C; 2]; 2] = [
    [C::new(2., 1.), C::new(1., -2.)],
    [C::new(3., 1.), C::new(4., -1.)],
];
const HPD: [[C; 2]; 2] = [
    [C::new(4., 0.), C::new(1., 1.)],
    [C::new(1., -1.), C::new(3., 0.)],
];

#[derive(Debug)]
struct Matrix<S, const N: usize>([[S; N]; N]);

impl<S: KrylovScalar, const N: usize> LinearOperator for Matrix<S, N> {
    type Scalar = S;
    fn rows(&self) -> usize {
        N
    }
    fn columns(&self) -> usize {
        N
    }
    fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
        self.apply_oriented(Orientation::Normal, input, output)
    }
    fn diagonal(&self, output: &mut [S]) -> Result<DiagonalAvailability, Diagnostic> {
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.0[i][i];
        }
        Ok(DiagonalAvailability::Available)
    }
}
impl<S: KrylovScalar, const N: usize> OrientedLinearOperator for Matrix<S, N> {
    fn supports_orientation(&self, _: Orientation) -> bool {
        true
    }
    fn apply_oriented(
        &self,
        orientation: Orientation,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        if input.len() != N || output.len() != N {
            return Err(solve_failed("fixture shape mismatch"));
        }
        for (i, value) in output.iter_mut().enumerate() {
            *value = input.iter().enumerate().fold(S::zero(), |sum, (j, x)| {
                let a = match orientation {
                    Orientation::Normal => self.0[i][j],
                    Orientation::Transposed => self.0[j][i],
                    Orientation::ConjugateTransposed => self.0[j][i].conj(),
                };
                sum + a * *x
            });
        }
        Ok(())
    }
}

fn plan(algorithm: LinearSolver, preconditioner: PreconditionerPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-13, 1e-13, 16.try_into().unwrap())
        .unwrap()
        .with_reduction(ReductionPolicy::Fast)
        .with_preconditioner(preconditioner)
}
fn invoke<S: KrylovScalar>(
    problem: &LinearProblem<'_, S>,
    plan: SolverPlan,
) -> Result<LinearSolution<S>, Diagnostic> {
    let inverse = inverse_diagonal(problem, plan)?;
    match plan.algorithm() {
        LinearSolver::ConjugateGradient => cg::solve(FAER_SOLVER_PROVIDER, problem, plan, inverse),
        LinearSolver::BiConjugateGradientStabilized => {
            bicgstab::solve(FAER_SOLVER_PROVIDER, problem, plan, inverse)
        }
        _ => unreachable!("private Krylov fixture"),
    }
}
fn assert_values(actual: &[C], expected: &[C]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).norm() <= TOLERANCE,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn native_complex_cg_and_bicg_solve_normal_and_exact_adjoint_actions() {
    for algorithm in [
        LinearSolver::ConjugateGradient,
        LinearSolver::BiConjugateGradientStabilized,
    ] {
        let (a, properties, normal_rhs, adjoint_rhs) =
            if algorithm == LinearSolver::ConjugateGradient {
                // Leading minors4 and10 prove HPD, with non-real off-diagonals.
                (
                    HPD,
                    LinearOperatorProperties::HermitianPositiveDefinite,
                    [C::new(1., 7.), C::new(-3., 4.)],
                    [C::new(1., 7.), C::new(-3., 4.)],
                )
            } else {
                (
                    GENERAL,
                    LinearOperatorProperties::General,
                    [C::new(0., 10.), C::new(-6., 13.)],
                    [C::new(-1., 8.), C::new(-12., 6.)],
                )
            };
        let source = Matrix(a);
        for (orientation, rhs) in [
            (Orientation::Normal, normal_rhs),
            (Orientation::ConjugateTransposed, adjoint_rhs),
        ] {
            let oriented = Oriented::new(&source, orientation).unwrap();
            let problem = LinearProblem::new(&oriented, &rhs, properties).unwrap();
            for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
                let request = plan(algorithm, preconditioner);
                let solution = invoke(&problem, request).unwrap();
                assert_values(solution.values(), &X);
                assert_eq!(solution.report().orientation(), orientation);
                assert_eq!(solution.report().solver_plan(), request);
                assert_eq!(solution.report().solver_provider(), FAER_SOLVER_PROVIDER);
                assert!(
                    solution.report().completed_iterations() <= request.maximum_iterations().get()
                );
                assert!(
                    solution.report().true_residual_norm() <= solution.report().residual_target()
                );
            }
        }
    }
}

#[test]
fn faer_coefficient_conjugation_is_not_the_eqiora_adjoint() {
    let source = Matrix(GENERAL);
    let input = Mat::from_fn(2, 2, |i, j| {
        if j == 0 {
            X[i]
        } else {
            [C::new(1., 0.), C::new(0., 0.)][i]
        }
    });
    for (orientation, action, conjugated) in [
        (
            Orientation::Normal,
            [C::new(0., 10.), C::new(-6., 13.)],
            [C::new(0., 0.), C::new(-4., 7.)],
        ),
        (
            Orientation::ConjugateTransposed,
            [C::new(-1., 8.), C::new(-12., 6.)],
            [C::new(-7., 6.), C::new(-2., 6.)],
        ),
    ] {
        let oriented = Oriented::new(&source, orientation).unwrap();
        let failure = ApplyFailure::default();
        let adapter = FaerOperator::new(&oriented, failure.clone());
        let mut workspace = MemBuffer::new(adapter.apply_scratch(2, Par::Seq));
        let mut output = Mat::zeros(2, 2);
        adapter.apply(
            output.as_mut(),
            input.as_ref(),
            Par::Seq,
            MemStack::new(&mut workspace),
        );
        failure.take().unwrap();
        assert_values(output.col_as_slice(0), &action);
        adapter.conj_apply(
            output.as_mut(),
            input.as_ref(),
            Par::Seq,
            MemStack::new(&mut workspace),
        );
        failure.take().unwrap();
        assert_values(output.col_as_slice(0), &conjugated);
        let column = if orientation == Orientation::Normal {
            [C::new(2., -1.), C::new(3., -1.)]
        } else {
            [C::new(2., 1.), C::new(1., -2.)]
        };
        assert_values(output.col_as_slice(1), &column);
    }
}

#[test]
fn implicit_complex_parameter_product_and_real_dual_pairing_match_rationals() {
    // A(p)=GENERAL+p*diag(1,2), b fixed, x(0)=X. Thus A*x'=-diag(1,2)*X.
    let source = Matrix(GENERAL);
    let rhs = [C::new(-1., -2.), C::new(4., -2.)];
    let problem = LinearProblem::new(&source, &rhs, LinearOperatorProperties::General).unwrap();
    let request = plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Jacobi,
    );
    let tangent = invoke(&problem, request).unwrap();
    assert_values(
        tangent.values(),
        &[C::new(-3. / 65., 54. / 65.), C::new(93. / 65., -49. / 65.)],
    );
    let adjoint = Oriented::new(&source, Orientation::ConjugateTransposed).unwrap();
    let w = [C::new(1., -1.), C::new(2., 1.)];
    let reverse = invoke(
        &LinearProblem::new(&adjoint, &w, LinearOperatorProperties::General).unwrap(),
        request,
    )
    .unwrap();
    assert_values(
        reverse.values(),
        &[C::new(4. / 13., -6. / 13.), C::new(3. / 13., 2. / 13.)],
    );
    let pairing = w
        .iter()
        .zip(tangent.values())
        .map(|(w, dx)| (w.conj() * dx).re)
        .sum::<f64>();
    let gradient = reverse
        .values()
        .iter()
        .zip([X[0], C::new(2., 0.) * X[1]])
        .map(|(lambda, bx)| -(lambda.conj() * bx).re)
        .sum::<f64>();
    assert!((pairing - 16. / 13.).abs() < TOLERANCE);
    assert!((gradient - 16. / 13.).abs() < TOLERANCE);
}
