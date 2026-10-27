//! Shared typed boundary for Faer's real and complex native Krylov kernels.
use super::*;
use eqiora_core::Scalar;
use faer::complex::ComplexFloat;
use faer::dyn_stack::StackReq;
use faer::mat::{AsMatMut, MatMut, MatRef};
use faer::matrix_free::{LinOp, Precond};
use faer::reborrow::ReborrowMut;
use faer::traits::ComplexField;

pub(super) trait KrylovScalar:
    Scalar + ComplexFloat<Real = f64> + ComplexField<Real = f64> + Sync
{
}
impl<S: Scalar + ComplexFloat<Real = f64> + ComplexField<Real = f64> + Sync> KrylovScalar for S {}

pub(super) fn inverse_diagonal<S: KrylovScalar>(
    problem: &LinearProblem<'_, S>,
    plan: SolverPlan,
) -> Result<Option<Vec<S>>, Diagnostic> {
    if plan.preconditioner() == PreconditionerPolicy::Identity {
        return Ok(None);
    }
    let mut diagonal = vec![S::zero(); problem.operator().rows()];
    if problem.operator().diagonal(&mut diagonal)? == DiagonalAvailability::Unavailable {
        return Err(invalid_realization(
            "faer Jacobi preconditioning requires an available operator diagonal",
        ));
    }
    if diagonal.iter().any(|value| {
        !value.is_finite()
            || *value == S::zero()
            || (plan.algorithm() == LinearSolver::ConjugateGradient
                && (value.im() != 0. || value.re() <= 0.))
    }) {
        return Err(solve_failed(
            "Jacobi diagonal must be finite and nonzero, and real positive for CG",
        ));
    }
    // Faer's typed reciprocal scales complex components before squaring.
    // No real-coordinate block diagonal or second preconditioner is introduced.
    let inverse = diagonal.iter().map(S::recip_impl).collect::<Vec<_>>();
    if inverse
        .iter()
        .any(|value| !value.is_finite() || *value == S::zero())
    {
        return Err(solve_failed("Jacobi reciprocal must be finite and nonzero"));
    }
    Ok(Some(inverse))
}

pub(super) fn initial_matrix<S: KrylovScalar>(problem: &LinearProblem<'_, S>) -> Mat<S> {
    Mat::from_fn(problem.operator().columns(), 1, |row, _| {
        problem
            .initial_guess()
            .map_or(S::zero(), |values| values[row])
    })
}

pub(super) fn right_hand_side_matrix<S: KrylovScalar>(problem: &LinearProblem<'_, S>) -> Mat<S> {
    Mat::from_fn(problem.operator().rows(), 1, |row, _| {
        problem.right_hand_side()[row]
    })
}

#[derive(Debug, Clone)]
pub(super) struct FaerOperator<'a, S = f64> {
    operator: &'a dyn LinearOperator<Scalar = S>,
    failure: ApplyFailure,
}

impl<'a, S: KrylovScalar> FaerOperator<'a, S> {
    pub(super) const fn new(
        operator: &'a dyn LinearOperator<Scalar = S>,
        failure: ApplyFailure,
    ) -> Self {
        Self { operator, failure }
    }

    fn apply_typed(
        &self,
        mut output: MatMut<'_, S>,
        input: MatRef<'_, S>,
        conjugate: bool,
        stack: &mut MemStack,
    ) {
        output.fill(S::nan_impl());
        let Some(input) = input.try_as_col_major() else {
            self.failure.record(solve_failed(
                "faer supplied a non-contiguous operator input",
            ));
            return;
        };
        let Some(mut output) = output.try_as_col_major_mut() else {
            self.failure.record(solve_failed(
                "faer supplied a non-contiguous operator output",
            ));
            return;
        };
        // The exact Eqiora action already owns Normal/Transpose/Adjoint.
        // Faer's conj_apply asks for coefficient conjugation of that action:
        // conj(L conj(x)), not its adjoint. Reuse one advertised scratch column.
        if conjugate && S::DOMAIN == ScalarDomain::Complex {
            let (mut conjugated, _) =
                faer::linalg::temp_mat_zeroed::<S, _, _>(self.operator.columns(), 1, stack);
            for column in 0..input.ncols() {
                let input_column = input.col(column).as_slice();
                if input_column.iter().any(|value| !value.is_finite()) {
                    self.failure.record_nonfinite_input();
                    return;
                }
                let mut values = conjugated
                    .as_mat_mut()
                    .try_as_col_major_mut()
                    .expect("faer temporary column is contiguous");
                for (row, value) in input_column.iter().enumerate() {
                    values[(row, 0)] = value.conj();
                }
                let output_column = output.rb_mut().col_mut(column).as_slice_mut();
                if let Err(diagnostic) = self
                    .operator
                    .apply(values.rb_mut().col_mut(0).as_slice_mut(), output_column)
                {
                    self.failure.record(diagnostic);
                    return;
                }
                for value in output_column {
                    *value = value.conj();
                }
            }
        } else {
            for column in 0..input.ncols() {
                let input_column = input.col(column).as_slice();
                let output_column = output.rb_mut().col_mut(column).as_slice_mut();
                if input_column.iter().any(|value| !value.is_finite()) {
                    self.failure.record_nonfinite_input();
                    return;
                }
                if let Err(diagnostic) = self.operator.apply(input_column, output_column) {
                    self.failure.record(diagnostic);
                    return;
                }
            }
        }
    }
}

impl<S: KrylovScalar> LinOp<S> for FaerOperator<'_, S> {
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        if S::DOMAIN == ScalarDomain::Complex {
            faer::linalg::temp_mat_scratch::<S>(self.operator.columns(), 1)
        } else {
            StackReq::EMPTY
        }
    }
    fn nrows(&self) -> usize {
        self.operator.rows()
    }
    fn ncols(&self) -> usize {
        self.operator.columns()
    }
    fn apply(&self, output: MatMut<'_, S>, input: MatRef<'_, S>, _: Par, stack: &mut MemStack) {
        self.apply_typed(output, input, false, stack);
    }
    fn conj_apply(
        &self,
        output: MatMut<'_, S>,
        input: MatRef<'_, S>,
        _: Par,
        stack: &mut MemStack,
    ) {
        self.apply_typed(output, input, true, stack);
    }
}

#[derive(Debug, Clone)]
pub(super) struct FaerPreconditioner<S = f64> {
    dimension: usize,
    inverse_diagonal: Option<Vec<S>>,
}

impl<S: KrylovScalar> FaerPreconditioner<S> {
    pub(super) const fn new(dimension: usize, inverse_diagonal: Option<Vec<S>>) -> Self {
        Self {
            dimension,
            inverse_diagonal,
        }
    }
    fn apply_typed(&self, mut output: MatMut<'_, S>, input: MatRef<'_, S>, conjugate: bool) {
        for column in 0..input.ncols() {
            for row in 0..self.dimension {
                let inverse = self
                    .inverse_diagonal
                    .as_ref()
                    .map_or(S::one(), |values| values[row]);
                output[(row, column)] =
                    input[(row, column)] * if conjugate { inverse.conj() } else { inverse };
            }
        }
    }
}

impl<S: KrylovScalar> LinOp<S> for FaerPreconditioner<S> {
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn nrows(&self) -> usize {
        self.dimension
    }
    fn ncols(&self) -> usize {
        self.dimension
    }
    fn apply(&self, output: MatMut<'_, S>, input: MatRef<'_, S>, _: Par, _: &mut MemStack) {
        self.apply_typed(output, input, false);
    }
    fn conj_apply(&self, output: MatMut<'_, S>, input: MatRef<'_, S>, _: Par, _: &mut MemStack) {
        self.apply_typed(output, input, true);
    }
}
impl<S: KrylovScalar> Precond<S> for FaerPreconditioner<S> {}
