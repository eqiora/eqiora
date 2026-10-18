use std::fmt::Debug;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;

use crate::{ExecutionId, ExecutionProvider, ExecutionReport, LinearOperator, ReductionPolicy};

/// One fixed-order conjugate-first inner product lowered into independent partials.
/// Real and complex binary64 values share the same indexed reduction tree.
///
/// Execution adapters may evaluate partials concurrently, but their indices,
/// local left-to-right arithmetic, and final left-to-right fold are fixed by
/// this Eqiora-owned action.
#[derive(Debug, Clone, Copy)]
pub struct FixedOrderInnerProduct<'a, S = f64> {
    left: &'a [S],
    right: &'a [S],
}

impl<'a, S: Scalar + ComplexFloat<Real = f64>> FixedOrderInnerProduct<'a, S> {
    /// Logical element count in each reproducible partial, independent of scheduling.
    /// Serial and parallel executions use these same partials and final fold.
    pub const CHUNK_LENGTH: usize = 1_024;

    /// Bind equal-length complete resident vectors.
    ///
    /// # Errors
    /// Returns `EQ0802` when the vector shapes differ.
    pub fn new(left: &'a [S], right: &'a [S]) -> Result<Self, Diagnostic> {
        if left.len() != right.len() {
            return Err(solve_failed(format!(
                "inner product received vector sizes {} and {}",
                left.len(),
                right.len()
            )));
        }
        Ok(Self { left, right })
    }

    /// Number of fixed logical partials.
    #[must_use]
    pub fn partial_count(self) -> usize {
        self.left.len().div_ceil(Self::CHUNK_LENGTH)
    }

    /// Evaluate one logical partial in its fixed local order.
    ///
    /// # Errors
    /// Returns `EQ0802` for an invalid partial index or non-finite arithmetic.
    pub fn evaluate_partial(self, index: usize) -> Result<S, Diagnostic> {
        if index >= self.partial_count() {
            return Err(solve_failed(format!(
                "inner-product partial {index} is outside {} partial(s)",
                self.partial_count()
            )));
        }
        let start = index * Self::CHUNK_LENGTH;
        let end = (start + Self::CHUNK_LENGTH).min(self.left.len());
        self.left[start..end]
            .iter()
            .zip(&self.right[start..end])
            .try_fold(S::zero(), |sum, (left, right)| {
                let next = sum + left.conj() * *right;
                (next.re().is_finite() && next.im().is_finite())
                    .then_some(next)
                    .ok_or_else(|| solve_failed("linear reduction overflowed"))
            })
    }

    /// Compose indexed partials in the fixed final order.
    ///
    /// # Errors
    /// Returns `EQ0802` for a partial-count mismatch or non-finite arithmetic.
    pub fn finish(self, partials: &[S]) -> Result<S, Diagnostic> {
        if partials.len() != self.partial_count() {
            return Err(solve_failed(format!(
                "inner product requires {} partial(s), but received {}",
                self.partial_count(),
                partials.len()
            )));
        }
        partials.iter().try_fold(S::zero(), |sum, partial| {
            let next = sum + *partial;
            (next.re().is_finite() && next.im().is_finite())
                .then_some(next)
                .ok_or_else(|| solve_failed("linear reduction overflowed"))
        })
    }

    /// Evaluate a Euclidean norm using the execution's fixed-order inner-product action.
    ///
    /// The norm and finite-real check remain owned here, independently of the adapter.
    ///
    /// # Errors
    /// Rejects non-finite, nonreal, or negative squared norms from the execution.
    pub fn norm(
        execution: &dyn ReplicatedLinearExecution<S>,
        values: &[S],
    ) -> Result<f64, Diagnostic> {
        let squared = execution.inner_product(FixedOrderInnerProduct::new(values, values)?)?;
        if !squared.re().is_finite() || squared.im() != 0.0 || squared.re() < 0.0 {
            return Err(solve_failed(
                "squared Euclidean norm must be finite, real, and nonnegative",
            ));
        }
        Ok(squared.re().sqrt())
    }

    fn evaluate_serial(self) -> Result<S, Diagnostic> {
        let partials = (0..self.partial_count())
            .map(|index| self.evaluate_partial(index))
            .collect::<Result<Vec<_>, _>>()?;
        self.finish(&partials)
    }
}

/// Execution of complete host-local vectors under one resolved placement.
///
/// The mathematical operator, solver algorithm, and convergence policy remain
/// separate. This trait is intentionally not a distributed-vector or device-
/// residency abstraction.
pub trait ReplicatedLinearExecution<S = f64>: Debug + Sync {
    /// Stable identity and declared release/dependency inventory of this provider.
    fn provider(&self) -> ExecutionProvider;

    /// Placement evidence for every action performed through this execution.
    fn report(&self) -> ExecutionReport;

    /// Validate a floating-point reduction policy before numerical work.
    ///
    /// # Errors
    /// Returns `EQ0807` when the execution cannot honor the selected policy.
    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic>;

    /// Apply one operator under this placement.
    ///
    /// # Errors
    /// Returns a numerical or capability diagnostic from the execution or
    /// operator.
    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = S>,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic>;

    /// Evaluate the Eqiora-owned fixed-order inner-product action.
    ///
    /// # Errors
    /// Returns a numerical diagnostic for invalid shape, ordering, or finite
    /// arithmetic.
    fn inner_product(&self, action: FixedOrderInnerProduct<'_, S>) -> Result<S, Diagnostic>;
}

/// Direct one-worker replicated execution used by the reference path.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialLinearExecution;

/// Shared direct one-worker execution.
pub const SERIAL_LINEAR_EXECUTION: SerialLinearExecution = SerialLinearExecution;

/// Exact declared release identity of direct one-worker execution.
pub const SERIAL_EXECUTION_PROVIDER: ExecutionProvider = ExecutionProvider::new(
    ExecutionId::new("eqiora.host.serial"),
    env!("CARGO_PKG_VERSION"),
    &[],
);

impl SerialLinearExecution {
    /// Exact release identity, independent of the scalar domain.
    #[must_use]
    pub const fn provider(&self) -> ExecutionProvider {
        SERIAL_EXECUTION_PROVIDER
    }

    /// Direct one-worker placement, independent of the scalar domain.
    #[must_use]
    pub const fn report(&self) -> ExecutionReport {
        ExecutionReport::host_serial()
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + Sync> ReplicatedLinearExecution<S>
    for SerialLinearExecution
{
    fn provider(&self) -> ExecutionProvider {
        self.provider()
    }

    fn report(&self) -> ExecutionReport {
        self.report()
    }

    fn require_reduction(&self, _policy: ReductionPolicy) -> Result<(), Diagnostic> {
        Ok(())
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = S>,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        operator.apply(input, output)
    }

    fn inner_product(&self, action: FixedOrderInnerProduct<'_, S>) -> Result<S, Diagnostic> {
        action.evaluate_serial()
    }
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_order_inner_product_has_an_explicit_expression_tree() {
        let mut right = vec![1.0; 2 * FixedOrderInnerProduct::<f64>::CHUNK_LENGTH + 1];
        right[0] = 1.0e16;
        right[FixedOrderInnerProduct::<f64>::CHUNK_LENGTH] = -1.0e16;
        let left = vec![1.0; right.len()];
        let action = FixedOrderInnerProduct::new(&left, &right).unwrap();

        assert_eq!(action.partial_count(), 3);
        assert_eq!(SERIAL_LINEAR_EXECUTION.inner_product(action).unwrap(), 1.0);
        let reassociated = left
            .iter()
            .zip(&right)
            .fold(0.0, |sum, (left, right)| sum + left * right);
        assert_eq!(reassociated, 1_024.0);
    }

    #[test]
    fn fixed_order_inner_product_rejects_shape_and_nonfinite_arithmetic() {
        assert_eq!(
            FixedOrderInnerProduct::new(&[1.0], &[]).unwrap_err().code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
        let action = FixedOrderInnerProduct::new(&[f64::MAX], &[2.0]).unwrap();
        assert_eq!(
            SERIAL_LINEAR_EXECUTION
                .inner_product(action)
                .unwrap_err()
                .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
    }
    #[test]
    fn complex_inner_product_preserves_conjugate_pairing_and_finite_components() {
        use num_complex::Complex64 as C;
        let left = [C::new(1.0, 2.0), C::new(3.0, -4.0)];
        let right = [C::new(5.0, -6.0), C::new(7.0, 8.0)];
        // (1-2i)(5-6i) + (3+4i)(7+8i) = -18+36i.
        let paired = SERIAL_LINEAR_EXECUTION
            .inner_product(FixedOrderInnerProduct::new(&left, &right).unwrap())
            .unwrap();
        assert_eq!(paired, C::new(-18.0, 36.0));
        let squared = SERIAL_LINEAR_EXECUTION
            .inner_product(FixedOrderInnerProduct::new(&left, &left).unwrap())
            .unwrap();
        assert_eq!(squared, C::new(30.0, 0.0));
        let invalid = [C::new(0.0, f64::INFINITY)];
        assert!(
            SERIAL_LINEAR_EXECUTION
                .inner_product(FixedOrderInnerProduct::new(&invalid, &invalid).unwrap())
                .is_err()
        );
    }
}
