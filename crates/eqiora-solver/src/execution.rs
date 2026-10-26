use std::fmt::Debug;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar, ScalarDomain};
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
    norm_scale: Option<f64>,
}

impl<'a, S: Scalar + ComplexFloat<Real = f64>> FixedOrderInnerProduct<'a, S> {
    /// Logical element count per inner-product partial, independent of scheduling.
    /// Norms count real coordinates, so complex norms use half as many elements.
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
        Ok(Self {
            left,
            right,
            norm_scale: None,
        })
    }

    /// Number of fixed logical partials.
    #[must_use]
    pub fn partial_count(self) -> usize {
        self.left.len().div_ceil(self.chunk_length())
    }

    fn chunk_length(self) -> usize {
        if self.norm_scale.is_some() && S::DOMAIN == ScalarDomain::Complex {
            Self::CHUNK_LENGTH / 2
        } else {
            Self::CHUNK_LENGTH
        }
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
        let start = index * self.chunk_length();
        let end = (start + self.chunk_length()).min(self.left.len());
        self.left[start..end]
            .iter()
            .zip(&self.right[start..end])
            .try_fold(S::zero(), |sum, (left, right)| {
                let next = match self.norm_scale {
                    Some(scale) => {
                        // Scale real components directly: complex division by a
                        // tiny real denominator can itself square that denominator.
                        let real = left.re() / scale;
                        let imaginary = left.im() / scale;
                        // Match adjacent real/imaginary coordinates exactly,
                        // including chunk boundaries; do not group each pair first.
                        S::from((sum.re() + real * real) + imaginary * imaginary)
                            .expect("binary64 real component")
                    }
                    None => sum + left.conj() * *right,
                };
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
    /// Components are scaled before squaring, using the same indexed reduction
    /// for every execution. No temporary scaled vector is allocated. Ordinary inner
    /// products retain their original arithmetic. The norm and finite-real check
    /// remain owned here, independently of the adapter. Complex values use the
    /// exact order of their adjacent real/imaginary coordinates, including partial
    /// boundaries, so coordinate conversion preserves the norm bit for bit.
    ///
    /// # Errors
    /// Rejects non-finite inputs, non-finite/nonreal/negative scaled squared norms
    /// from the execution, or a norm outside the binary64 range.
    pub fn norm(
        execution: &dyn ReplicatedLinearExecution<S>,
        values: &[S],
    ) -> Result<f64, Diagnostic> {
        let (action, scale) = Self::norm_action(values)?;
        Self::finish_norm(execution.inner_product(action)?, scale)
    }

    /// Evaluate the same serial Euclidean norm with admitted partial buffers.
    ///
    /// This allocates no workspace and shares the exact scaled action and final
    /// arithmetic with [`Self::norm`]. Receipt replay can therefore remain
    /// independent of a producer without duplicating its norm formula.
    ///
    /// # Errors
    /// Returns the errors of [`Self::norm`], or rejects a partial count different
    /// from the real-coordinate count divided by [`Self::CHUNK_LENGTH`] (rounded
    /// up) before writing the buffer. Each complex value has two real coordinates.
    pub fn norm_with_partials(values: &[S], partials: &mut [S]) -> Result<f64, Diagnostic> {
        let (action, scale) = Self::norm_action(values)?;
        if partials.len() != action.partial_count() {
            return Err(solve_failed(
                "Euclidean norm partial workspace has the wrong shape",
            ));
        }
        for (index, partial) in partials.iter_mut().enumerate() {
            *partial = action.evaluate_partial(index)?;
        }
        Self::finish_norm(action.finish(partials)?, scale)
    }

    fn norm_action(values: &[S]) -> Result<(FixedOrderInnerProduct<'_, S>, f64), Diagnostic> {
        let scale = values.iter().try_fold(0_f64, |scale, value| {
            if !value.is_finite() {
                return Err(solve_failed("Euclidean norm input must be finite"));
            }
            Ok(scale.max(value.re().abs()).max(value.im().abs()))
        })?;
        let scale = if scale == 0.0 { 1.0 } else { scale };
        Ok((
            FixedOrderInnerProduct {
                left: values,
                right: values,
                norm_scale: Some(scale),
            },
            scale,
        ))
    }

    fn finish_norm(squared: S, scale: f64) -> Result<f64, Diagnostic> {
        if !squared.re().is_finite() || squared.im() != 0.0 || squared.re() < 0.0 {
            return Err(solve_failed(
                "squared Euclidean norm must be finite, real, and nonnegative",
            ));
        }
        let norm = squared.re().sqrt() * scale;
        if !norm.is_finite() {
            return Err(solve_failed("Euclidean norm exceeds the binary64 range"));
        }
        Ok(norm)
    }

    fn evaluate_serial(self) -> Result<S, Diagnostic> {
        // Request exactly the known numeric payload; Result collection has no
        // exact-size lower bound and can otherwise grow its allocation.
        let mut partials = Vec::with_capacity(self.partial_count());
        for index in 0..self.partial_count() {
            partials.push(self.evaluate_partial(index)?);
        }
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
    fn explicitly_sized_serial_partials_keep_cancellation_and_boundary_bits() {
        for (count, expected) in [
            (0, 0_f64),
            (1, 1e16),
            (2, 1e16),
            (3, 0.),
            (4, 3.),
            (1023, 3.),
            (1024, 3.),
            (1025, 8.),
            (2049, 21.),
        ] {
            let left = vec![1.; count];
            let mut right = vec![0.; count];
            for (index, value) in [(0, 1e16), (1, 1.), (2, -1e16), (3, 3.), (1024, 5.)] {
                if index < count {
                    right[index] = value;
                }
            }
            if count == 2049 {
                right[1024..1028].copy_from_slice(&[1e16, 1., -1e16, 7.]);
                right[2048] = 11.;
            }
            // The local left-to-right folds lose each isolated +1 between
            // +/-1e16. The final fold is 3+5, or 3+7+11 at two boundaries.
            assert_eq!(
                FixedOrderInnerProduct::new(&left, &right)
                    .unwrap()
                    .evaluate_serial()
                    .unwrap()
                    .to_bits(),
                expected.to_bits()
            );
        }
        assert!(
            FixedOrderInnerProduct::new(&[f64::MAX; 2], &[1.; 2])
                .unwrap()
                .evaluate_serial()
                .is_err()
        );
    }

    #[test]
    fn norm_is_identical_in_complex_and_real_coordinates() {
        use num_complex::Complex64 as C;
        // Coordinate conversion is an exact Euclidean isometry. Exercise both
        // local summation and partial boundaries with unequal components.
        for count in [2, 511, 512, 513, 1025] {
            let complex = (0..count)
                .map(|i| C::new(1., if i % 2 == 0 { 1. } else { 3. }))
                .collect::<Vec<_>>();
            let real = complex
                .iter()
                .flat_map(|z| [z.re, z.im])
                .collect::<Vec<_>>();
            let expected = FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &real).unwrap();
            assert_eq!(
                FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &complex)
                    .unwrap()
                    .to_bits(),
                expected.to_bits(),
                "{count} complex coordinates",
            );
            let mut partials = vec![C::new(0., 0.); real.len().div_ceil(1024)];
            assert_eq!(
                FixedOrderInnerProduct::norm_with_partials(&complex, &mut partials)
                    .unwrap()
                    .to_bits(),
                expected.to_bits(),
            );
        }
    }

    #[test]
    fn norm_preserves_dyadic_scales_and_complex_components() {
        use num_complex::Complex64 as C;
        // The 3-4-5 triangle is exact at every selected binary scale.
        for scale in [
            f64::from_bits(16), // 2^-1070, represented without an intermediate reciprocal.
            2_f64.powi(-700),
            1.,
            2_f64.powi(700),
            2_f64.powi(1020),
        ] {
            let real = [3. * scale, 4. * scale];
            let complex = [C::new(3. * scale, 4. * scale)];
            assert_eq!(
                FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &real).unwrap(),
                5. * scale,
            );
            assert_eq!(
                FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &complex).unwrap(),
                5. * scale,
            );
            assert_eq!(
                FixedOrderInnerProduct::norm_with_partials(&real, &mut [0.]).unwrap(),
                5. * scale,
            );
            assert_eq!(
                FixedOrderInnerProduct::norm_with_partials(&complex, &mut [C::new(0., 0.)])
                    .unwrap(),
                5. * scale,
            );
        }
        let mut wrong_shape = [7., 9.];
        assert!(FixedOrderInnerProduct::norm_with_partials(&[3., 4.], &mut wrong_shape).is_err());
        assert_eq!(wrong_shape, [7., 9.]);
        assert_eq!(
            FixedOrderInnerProduct::<f64>::norm_with_partials(&[], &mut []).unwrap(),
            0.
        );
        let smallest = f64::from_bits(1);
        assert_eq!(
            FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[smallest]).unwrap(),
            smallest,
        );
        assert_eq!(
            FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[f64::MAX]).unwrap(),
            f64::MAX,
        );
        assert!(FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[f64::MAX; 2]).is_err());
        for invalid in [f64::INFINITY, f64::NAN] {
            assert!(FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[invalid]).is_err());
            assert!(
                FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &[C::new(0., invalid)],)
                    .is_err()
            );
        }
    }

    #[test]
    fn acceptance_distinguishes_zero_from_tiny_real_and_complex_solutions() {
        use crate::{
            ConvergenceReason, LinearOperatorProperties, LinearProblem, LinearSolver,
            REFERENCE_SOLVER_PROVIDER, SolverPlan, accept_linear_solution,
        };
        use num_complex::Complex64 as C;
        use std::{marker::PhantomData, num::NonZeroUsize};

        #[derive(Debug)]
        struct Identity<S>(PhantomData<S>);
        impl<S: Scalar + ComplexFloat<Real = f64> + Sync> LinearOperator for Identity<S> {
            type Scalar = S;
            fn rows(&self) -> usize {
                1
            }
            fn columns(&self) -> usize {
                1
            }
            fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
                output.copy_from_slice(input);
                Ok(())
            }
        }

        fn check<S: Scalar + ComplexFloat<Real = f64> + Sync>(rhs: S, norm: f64) {
            let operator = Identity::<S>(PhantomData);
            let right = [rhs];
            let problem =
                LinearProblem::new(&operator, &right, LinearOperatorProperties::General).unwrap();
            let plan = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                2_f64.powi(-20),
                0.,
                NonZeroUsize::MIN,
            )
            .unwrap();
            let accept = |value| {
                accept_linear_solution(
                    &problem,
                    plan,
                    REFERENCE_SOLVER_PROVIDER,
                    ConvergenceReason::ResidualToleranceSatisfied,
                    1,
                    0.,
                    vec![value],
                )
            };
            assert!(accept(S::zero()).is_err());
            let solved = accept(rhs).unwrap();
            assert_eq!(solved.report().initial_residual_norm(), norm);
            assert_eq!(solved.report().residual_target(), norm * 2_f64.powi(-20));
            assert_eq!(solved.report().true_residual_norm(), 0.);
        }
        for exponent in [-700, 0, 700] {
            let scale = 2_f64.powi(exponent);
            check(scale, scale);
            check(C::new(3. * scale, 4. * scale), 5. * scale);
        }
    }

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
