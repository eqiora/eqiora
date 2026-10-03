mod oriented;
pub use oriented::{Oriented, OrientedLinearOperator};

use std::fmt::Debug;
use std::ops::Range;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::csr::CanonicalCsrSystemView;

/// Result of requesting an operator diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalAvailability {
    /// Every diagonal value was written to the caller-owned output.
    Available,
    /// The operator does not expose a diagonal under this realization.
    Unavailable,
}

/// Orientation of the linear action accepted by a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearOperatorOrientation {
    /// Apply the operator as supplied.
    Normal,
    /// Apply the mathematical transpose of a source operator.
    Transposed,
    /// Apply the conjugate transpose (Euclidean adjoint), without a metric claim.
    ConjugateTransposed,
}

/// Independent action on a contiguous subset of output rows.
///
/// This is an optional host-local execution capability, not a distributed
/// layout. `input` is still the complete resident vector; `output` contains
/// exactly `rows.len()` values corresponding to the ordered global row range.
/// Disjoint ranges may be evaluated concurrently.
pub trait RowLinearAction: Debug + Sync {
    /// Value stored in each input and output coordinate.
    type Scalar;

    /// Compute one contiguous output-row range without allocating.
    ///
    /// # Errors
    /// Returns a numerical diagnostic for an invalid range, shape mismatch,
    /// or non-finite result.
    fn apply_rows(
        &self,
        rows: Range<usize>,
        input: &[Self::Scalar],
        output: &mut [Self::Scalar],
    ) -> Result<(), Diagnostic>;
}

/// A host-local linear action over complete typed finite vectors.
///
/// This v0 trait is intentionally not a distributed-vector abstraction. The
/// complete input is resident in one process, the caller owns both buffers,
/// and `apply` must not allocate merely to return its output.
pub trait LinearOperator: Debug + Sync {
    /// Arithmetic value in each mathematical coordinate.
    type Scalar;

    /// Output dimension.
    fn rows(&self) -> usize;

    /// Input dimension.
    fn columns(&self) -> usize;

    /// Compute `output = self * input`.
    ///
    /// # Errors
    /// Returns a numerical diagnostic for shape or non-finite failures.
    fn apply(&self, input: &[Self::Scalar], output: &mut [Self::Scalar]) -> Result<(), Diagnostic>;

    /// Expose independently executable row ranges when the realization has
    /// that capability.
    ///
    /// The default is an explicit absence. Threaded adapters must fail closed
    /// rather than silently running an unpartitionable operator serially.
    fn row_action(&self) -> Option<&dyn RowLinearAction<Scalar = Self::Scalar>> {
        None
    }

    /// Orientation represented by this callable action.
    ///
    /// Ordinary operators are normal. Eqiora's [`Oriented`] view overrides
    /// this metadata so solve evidence distinguishes `A x = b` from
    /// `A^T x = b` without duplicating the solver plan.
    fn orientation(&self) -> LinearOperatorOrientation {
        LinearOperatorOrientation::Normal
    }

    /// Write the operator diagonal when it is naturally available.
    ///
    /// The default reports `Unavailable` without modifying `output`.
    ///
    /// # Errors
    /// Implementations return a numerical diagnostic for shape or non-finite
    /// failures.
    fn diagonal(&self, _output: &mut [Self::Scalar]) -> Result<DiagonalAvailability, Diagnostic> {
        Ok(DiagonalAvailability::Unavailable)
    }
}

/// Mathematical properties asserted by the selected realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinearOperatorProperties {
    /// A square operator with no symmetry or definiteness assertion.
    General,
    /// A square symmetric positive-definite operator.
    SymmetricPositiveDefinite,
    /// A square symmetric operator known to be indefinite.
    SymmetricIndefinite,
}

/// One validated host-local linear problem.
#[derive(Debug)]
pub struct LinearProblem<'a> {
    operator: &'a dyn LinearOperator<Scalar = f64>,
    right_hand_side: &'a [f64],
    initial_guess: Option<&'a [f64]>,
    properties: LinearOperatorProperties,
    canonical_csr_system: Option<&'a CanonicalCsrSystemView>,
}

impl<'a> LinearProblem<'a> {
    /// Construct a square problem with an implicit zero initial guess.
    ///
    /// # Errors
    /// Returns `EQ0802` for empty/non-square shape, right-hand-side mismatch,
    /// or non-finite data.
    pub fn new(
        operator: &'a dyn LinearOperator<Scalar = f64>,
        right_hand_side: &'a [f64],
        properties: LinearOperatorProperties,
    ) -> Result<Self, Diagnostic> {
        if operator.rows() == 0 || operator.rows() != operator.columns() {
            return Err(solve_failed(
                "a linear solve requires a nonempty square operator",
            ));
        }
        if right_hand_side.len() != operator.rows() {
            return Err(solve_failed(format!(
                "operator has {} rows but the right-hand side has {} values",
                operator.rows(),
                right_hand_side.len()
            )));
        }
        if right_hand_side.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed("right-hand side must contain finite values"));
        }
        Ok(Self {
            operator,
            right_hand_side,
            initial_guess: None,
            properties,
            canonical_csr_system: None,
        })
    }

    pub(crate) fn from_canonical(system: &'a CanonicalCsrSystemView) -> Result<Self, Diagnostic> {
        let mut problem = Self::new(system, system.right_hand_side(), system.properties())?;
        problem.canonical_csr_system = Some(system);
        Ok(problem)
    }

    pub(crate) fn from_oriented_canonical(
        operator: &'a dyn LinearOperator<Scalar = f64>,
        system: &'a CanonicalCsrSystemView,
        right_hand_side: &'a [f64],
    ) -> Result<Self, Diagnostic> {
        if operator.rows() != system.rows() || operator.columns() != system.columns() {
            return Err(solve_failed(
                "oriented canonical action and captured CSR source must have identical dimensions",
            ));
        }
        let mut problem = Self::new(operator, right_hand_side, system.properties())?;
        problem.canonical_csr_system = Some(system);
        Ok(problem)
    }

    /// Attach an explicit initial guess.
    ///
    /// # Errors
    /// Returns `EQ0802` for a shape mismatch or non-finite value.
    pub fn with_initial_guess(mut self, initial_guess: &'a [f64]) -> Result<Self, Diagnostic> {
        if initial_guess.len() != self.operator.columns()
            || initial_guess.iter().any(|value| !value.is_finite())
        {
            return Err(solve_failed(
                "initial guess must match the operator and contain finite values",
            ));
        }
        self.initial_guess = Some(initial_guess);
        Ok(self)
    }

    /// Operator action.
    #[must_use]
    pub const fn operator(&self) -> &'a dyn LinearOperator<Scalar = f64> {
        self.operator
    }

    /// Right-hand-side values.
    #[must_use]
    pub const fn right_hand_side(&self) -> &'a [f64] {
        self.right_hand_side
    }

    /// Explicit initial guess, or `None` for the zero vector.
    #[must_use]
    pub const fn initial_guess(&self) -> Option<&'a [f64]> {
        self.initial_guess
    }

    /// Asserted mathematical properties.
    #[must_use]
    pub const fn properties(&self) -> LinearOperatorProperties {
        self.properties
    }

    /// Exact captured CSR coefficient source associated with this problem.
    ///
    /// The problem right-hand side may be derivative-specific and therefore
    /// differ from the source's captured primal right-hand side. Hand-built
    /// problems return `None`; a backend requiring materialized sparse storage
    /// must fail closed in that case.
    #[must_use]
    pub const fn canonical_csr_system(&self) -> Option<&'a CanonicalCsrSystemView> {
        self.canonical_csr_system
    }
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Rectangular;

    impl LinearOperator for Rectangular {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            3
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if input.len() != 3 || output.len() != 2 {
                return Err(solve_failed("normal test action shape mismatch"));
            }
            output[0] = input[0] + 2.0 * input[1];
            output[1] = 3.0 * input[1] + 4.0 * input[2];
            Ok(())
        }
    }

    impl OrientedLinearOperator for Rectangular {
        fn supports_orientation(&self, _orientation: crate::LinearOperatorOrientation) -> bool {
            true
        }

        fn apply_oriented(
            &self,
            orientation: crate::LinearOperatorOrientation,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), Diagnostic> {
            if orientation == crate::LinearOperatorOrientation::Normal {
                return self.apply(input, output);
            }
            if input.len() != 2 || output.len() != 3 {
                return Err(solve_failed("transpose test action shape mismatch"));
            }
            output[0] = input[0];
            output[1] = 2.0 * input[0] + 3.0 * input[1];
            output[2] = 4.0 * input[1];
            Ok(())
        }
    }

    #[test]
    fn transposed_view_swaps_spaces_and_uses_the_explicit_capability() {
        let transposed =
            Oriented::new(&Rectangular, crate::LinearOperatorOrientation::Transposed).unwrap();
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.columns(), 2);
        assert_eq!(
            transposed.orientation(),
            LinearOperatorOrientation::Transposed
        );
        let mut output = [0.0; 3];
        transposed.apply(&[5.0, 7.0], &mut output).unwrap();
        assert_eq!(output, [5.0, 31.0, 28.0]);
    }
}
