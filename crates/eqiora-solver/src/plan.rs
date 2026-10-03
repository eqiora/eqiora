use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

/// Linear algorithm selected independently from model meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinearSolver {
    /// Conjugate gradients for an asserted symmetric positive-definite operator.
    ConjugateGradient,
    /// Minimum residual iteration for a symmetric definite or indefinite operator.
    MinimumResidual,
    /// Bi-conjugate gradient stabilized for a square general operator.
    BiConjugateGradientStabilized,
    /// Partial-pivot sparse LU for an explicitly captured square operator.
    SparseLu,
}

/// Preconditioner policy selected independently from the linear algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PreconditionerPolicy {
    /// No transformation of the residual.
    Identity,
    /// Inverse of the operator diagonal.
    Jacobi,
}

/// Floating-point reduction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReductionPolicy {
    /// A specified stable reduction order, independent of worker availability.
    Reproducible,
    /// The backend-native reduction order, which may vary with placement.
    Fast,
}

/// Complete numerical policy for one linear solve.
///
/// This is the only solver-control type. Numerical realization crates consume
/// it directly; they do not translate it into method-specific configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverPlan {
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
}

impl SolverPlan {
    /// Construct a validated plan with identity preconditioning and a
    /// reproducible reduction policy.
    ///
    /// # Errors
    /// Returns `EQ0807` for non-finite, negative, or jointly zero tolerances.
    pub fn new(
        algorithm: LinearSolver,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
    ) -> Result<Self, Diagnostic> {
        if !relative_tolerance.is_finite()
            || !absolute_tolerance.is_finite()
            || relative_tolerance < 0.0
            || absolute_tolerance < 0.0
            || (relative_tolerance == 0.0 && absolute_tolerance == 0.0)
        {
            return Err(invalid_plan(
                "solver tolerances must be finite and non-negative, with at least one positive",
            ));
        }
        Ok(Self {
            algorithm,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
        })
    }

    /// Select a preconditioner without changing the other controls.
    #[must_use]
    pub const fn with_preconditioner(mut self, preconditioner: PreconditionerPolicy) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// Select a reduction policy without changing the other controls.
    #[must_use]
    pub const fn with_reduction(mut self, reduction: ReductionPolicy) -> Self {
        self.reduction = reduction;
        self
    }

    /// Solver algorithm.
    #[must_use]
    pub const fn algorithm(self) -> LinearSolver {
        self.algorithm
    }

    /// Preconditioner policy.
    #[must_use]
    pub const fn preconditioner(self) -> PreconditionerPolicy {
        self.preconditioner
    }

    /// Floating-point reduction policy.
    #[must_use]
    pub const fn reduction(self) -> ReductionPolicy {
        self.reduction
    }

    /// Relative convergence tolerance.
    #[must_use]
    pub const fn relative_tolerance(self) -> f64 {
        self.relative_tolerance
    }

    /// Absolute convergence tolerance.
    #[must_use]
    pub const fn absolute_tolerance(self) -> f64 {
        self.absolute_tolerance
    }

    /// Iteration limit or direct-solve work bound.
    ///
    /// The `eqiora.reference` minimum-residual implementation caps its retained
    /// full-H projection at the smaller of this value and the operator
    /// dimension. Other providers interpret this common bound independently.
    ///
    /// Sparse LU retains this field as part of the common plan identity,
    /// ignores it as a factorization control, and reports at most one completed
    /// factor-and-solve attempt.
    #[must_use]
    pub const fn maximum_iterations(self) -> NonZeroUsize {
        self.maximum_iterations
    }

    /// Accepted Euclidean residual threshold for a right-hand-side norm.
    ///
    /// # Errors
    /// Returns `EQ0802` if the norm is negative/non-finite or the target
    /// overflows.
    pub fn residual_target(self, right_hand_side_norm: f64) -> Result<f64, Diagnostic> {
        if !right_hand_side_norm.is_finite() || right_hand_side_norm < 0.0 {
            return Err(solve_failed(
                "right-hand-side norm must be finite and non-negative",
            ));
        }
        let target = self
            .absolute_tolerance
            .max(self.relative_tolerance * right_hand_side_norm);
        if !target.is_finite() {
            return Err(solve_failed("linear residual target overflowed"));
        }
        Ok(target)
    }
}

impl LinearSolver {
    /// Whether this algorithm is mathematically admissible for an asserted operator class.
    ///
    /// This is independent of whether a particular backend implements the
    /// requested scalar, preconditioner, or reduction tuple.
    #[must_use]
    pub const fn accepts(self, properties: crate::LinearOperatorProperties) -> bool {
        match self {
            Self::ConjugateGradient => matches!(
                properties,
                crate::LinearOperatorProperties::SymmetricPositiveDefinite
                    | crate::LinearOperatorProperties::HermitianPositiveDefinite
            ),
            Self::MinimumResidual => matches!(
                properties,
                crate::LinearOperatorProperties::SymmetricPositiveDefinite
                    | crate::LinearOperatorProperties::SymmetricIndefinite
                    | crate::LinearOperatorProperties::Symmetric
                    | crate::LinearOperatorProperties::Hermitian
                    | crate::LinearOperatorProperties::HermitianPositiveDefinite
            ),
            Self::BiConjugateGradientStabilized => true,
            Self::SparseLu => true,
        }
    }
}

fn invalid_plan(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_is_the_complete_validated_control() {
        let plan = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-8,
            1.0e-12,
            NonZeroUsize::new(300).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Jacobi)
        .with_reduction(ReductionPolicy::Fast);
        assert_eq!(
            plan.algorithm(),
            LinearSolver::BiConjugateGradientStabilized
        );
        assert_eq!(plan.preconditioner(), PreconditionerPolicy::Jacobi);
        assert_eq!(plan.reduction(), ReductionPolicy::Fast);
        assert_eq!(plan.maximum_iterations().get(), 300);
    }

    #[test]
    fn invalid_tolerances_fail_at_the_plan_boundary() {
        let error = SolverPlan::new(LinearSolver::ConjugateGradient, 0.0, 0.0, NonZeroUsize::MIN)
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
    }

    #[test]
    fn algorithm_admission_is_one_backend_neutral_mathematical_rule() {
        assert!(
            LinearSolver::ConjugateGradient
                .accepts(crate::LinearOperatorProperties::SymmetricPositiveDefinite)
        );
        assert!(
            !LinearSolver::ConjugateGradient
                .accepts(crate::LinearOperatorProperties::SymmetricIndefinite)
        );
        assert!(
            LinearSolver::MinimumResidual
                .accepts(crate::LinearOperatorProperties::SymmetricIndefinite)
        );
        assert!(
            LinearSolver::BiConjugateGradientStabilized
                .accepts(crate::LinearOperatorProperties::General)
        );
        assert!(LinearSolver::SparseLu.accepts(crate::LinearOperatorProperties::General));
    }
}
