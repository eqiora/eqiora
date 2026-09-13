//! Explicit bounded active-set realization of finite affine real constraints.
//! Model conditions remain unchanged; numerical acceptance never implies exact satisfaction.
mod configuration;
mod preparation;
mod solve;
#[cfg(test)]
mod tests;

use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, entity::kinds};
use eqiora_schema::kernel::{ExprDag, RelationConditionKind, SymbolRef};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolveRequest, SolveReport, SolverPlan};

pub use configuration::{ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement};
pub(crate) use preparation::lower_finite_constraints;
pub(crate) use solve::solve_finite_constraints;

/// Original finite Model plus its explicitly selected enforcement policy.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FiniteConstraintProblem {
    kernel: KernelProgram,
    symbols: Vec<SymbolRef>,
    dimensions: Vec<DimExponents>,
    bindings: Vec<(SymbolRef, f64)>,
    relations: Vec<RelationOperands>,
    enforcement: FiniteConstraintEnforcement,
    complementarity_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct RelationOperands {
    id: Id<kinds::Relation>,
    expression: ExprDag,
    conditions: Vec<RelationConditionKind>,
    dimensions: Vec<(DimExponents, DimExponents)>,
}

impl FiniteConstraintProblem {
    /// Canonical scalar Field unknown ordering.
    #[must_use]
    pub fn symbols(&self) -> &[SymbolRef] {
        &self.symbols
    }
    /// Coherent SI dimensions in exact unknown ordering.
    #[must_use]
    pub fn dimensions(&self) -> &[DimExponents] {
        &self.dimensions
    }
    /// Explicit numerical enforcement; never part of Model meaning.
    #[must_use]
    pub const fn enforcement(&self) -> &FiniteConstraintEnforcement {
        &self.enforcement
    }
    /// Independently check original DAG conditions at a candidate vector.
    ///
    /// # Errors
    /// Rejects wrong/nonfinite vectors, mismatched original values, failed equalities,
    /// negative unilateral operands or unsupported approximate complementarity.
    pub fn validate_values(
        &self,
        values: &[f64],
        plan: SolverPlan,
        active_set_mask: u32,
    ) -> Result<ConstraintAssessment, Diagnostic> {
        solve::validate_values(self, values, plan, active_set_mask)
    }
}

/// Measured activity within the explicitly retained operand tolerances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintActivity {
    /// The first operand is zero within its own tolerance.
    Active,
    /// The second operand is zero within its own tolerance.
    Inactive,
    /// Both operands are zero within their respective tolerances.
    Biactive,
    /// One independently checked ordered inequality.
    Inequality,
}

/// One original condition measurement, with physical units and numerical acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintMeasurement {
    reference: ConstraintRef,
    left: DynQuantity,
    right: DynQuantity,
    activity: ConstraintActivity,
}
impl ConstraintMeasurement {
    /// Exact original Relation and condition ordinal.
    #[must_use]
    pub const fn reference(&self) -> ConstraintRef {
        self.reference
    }
    /// Original first operand, including its distinct physical dimension.
    #[must_use]
    pub const fn left(&self) -> DynQuantity {
        self.left
    }
    /// Original second operand, including its distinct physical dimension.
    #[must_use]
    pub const fn right(&self) -> DynQuantity {
        self.right
    }
    /// Approximate activity; this is not an exact symbolic proof.
    #[must_use]
    pub const fn activity(&self) -> ConstraintActivity {
        self.activity
    }
}

/// Independent reevaluation of equality residuals and original constraint operands.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConstraintAssessment {
    measurements: Vec<ConstraintMeasurement>,
    equality_residual_norm: f64,
    residual_target: f64,
    active_set_mask: u32,
}
impl ConstraintAssessment {
    /// Target recomputed from the exact selected branch RHS and SolverPlan.
    #[must_use]
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }
    /// Selected bounded active-set mask; its conditions are independently rechecked.
    #[must_use]
    pub const fn active_set_mask(&self) -> u32 {
        self.active_set_mask
    }
    /// Original constraints in canonical Relation/ordinal order.
    #[must_use]
    pub fn measurements(&self) -> &[ConstraintMeasurement] {
        &self.measurements
    }
    /// Coherent-SI numeric equality residual norm, checked against SolverPlan target.
    #[must_use]
    pub const fn equality_residual_norm(&self) -> f64 {
        self.equality_residual_norm
    }
}

/// Accepted finite candidate plus solver and original-Model evidence.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FiniteConstraintSolution {
    values: Vec<f64>,
    report: SolveReport,
    assessment: ConstraintAssessment,
}
impl FiniteConstraintSolution {
    /// Accepted values in the problem's exact symbol order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Selected linear backend's report, distinct from constraint evidence.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }
    /// Independently recomputed original conditions.
    #[must_use]
    pub const fn assessment(&self) -> &ConstraintAssessment {
        &self.assessment
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
fn failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::NUMERICAL_SOLVE_FAILED,
        message,
    )
}
