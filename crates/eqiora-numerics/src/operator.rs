use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_meshing::QuadratureRule;

/// Pure entity-local operator contract.
///
/// `C` is a geometry/coefficient context for one cell, interior facet, or
/// boundary facet. The quadrature rule carries the integration-domain
/// dimension at runtime. The operator has no global numbering and performs no
/// scatter; lowering may specialize a validated rule later.
/// `S` is the coefficient domain; physical geometry and quadrature stay real.
pub(crate) trait LocalOperator<C, S: Scalar = f64> {
    /// Evaluate one local contribution using an explicit quadrature rule.
    ///
    /// # Errors
    /// Returns a numerical diagnostic if context, quadrature, or evaluation
    /// violates the operator contract.
    fn evaluate(
        &self,
        context: &C,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution<S>, Diagnostic>;
}
