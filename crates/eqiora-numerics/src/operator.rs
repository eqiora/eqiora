use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::QuadratureRule;

/// Pure entity-local operator contract.
///
/// `C` is a geometry/coefficient context for one cell, interior facet, or
/// boundary facet. The quadrature rule carries the integration-domain
/// dimension at runtime. The operator has no global numbering and performs no
/// scatter; lowering may specialize a validated rule later.
pub trait LocalOperator<C> {
    /// Evaluate one local contribution using an explicit quadrature rule.
    ///
    /// # Errors
    /// Returns a numerical diagnostic if context, quadrature, or evaluation
    /// violates the operator contract.
    fn evaluate(
        &self,
        context: &C,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution<f64>, Diagnostic>;
}
