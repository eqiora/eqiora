//! Trace-theorem checks for boundary traces, separate from the choice of numerical basis.
use super::*;
use crate::kernel::{
    FiniteUnaryOperation, SpatialRegularity,
    pure_operator::{CalculusNode, PureOperatorDefinition},
};
use std::convert::Infallible;

mod checker;
pub use checker::TraceRegularityChecker;

impl<I: Clone + Eq> TypedResidual<I> {
    /// Check from the authored regularity of each symbol whether a boundary trace exists.
    ///
    /// Sums and smooth scalar multipliers preserve regularity. Full traces of H1, normal
    /// traces of H(div), and tangential traces of H(curl) are allowed; traces of
    /// derivatives require smooth operands. No numerical basis, shared interface value,
    /// or general product of distributions is inferred.
    ///
    /// A tangential trace is written as the skew lift followed by normal contraction.
    /// The check recognizes that closed calculus definition by content, not by name, so
    /// other component maps cannot gain this trace theorem.
    pub fn validate_trace_regularity(
        &self,
        mut symbol_regularity: impl FnMut(SymbolRef) -> SpatialRegularity,
    ) -> Result<(), Vec<TypedResidualError<I, Infallible>>> {
        if !self.expression.nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Trace { .. } | ExprNode::NormalComponent { .. }
            )
        }) {
            return Ok(());
        }
        let mut checker = TraceRegularityChecker::default();
        let mut errors = Vec::new();
        for (index, node) in self.expression.nodes().iter().enumerate() {
            if let Err(error) = checker.check_node(
                node,
                &self.node_types[index],
                |id| &self.node_types[id.index() as usize],
                &mut symbol_regularity,
                |digest| {
                    self.expression
                        .definition(digest)
                        .expect("closed definition table")
                },
            ) {
                errors.push(TypedResidualError::Type {
                    node_index: index as u32,
                    error,
                });
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests;
