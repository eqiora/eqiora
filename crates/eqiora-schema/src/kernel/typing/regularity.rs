//! Bounded trace-theorem admission, separate from numerical basis selection.
use super::*;
use crate::kernel::{
    FiniteUnaryOperation, SpatialRegularity,
    pure_operator::{CalculusNode, PureOperatorDefinition},
};
use std::convert::Infallible;

mod checker;
pub use checker::TraceRegularityChecker;

impl<I: Clone + Eq> TypedResidual<I> {
    /// Check boundary trace availability from authored symbol regularity.
    ///
    /// This bounded profile preserves sums and smooth scalar multipliers,
    /// admits full H1, normal H(div), and tangential H(curl) traces, and requires
    /// smooth operands for traces of derivatives. No numerical basis, shared
    /// interface value, or general distribution product is inferred.
    ///
    /// Tangential traces retain the existing exact skew lift followed by normal
    /// contraction. Admission checks that closed calculus definition, never its
    /// source name. Other component maps cannot acquire this trace theorem.
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
