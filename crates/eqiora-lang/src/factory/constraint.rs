//! Typed source construction retains the mathematical condition discriminator.
use super::*;

impl SourceAstFactory {
    /// Construct a mathematical condition without disguising it as a Boolean or equality.
    /// Complementarity operands are explicit nonnegativity predicates; the compiler checks
    /// their literal zero, real order, units and exact supports.
    ///
    /// # Errors
    /// Rejects malformed expressions or ranges; mathematical admission remains compiler-owned.
    pub fn condition(
        kind: eqiora_schema::kernel::RelationConditionKind,
        left: Expr,
        right: Expr,
        range: TextRange,
    ) -> Result<RelationCondition, AstConstructionError> {
        validate_expression(&left)?;
        validate_expression(&right)?;
        Ok(RelationCondition {
            kind,
            left,
            right,
            range: checked_range(range)?,
        })
    }
}
