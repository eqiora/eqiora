//! Checked native construction of the shared private event declaration.
use super::{
    AstConstructionError, SourceAstFactory, checked_identifier, checked_range, validate_expression,
};
use crate::{EventDecl, Expr, TextRange};
use eqiora_schema::kernel::EventDirection;

impl SourceAstFactory {
    /// Construct an event with inferred guard type, crossing direction and static priority.
    ///
    /// # Errors
    /// Rejects invalid names, ranges, and unbounded or malformed guard syntax.
    /// Scalar guard type and event scope are validated by the common compiler.
    pub fn event(
        name: impl Into<String>,
        guard: Expr,
        direction: EventDirection,
        priority: i64,
        range: TextRange,
    ) -> Result<EventDecl, AstConstructionError> {
        validate_expression(&guard)?;
        Ok(EventDecl {
            comments: Default::default(),
            name: checked_identifier(name, "event name")?,
            guard,
            direction,
            priority,
            range: checked_range(range)?,
        })
    }
}
