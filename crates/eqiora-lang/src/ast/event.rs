//! Private typed crossing-event source declarations.
use super::{Expr, TextRange};
use eqiora_schema::kernel::EventDirection;

/// A private event with an inferred guard type and explicit crossing direction.
#[derive(Debug, Clone, PartialEq)]
pub struct EventDecl {
    pub(crate) comments: super::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) guard: Expr,
    pub(crate) direction: EventDirection,
    pub(crate) priority: i64,
    pub(crate) range: TextRange,
}

impl EventDecl {
    /// Name of the event in its owning Model or Component.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Authored scalar zero-crossing guard; its complete type is inferred.
    #[must_use]
    pub const fn guard(&self) -> &Expr {
        &self.guard
    }
    /// Explicit negative-to-positive, positive-to-negative, or either crossing.
    #[must_use]
    pub const fn direction(&self) -> EventDirection {
        self.direction
    }
    /// Static event priority; higher values win overlapping event resets.
    #[must_use]
    pub const fn priority(&self) -> i64 {
        self.priority
    }
    /// Full declaration range, including the terminating semicolon.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}
