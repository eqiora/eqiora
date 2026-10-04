use super::{Expr, TextRange, ValueTypeSyntax};

/// A typed derived value, separate from unknowns and equations.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservableDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) value_type: ValueTypeSyntax,
    pub(crate) domain: Option<String>,
    pub(crate) value: Expr,
    pub(crate) range: TextRange,
}

impl ObservableDecl {
    /// Returns the declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the declared dimension expression.
    #[must_use]
    pub fn dimension(&self) -> Option<&Expr> {
        self.value_type.dimension()
    }

    /// Complete declared mathematical type.
    #[must_use]
    pub const fn value_type(&self) -> &ValueTypeSyntax {
        &self.value_type
    }

    /// Exact declared output support, absent for a lumped value.
    #[must_use]
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// Returns the derived expression or explicit spatial integral.
    #[must_use]
    pub const fn value(&self) -> &Expr {
        &self.value
    }

    /// Returns the declaration's source range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}
