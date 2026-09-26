use super::{Expr, TextRange, ValueTypeSyntax};

/// Parameter source declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) value_type: ValueTypeSyntax,
    pub(crate) value: Expr,
    pub(crate) range: TextRange,
}

impl ParameterDecl {
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

    /// Returns the numeric value, including any explicit input unit.
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

/// Named expression definition. Its containing declaration determines meaning and allowed assertions.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedDefinitionDecl {
    pub(crate) visibility: super::VisibilitySyntax,
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) value_type: Option<ValueTypeSyntax>,
    pub(crate) domain: Option<String>,
    pub(crate) activation: Option<String>,
    pub(crate) value: Expr,
    pub(crate) activation_name_range: Option<TextRange>,
    pub(crate) range: TextRange,
}

impl NamedDefinitionDecl {
    /// Exact authored activation-name token; absent without an assertion or source tokens.
    #[must_use]
    pub const fn activation_name_range(&self) -> Option<TextRange> {
        self.activation_name_range
    }

    /// Visibility of this named definition in its compilation unit.
    #[must_use]
    pub const fn visibility(&self) -> super::VisibilitySyntax {
        self.visibility
    }

    pub(crate) fn plain(
        name: String,
        value: Expr,
        range: TextRange,
        visibility: super::VisibilitySyntax,
    ) -> Self {
        Self {
            comments: Default::default(),
            visibility,
            name,
            value,
            range,
            value_type: None,
            domain: None,
            activation: None,
            activation_name_range: None,
        }
    }

    /// Returns the declared alias name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the optional complete mathematical type assertion.
    #[must_use]
    pub const fn value_type(&self) -> Option<&ValueTypeSyntax> {
        self.value_type.as_ref()
    }

    /// Returns the optional assertion of the expression's intrinsic support.
    #[must_use]
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// Returns the optional named assertion of the expression's activation.
    #[must_use]
    pub fn activation(&self) -> Option<&str> {
        self.activation.as_deref()
    }

    /// Returns the immutable expression.
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
