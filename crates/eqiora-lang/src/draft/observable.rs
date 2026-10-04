//! Derived output declarations share the ordinary native expression graph.
use super::*;

/// Immutable typed derived output; it introduces no solve unknown.
#[derive(Debug, Clone)]
pub struct DraftObservable {
    pub(super) symbol: DraftSymbol,
    pub(super) name: String,
    pub(super) value_type: ValueType,
    pub(super) domain: Option<DraftSpatialDomain>,
    pub(super) expression: DraftExpression,
}

impl DraftObservable {
    /// Declare one typed output expression.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        value_type: ValueType,
        expression: DraftExpression,
    ) -> Self {
        Self {
            symbol: DraftSymbol::new(),
            name: name.into(),
            value_type,
            domain: None,
            expression,
        }
    }

    /// Declare the exact support retained by this output.
    #[must_use]
    pub fn on(mut self, domain: &DraftSpatialDomain) -> Self {
        self.domain = Some(domain.clone());
        self
    }

    /// Declaration name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Complete mathematical output type.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    /// Authored output expression, without an Observable reference symbol.
    #[must_use]
    pub const fn expression(&self) -> &DraftExpression {
        &self.expression
    }
}

impl From<DraftObservable> for DraftDeclaration {
    fn from(value: DraftObservable) -> Self {
        Self::Observable(value)
    }
}

impl DraftExpression {
    /// Reference one exact reduced Observable declared in the same native Module.
    #[must_use]
    pub fn observable(observable: &DraftObservable) -> Self {
        Self::reference(
            observable.symbol.clone(),
            observable.name.clone(),
            DraftSymbolKind::Observable,
        )
    }

    /// Integral expression with an explicit measure(domain) expression.
    #[must_use]
    pub fn integral(value: Self, measure: Self) -> Self {
        Self::call("integral", vec![value, measure])
    }

    /// Exact Domain measure; its volume or boundary kind is resolved during typing.
    #[must_use]
    pub fn measure(domain: &DraftSpatialDomain) -> Self {
        let mut reference = Self::leaf(ExprKind::Name(domain.name().to_owned()));
        reference.references =
            std::sync::Arc::new(vec![expression::NativeReference::Domain(domain.clone())]);
        Self::call("measure", vec![reference])
    }
}
