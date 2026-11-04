//! Shared external contracts for Component and Model declarations.

use super::*;

/// One public requirement or occurrence-owned exposed value in a typed signature.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SignatureItem {
    /// Required or defaulted static value.
    Parameter(ComponentParameterDecl),
    /// Borrowed typed expression; the shared value head allocates no Field.
    Expression(FieldDecl),
    /// Borrowed exact nominal spatial support.
    Support(SupportSlotDecl),
    /// Borrowed algebraic unknown or state; the declaration role is retained.
    Field(FieldDecl),
    /// Borrowed exact nominal periodic clock.
    Clock(ClockRequirementDecl),
    /// Exact property release requirement.
    Property(crate::ast_property::ComponentPropertyDecl),
    /// Externally driven causal value, with its declared support and activation.
    Input(FieldDecl),
    /// Occurrence-owned causal output, with its declared support and activation.
    Output(FieldDecl),
    /// Occurrence-owned physical connector endpoint.
    Port(ComponentPortDecl),
    /// Occurrence-owned physical connector family over an exact complete exterior.
    PortFamily(ComponentPortFamilyDecl),
}

impl SignatureItem {
    /// Name in the shared signature and body declaration namespace.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Parameter(value) => value.name(),
            Self::Support(value) => value.name(),
            Self::Expression(value)
            | Self::Field(value)
            | Self::Input(value)
            | Self::Output(value) => value.name(),
            Self::Clock(value) => value.name(),
            Self::Property(value) => value.name(),
            Self::Port(value) => value.name(),
            Self::PortFamily(value) => value.port().name(),
        }
    }

    /// Exact source range of this signature entry.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        match self {
            Self::Parameter(value) => value.range(),
            Self::Support(value) => value.range(),
            Self::Expression(value)
            | Self::Field(value)
            | Self::Input(value)
            | Self::Output(value) => value.range(),
            Self::Clock(value) => value.range(),
            Self::Property(value) => value.range(),
            Self::Port(value) => value.range(),
            Self::PortFamily(value) => value.range(),
        }
    }
}

impl SignatureItem {
    pub(crate) fn source_comments(&self) -> &comments::SourceComments {
        match self {
            Self::Parameter(value) => &value.comments,
            Self::Support(value) => &value.comments,
            Self::Expression(value)
            | Self::Field(value)
            | Self::Input(value)
            | Self::Output(value) => &value.comments,
            Self::Clock(value) => &value.comments,
            Self::Property(value) => &value.comments,
            Self::Port(value) => &value.comments,
            Self::PortFamily(value) => &value.port.comments,
        }
    }
}
