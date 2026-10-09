use crate::ast_property::{MaterialCompositionDecl, PropertyContractDecl, PropertyReleaseDecl};

use super::{
    ComponentDecl, ConnectorDecl, Item, NamePath, NamedDefinitionDecl, PureOperatorDecl, TextRange,
    VisibilitySyntax,
};

/// A named model and its declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) signature: Vec<super::SignatureItem>,
    pub(crate) items: Vec<Item>,
    pub(crate) formulations: Vec<super::formulation::FormulationDecl>,
    pub(crate) range: TextRange,
}

impl ModelDecl {
    /// Public requirements and occurrence-owned exposed values.
    #[must_use]
    pub fn signature(&self) -> &[super::SignatureItem] {
        &self.signature
    }

    /// Module visibility.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Declarations in source order.
    #[must_use]
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// Full model declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// One explicit, side-effect-free semantic module import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImportDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) module: NamePath,
    pub(crate) alias: String,
    pub(crate) range: TextRange,
}

/// One parsed source file.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub(crate) comments: super::comments::SourceComments,
    pub(crate) imports: Vec<ImportDecl>,
    pub(crate) records: Vec<super::RecordDecl>,
    pub(crate) enumerations: Vec<super::EnumDecl>,
    pub(crate) finite_spaces: Vec<super::NamedDefinitionDecl>,
    pub(crate) dimensions: Vec<NamedDefinitionDecl>,
    pub(crate) property_contracts: Vec<PropertyContractDecl>,
    pub(crate) property_releases: Vec<PropertyReleaseDecl>,
    pub(crate) material_compositions: Vec<MaterialCompositionDecl>,
    pub(crate) connectors: Vec<ConnectorDecl>,
    pub(crate) components: Vec<ComponentDecl>,
    pub(crate) pure_operators: Vec<PureOperatorDecl>,
    pub(crate) models: Vec<ModelDecl>,
}

impl Document {
    /// Closed record declarations in authored order.
    #[must_use]
    pub fn records(&self) -> &[super::RecordDecl] {
        &self.records
    }

    /// Module-level enum declarations in authored order.
    #[must_use]
    pub fn enumerations(&self) -> &[super::EnumDecl] {
        &self.enumerations
    }

    /// Ordered nominal atomic finite-space declarations.
    #[must_use]
    pub fn finite_spaces(&self) -> &[super::NamedDefinitionDecl] {
        &self.finite_spaces
    }
    /// Explicit semantic imports in authored order.
    pub fn imports(&self) -> impl ExactSizeIterator<Item = (&NamePath, &str, TextRange)> {
        self.imports
            .iter()
            .map(|import| (&import.module, import.alias.as_str(), import.range))
    }

    /// Structural dimension declarations, including their export visibility.
    #[must_use]
    pub fn dimensions(&self) -> &[NamedDefinitionDecl] {
        &self.dimensions
    }

    /// Compilation-unit connector declarations in source order.
    #[must_use]
    pub fn connectors(&self) -> &[ConnectorDecl] {
        &self.connectors
    }

    /// Compilation-unit component declarations in source order.
    #[must_use]
    pub fn components(&self) -> &[ComponentDecl] {
        &self.components
    }

    /// Compilation-unit pure operator declarations in source order.
    #[must_use]
    pub fn pure_operators(&self) -> &[PureOperatorDecl] {
        &self.pure_operators
    }

    /// Model declarations in source order.
    #[must_use]
    pub fn models(&self) -> &[ModelDecl] {
        &self.models
    }
}
