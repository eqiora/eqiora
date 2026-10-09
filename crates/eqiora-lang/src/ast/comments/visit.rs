//! Traverse comment owners in the syntax tree, including signature entries.

use super::SourceComments;
use crate::ast::{ComponentItem, Document, Item, TextRange};

macro_rules! instance {
    ($node:expr, $visit:ident $(, $mutable:tt)?) => {{
        for node in &$($mutable)? $node.bindings {
            $visit(node.range, &$($mutable)? node.comments);
        }
    }};
}

macro_rules! signature {
    ($node:expr, $visit:ident $(, $mutable:tt)?) => {{
        for item in &$($mutable)? $node.signature {
            match item {
                crate::SignatureItem::Parameter(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::Support(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::Field(value) | crate::SignatureItem::Input(value) | crate::SignatureItem::Output(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::Clock(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::Property(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::Port(value) => $visit(value.range, &$($mutable)? value.comments),
                crate::SignatureItem::PortFamily(value) => $visit(value.port.range, &$($mutable)? value.port.comments),
            }
        }
    }};
}

macro_rules! owners {
    ($document:expr, $visit:ident $(, $mutable:tt)?) => {{
        for node in &$($mutable)? $document.records {
            $visit(node.range, &$($mutable)? node.comments);
            for member in &$($mutable)? node.members {
                $visit(member.range, &$($mutable)? member.comments);
            }
        }
        for node in &$($mutable)? $document.enumerations {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.finite_spaces {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.imports {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.dimensions {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.property_contracts {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.property_releases {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.material_compositions {
            $visit(node.range, &$($mutable)? node.comments);
            for property in &$($mutable)? node.properties {
                $visit(property.range, &$($mutable)? property.comments);
            }
        }
        for node in &$($mutable)? $document.connectors {
            $visit(node.range, &$($mutable)? node.comments);
        }
        for node in &$($mutable)? $document.pure_operators {
            $visit(node.range, &$($mutable)? node.comments);
            for formal in &$($mutable)? node.formals {
                $visit(formal.range, &$($mutable)? formal.comments);
            }
        }
        for node in &$($mutable)? $document.components {
            signature!(node, $visit $(, $mutable)?);
            $visit(node.range, &$($mutable)? node.comments);
            for form in &$($mutable)? node.formulations {
                $visit(form.range, &$($mutable)? form.comments);
                if let crate::FormulationBinding::Harmonic { amplitudes, .. } = &$($mutable)? form.binding {
                    for (amplitude, _) in amplitudes {
                        $visit(amplitude.range, &$($mutable)? amplitude.comments);
                    }
                }
            }
            for item in &$($mutable)? node.items {
                match item {
                    ComponentItem::Domain(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::IndexSet(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Let(value) | ComponentItem::Coordinate(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Parameter(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Port(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::PortFamily(value) => $visit(value.port.range, &$($mutable)? value.port.comments),
                    ComponentItem::Field(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Observable(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Initial(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Event(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Clock(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Relation(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::RelationFamily(value) => $visit(value.relation.range, &$($mutable)? value.relation.comments),
                    ComponentItem::Connection(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::BoundaryConnection(value) => $visit(value.range, &$($mutable)? value.comments),
                    ComponentItem::Instance(value) => {
                        $visit(value.range, &$($mutable)? value.comments);
                        instance!(value, $visit $(, $mutable)?);
                    }
                }
            }
        }
        for node in &$($mutable)? $document.models {
            signature!(node, $visit $(, $mutable)?);
            $visit(node.range, &$($mutable)? node.comments);
            for form in &$($mutable)? node.formulations {
                $visit(form.range, &$($mutable)? form.comments);
                if let crate::FormulationBinding::Harmonic { amplitudes, .. } = &$($mutable)? form.binding {
                    for (amplitude, _) in amplitudes {
                        $visit(amplitude.range, &$($mutable)? amplitude.comments);
                    }
                }
            }
            for item in &$($mutable)? node.items {
                match item {
                    Item::IndexSet(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Domain(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Field(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Observable(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Initial(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Parameter(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Let(value) | Item::Coordinate(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Port(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Event(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Clock(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Relation(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::RelationFamily(value) => $visit(value.relation.range, &$($mutable)? value.relation.comments),
                    Item::Connection(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::BoundaryConnection(value) => $visit(value.range, &$($mutable)? value.comments),
                    Item::Instance(value) => {
                        $visit(value.range, &$($mutable)? value.comments);
                        instance!(value, $visit $(, $mutable)?);
                    }
                }
            }
        }
    }};
}

impl Document {
    /// Attach builder documentation and notation to one exact declaration.
    /// Builder ranges identify AST owners; they do not claim a source-file span.
    ///
    /// # Errors
    /// Rejects missing or ambiguous owners and oversized documentation.
    pub fn with_declaration_metadata(
        mut self,
        declaration: TextRange,
        doc: Option<String>,
        notation: Option<crate::Notation>,
    ) -> Result<Self, crate::AstConstructionError> {
        let mut count = 0;
        self.visit_comments(|range, _| {
            if range == declaration {
                count += 1;
            }
        });
        if count != 1 {
            return Err(crate::AstConstructionError::new(
                "metadata needs one exact declaration owner",
            ));
        }
        let doc = doc
            .map(|text| {
                super::DocComment::new(text, declaration).ok_or_else(|| {
                    crate::AstConstructionError::new("documentation exceeds 16384 bytes")
                })
            })
            .transpose()?;
        self.visit_comments_mut(|range, owner| {
            if range == declaration {
                owner.doc = doc.clone();
                owner.notation = notation.clone();
            }
        });
        Ok(self)
    }

    /// Typed declaration notations paired with their original declaration ranges.
    pub fn notations(&self) -> impl ExactSizeIterator<Item = (TextRange, &crate::Notation)> {
        let mut result = Vec::new();
        self.visit_comments(|range, metadata| {
            if let Some(notation) = &metadata.notation {
                result.push((range, notation));
            }
        });
        result.into_iter()
    }

    /// Notation belonging to this exact declaration in this source file.
    #[must_use]
    pub fn notation(&self, declaration: TextRange) -> Option<&crate::Notation> {
        let mut result = None;
        self.visit_comments(|range, metadata| {
            if range == declaration {
                result = metadata.notation.as_ref();
            }
        });
        result
    }

    /// Attached documentation paired with each declaration's original source range.
    ///
    /// One traversal lets editor and documentation clients index a complete source
    /// without searching the syntax tree separately for every declaration.
    pub fn doc_comments(&self) -> impl ExactSizeIterator<Item = (TextRange, &super::DocComment)> {
        let mut comments = Vec::new();
        self.visit_comments(|range, owner| {
            if let Some(doc) = &owner.doc {
                comments.push((range, doc));
            }
        });
        comments.into_iter()
    }

    pub(crate) fn visit_comments<'a>(
        &'a self,
        mut visit: impl FnMut(TextRange, &'a SourceComments),
    ) {
        owners!(self, visit);
    }

    pub(crate) fn visit_comments_mut(
        &mut self,
        mut visit: impl FnMut(TextRange, &mut SourceComments),
    ) {
        owners!(self, visit, mut);
    }

    /// Documentation on the declaration with this exact original source range.
    ///
    /// The lookup is source-local; editor consumers first resolve the declaration
    /// and its source file. Generated source must be parsed for generated ranges.
    #[must_use]
    pub fn doc_comment(&self, declaration: TextRange) -> Option<&super::DocComment> {
        let mut result = None;
        self.visit_comments(|range, comments| {
            if range == declaration {
                result = comments.doc.as_ref();
            }
        });
        result
    }
}

macro_rules! notation_owner {
    ($($owner:ty),+ $(,)?) => { $(
        impl $owner {
            /// Attach validated presentation notation without changing the declaration name or type.
            #[must_use]
            pub fn with_notation(mut self, notation: crate::Notation) -> Self {
                self.comments.notation = Some(notation);
                self
            }
            /// Explicit declaration notation, independent of physical meaning.
            #[must_use]
            pub const fn notation(&self) -> Option<&crate::Notation> {
                self.comments.notation.as_ref()
            }
        }
    )+ };
}

notation_owner!(
    crate::ModelDecl,
    crate::ComponentDecl,
    crate::ConnectorDecl,
    crate::EnumDecl,
    crate::RecordDecl,
    crate::RecordMemberDecl,
    crate::NamedDefinitionDecl,
    crate::ParameterDecl,
    crate::FieldDecl,
    crate::DomainDecl,
    crate::PortDecl,
    crate::ComponentParameterDecl,
    crate::ComponentPortDecl,
    crate::SupportSlotDecl,
    crate::ast::ClockRequirementDecl,
    crate::ClockDecl,
    crate::PureOperatorDecl,
    crate::PureOperatorFormal,
    crate::RelationDecl,
    crate::EventDecl,
    crate::ObservableDecl,
    crate::InstanceDecl,
    crate::ComponentPropertyDecl,
);

impl Item {
    pub(crate) fn source_comments(&self) -> &SourceComments {
        match self {
            Self::Domain(node) => &node.comments,
            Self::Observable(node) => &node.comments,
            Self::Field(node) => &node.comments,
            Self::Initial(node) => &node.comments,
            Self::Parameter(node) => &node.comments,
            Self::Let(node) | Self::Coordinate(node) => &node.comments,
            Self::Port(node) => &node.comments,
            Self::Event(node) => &node.comments,
            Self::Clock(node) => &node.comments,
            Self::Relation(node) => &node.comments,
            Self::RelationFamily(node) => &node.relation.comments,
            Self::Connection(node) => &node.comments,
            Self::BoundaryConnection(node) => &node.comments,
            Self::IndexSet(node) => &node.comments,
            Self::Instance(node) => &node.comments,
        }
    }
}

impl ComponentItem {
    pub(crate) fn source_comments(&self) -> &SourceComments {
        match self {
            Self::Domain(node) => &node.comments,
            Self::Let(node) | Self::Coordinate(node) => &node.comments,
            Self::Parameter(node) => &node.comments,
            Self::Port(node) => &node.comments,
            Self::PortFamily(node) => &node.port.comments,
            Self::Observable(node) => &node.comments,
            Self::Field(node) => &node.comments,
            Self::Initial(node) => &node.comments,
            Self::Event(node) => &node.comments,
            Self::Clock(node) => &node.comments,
            Self::Relation(node) => &node.comments,
            Self::RelationFamily(node) => &node.relation.comments,
            Self::Connection(node) => &node.comments,
            Self::BoundaryConnection(node) => &node.comments,
            Self::IndexSet(node) => &node.comments,
            Self::Instance(node) => &node.comments,
        }
    }
}
