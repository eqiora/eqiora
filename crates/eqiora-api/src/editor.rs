//! In-process source analysis for editor adapters.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ComponentItem, DocComment, Document, Item, SignatureItem, TextRange, format, parse,
};

mod assistance;
mod workspace;
pub use assistance::{
    EditorCall, EditorCandidate, EditorCompletion, EditorCompletionContext, EditorParameter,
};

pub use workspace::{
    EditorDefinition, EditorReference, EditorWorkspaceService, EditorWorkspaceSnapshot,
};

const MAX_EDITOR_SOURCE_BYTES: usize = 16 * 1024 * 1024;

/// Zero-based editor position using UTF-16 code units within a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EditorPosition {
    line: u32,
    character: u32,
}

impl EditorPosition {
    /// Construct a zero-based line and UTF-16 character position.
    #[must_use]
    pub const fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }

    /// Zero-based line number.
    #[must_use]
    pub const fn line(self) -> u32 {
        self.line
    }

    /// Zero-based UTF-16 code-unit offset within the line.
    #[must_use]
    pub const fn character(self) -> u32 {
        self.character
    }
}

/// Declaration category exposed to editor symbol views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EditorSymbolKind {
    /// Imported module alias.
    Import,
    /// Structural dimension alias.
    Dimension,
    /// Typed property contract or release.
    Property,
    /// Material composition.
    Material,
    /// Conserving interface declaration.
    Connector,
    /// Reusable component definition.
    Component,
    /// Pure compile-time operator.
    Operator,
    /// Declaration-owned finite enumeration.
    Enum,
    /// Closed nominal record type.
    Record,
    /// Named finite index space.
    FiniteSpace,
    /// Ordered member of a finite enumeration.
    EnumMember,
    /// Executable model.
    Model,
    /// Spatial domain.
    Domain,
    /// Scalar parameter.
    Parameter,
    /// Compile-time expression alias.
    Let,
    /// Pure-operator formal argument.
    Formal,
    /// Spatial support.
    Support,
    /// Field or required field slot.
    Field,
    /// Causal or conserving port.
    Port,
    /// Exact periodic clock.
    Clock,
    /// Residual relation.
    Relation,
    /// Component instance.
    Instance,
}

/// One recovered named declaration and its nested declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorSymbol {
    kind: EditorSymbolKind,
    name: String,
    range: TextRange,
    children: Vec<Self>,
    doc_comment: Option<DocComment>,
}

impl EditorSymbol {
    fn leaf(kind: EditorSymbolKind, name: impl Into<String>, range: TextRange) -> Self {
        Self {
            kind,
            name: name.into(),
            range,
            children: Vec::new(),
            doc_comment: None,
        }
    }

    fn branch(
        kind: EditorSymbolKind,
        name: impl Into<String>,
        range: TextRange,
        mut children: Vec<Self>,
    ) -> Self {
        children.sort_by_key(|symbol| (symbol.range.start(), symbol.range.end()));
        Self {
            kind,
            name: name.into(),
            range,
            children,
            doc_comment: None,
        }
    }

    /// Declaration category.
    #[must_use]
    pub const fn kind(&self) -> EditorSymbolKind {
        self.kind
    }

    /// Source-declared name or import alias.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Complete UTF-8 byte range of the declaration.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }

    /// Directly nested named declarations in source order.
    #[must_use]
    pub fn children(&self) -> &[Self] {
        &self.children
    }

    /// Source documentation attached to this exact recovered declaration.
    #[must_use]
    pub const fn doc_comment(&self) -> Option<&DocComment> {
        self.doc_comment.as_ref()
    }
}

/// Immutable analysis of one exact source version.
#[derive(Debug, Clone, PartialEq)]
pub struct EditorSnapshot {
    version: u64,
    source: String,
    line_starts: Vec<u32>,
    diagnostics: Vec<Diagnostic>,
    formatted: Option<String>,
    symbols: Vec<EditorSymbol>,
    syntax: Option<Document>,
}

impl EditorSnapshot {
    /// Maximum UTF-8 source bytes analyzed for one editor document.
    pub const MAX_SOURCE_BYTES: usize = MAX_EDITOR_SOURCE_BYTES;

    fn analyze(file: &str, version: u64, source: String) -> Self {
        if source.len() > MAX_EDITOR_SOURCE_BYTES {
            return Self {
                version,
                source,
                line_starts: Vec::new(),
                diagnostics: vec![Diagnostic::error(
                    codes::INVALID_TOKEN,
                    format!(
                        "editor source exceeds the {MAX_EDITOR_SOURCE_BYTES}-byte analysis limit"
                    ),
                )],
                formatted: None,
                symbols: Vec::new(),
                syntax: None,
            };
        }

        let parsed = parse(file, &source);
        let formatted = parsed
            .diagnostics()
            .is_empty()
            .then(|| parsed.document().map(format))
            .flatten();
        let symbols = parsed.document().map_or_else(Vec::new, document_symbols);
        let diagnostics = if parsed.diagnostics().is_empty()
            && parsed
                .document()
                .is_some_and(|document| !document.models().is_empty())
        {
            eqiora_compiler::compile(file, &source)
                .err()
                .unwrap_or_default()
        } else {
            parsed.diagnostics().to_vec()
        };
        let line_starts = line_starts(&source);

        Self {
            version,
            source,
            line_starts,
            diagnostics,
            formatted,
            symbols,
            syntax: parsed.document().cloned(),
        }
    }

    fn from_resolved_source(version: u64, file: &str, source: String) -> Self {
        let parsed = parse(file, &source);
        debug_assert!(parsed.diagnostics().is_empty());
        let document = parsed
            .document()
            .expect("compiler-accepted source reparses to one document");
        Self {
            version,
            line_starts: line_starts(&source),
            source,
            diagnostics: Vec::new(),
            formatted: Some(format(document)),
            symbols: document_symbols(document),
            syntax: Some(document.clone()),
        }
    }

    fn from_recovered_source(
        version: u64,
        file: &str,
        source: String,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        if source.len() > MAX_EDITOR_SOURCE_BYTES {
            return Self {
                version,
                source,
                line_starts: Vec::new(),
                diagnostics,
                formatted: None,
                symbols: Vec::new(),
                syntax: None,
            };
        }
        let parsed = parse(file, &source);
        let formatted = parsed
            .diagnostics()
            .is_empty()
            .then(|| parsed.document().map(format))
            .flatten();
        let symbols = parsed.document().map_or_else(Vec::new, document_symbols);
        Self {
            version,
            line_starts: line_starts(&source),
            source,
            diagnostics,
            formatted,
            symbols,
            syntax: parsed.document().cloned(),
        }
    }

    /// Exact monotonically increasing document version supplied by the client.
    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Recovered lexical, syntactic, and available semantic diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Canonical whole-document text when parsing completed without errors.
    #[must_use]
    pub fn formatted(&self) -> Option<&str> {
        self.formatted.as_deref()
    }

    /// Named declarations recovered from this document in source order.
    #[must_use]
    pub fn symbols(&self) -> &[EditorSymbol] {
        &self.symbols
    }

    /// Convert an exact UTF-8 byte boundary to an editor position.
    #[must_use]
    pub fn position(&self, byte_offset: u32) -> Option<EditorPosition> {
        let offset = usize::try_from(byte_offset).ok()?;
        if offset > self.source.len() || !self.source.is_char_boundary(offset) {
            return None;
        }
        let line_index = self
            .line_starts
            .partition_point(|start| *start <= byte_offset)
            .checked_sub(1)?;
        let start = usize::try_from(self.line_starts[line_index]).ok()?;
        if offset > self.line_end(line_index)? {
            return None;
        }
        let character = self.source[start..offset].encode_utf16().count();
        Some(EditorPosition::new(
            u32::try_from(line_index).ok()?,
            u32::try_from(character).ok()?,
        ))
    }

    /// Convert an editor position to an exact UTF-8 byte boundary.
    #[must_use]
    pub fn byte_offset(&self, position: EditorPosition) -> Option<u32> {
        let line = usize::try_from(position.line).ok()?;
        let start = usize::try_from(*self.line_starts.get(line)?).ok()?;
        let end = self.line_end(line)?;

        let target = usize::try_from(position.character).ok()?;
        let mut utf16 = 0_usize;
        for (relative, character) in self.source[start..end].char_indices() {
            if utf16 == target {
                return u32::try_from(start + relative).ok();
            }
            utf16 = utf16.checked_add(character.len_utf16())?;
            if utf16 > target {
                return None;
            }
        }
        (utf16 == target).then(|| u32::try_from(end).ok()).flatten()
    }

    fn line_end(&self, line: usize) -> Option<usize> {
        let start = usize::try_from(*self.line_starts.get(line)?).ok()?;
        let mut end = match self.line_starts.get(line + 1) {
            Some(next) => usize::try_from(*next).ok()?,
            None => self.source.len(),
        };
        if end > start && self.source.as_bytes().get(end - 1) == Some(&b'\n') {
            end -= 1;
        }
        if end > start && self.source.as_bytes().get(end - 1) == Some(&b'\r') {
            end -= 1;
        }
        Some(end)
    }
}

/// Version owner for one editor document.
#[derive(Debug, Clone, PartialEq)]
pub struct EditorService {
    file: String,
    current: EditorSnapshot,
}

impl EditorService {
    /// Analyze the initial document version.
    #[must_use]
    pub fn new(file: impl Into<String>, version: u64, source: impl Into<String>) -> Self {
        let file = file.into();
        let current = EditorSnapshot::analyze(&file, version, source.into());
        Self { file, current }
    }

    /// Replace the source with a strictly newer version and analyze it atomically.
    ///
    /// # Errors
    /// Returns a precondition diagnostic without changing the current snapshot
    /// when `version` is not newer.
    pub fn replace(
        &mut self,
        version: u64,
        source: impl Into<String>,
    ) -> Result<&EditorSnapshot, Diagnostic> {
        if version <= self.current.version {
            return Err(stale_version(version, self.current.version));
        }
        self.current = EditorSnapshot::analyze(&self.file, version, source.into());
        Ok(&self.current)
    }

    /// Return the current snapshot only when it still matches the request version.
    ///
    /// # Errors
    /// Returns a precondition diagnostic when the request is stale or refers to
    /// an unknown future version.
    pub fn snapshot(&self, version: u64) -> Result<&EditorSnapshot, Diagnostic> {
        if version == self.current.version {
            Ok(&self.current)
        } else {
            Err(stale_version(version, self.current.version))
        }
    }

    /// Current source analysis.
    #[must_use]
    pub const fn current(&self) -> &EditorSnapshot {
        &self.current
    }
}

fn stale_version(requested: u64, current: u64) -> Diagnostic {
    Diagnostic::error(
        codes::PRECONDITION_FAILED,
        format!("editor request version {requested} does not match current version {current}"),
    )
}

fn line_starts(source: &str) -> Vec<u32> {
    let mut starts = vec![0];
    let bytes = source.as_bytes();
    let mut offset = 0;
    while offset < bytes.len() {
        let width = match bytes[offset] {
            b'\r' if bytes.get(offset + 1) == Some(&b'\n') => 2,
            b'\r' | b'\n' => 1,
            _ => {
                offset += 1;
                continue;
            }
        };
        offset += width;
        if let Ok(start) = u32::try_from(offset) {
            starts.push(start);
        }
    }
    starts
}

fn document_symbols(document: &Document) -> Vec<EditorSymbol> {
    let mut symbols = Vec::new();
    symbols.extend(document.records().iter().map(|r| {
        EditorSymbol::branch(
            EditorSymbolKind::Record,
            r.name(),
            r.range(),
            r.members()
                .iter()
                .map(|m| EditorSymbol::leaf(EditorSymbolKind::Field, m.name(), m.range()))
                .collect(),
        )
    }));
    symbols.extend(
        document
            .finite_spaces()
            .iter()
            .map(|s| EditorSymbol::leaf(EditorSymbolKind::FiniteSpace, s.name(), s.range())),
    );
    symbols.extend(
        document
            .imports()
            .map(|(_, alias, range)| EditorSymbol::leaf(EditorSymbolKind::Import, alias, range)),
    );
    symbols.extend(
        document.dimensions().iter().map(|value| {
            EditorSymbol::leaf(EditorSymbolKind::Dimension, value.name(), value.range())
        }),
    );
    symbols.extend(
        document
            .property_contract_syntax()
            .map(|(_, name, _, range)| EditorSymbol::leaf(EditorSymbolKind::Property, name, range)),
    );
    symbols.extend(
        document
            .property_release_syntax()
            .map(|(_, name, _, _, _, _, _, _, range)| {
                EditorSymbol::leaf(EditorSymbolKind::Property, name, range)
            }),
    );
    symbols.extend(
        document
            .material_composition_syntax()
            .map(|(_, name, _, range)| EditorSymbol::leaf(EditorSymbolKind::Material, name, range)),
    );
    symbols.extend(document.enumerations().iter().map(|declaration| {
        EditorSymbol::branch(
            EditorSymbolKind::Enum,
            declaration.name(),
            declaration.range(),
            declaration
                .tags()
                .iter()
                .map(|tag| {
                    EditorSymbol::leaf(EditorSymbolKind::EnumMember, tag.as_str(), tag.range())
                })
                .collect(),
        )
    }));
    symbols.extend(document.connectors().iter().map(|declaration| {
        EditorSymbol::leaf(
            EditorSymbolKind::Connector,
            declaration.name(),
            declaration.range(),
        )
    }));
    symbols.extend(document.components().iter().map(component_symbol));
    symbols.extend(document.pure_operators().iter().map(|declaration| {
        EditorSymbol::branch(
            EditorSymbolKind::Operator,
            declaration.name(),
            declaration.range(),
            declaration
                .formals()
                .iter()
                .map(|formal| {
                    EditorSymbol::leaf(EditorSymbolKind::Formal, formal.name(), formal.range())
                })
                .collect(),
        )
    }));
    symbols.extend(document.models().iter().map(|declaration| {
        EditorSymbol::branch(
            EditorSymbolKind::Model,
            declaration.name(),
            declaration.range(),
            declaration
                .signature()
                .iter()
                .filter_map(signature_item_symbol)
                .chain(declaration.items().iter().filter_map(model_item_symbol))
                .collect(),
        )
    }));
    symbols.sort_by_key(|symbol| (symbol.range.start(), symbol.range.end()));
    fn attach(
        symbols: &mut [EditorSymbol],
        documentation: &std::collections::HashMap<TextRange, &DocComment>,
    ) {
        for symbol in symbols {
            symbol.doc_comment = documentation.get(&symbol.range).map(|doc| (*doc).clone());
            attach(&mut symbol.children, documentation);
        }
    }
    attach(&mut symbols, &document.doc_comments().collect());
    symbols
}

fn component_symbol(component: &eqiora_lang::ComponentDecl) -> EditorSymbol {
    let mut children = component
        .signature()
        .iter()
        .filter_map(signature_item_symbol)
        .collect::<Vec<_>>();
    children.extend(component.items().iter().filter_map(component_item_symbol));
    EditorSymbol::branch(
        EditorSymbolKind::Component,
        component.name(),
        component.range(),
        children,
    )
}

fn signature_item_symbol(item: &SignatureItem) -> Option<EditorSymbol> {
    let kind = match item {
        SignatureItem::Parameter(_) => EditorSymbolKind::Parameter,
        SignatureItem::Support(_) => EditorSymbolKind::Support,
        SignatureItem::Field(_) => EditorSymbolKind::Field,
        SignatureItem::Clock(_) => EditorSymbolKind::Clock,
        SignatureItem::Property(_) => EditorSymbolKind::Property,
        SignatureItem::Input(_)
        | SignatureItem::Output(_)
        | SignatureItem::Port(_)
        | SignatureItem::PortFamily(_) => EditorSymbolKind::Port,
        _ => return None,
    };
    Some(EditorSymbol::leaf(kind, item.name(), item.range()))
}

fn component_item_symbol(item: &ComponentItem) -> Option<EditorSymbol> {
    let symbol = match item {
        ComponentItem::Parameter(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Parameter, value.name(), value.range())
        }
        ComponentItem::Port(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Port, value.name(), value.range())
        }
        ComponentItem::PortFamily(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Port, value.port().name(), value.range())
        }
        ComponentItem::Field(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Field, value.name(), value.range())
        }
        ComponentItem::Clock(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Clock, value.name(), value.range())
        }
        ComponentItem::Initial(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Relation, "initial", value.range())
        }
        ComponentItem::Relation(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Relation, value.name(), value.range())
        }
        ComponentItem::RelationFamily(value) => EditorSymbol::leaf(
            EditorSymbolKind::Relation,
            value.relation().name(),
            value.range(),
        ),
        ComponentItem::Instance(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Instance, value.name(), value.range())
        }
        ComponentItem::Connection(_) | ComponentItem::BoundaryConnection(_) => return None,
        _ => return None,
    };
    Some(symbol)
}

fn model_item_symbol(item: &Item) -> Option<EditorSymbol> {
    let symbol = match item {
        Item::Domain(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Domain, value.name(), value.range())
        }
        Item::Field(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Field, value.name(), value.range())
        }
        Item::Parameter(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Parameter, value.name(), value.range())
        }
        Item::Let(value) => EditorSymbol::leaf(EditorSymbolKind::Let, value.name(), value.range()),
        Item::Port(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Port, value.name(), value.range())
        }
        Item::Clock(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Clock, value.name(), value.range())
        }
        Item::Initial(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Relation, "initial", value.range())
        }
        Item::Relation(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Relation, value.name(), value.range())
        }
        Item::Instance(value) => {
            EditorSymbol::leaf(EditorSymbolKind::Instance, value.name(), value.range())
        }
        Item::Connection(_) | Item::BoundaryConnection(_) => return None,
        _ => return None,
    };
    Some(symbol)
}

#[cfg(test)]
mod enumerations;
