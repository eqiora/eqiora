//! Compiler-owned input boundary for exact, already-resolved package graphs.
//!
//! Package storage, version selection, and digest verification remain outside
//! the compiler. This module accepts only an explicit set of source units and
//! direct dependencies and source-authored imports. Analysis is a separate phase so callers can
//! compare canonical declarations with a signed or locked package record
//! before hierarchy elaboration creates a graph transaction.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{Document, SourceAstFactory, TextRange, VisibilitySyntax};

use crate::CompiledModel;
use crate::hierarchy::HierarchyLimits;
use crate::source_identity::{LocalSourceIdentity, ResolvedAliasTarget};

mod analyze;
mod completion;
mod declaration;
mod graph;
mod source;
pub use analyze::analyze_resolved_hierarchy;
pub use declaration::{CanonicalDeclarationIdentity, CanonicalDeclarationKind};

const MAX_NAMESPACE_SEGMENTS: usize = 31;
const MAX_NAMESPACE_SEGMENT_BYTES: usize = 4_096;
const MAX_MODULE_SEGMENTS: usize = 31;
const MAX_MODULE_SEGMENT_BYTES: usize = 512;
const MAX_SOURCE_UNITS: usize = 1_000_000;
const MAX_ALIASES: usize = 1_000_000;
const MAX_TOTAL_SOURCE_BYTES: usize = 256 * 1_024 * 1_024;

#[derive(Clone, Copy)]
struct ResolvedHierarchyResourceLimits {
    source_units: usize,
    aliases: usize,
    source_unit_bytes: usize,
    total_source_bytes: usize,
}

fn resolved_hierarchy_limits() -> ResolvedHierarchyResourceLimits {
    ResolvedHierarchyResourceLimits {
        source_units: MAX_SOURCE_UNITS,
        aliases: MAX_ALIASES,
        source_unit_bytes: HierarchyLimits::default().max_source_bytes,
        total_source_bytes: MAX_TOTAL_SOURCE_BYTES,
    }
}

/// Deterministic identity of one package compilation namespace.
///
/// The first segment is the canonical dotted package name used by source
/// imports. Remaining segments are opaque exact-resolution identity supplied
/// by L4.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompilationNamespaceId(Box<[String]>);

impl CompilationNamespaceId {
    /// Construct one nonempty bounded namespace identity.
    ///
    /// # Errors
    /// Returns a diagnostic for an empty namespace, empty segment, embedded
    /// NUL, or a v1 resource-limit violation.
    pub fn new<I, S>(segments: I) -> Result<Self, Diagnostic>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut result = Vec::new();
        for segment in segments {
            if result.len() >= MAX_NAMESPACE_SEGMENTS {
                return Err(resolved_error(format!(
                    "compilation namespace exceeds the {MAX_NAMESPACE_SEGMENTS} segment limit"
                )));
            }
            let segment = segment.into();
            if segment.is_empty()
                || segment.len() > MAX_NAMESPACE_SEGMENT_BYTES
                || segment.contains('\0')
            {
                return Err(resolved_error(format!(
                    "compilation namespace segments must be nonempty NUL-free UTF-8 within {MAX_NAMESPACE_SEGMENT_BYTES} bytes"
                )));
            }
            result
                .try_reserve(1)
                .map_err(|_| resolved_error("cannot reserve compilation namespace"))?;
            result.push(segment);
        }
        if result.is_empty() {
            return Err(resolved_error("compilation namespace must be nonempty"));
        }
        if !result[0].split('.').all(is_identifier) {
            return Err(resolved_error(format!(
                "canonical package name `{}` must contain only dotted Eqiora identifiers",
                result[0]
            )));
        }
        Ok(Self(result.into_boxed_slice()))
    }

    /// Canonical dotted package name used as the source import root.
    #[must_use]
    pub fn package_name(&self) -> &str {
        &self.0[0]
    }

    /// Opaque identity segments in canonical order.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for CompilationNamespaceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, segment) in self.0.iter().enumerate() {
            if index != 0 {
                formatter.write_str("/")?;
            }
            formatter.write_str(segment)?;
        }
        Ok(())
    }
}

/// Logical source module name below one compilation/package owner.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ModuleName(Box<[String]>);

impl ModuleName {
    /// Construct one nonempty dotted logical module name.
    ///
    /// # Errors
    /// Rejects invalid Eqiora identifiers and bounded-name violations.
    pub fn new<I, S>(segments: I) -> Result<Self, Diagnostic>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut result = Vec::new();
        for segment in segments {
            if result.len() >= MAX_MODULE_SEGMENTS {
                return Err(resolved_error(format!(
                    "logical module name exceeds the {MAX_MODULE_SEGMENTS} segment limit"
                )));
            }
            let segment = segment.into();
            if segment.len() > MAX_MODULE_SEGMENT_BYTES || !is_identifier(&segment) {
                return Err(resolved_error(format!(
                    "logical module segment `{segment}` must be an Eqiora identifier within {MAX_MODULE_SEGMENT_BYTES} bytes"
                )));
            }
            result.push(segment);
        }
        if result.is_empty() {
            return Err(resolved_error("logical module name must be nonempty"));
        }
        Ok(Self(result.into_boxed_slice()))
    }

    fn main() -> Self {
        Self(vec!["main".to_owned()].into_boxed_slice())
    }

    /// Identifier segments in canonical order.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for ModuleName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, segment) in self.0.iter().enumerate() {
            if index != 0 {
                formatter.write_str(".")?;
            }
            formatter.write_str(segment)?;
        }
        Ok(())
    }
}

/// Exact logical module inside one resolved compilation/package owner.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct CompilationModuleId {
    owner: CompilationNamespaceId,
    name: ModuleName,
}

impl CompilationModuleId {
    pub(super) fn import_path(&self) -> String {
        format!("{}.{}", self.owner().package_name(), self.name())
    }

    /// Bind one logical module to its exact compilation owner.
    #[must_use]
    pub const fn new(owner: CompilationNamespaceId, name: ModuleName) -> Self {
        Self { owner, name }
    }

    fn main(owner: CompilationNamespaceId) -> Self {
        Self::new(owner, ModuleName::main())
    }

    /// Exact package/source compilation owner.
    #[must_use]
    pub const fn owner(&self) -> &CompilationNamespaceId {
        &self.owner
    }

    /// Logical module name.
    #[must_use]
    pub const fn name(&self) -> &ModuleName {
        &self.name
    }
}

impl fmt::Display for CompilationModuleId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.owner.fmt(formatter)?;
        write!(formatter, "::{}", self.name)
    }
}

/// One exact UTF-8 source unit owned by a resolved package namespace.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSourceUnit {
    module: CompilationModuleId,
    file: String,
    source: String,
    authored: Option<eqiora_lang::Module>,
    resolved_arrays:
        std::sync::Arc<BTreeMap<String, eqiora_schema::resolved_array::ResolvedF64Array>>,
    resolved_documents: std::sync::Arc<BTreeMap<String, std::sync::Arc<[u8]>>>,
    input_bytes: usize,
}

impl ResolvedSourceUnit {
    /// Construct a source unit from its portable path below the `src/` root.
    ///
    /// The path is the sole logical module identity. For example,
    /// `src/models/main.eqi` belongs to `models.main`.
    ///
    /// # Errors
    /// Rejects paths outside `src/`, non-`.eqi` files, and path segments that
    /// are not Eqiora identifiers.
    pub fn new(
        namespace: CompilationNamespaceId,
        file: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        let file = file.into();
        let relative = file.strip_prefix("src/").ok_or_else(|| {
            resolved_error(format!(
                "model source `{file}` must be below the `src/` source root"
            ))
        })?;
        let without_extension = relative.strip_suffix(".eqi").ok_or_else(|| {
            resolved_error(format!(
                "model source `{file}` must have an `.eqi` extension"
            ))
        })?;
        if without_extension.is_empty() {
            return Err(resolved_error(format!(
                "model source `{file}` must name a module below `src/`"
            )));
        }
        let module = ModuleName::new(without_extension.split('/'))?;
        let source = source.into();
        let input_bytes = source.len();
        Ok(Self {
            module: CompilationModuleId::new(namespace, module),
            file,
            source,
            authored: None,
            resolved_arrays: Default::default(),
            resolved_documents: Default::default(),
            input_bytes,
        })
    }

    /// Attach the owning package's verified immutable data assets.
    ///
    /// # Errors
    /// Rejects invalid names, array work excess, repeated attachment or hierarchy input excess.
    pub fn with_resolved_arrays(
        mut self,
        arrays: std::sync::Arc<BTreeMap<String, eqiora_schema::resolved_array::ResolvedF64Array>>,
    ) -> Result<Self, Diagnostic> {
        if !self.resolved_arrays.is_empty() || arrays.len() > 65_536 {
            return Err(resolved_error(
                "resolved array closure is already attached or exceeds its entry bound",
            ));
        }
        let mut bytes = self.input_bytes;
        let mut portable = BTreeSet::new();
        for (name, array) in arrays.iter() {
            ModuleName::new(name.split('.'))?;
            if !portable.insert(name.to_ascii_lowercase()) {
                return Err(resolved_error(
                    "resolved array names collide under ASCII case folding",
                ));
            }
            if name.len() > 512 || array.shape().len() > 8 || array.values().len() > 16_000_000 {
                return Err(resolved_error(
                    "resolved array asset exceeds name, rank or scalar limits",
                ));
            }
            bytes = array
                .values()
                .len()
                .checked_add(array.shape().len())
                .and_then(|count| count.checked_mul(8))
                .and_then(|size| size.checked_add(name.len()))
                .and_then(|size| size.checked_add(bytes))
                .ok_or_else(|| resolved_error("resolved array asset byte count overflow"))?;
        }
        preflight_resolved_hierarchy([bytes], 0)?;
        self.input_bytes = bytes;
        self.resolved_arrays = arrays;
        Ok(self)
    }

    /// Exact package-owned array names and immutable payloads available to this unit.
    #[must_use]
    pub fn resolved_arrays(
        &self,
    ) -> &BTreeMap<String, eqiora_schema::resolved_array::ResolvedF64Array> {
        &self.resolved_arrays
    }

    /// Attach exact package-owned UTF-8 attribution payloads, shared across modules.
    ///
    /// # Errors
    /// Rejects invalid/ambiguous names, invalid UTF-8, repeated attachment or input excess.
    pub fn with_resolved_documents(
        mut self,
        documents: std::sync::Arc<BTreeMap<String, std::sync::Arc<[u8]>>>,
    ) -> Result<Self, Diagnostic> {
        if !self.resolved_documents.is_empty() || documents.len() > 65_536 {
            return Err(resolved_error(
                "documentation closure is already attached or exceeds its entry bound",
            ));
        }
        let mut bytes = self.input_bytes;
        let mut portable = BTreeSet::new();
        for (name, content) in documents.iter() {
            ModuleName::new(name.split('.'))?;
            if name.len() > 512 || !portable.insert(name.to_ascii_lowercase()) {
                return Err(resolved_error(
                    "documentation asset names exceed bounds or collide under ASCII case folding",
                ));
            }
            std::str::from_utf8(content)
                .map_err(|_| resolved_error("documentation asset must be UTF-8"))?;
            bytes = content
                .len()
                .checked_add(name.len())
                .and_then(|size| bytes.checked_add(size))
                .ok_or_else(|| resolved_error("documentation asset byte count overflow"))?;
        }
        preflight_resolved_hierarchy([bytes], 0)?;
        self.input_bytes = bytes;
        self.resolved_documents = documents;
        Ok(self)
    }

    /// Exact package-owned attribution names and immutable UTF-8 payloads.
    #[must_use]
    pub fn resolved_documents(&self) -> &BTreeMap<String, std::sync::Arc<[u8]>> {
        &self.resolved_documents
    }

    /// Admit a directly authored module under the same explicit path and package
    /// identity as a parsed source unit. Compilation consumes the owned AST;
    /// formatting is performed only when a consumer explicitly requests text.
    ///
    /// # Errors
    /// Rejects invalid paths and the existing compiler AST/source resource limits.
    pub fn from_module(
        namespace: CompilationNamespaceId,
        file: impl Into<String>,
        module: eqiora_lang::Module,
    ) -> Result<Self, Diagnostic> {
        let input_bytes = crate::source_identity::module_input_bytes(&module)?;
        preflight_resolved_hierarchy([input_bytes], 0)?;
        let mut unit = Self::new(namespace, file, "")?;
        unit.authored = Some(module);
        unit.input_bytes = input_bytes;
        Ok(unit)
    }

    /// Owning package namespace.
    #[must_use]
    pub const fn namespace(&self) -> &CompilationNamespaceId {
        self.module.owner()
    }

    /// Exact logical module owning this source unit.
    #[must_use]
    pub(crate) const fn module(&self) -> &CompilationModuleId {
        &self.module
    }

    /// Path-derived logical module segments below the source root.
    #[must_use]
    pub fn module_segments(&self) -> &[String] {
        self.module.name().segments()
    }

    /// Canonical source import spelling; exact ownership remains in `namespace`.
    #[must_use]
    pub fn import_path(&self) -> String {
        self.module.import_path()
    }

    /// Provenance path supplied by the exact source bundle.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Original UTF-8 source, or explicitly requested canonical AST emission.
    #[must_use]
    pub fn source(&self) -> std::borrow::Cow<'_, str> {
        self.authored.as_ref().map_or_else(
            || std::borrow::Cow::Borrowed(self.source.as_str()),
            |module| std::borrow::Cow::Owned(eqiora_lang::format(module.document())),
        )
    }

    /// Bounded input size: UTF-8 bytes for parsed text or canonical structural
    /// bytes for native AST. Neither path skips the compiler's AST budgets.
    #[must_use]
    pub const fn input_bytes(&self) -> usize {
        self.input_bytes
    }

    /// Package-qualified source label used by compiler diagnostics.
    ///
    /// The host-assigned module identity participates in the label.
    #[must_use]
    pub fn diagnostic_file(&self) -> String {
        resolved_source_label(self.module(), &self.file)
    }
}

/// One direct package dependency authorized by an already-resolved graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedDependency {
    declaring: CompilationNamespaceId,
    target: CompilationNamespaceId,
}

impl ResolvedDependency {
    /// Bind one declaring package to one exact direct dependency.
    #[must_use]
    pub const fn new(declaring: CompilationNamespaceId, target: CompilationNamespaceId) -> Self {
        Self { declaring, target }
    }

    /// Package that declares the dependency.
    #[must_use]
    pub const fn declaring(&self) -> &CompilationNamespaceId {
        &self.declaring
    }

    /// Exact package made directly importable.
    #[must_use]
    pub const fn target(&self) -> &CompilationNamespaceId {
        &self.target
    }
}

/// One source-authored local name for an imported canonical module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedAlias {
    declaring: CompilationModuleId,
    alias: String,
    target: CompilationModuleId,
    source_span: (String, TextRange),
}

impl ResolvedAlias {
    fn authored_import(
        declaring: CompilationModuleId,
        alias: impl Into<String>,
        target: CompilationModuleId,
        file: impl Into<String>,
        range: TextRange,
    ) -> Self {
        Self {
            declaring,
            alias: alias.into(),
            target,
            source_span: (file.into(), range),
        }
    }

    pub(crate) fn source_span(&self) -> (&str, TextRange) {
        (self.source_span.0.as_str(), self.source_span.1)
    }

    /// Logical module containing the import.
    #[must_use]
    pub(crate) const fn declaring_module(&self) -> &CompilationModuleId {
        &self.declaring
    }

    /// Exact source identifier used as the first path segment.
    #[must_use]
    pub fn alias(&self) -> &str {
        &self.alias
    }

    /// Exact logical module selected by the import.
    #[must_use]
    pub(crate) const fn target_module(&self) -> &CompilationModuleId {
        &self.target
    }
}

/// Closed input for one exact multi-package hierarchy analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedHierarchyInput {
    root: CompilationModuleId,
    units: Vec<ResolvedSourceUnit>,
    dependencies: Vec<ResolvedDependency>,
}

impl ResolvedHierarchyInput {
    /// Construct an input graph. Structural validation intentionally belongs
    /// to the joint analysis phase.
    #[must_use]
    pub fn new(
        root: CompilationNamespaceId,
        units: Vec<ResolvedSourceUnit>,
        dependencies: Vec<ResolvedDependency>,
    ) -> Self {
        Self {
            root: CompilationModuleId::main(root),
            units,
            dependencies,
        }
    }

    /// Construct an input graph with an explicit root logical module.
    pub fn with_root_module<I, S>(
        owner: CompilationNamespaceId,
        module: I,
        units: Vec<ResolvedSourceUnit>,
        dependencies: Vec<ResolvedDependency>,
    ) -> Result<Self, Diagnostic>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Ok(Self {
            root: CompilationModuleId::new(owner, ModuleName::new(module)?),
            units,
            dependencies,
        })
    }

    /// Exact namespace containing the selected executable model.
    #[must_use]
    pub const fn root(&self) -> &CompilationNamespaceId {
        self.root.owner()
    }

    /// Source units in resolver-provided order. Meaning does not depend on
    /// this order.
    #[must_use]
    pub fn units(&self) -> &[ResolvedSourceUnit] {
        &self.units
    }

    /// Exact direct dependencies in resolver-provided order.
    #[must_use]
    pub fn dependencies(&self) -> &[ResolvedDependency] {
        &self.dependencies
    }

    /// Analyze this graph while observing cooperative cancellation between
    /// source units and semantic analysis stages.
    ///
    /// `Ok(None)` publishes no partial analysis when `is_cancelled` returns
    /// true. Parser or semantic failures remain ordered diagnostics.
    ///
    /// # Errors
    /// Returns ordered parser or semantic diagnostics observed before
    /// cancellation.
    pub fn analyze_with_cancellation(
        self,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<Option<AnalyzedResolvedHierarchy>, Vec<Diagnostic>> {
        analyze::analyze_resolved_hierarchy_with_cancellation(self, is_cancelled)
    }
}

/// Validate the allocation footprint of one resolved hierarchy before source
/// strings or alias records are copied into compiler-owned inputs.
///
/// This is the compiler-owned resource authority used by composition layers
/// that borrow source bundles. Passing this preflight does not replace the
/// structural and semantic checks in [`analyze_resolved_hierarchy`].
///
/// # Errors
///
/// Returns a diagnostic when the source-unit count, alias count, one source
/// length, aggregate source length, or a checked counter exceeds the compiler
/// v1 hierarchy bounds.
pub fn preflight_resolved_hierarchy<I>(
    source_byte_lengths: I,
    alias_count: usize,
) -> Result<(), Diagnostic>
where
    I: IntoIterator<Item = usize>,
{
    preflight_resolved_hierarchy_with_limits(
        source_byte_lengths,
        alias_count,
        resolved_hierarchy_limits(),
    )
}

fn preflight_resolved_hierarchy_with_limits<I>(
    source_byte_lengths: I,
    alias_count: usize,
    limits: ResolvedHierarchyResourceLimits,
) -> Result<(), Diagnostic>
where
    I: IntoIterator<Item = usize>,
{
    if alias_count > limits.aliases {
        return Err(resolved_error(format!(
            "resolved hierarchy exceeds the {} module-link limit",
            limits.aliases
        )));
    }
    let mut source_units = 0_usize;
    let mut total_source_bytes = 0_usize;
    for source_bytes in source_byte_lengths {
        source_units = source_units
            .checked_add(1)
            .ok_or_else(|| resolved_error("resolved hierarchy source-unit count overflow"))?;
        if source_units > limits.source_units {
            return Err(resolved_error(format!(
                "resolved hierarchy exceeds the {} source-unit limit",
                limits.source_units
            )));
        }
        if source_bytes > limits.source_unit_bytes {
            return Err(resolved_error(format!(
                "source requires {source_bytes} bytes, exceeding the {} byte hierarchy limit",
                limits.source_unit_bytes
            )));
        }
        total_source_bytes = total_source_bytes
            .checked_add(source_bytes)
            .ok_or_else(|| resolved_error("resolved hierarchy source-byte count overflow"))?;
        if total_source_bytes > limits.total_source_bytes {
            return Err(resolved_error(format!(
                "resolved hierarchy exceeds the {} total source-byte limit",
                limits.total_source_bytes
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct AnalyzedSourceUnit {
    pub(crate) native: Option<std::sync::Arc<eqiora_lang::Module>>,
    pub(crate) resolved_documents: std::sync::Arc<BTreeMap<String, std::sync::Arc<[u8]>>>,
    pub(crate) resolved_arrays:
        std::sync::Arc<BTreeMap<String, eqiora_schema::resolved_array::ResolvedF64Array>>,
    pub(crate) module: CompilationModuleId,
    pub(crate) file: String,
    pub(crate) source_bytes: usize,
    pub(crate) document: Document,
    pub(crate) authored_document: std::sync::Arc<Document>,
}

/// Completely parsed and globally indexed package hierarchy.
///
/// This value is the comparison barrier: callers inspect
/// [`Self::canonical_declarations`], compare package semantic content, and
/// cross [`Self::validate_definitions`] before
/// [`ValidatedResolvedHierarchy::compile_root`] becomes available.
#[derive(Clone, Debug)]
pub struct AnalyzedResolvedHierarchy {
    pub(crate) completion: std::sync::Arc<crate::hierarchy::CompletionIndex>,
    pub(crate) root: CompilationModuleId,
    pub(crate) units: Vec<AnalyzedSourceUnit>,
    pub(crate) aliases: Vec<ResolvedAlias>,
    canonical_declarations: Box<[CanonicalDeclarationIdentity]>,
    declaration_locations: Box<[(String, TextRange)]>,
    reference_locations: Box<[(usize, String, TextRange)]>,
    property_bindings: Box<[crate::property::ResolvedPropertyBinding]>,
    pub(crate) property_catalog: std::sync::Arc<crate::property::catalog::Catalog>,
}

impl AnalyzedResolvedHierarchy {
    /// Exact namespace containing the selected executable Model.
    #[must_use]
    pub const fn root(&self) -> &CompilationNamespaceId {
        self.root.owner()
    }

    /// Package-qualified source labels in resolved-graph order.
    #[must_use]
    pub fn resolved_source_files(&self) -> impl ExactSizeIterator<Item = &str> {
        self.units.iter().map(|unit| unit.file.as_str())
    }

    /// Compiler-resolved top-level declarations and their definition locations.
    #[must_use]
    pub fn resolved_declarations(
        &self,
    ) -> impl ExactSizeIterator<Item = (&CanonicalDeclarationIdentity, &str, TextRange)> {
        self.canonical_declarations
            .iter()
            .zip(self.declaration_locations.iter())
            .map(|(identity, (file, range))| (identity, file.as_str(), *range))
    }

    /// Resolve a structural dimension in the root module or one direct public import.
    ///
    /// The result is a structural SI dimension, never a nominal declaration identity.
    /// # Errors
    /// Rejects absent, private, wrong-kind, or transitive declaration paths.
    pub fn dimension_alias(&self, name: &str) -> Result<eqiora_core::DimExponents, Diagnostic> {
        let parts = name.splitn(3, '.').collect::<Vec<_>>();
        let selected = match parts.as_slice() {
            [_] => Some((&self.root, name, false)),
            [alias, local] => self
                .aliases
                .iter()
                .find(|edge| edge.declaring_module() == &self.root && edge.alias() == *alias)
                .map(|edge| (edge.target_module(), *local, true)),
            _ => None,
        };
        if let Some((module, local, imported)) = selected
            && let Some(unit) = self.units.iter().find(|unit| &unit.module == module)
            && let Some(declaration) = unit
                .document
                .dimensions()
                .iter()
                .find(|declaration| declaration.name() == local)
            && (!imported || declaration.visibility() == VisibilitySyntax::Public)
        {
            return crate::dimensions::lower_dimension(&unit.file, declaration.value());
        }
        Err(crate::diagnostics::source_error(
            codes::LANGUAGE_TYPE_ERROR,
            "modules",
            TextRange::new(0, 0),
            format!(
                "unknown or private dimension alias `{name}`; expected local or direct imported declaration"
            ),
        ))
    }

    /// Read-only nominal property bindings retained through elaboration.
    #[must_use]
    pub fn property_bindings(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            Option<&str>,
            &str,
            &str,
            &str,
            &str,
            &eqiora_schema::kernel::PropertyMeaning,
            &str,
            &str,
            &str,
        ),
    > {
        self.property_bindings.iter().map(|value| {
            (
                value.composition(),
                value.contract(),
                value.release(),
                value.component(),
                value.requirement(),
                value.meaning(),
                value.validity(),
                value.citation(),
                value.license(),
            )
        })
    }

    /// Validate every distributable definition before selected-root
    /// elaboration.
    ///
    /// # Errors
    /// Returns accumulated definition diagnostics. No Model, graph
    /// Transaction, occurrence identity, or provenance entry is created.
    pub fn validate_definitions(self) -> Result<ValidatedResolvedHierarchy, Vec<Diagnostic>> {
        let checked =
            crate::hierarchy::validate_resolved_definitions(&self, HierarchyLimits::default())
                .map_err(|errors| self.native_diagnostics(errors))?;
        Ok(ValidatedResolvedHierarchy {
            analysis: self,
            checked,
        })
    }

    pub(crate) fn native_diagnostics(&self, errors: Vec<Diagnostic>) -> Vec<Diagnostic> {
        let modules = self
            .units
            .iter()
            .filter_map(|unit| {
                unit.native
                    .as_ref()
                    .filter(|module| module.source_file().is_none())
                    .map(|module| (unit.file.clone(), module.as_ref().clone()))
            })
            .collect::<Vec<_>>();
        source::native_diagnostics(&modules, errors)
    }

    pub(crate) fn retain_authored_provenance(&self, model: CompiledModel) -> CompiledModel {
        let synthetic = self
            .units
            .iter()
            .filter(|unit| {
                unit.native
                    .as_ref()
                    .is_some_and(|module| module.source_file().is_none())
            })
            .map(|unit| unit.file.as_str())
            .collect::<BTreeSet<_>>();
        model.retain_source_files(|file| !synthetic.contains(file))
    }
}

/// A resolved hierarchy whose complete distributable definition graph has
/// crossed the compiler-owned static validation barrier.
#[derive(Clone, Debug)]
pub struct ValidatedResolvedHierarchy {
    pub(crate) analysis: AnalyzedResolvedHierarchy,
    pub(crate) checked: crate::hierarchy::CheckedDefinitionGraph,
}

impl ValidatedResolvedHierarchy {
    /// Exact namespace containing the selected executable Model.
    #[must_use]
    pub const fn root(&self) -> &CompilationNamespaceId {
        self.analysis.root()
    }

    /// Compiler-canonical declarations in `(namespace, path, kind)` order.
    #[must_use]
    pub fn canonical_declarations(&self) -> &[CanonicalDeclarationIdentity] {
        self.analysis.canonical_declarations()
    }

    /// Elaborate one root-local or directly imported public Model.
    ///
    /// An imported entry uses exactly `alias.Model`, where `alias` is declared
    /// by the root module or exact root package. Transitive qualification and
    /// private Models fail closed.
    ///
    /// # Errors
    /// Returns accumulated occurrence diagnostics. No partial transaction is
    /// returned.
    pub fn compile_root(&self, model: &str) -> Result<CompiledModel, Vec<Diagnostic>> {
        crate::hierarchy::compile_resolved_hierarchy(
            &self.analysis,
            &self.checked,
            model,
            HierarchyLimits::default(),
        )
        .map(|model| self.analysis.retain_authored_provenance(model))
        .map_err(|errors| self.analysis.native_diagnostics(errors))
    }
}

mod canonical;
use canonical::{canonical_declaration_path, collect_canonical_declarations};

fn is_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

const fn visibility_rank(visibility: VisibilitySyntax) -> u8 {
    match visibility {
        VisibilitySyntax::Private => 0,
        VisibilitySyntax::Public => 1,
    }
}

fn declaration_identity_form(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(105);
    result.push_str("eqiora.source-declaration.v1:sha256:");
    for byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn pure_operator_identity_form(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(111);
    result.push_str("eqiora.pure-operator-definition.v1:sha256:");
    for byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn resolved_source_label(module: &CompilationModuleId, file: &str) -> String {
    use core::fmt::Write;

    let mut label = String::new();
    write!(
        label,
        "eqiora-package-v1:{}:",
        module.owner().segments().len()
    )
    .expect("String writes cannot fail");
    for segment in module.owner().segments() {
        write!(label, "{}:{segment}:", segment.len()).expect("String writes cannot fail");
    }
    write!(label, "module:{}:", module.name().segments().len()).expect("String writes cannot fail");
    for segment in module.name().segments() {
        write!(label, "{}:{segment}:", segment.len()).expect("String writes cannot fail");
    }
    label.push_str(file);
    label
}

fn resolved_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}

#[cfg(test)]
mod tests;
