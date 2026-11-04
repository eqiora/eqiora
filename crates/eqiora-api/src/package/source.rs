//! Ordinary source compilation with explicit import policy and retained lineage.

use std::collections::BTreeMap;

use eqiora_compiler::StaticBindingValue;
use eqiora_core::{Diagnostic, Span, diagnostic::codes};
use eqiora_package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1, QualifiedName,
    ResolutionRecordV1, SourceFileV1,
};

use super::{
    PackageCompilationError, PackagePreparationError, PackagedModelDocument,
    bundled_package_names, bundled_package_releases, prepare_package_release_v1,
};
use crate::ModelDocument;

const SOURCE_PACKAGE: &str = "Eqiora.Source.Root";
const SOURCE_PATH: &str = "src/main.eqi";

/// Import policy for ordinary source compilation. Neither policy searches disk or the network.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SourceImportContext {
    /// Resolve imports using only the exact releases embedded in this distribution.
    #[default]
    Bundled,
    /// Use the existing single-source compiler without implicit package inclusion.
    Closed,
}

/// The complete result of ordinary compilation, including package lineage when imports use it.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceCompilation {
    /// A source compiled without implicit package inclusion.
    Local(ModelDocument),
    /// A source compiled with a verified exact embedded dependency graph.
    Packaged(PackagedModelDocument),
}

impl SourceCompilation {
    /// The admitted model, without discarding the containing result's sidecars.
    #[must_use]
    pub const fn model(&self) -> &ModelDocument {
        match self {
            Self::Local(model) => model,
            Self::Packaged(packaged) => packaged.model(),
        }
    }
}

/// Typed failure from source admission, exact release preparation, or compilation.
#[derive(Debug)]
pub enum SourceCompilationError {
    /// The exact source release or dependency graph could not be prepared.
    Preparation(PackagePreparationError),
    /// Exact resolution, compilation, or artifact admission failed.
    Compilation(PackageCompilationError),
}

impl SourceCompilationError {
    /// Compiler diagnostics when the error originated in source analysis.
    #[must_use]
    pub fn diagnostics(&self) -> Option<&[Diagnostic]> {
        match self {
            Self::Preparation(PackagePreparationError::Diagnostics(diagnostics))
            | Self::Compilation(PackageCompilationError::Diagnostics(diagnostics)) => Some(diagnostics),
            _ => None,
        }
    }
}

impl std::fmt::Display for SourceCompilationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(error) => error.fmt(formatter),
            Self::Compilation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SourceCompilationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preparation(error) => Some(error),
            Self::Compilation(error) => Some(error),
        }
    }
}

impl From<PackagePreparationError> for SourceCompilationError {
    fn from(error: PackagePreparationError) -> Self { Self::Preparation(error) }
}
impl From<PackageCompilationError> for SourceCompilationError {
    fn from(error: PackageCompilationError) -> Self { Self::Compilation(error) }
}
impl From<eqiora_package::ContractError> for SourceCompilationError {
    fn from(error: eqiora_package::ContractError) -> Self {
        Self::Preparation(error.into())
    }
}
impl From<Vec<Diagnostic>> for SourceCompilationError {
    fn from(error: Vec<Diagnostic>) -> Self {
        Self::Compilation(error.into())
    }
}

/// Compile ordinary source through the shared application boundary.
///
/// Without imports, the existing single-source path and identities are preserved.
/// With bundled imports, a deterministic in-memory root release at `src/main.eqi`
/// depends only on declared embedded packages and their complete exact closures.
/// `filename` affects diagnostics, never the packaged source identity or provenance.
/// An omitted entry selects exactly one root-local Model, never a dependency Model.
/// Explicit locked compilation remains [`PackagedModelDocument::compile_selected`].
///
/// # Errors
/// Returns typed source, import, visibility, exact resolution, binding, or artifact failures.
pub fn compile_source(
    filename: &str,
    source: &str,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
    context: SourceImportContext,
) -> Result<SourceCompilation, SourceCompilationError> {
    if context == SourceImportContext::Closed {
        return compile_local(filename, source, entry, bindings);
    }
    let document = eqiora_lang::parse(filename, source).into_document()?;
    if document.imports().len() == 0 {
        return compile_local(filename, source, entry, bindings);
    }
    let selected = match entry {
        Some(entry) => entry,
        None => match document.models() {
            [model] => model.name(),
            models => return Err(vec![Diagnostic::error(
                codes::LANGUAGE_LOWERING_ERROR,
                format!("compile requires exactly one root-local model declaration, found {}", models.len()),
            )].into()),
        },
    };
    let mut dependencies = BTreeMap::new();
    let mut direct = BTreeMap::new();
    for (module, _, range) in document.imports() {
        let name = bundled_package_names().iter().copied().filter(|name| {
            module.as_str() == *name || module.as_str().strip_prefix(name).is_some_and(|suffix| suffix.starts_with('.'))
        }).max_by_key(|name| name.len()).ok_or_else(|| {
            vec![Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR,
                format!("import `{module}` does not name a package embedded in this distribution"))
                .with_span(Span { file: filename.to_owned(), start: range.start(), end: range.end() })]
        })?;
        let closure = bundled_package_releases(name)?;
        for release in closure {
            let identity = release.package_identity()?;
            if identity.name.as_str() == SOURCE_PACKAGE {
                return Err(PackagePreparationError::BundledCatalog(format!("embedded package collides with reserved source root `{SOURCE_PACKAGE}`")).into());
            }
            if identity.name.as_str() == name {
                direct.insert(identity.name.clone(), PackageDependencyV1::new(identity.clone()));
            }
            if let Some(previous) = dependencies.insert(identity.name.clone(), release.clone()) {
                if previous != release {
                    return Err(PackagePreparationError::BundledCatalog(format!("embedded closures disagree on exact package `{}`", identity.name)).into());
                }
            }
        }
    }
    let dependencies = dependencies.into_values().collect::<Vec<PackageReleaseV1>>();
    let path = NormalizedRelativePath::parse(SOURCE_PATH)?;
    let manifest = PackageManifestV1::new(
        "main", QualifiedName::parse(SOURCE_PACKAGE)?, ExactVersion::parse("0.0.0")?,
        direct.into_values().collect(), vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )?;
    let sources = PackageSourcesV1::new(manifest, vec![SourceFileV1::new(path, BundleRoleV1::ModelSource, source.as_bytes().to_vec())])?;
    let root = prepare_package_release_v1(sources, &dependencies)?;
    let resolution = ResolutionRecordV1::from_exact_releases(&root, &dependencies)
        .map_err(PackagePreparationError::from)?;
    let mut store = InMemoryPackageStore::default();
    for release in std::iter::once(&root).chain(&dependencies) {
        store.insert(release).map_err(|error| PackagePreparationError::Resolution(error.into()))?;
    }
    PackagedModelDocument::compile_selected(&store, &resolution, selected, bindings)
        .map(SourceCompilation::Packaged).map_err(Into::into)
}

fn compile_local(
    filename: &str, source: &str, entry: Option<&str>, bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<SourceCompilation, SourceCompilationError> {
    match entry {
        Some(entry) => ModelDocument::compile_selected(filename, source, entry, bindings),
        None if bindings.is_empty() => ModelDocument::compile(filename, source),
        None => Err(vec![Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, "static bindings require an explicit entry")]),
    }.map(SourceCompilation::Local).map_err(Into::into)
}
