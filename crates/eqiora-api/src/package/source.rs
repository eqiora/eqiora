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
    PackageCompilationError, PackagePreparationError, PackagedModelDocument, bundled_package_names,
    bundled_package_releases, prepare_package_release_v1,
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
    Local(Box<ModelDocument>),
    /// A source compiled with a verified exact embedded dependency graph.
    Packaged(Box<PackagedModelDocument>),
}

impl SourceCompilation {
    /// The admitted model, without discarding the containing result's sidecars.
    #[must_use]
    pub fn model(&self) -> &ModelDocument {
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
            | Self::Compilation(PackageCompilationError::Diagnostics(diagnostics)) => {
                Some(diagnostics)
            }
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
    fn from(error: PackagePreparationError) -> Self {
        Self::Preparation(error)
    }
}
impl From<PackageCompilationError> for SourceCompilationError {
    fn from(error: PackageCompilationError) -> Self {
        Self::Compilation(error)
    }
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
    if entry.is_none() && !bindings.is_empty() {
        return Err(vec![Diagnostic::error(
            codes::LANGUAGE_LOWERING_ERROR,
            "static bindings require an explicit entry",
        )]
        .into());
    }
    if context == SourceImportContext::Closed {
        return compile_local(filename, source, entry, bindings);
    }
    let parsed = eqiora_lang::parse(filename, source);
    if parsed
        .document()
        .is_none_or(|document| document.imports().len() == 0)
    {
        return compile_local(filename, source, entry, bindings);
    }
    let document = parsed.into_document()?;
    let selected = match entry {
        Some(entry) => entry,
        None => match document.models() {
            [model] => model.name(),
            models => {
                return Err(vec![Diagnostic::error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    format!(
                        "compile requires exactly one root-local model declaration, found {}",
                        models.len()
                    ),
                )]
                .into());
            }
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
                return Err(PackagePreparationError::BundledCatalog(format!(
                    "embedded package collides with reserved source root `{SOURCE_PACKAGE}`"
                ))
                .into());
            }
            if identity.name.as_str() == name {
                direct.insert(
                    identity.name.clone(),
                    PackageDependencyV1::new(identity.clone()),
                );
            }
            if let Some(previous) = dependencies.insert(identity.name.clone(), release.clone())
                && previous != release
            {
                return Err(PackagePreparationError::BundledCatalog(format!(
                    "embedded closures disagree on exact package `{}`",
                    identity.name
                ))
                .into());
            }
        }
    }
    let dependencies = dependencies
        .into_values()
        .collect::<Vec<PackageReleaseV1>>();
    let path = NormalizedRelativePath::parse(SOURCE_PATH)?;
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(SOURCE_PACKAGE)?,
        ExactVersion::parse("0.0.0")?,
        direct.into_values().collect(),
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )?;
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )?;
    let preparation_label = eqiora_compiler::ResolvedSourceUnit::new(
        super::authoring_namespace(sources.manifest()).map_err(|error| vec![error])?,
        SOURCE_PATH,
        source,
    )
    .map_err(|error| vec![error])?
    .diagnostic_filename();
    let root = prepare_package_release_v1(sources, &dependencies)
        .map_err(|error| display_error(error.into(), &preparation_label, filename))?;
    let exact_label = eqiora_compiler::ResolvedSourceUnit::new(
        super::exact_namespace(&root.package_identity()?).map_err(|error| vec![error])?,
        SOURCE_PATH,
        source,
    )
    .map_err(|error| vec![error])?
    .diagnostic_filename();
    let resolution = ResolutionRecordV1::from_exact_releases(&root, &dependencies)
        .map_err(PackagePreparationError::from)?;
    let mut store = InMemoryPackageStore::default();
    for release in std::iter::once(&root).chain(&dependencies) {
        store
            .insert(release)
            .map_err(|error| PackagePreparationError::Resolution(error.into()))?;
    }
    PackagedModelDocument::compile_selected(&store, &resolution, selected, bindings)
        .map(|packaged| SourceCompilation::Packaged(Box::new(packaged)))
        .map_err(|error| display_error(error.into(), &exact_label, filename))
}

fn display_error(
    mut error: SourceCompilationError,
    root_label: &str,
    filename: &str,
) -> SourceCompilationError {
    let diagnostics = match &mut error {
        SourceCompilationError::Preparation(PackagePreparationError::Diagnostics(diagnostics))
        | SourceCompilationError::Compilation(PackageCompilationError::Diagnostics(diagnostics)) => {
            diagnostics
        }
        _ => return error,
    };
    for diagnostic in diagnostics {
        if let Some(span) = diagnostic
            .source_span()
            .filter(|span| span.file == root_label)
        {
            let span = Span {
                file: filename.to_owned(),
                start: span.start,
                end: span.end,
            };
            *diagnostic = diagnostic.clone().with_span(span);
        }
    }
    error
}

fn compile_local(
    filename: &str,
    source: &str,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<SourceCompilation, SourceCompilationError> {
    match entry {
        Some(entry) => ModelDocument::compile_selected(filename, source, entry, bindings),
        None => ModelDocument::compile(filename, source),
    }
    .map(|model| SourceCompilation::Local(Box::new(model)))
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_package::ExactResolver;

    const CIRCUIT: &str = include_str!("../../../../examples/voltage-divider/src/main.eqi");

    fn compile(source: &str) -> Result<SourceCompilation, SourceCompilationError> {
        compile_source(
            "display.eqi",
            source,
            None,
            &[],
            SourceImportContext::Bundled,
        )
    }

    #[test]
    fn catalog_inventory_is_exact_and_filesystem_independent() {
        for name in bundled_package_names() {
            let closure = bundled_package_releases(name).unwrap();
            let (root, dependencies) = closure.split_last().unwrap();
            assert_eq!(root.package_identity().unwrap().name.as_str(), *name);
            let record = ResolutionRecordV1::from_exact_releases(root, dependencies).unwrap();
            ExactResolver
                .resolve_releases(&record, root, dependencies)
                .unwrap();
        }
        assert!(matches!(
            bundled_package_releases("Eqiora.NotEmbedded"),
            Err(PackagePreparationError::BundledCatalog(_))
        ));
    }

    #[test]
    fn ordinary_import_matches_explicit_exact_graph_and_ignores_display_path() {
        let SourceCompilation::Packaged(actual) = compile(CIRCUIT).unwrap() else {
            panic!("package lineage required")
        };
        let moved = compile_source(
            "/another/host/moved.eqi",
            CIRCUIT,
            None,
            &[],
            SourceImportContext::Bundled,
        )
        .unwrap();
        assert_eq!(moved, SourceCompilation::Packaged(actual.clone()));
        assert!(!actual.provenance().is_empty());

        // Construct the same exact graph explicitly through the public package contracts.
        let dependencies = bundled_package_releases("Eqiora.Electrical.Basic").unwrap();
        let path = NormalizedRelativePath::parse("src/main.eqi").unwrap();
        let manifest = PackageManifestV1::new(
            "main",
            QualifiedName::parse("Eqiora.Source.Root").unwrap(),
            ExactVersion::parse("0.0.0").unwrap(),
            vec![PackageDependencyV1::new(
                dependencies[0].package_identity().unwrap(),
            )],
            vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
        )
        .unwrap();
        let sources = PackageSourcesV1::new(
            manifest,
            vec![SourceFileV1::new(
                path,
                BundleRoleV1::ModelSource,
                CIRCUIT.as_bytes().to_vec(),
            )],
        )
        .unwrap();
        let root = prepare_package_release_v1(sources, &dependencies).unwrap();
        let record = ResolutionRecordV1::from_exact_releases(&root, &dependencies).unwrap();
        let mut store = InMemoryPackageStore::default();
        for release in std::iter::once(&root).chain(&dependencies) {
            store.insert(release).unwrap();
        }
        let explicit =
            PackagedModelDocument::compile_locked(&store, &record, "VoltageDivider").unwrap();
        assert_eq!(*actual, explicit);
        actual.compilation().validate_against(&record).unwrap();

        // Locked compilation cannot replace absent or altered bytes with the embedded catalog.
        assert!(
            PackagedModelDocument::compile_locked(
                &InMemoryPackageStore::default(),
                &record,
                "VoltageDivider"
            )
            .is_err()
        );
        let source_digest = dependencies[0].source_digest().unwrap();
        store.insert_unchecked(source_digest, b"{}".to_vec());
        assert!(PackagedModelDocument::compile_locked(&store, &record, "VoltageDivider").is_err());
    }

    #[test]
    fn no_imports_preserve_local_identity_and_closed_policy_is_strict() {
        let source = "model Main() { variable x: 1; relation balance { x = 0; } }";
        let local = ModelDocument::compile("display.eqi", source).unwrap();
        assert_eq!(
            compile(source).unwrap(),
            SourceCompilation::Local(Box::new(local.clone()))
        );
        assert_eq!(
            compile_source(
                "display.eqi",
                source,
                None,
                &[],
                SourceImportContext::Closed
            )
            .unwrap(),
            SourceCompilation::Local(Box::new(local))
        );
        for source in [
            "",
            "// comments only",
            "model Main( {",
            "component Library() {}",
        ] {
            let original = ModelDocument::compile("display.eqi", source).unwrap_err();
            let dispatched = compile(source).unwrap_err();
            assert_eq!(dispatched.diagnostics(), Some(original.as_slice()));
        }
        let closed = compile_source(
            "display.eqi",
            CIRCUIT,
            None,
            &[],
            SourceImportContext::Closed,
        )
        .unwrap_err();
        assert!(closed.diagnostics().is_some());
    }

    #[test]
    fn static_bindings_require_entry_under_every_import_policy() {
        let expression = eqiora_lang::DraftExpression::constant(
            eqiora_lang::DecimalLiteral::parse("1").unwrap(),
        )
        .source_ast(|_| None, |_| None)
        .unwrap();
        let bindings = [("value", StaticBindingValue::Expression(&expression))];
        let mut expected = None;
        for source in ["model Main() {}", CIRCUIT] {
            for context in [SourceImportContext::Bundled, SourceImportContext::Closed] {
                let error =
                    compile_source("display.eqi", source, None, &bindings, context).unwrap_err();
                let diagnostics = error.diagnostics().unwrap().to_vec();
                if let Some(expected) = &expected {
                    assert_eq!(&diagnostics, expected);
                } else {
                    expected = Some(diagnostics);
                }
            }
        }
    }

    #[test]
    fn imports_require_embedded_package_and_real_module_and_public_member() {
        for source in [
            "import Eqiora.NotEmbedded.main as unknown; model Main() {}",
            "import Eqiora.Electrical.Basic.missing as unknown; model Main() {}",
            "import Eqiora.Electrical.Basic.basic as electrical; model Main() { instance x: electrical.Missing(); }",
            "import Eqiora.Electrical.Basic.basic as electrical; model First() {} model Second() {}",
            "import Eqiora.Electrical.Basic.basic as electrical; public component Only() {}",
        ] {
            assert!(compile(source).is_err(), "must reject {source}");
        }
    }

    #[test]
    fn packaged_error_displays_caller_filename_without_rewriting_provenance() {
        let source = "import Eqiora.Electrical.Basic.missing as electrical; model Main() {}";
        for filename in ["one.eqi", "/moved/two.eqi"] {
            let error = compile_source(filename, source, None, &[], SourceImportContext::Bundled)
                .unwrap_err();
            assert!(error.diagnostics().unwrap().iter().any(|diagnostic| {
                diagnostic
                    .source_span()
                    .is_some_and(|span| span.file == filename)
            }));
        }
    }

    #[test]
    fn selected_component_retains_geometry_and_package_provenance() {
        let source = r"import Eqiora.Fluid.Incompressible.incompressible as fluid;
public component Flow(support body: volume(ambient_dimension = 2), parameter viscosity: kg / (m * s)) {
  variable velocity: vector<m / s, 2> on body;
  variable pressure: kg / (m * s ^ 2) on body;
  instance balance: fluid.SteadyStokes2d(body = body, velocity = velocity, pressure = pressure, dynamic_viscosity = viscosity);
}";
        let geometry = eqiora_geometry::CanonicalGeometryV1::from_circular_hole_named_roles(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            "body",
            "inlet",
            "outlet",
            "walls",
            "walls",
            "cylinder",
        )
        .unwrap();
        let expression = eqiora_lang::parse(
            "binding.eqi",
            "model Main() { parameter viscosity: kg / (m * s) = 0.001[Pa * s]; }",
        )
        .into_document()
        .unwrap();
        let eqiora_lang::Item::Parameter(parameter) = &expression.models()[0].items()[0] else {
            panic!("parameter")
        };
        let bindings = [
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("body").unwrap(),
                    parent: None,
                },
            ),
            (
                "viscosity",
                StaticBindingValue::Expression(parameter.value()),
            ),
        ];
        let private = source.replacen("public component", "component", 1);
        let private_error = compile_source(
            "private-display.eqi",
            &private,
            Some("Flow"),
            &bindings,
            SourceImportContext::Bundled,
        )
        .unwrap_err();
        assert!(
            private_error
                .diagnostics()
                .unwrap()
                .iter()
                .any(
                    |diagnostic| diagnostic.message().contains("must be declared public")
                        && diagnostic
                            .source_span()
                            .is_some_and(|span| span.file == "private-display.eqi")
                )
        );
        let SourceCompilation::Packaged(compiled) = compile_source(
            "display.eqi",
            source,
            Some("Flow"),
            &bindings,
            SourceImportContext::Bundled,
        )
        .unwrap() else {
            panic!("package lineage")
        };
        assert!(!compiled.provenance().is_empty());
        assert!(!compiled.model().canonical_json().unwrap().is_empty());
        assert_eq!(
            compiled.model().aliases()["definition.viscosity"].kind(),
            eqiora_core::EntityKind::Parameter
        );
        assert_eq!(
            compiled.compilation().model_digest(),
            eqiora_package::CanonicalModelDigest::parse(&compiled.model().digest().unwrap())
                .unwrap()
        );
    }
}
