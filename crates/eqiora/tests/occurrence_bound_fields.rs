use std::collections::BTreeSet;

use eqiora::compiler::source_identity::{LocalSourceIdentity, LocalSourceIdentityLimits};
use eqiora::compiler::{CompiledModel, compile};
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora::kernel::{ExprNode, KernelNode, SymbolRef};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageCompilationError, PackageDependencyV1, PackageManifestV1, PackagePreparationError,
    PackageReleaseV1, PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1,
    SourceFileV1, prepare_package_release_v1,
};
use eqiora::sem::KernelProgram;
use eqiora::{Entity, Id, RawId};

const LOCAL_NESTED: &str =
    include_str!("../../../verify/packages/occurrence-bound-fields/models/nested-fields.eqi");
const VERSION: &str = "0.1.0";
const COMPONENT_PACKAGE: &str = r#"
public component FieldLaw(
  support body: volume(ambient_dimension = 2),
  variable scalar_state: 1 on body,
  variable displacement: vector<m, 2> on body,
) {

  relation scalar_identity on body {
    scalar_state - scalar_state = 0;
  }
  relation vector_identity on body {
    displacement - displacement = 0;
  }
}

public component FieldLawWrapper(
  support body: volume(ambient_dimension = 2),
  variable scalar_state: 1 on body,
  variable displacement: vector<m, 2> on body,
) {

  instance inner: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#;
const COMPONENT_PACKAGE_PERMUTED: &str = r#"
public component FieldLawWrapper(
  variable displacement: vector<m, 2> on body,
  variable scalar_state: 1 on body,
  support body: volume(ambient_dimension = 2),
) {

  instance inner: FieldLaw(
    displacement = displacement,
    scalar_state = scalar_state,
    body = body
  );
}

public component FieldLaw(
  variable displacement: vector<m, 2> on body,
  variable scalar_state: 1 on body,
  support body: volume(ambient_dimension = 2),
) {
  relation vector_identity on body {
    displacement - displacement = 0;
  }
  relation scalar_identity on body {
    scalar_state - scalar_state = 0;
  }

}
"#;

fn compile_one(file: &str, source: &str) -> CompiledModel {
    let mut compiled = compile(file, source).expect("Field-slot fixture compiles");
    assert_eq!(compiled.len(), 1, "fixture has exactly one root Model");
    compiled.pop().expect("one compiled Model")
}

fn typed<I>(compiled: &CompiledModel, name: &str) -> Id<I>
where
    I: Entity,
{
    compiled
        .symbols()
        .get(name)
        .and_then(RawId::downcast)
        .unwrap_or_else(|| panic!("missing typed symbol `{name}`"))
}

fn has_edge(
    program: &KernelProgram,
    kind: EdgeKind,
    from: impl Into<RawId>,
    to: impl Into<RawId>,
) -> bool {
    let from = from.into();
    let to = to.into();
    program
        .edges()
        .iter()
        .any(|edge| edge.kind() == kind && edge.from() == from && edge.to() == to)
}

fn admitted(compiled: CompiledModel) -> KernelProgram {
    let model = compiled.model();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("atomic graph admission");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("accepted Kernel program")
}

fn relation_field_symbols(
    program: &KernelProgram,
    relation: Id<kinds::Relation>,
) -> BTreeSet<RawId> {
    program
        .typed_relation_residual(relation)
        .expect("accepted Relation retains a typing proof")
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Field(field)) => Some(field.erase()),
            _ => None,
        })
        .collect()
}

fn assert_diagnostic_contains(diagnostics: &[eqiora::Diagnostic], fragments: &[&str]) {
    assert!(
        diagnostics.iter().any(|diagnostic| fragments
            .iter()
            .all(|fragment| diagnostic.message().contains(fragment))),
        "expected one diagnostic containing {fragments:?}, got {diagnostics:#?}"
    );
}

fn assert_compile_rejects_without_graph_mutation(name: &str, source: &str, expected: &[&str]) {
    let store = InMemoryGraphStore::new();
    let before = store.snapshot();
    let diagnostics = compile(&format!("{name}.eqi"), source).expect_err(name);
    assert_diagnostic_contains(&diagnostics, expected);
    let after = store.snapshot();
    assert_eq!(
        after.revision(),
        before.revision(),
        "{name}: graph revision"
    );
    assert_eq!(after.nodes().len(), before.nodes().len(), "{name}: nodes");
    assert_eq!(after.edges().len(), before.edges().len(), "{name}: edges");
    assert_eq!(
        after.ontology_views().len(),
        before.ontology_views().len(),
        "{name}: ontology views"
    );
}

fn source_identity(source: &str) -> LocalSourceIdentity {
    let document = eqiora::language::parse("identity.eqi", source)
        .into_document()
        .expect("identity fixture parses");
    let identity = LocalSourceIdentity::from_document(&document).expect("bounded source identity");
    assert_eq!(
        identity.namespace().unwrap().segments()[0],
        "local-source-v18"
    );
    identity
}

fn package_sources(
    name: &str,
    dependencies: Vec<PackageDependencyV1>,
    source: &str,
    reverse_files: bool,
) -> PackageSourcesV1 {
    let model_path = NormalizedRelativePath::parse("src/model.eqi").expect("model source path");
    let readme_path = NormalizedRelativePath::parse("README.md").expect("documentation path");
    let manifest = PackageManifestV1::new(
        "model",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(VERSION).expect("package version"),
        dependencies,
        vec![
            BundleEntryV1::new(model_path.clone(), BundleRoleV1::ModelSource),
            BundleEntryV1::new(readme_path.clone(), BundleRoleV1::Documentation),
        ],
    )
    .expect("closed package manifest");
    let mut files = vec![
        SourceFileV1::new(
            model_path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        ),
        SourceFileV1::new(
            readme_path,
            BundleRoleV1::Documentation,
            b"Occurrence-bound Field evidence.\n".to_vec(),
        ),
    ];
    if reverse_files {
        files.reverse();
    }
    PackageSourcesV1::new(manifest, files).expect("closed package sources")
}

fn component_release(source: &str, reverse_files: bool) -> PackageReleaseV1 {
    prepare_package_release_v1(
        package_sources(
            "Eqiora.Verify.OccurrenceBoundFields",
            Vec::new(),
            source,
            reverse_files,
        ),
        &[],
    )
    .expect("occurrence-free Field-slot definitions validate")
}

fn root_source(alias: &str, permuted: bool) -> String {
    if permuted {
        format!(
            r#"
import Eqiora.Verify.OccurrenceBoundFields.model as {alias};

model Main() {{
  instance law: {alias}.FieldLawWrapper(
    displacement = displacement,
    scalar_state = scalar_state,
    body = body
  );
  variable displacement: vector<m, 2> on body;
  variable scalar_state: 1 on body;

  domain body = box(0, 1, 0, 1);
}}
"#
        )
    } else {
        format!(
            r#"
import Eqiora.Verify.OccurrenceBoundFields.model as {alias};

model Main() {{
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: vector<m, 2> on body;
  instance law: {alias}.FieldLawWrapper(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}}
"#
        )
    }
}

fn root_release(
    components: &PackageReleaseV1,
    alias: &str,
    permuted: bool,
    reverse_files: bool,
) -> PackageReleaseV1 {
    let dependency = PackageDependencyV1::new(
        components
            .package_identity()
            .expect("component package identity"),
    );
    prepare_package_release_v1(
        package_sources(
            "org.eqiora.verify.occurrence_bound_fields",
            vec![dependency],
            &root_source(alias, permuted),
            reverse_files,
        ),
        std::slice::from_ref(components),
    )
    .expect("root Field-slot package validates")
}

fn compile_locked(components: &PackageReleaseV1, root: &PackageReleaseV1) -> PackagedModelDocument {
    let resolution =
        ResolutionRecordV1::from_exact_releases(root, std::slice::from_ref(components))
            .expect("exact offline resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(components).expect("insert component release");
    store.insert(root).expect("insert root release");
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact package occurrence");
    packaged
        .compilation()
        .validate_against(&resolution)
        .expect("package compilation matches exact resolution");
    packaged
}

#[test]
fn nested_slots_disappear_into_exact_field_and_support_identity() {
    let compiled = compile_one("nested-fields.eqi", LOCAL_NESTED);
    let body = typed::<kinds::Domain>(&compiled, "body");
    let scalar = typed::<kinds::Field>(&compiled, "scalar_state");
    let displacement = typed::<kinds::Field>(&compiled, "displacement");
    let scalar_relation = typed::<kinds::Relation>(&compiled, "law.inner.scalar_identity");
    let vector_relation = typed::<kinds::Relation>(&compiled, "law.inner.vector_identity");

    for eliminated in [
        "law.body",
        "law.scalar_state",
        "law.displacement",
        "law.inner.body",
        "law.inner.scalar_state",
        "law.inner.displacement",
    ] {
        assert_eq!(
            compiled.symbols().get(eliminated),
            None,
            "slot `{eliminated}` must not become a display alias"
        );
    }

    let provenance = compiled.provenance().expect("hierarchy provenance");
    for relation in [scalar_relation, vector_relation] {
        let source = provenance
            .get_by_graph_id(relation.erase())
            .expect("expanded Relation provenance");
        let bindings = source
            .binding_spans()
            .iter()
            .map(|span| &LOCAL_NESTED[span.start as usize..span.end as usize])
            .collect::<Vec<_>>();
        assert_eq!(
            bindings.len(),
            6,
            "two levels each contribute three bindings"
        );
        for expected in [
            "body = body",
            "scalar_state = scalar_state",
            "displacement = displacement",
        ] {
            assert_eq!(
                bindings
                    .iter()
                    .filter(|binding| **binding == expected)
                    .count(),
                2,
                "inner and outer occurrences retain the `{expected}` span"
            );
        }
    }

    let program = admitted(compiled);
    assert_eq!(
        program
            .nodes()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count(),
        2,
        "only the two root-owned Fields enter the Kernel"
    );
    assert_eq!(
        program
            .nodes()
            .filter(|node| matches!(node, KernelNode::Relation(_)))
            .count(),
        2,
        "slot binding must not synthesize equality Relations"
    );
    assert_eq!(
        relation_field_symbols(&program, scalar_relation),
        BTreeSet::from([scalar.erase()])
    );
    assert_eq!(
        relation_field_symbols(&program, vector_relation),
        BTreeSet::from([displacement.erase()])
    );
    for field in [scalar, displacement] {
        assert!(has_edge(&program, EdgeKind::DefinedOn, field, body));
    }
    for (relation, field) in [(scalar_relation, scalar), (vector_relation, displacement)] {
        assert!(has_edge(&program, EdgeKind::AppliesOn, relation, body));
        assert!(has_edge(&program, EdgeKind::DependsOn, relation, field));
    }

    let document = eqiora::api::ModelDocument::compile("nested-fields.eqi", LOCAL_NESTED)
        .expect("Field slots cross the current Model wire");
    let wire = String::from_utf8(document.canonical_json().expect("canonical Model wire"))
        .expect("canonical JSON is UTF-8");
    assert!(!wire.contains("field_slot"));
    assert!(!wire.contains("field_binding"));
}

#[test]
fn rebinding_changes_source_identity_and_the_exact_relation_target() {
    let first = r#"
component C(
  support body: volume(ambient_dimension = 2),
  variable state: 1 on body,
) {
  relation identity on body { state - state = 0; }
}
model M() {
  domain body = box(0, 1, 0, 1);

  variable first: 1 on body;
  variable second: 1 on body;
  instance law: C(body = body, state = first);
}
"#;
    let second = first.replace("state = first", "state = second");
    assert_ne!(source_identity(first), source_identity(&second));

    for (name, source, selected, rejected) in [
        ("first", first, "first", "second"),
        ("second", second.as_str(), "second", "first"),
    ] {
        let compiled = compile_one(&format!("rebound-{name}.eqi"), source);
        let relation = typed::<kinds::Relation>(&compiled, "law.identity");
        let selected = typed::<kinds::Field>(&compiled, selected);
        let rejected = typed::<kinds::Field>(&compiled, rejected);
        let program = admitted(compiled);
        let symbols = relation_field_symbols(&program, relation);
        assert_eq!(symbols, BTreeSet::from([selected.erase()]));
        assert!(!symbols.contains(&rejected.erase()));
    }
}

#[test]
fn invalid_field_slots_fail_closed_before_transaction_or_graph_exposure() {
    let component = r#"
component FieldLaw(
  support body: volume(ambient_dimension = 2),
  variable scalar_state: 1 on body,
  variable displacement: vector<m, 2> on body,
) {
  relation scalar_identity on body { scalar_state - scalar_state = 0; }
  relation vector_identity on body { displacement - displacement = 0; }
}
"#;
    let cases = [
        (
            "missing",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: vector<m, 2> on body;
  instance law: FieldLaw(body = body, displacement = displacement);
}
"#,
            &["has no binding for required Field slot", "scalar_state"][..],
        ),
        (
            "duplicate",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: vector<m, 2> on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#,
            &["duplicate named binding", "scalar_state"][..],
        ),
        (
            "unknown",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: vector<m, 2> on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement,
    ghost = scalar_state
  );
}
"#,
            &["not a public requirement", "ghost", "FieldLaw"][..],
        ),
        (
            "wrong-kind",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  parameter gain: 1 = 0;
  variable displacement: vector<m, 2> on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = gain,
    displacement = displacement
  );
}
"#,
            &["gain", "is not an enclosing Field"][..],
        ),
        (
            "dimension",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: m on body;
  variable displacement: vector<m, 2> on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#,
            &["scalar_state", "disagree in physical dimension"][..],
        ),
        (
            "shape",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: m on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#,
            &["displacement", "exact value shape"][..],
        ),
        (
            "frame",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);

  variable scalar_state: 1 on body;
  variable displacement: array<m, 2> on body;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#,
            &["displacement", "array and spatial axis roles"][..],
        ),
        (
            "exact-support",
            r#"
model M() {
  domain body = box(0, 1, 0, 1);
  domain other = box(0, 1, 0, 1);

  variable scalar_state: 1 on other;
  variable displacement: vector<m, 2> on other;
  instance law: FieldLaw(
    body = body,
    scalar_state = scalar_state,
    displacement = displacement
  );
}
"#,
            &["scalar_state", "exact spatial support"][..],
        ),
    ];

    for (name, model, expected) in cases {
        assert_compile_rejects_without_graph_mutation(
            name,
            &format!("{component}{model}"),
            expected,
        );
    }

    assert_compile_rejects_without_graph_mutation(
        "non-continuum-slot",
        r#"
component C(
  support body: volume(ambient_dimension = 2),
) {
  public field slot state on body as discrete: 1;
}
model M() {}
"#,
        &["public requirements belong in the signature"],
    );
    assert_compile_rejects_without_graph_mutation(
        "ambient-dimension",
        &format!(
            r#"{component}
model M() {{
  domain line = box(0, 1);

  variable scalar_state: 1 on line;
  variable displacement: vector<m, 1> on line;
  instance law: FieldLaw(
    body = line,
    scalar_state = scalar_state,
    displacement = displacement
  );
}}
"#
        ),
        &["requires ambient dimension 2"],
    );
}

#[test]
fn parameter_support_and_field_bindings_share_one_bounded_identity_budget() {
    let source = r#"
component C(
  support body: volume(ambient_dimension = 2),
  variable state: 1 on body,
  parameter gain: 1
) {
  relation identity on body { gain * state - state = 0; }
}
model M() {
  domain body = box(0, 1, 0, 1);

  variable state: 1 on body;
  instance law: C(gain = 1, body = body, state = state);
}
"#;
    let document = eqiora::language::parse("binding-budget.eqi", source)
        .into_document()
        .expect("budget fixture parses");
    let limits = LocalSourceIdentityLimits {
        max_bindings_per_instance: 2,
        ..LocalSourceIdentityLimits::default()
    };
    let diagnostic = LocalSourceIdentity::from_document_with_limits(&document, limits)
        .expect_err("three binding families share one limit");
    assert!(
        diagnostic
            .message()
            .contains("instance named binding count exceeds resource limit")
    );
    LocalSourceIdentity::from_document_with_limits(
        &document,
        LocalSourceIdentityLimits {
            max_bindings_per_instance: 3,
            ..limits
        },
    )
    .expect("exactly three named bindings fit the shared budget");

    let store = InMemoryGraphStore::new();
    let snapshot = store.snapshot();
    assert_eq!(snapshot.revision(), store.snapshot().revision());
    assert_eq!(snapshot.nodes().len(), 0);
    assert_eq!(snapshot.edges().len(), 0);
}

#[test]
fn exact_packages_normalize_alias_declaration_binding_and_file_order() {
    let components = component_release(COMPONENT_PACKAGE, false);
    let permuted_components = component_release(COMPONENT_PACKAGE_PERMUTED, true);
    assert_eq!(
        components
            .package_identity()
            .expect("component package identity"),
        permuted_components
            .package_identity()
            .expect("permuted component identity"),
        "definition declaration, binding, and input-file order are non-semantic"
    );

    let root = root_release(&components, "laws", false, false);
    let permuted_root = root_release(&permuted_components, "constitutive_components", true, true);
    assert_eq!(
        root.package_identity().expect("root package identity"),
        permuted_root
            .package_identity()
            .expect("permuted root identity"),
        "an exact import alias and source order do not enter package meaning"
    );

    let packaged = compile_locked(&components, &root);
    let permuted = compile_locked(&permuted_components, &permuted_root);
    assert_eq!(
        packaged.model().canonical_json().expect("canonical Model"),
        permuted
            .model()
            .canonical_json()
            .expect("permuted canonical Model")
    );
    assert_eq!(
        packaged.model().digest().expect("canonical Model digest"),
        permuted.model().digest().expect("permuted Model digest")
    );
    for eliminated in [
        "law.body",
        "law.scalar_state",
        "law.displacement",
        "law.inner.body",
        "law.inner.scalar_state",
        "law.inner.displacement",
    ] {
        assert_eq!(packaged.model().aliases().get(eliminated), None);
    }

    let scalar: Id<kinds::Field> = packaged.model().aliases()["scalar_state"]
        .downcast()
        .expect("root scalar Field");
    let displacement: Id<kinds::Field> = packaged.model().aliases()["displacement"]
        .downcast()
        .expect("root displacement Field");
    let scalar_relation: Id<kinds::Relation> =
        packaged.model().aliases()["law.inner.scalar_identity"]
            .downcast()
            .expect("expanded scalar Relation");
    let vector_relation: Id<kinds::Relation> =
        packaged.model().aliases()["law.inner.vector_identity"]
            .downcast()
            .expect("expanded vector Relation");
    let provenance = packaged
        .provenance()
        .get_by_graph_id(scalar_relation.erase())
        .expect("cross-package Field-binding provenance");
    assert_eq!(provenance.binding_spans().len(), 6);
    assert!(
        provenance
            .binding_spans()
            .iter()
            .any(|span| span.file.contains("Eqiora.Verify.OccurrenceBoundFields"))
    );
    assert!(provenance.binding_spans().iter().any(|span| {
        span.file
            .contains("org.eqiora.verify.occurrence_bound_fields")
    }));
    assert_eq!(
        relation_field_symbols(packaged.model().program(), scalar_relation),
        BTreeSet::from([scalar.erase()])
    );
    assert_eq!(
        relation_field_symbols(packaged.model().program(), vector_relation),
        BTreeSet::from([displacement.erase()])
    );
}

#[test]
fn invalid_exact_package_binding_never_exposes_a_packaged_model() {
    let components = component_release(COMPONENT_PACKAGE, false);
    let alias = "laws";
    let invalid = root_source(alias, false).replace("    scalar_state = scalar_state,\n", "");
    let dependency = PackageDependencyV1::new(
        components
            .package_identity()
            .expect("component package identity"),
    );
    let root_sources = package_sources(
        "org.eqiora.verify.occurrence_bound_fields",
        vec![dependency],
        &invalid,
        false,
    );

    let diagnostics =
        match prepare_package_release_v1(root_sources, std::slice::from_ref(&components)) {
            Err(PackagePreparationError::Diagnostics(diagnostics)) => diagnostics,
            Err(error) => panic!("unexpected package preparation failure: {error}"),
            Ok(root) => {
                let resolution = ResolutionRecordV1::from_exact_releases(
                    &root,
                    std::slice::from_ref(&components),
                )
                .expect("exact offline resolution");
                let mut store = InMemoryPackageStore::default();
                store.insert(&components).expect("insert component release");
                store.insert(&root).expect("insert root release");
                match PackagedModelDocument::compile_locked(&store, &resolution, "Main") {
                    Err(PackageCompilationError::Diagnostics(diagnostics)) => diagnostics,
                    Err(error) => panic!("unexpected locked compilation failure: {error}"),
                    Ok(_) => panic!("invalid exact package exposed a packaged Model"),
                }
            }
        };
    assert_diagnostic_contains(
        &diagnostics,
        &["has no binding for required Field slot", "scalar_state"],
    );
}
