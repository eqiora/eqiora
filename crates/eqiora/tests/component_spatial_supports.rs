use eqiora::compiler::{CompiledModel, compile};
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::sem::KernelProgram;
use eqiora::{Id, RawId};

const TWO_BOUNDARIES: &str =
    include_str!("../../../verify/packages/component-spatial-supports/models/two-boundaries.eqi");
const PACKAGE_COMPONENT: &str = r#"
public component BoundaryState(
  support body: volume(ambient_dimension = 2),
  support interface: boundary(parent = body),
) {

  variable state: 1 on body;
  relation volume_law on body { state = 0; }
  relation interface_law on interface { trace(state) = 0; }
}

public component BoundaryWrapper(
  support body: volume(ambient_dimension = 2),
  support interface: boundary(parent = body),
) {
  instance inner: BoundaryState(
    body = body,
    interface = interface
  );
}
"#;
const VERSION: &str = "0.1.0";

fn compile_one(file: &str, source: &str) -> CompiledModel {
    let mut compiled = compile(file, source).expect("support fixture compiles");
    assert_eq!(compiled.len(), 1, "fixture has one root Model");
    compiled.pop().unwrap()
}

fn admit(compiled: CompiledModel) -> KernelProgram {
    let model = compiled.model();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("atomic model admission");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("accepted canonical program")
}

fn typed<I>(compiled: &CompiledModel, name: &str) -> Id<I>
where
    I: eqiora::Entity,
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

fn package_sources(
    name: &str,
    dependencies: Vec<PackageDependencyV1>,
    source: &str,
) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse("src/model.eqi").expect("normalized source path");
    let manifest = PackageManifestV1::new(
        "model",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(VERSION).expect("package version"),
        dependencies,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("package manifest");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("closed package sources")
}

fn component_release() -> PackageReleaseV1 {
    prepare_package_release_v1(
        package_sources(
            "Eqiora.Verify.SpatialSupport",
            Vec::new(),
            PACKAGE_COMPONENT,
        ),
        &[],
    )
    .expect("occurrence-free support definition validates before release identity")
}

fn packaged_model(components: &PackageReleaseV1, alias: &str) -> PackagedModelDocument {
    let dependency = PackageDependencyV1::new(
        components
            .package_identity()
            .expect("component package identity"),
    );
    let source = format!(
        r#"
import Eqiora.Verify.SpatialSupport.model as {alias};

model Main() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance probe: {alias}.BoundaryWrapper(
    body = fluid,
    interface = wall
  );
}}
"#
    );
    let root = prepare_package_release_v1(
        package_sources(
            "org.eqiora.verify.spatial_support",
            vec![dependency],
            &source,
        ),
        std::slice::from_ref(components),
    )
    .expect("root support package validates");
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(components))
            .expect("exact offline resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(components).expect("insert component release");
    store.insert(&root).expect("insert root release");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile locked support occurrence")
}

#[test]
fn support_slots_flatten_to_exact_existing_domains_without_entities_or_aliases() {
    let compiled = compile_one("two-boundaries.eqi", TWO_BOUNDARIES);
    let fluid = typed::<kinds::Domain>(&compiled, "fluid");
    let left = typed::<kinds::Domain>(&compiled, "left");
    let right = typed::<kinds::Domain>(&compiled, "right");
    let left_state = typed::<kinds::Field>(&compiled, "left_state.state");
    let right_state = typed::<kinds::Field>(&compiled, "right_state.state");
    let left_volume = typed::<kinds::Relation>(&compiled, "left_state.volume_law");
    let left_interface = typed::<kinds::Relation>(&compiled, "left_state.interface_law");
    let right_volume = typed::<kinds::Relation>(&compiled, "right_state.volume_law");
    let right_interface = typed::<kinds::Relation>(&compiled, "right_state.interface_law");

    assert_ne!(
        left_state, right_state,
        "occurrences retain distinct identity"
    );
    assert_eq!(compiled.symbols().get("left_state.body"), None);
    assert_eq!(compiled.symbols().get("left_state.interface"), None);
    assert_eq!(compiled.symbols().get("right_state.body"), None);
    assert_eq!(compiled.symbols().get("right_state.interface"), None);

    let provenance = compiled.provenance().expect("hierarchy provenance");
    assert_eq!(
        provenance
            .get_by_graph_id(left_state.erase())
            .expect("left Field provenance")
            .binding_spans()
            .len(),
        2,
        "both support bindings are complete occurrence provenance"
    );

    let program = admit(compiled);
    assert!(has_edge(&program, EdgeKind::BoundaryOf, left, fluid));
    assert!(has_edge(&program, EdgeKind::BoundaryOf, right, fluid));
    assert!(has_edge(&program, EdgeKind::DefinedOn, left_state, fluid));
    assert!(has_edge(&program, EdgeKind::DefinedOn, right_state, fluid));
    assert!(has_edge(&program, EdgeKind::AppliesOn, left_volume, fluid));
    assert!(has_edge(&program, EdgeKind::AppliesOn, right_volume, fluid));
    assert!(has_edge(
        &program,
        EdgeKind::AppliesOn,
        left_interface,
        left
    ));
    assert!(has_edge(
        &program,
        EdgeKind::AppliesOn,
        right_interface,
        right
    ));
}

#[test]
fn invalid_support_contracts_fail_before_a_transaction_exists() {
    let component = r#"
component BoundaryState(
  support body: volume(ambient_dimension = 2),
  support interface: boundary(parent = body),
) {

  variable state: 1 on body;
  relation law on interface { trace(state) = 0; }
}
"#;
    let cases = [
        (
            "missing",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  instance c: BoundaryState(body = fluid);
}
"#,
            "has no binding for required support slot `interface`",
        ),
        (
            "unknown",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance c: BoundaryState(
    body = fluid,
    interface = wall,
    ghost = wall
  );
}
"#,
            "`ghost` is not a public requirement of `BoundaryState`",
        ),
        (
            "duplicate",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance c: BoundaryState(
    body = fluid,
    body = fluid,
    interface = wall
  );
}
"#,
            "duplicate named binding `body`",
        ),
        (
            "volume-to-boundary",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance c: BoundaryState(body = wall, interface = wall);
}
"#,
            "support slot `body` requires volume support, found boundary",
        ),
        (
            "boundary-to-volume",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  instance c: BoundaryState(body = fluid, interface = fluid);
}
"#,
            "support slot `interface` requires boundary support, found volume",
        ),
        (
            "wrong-parent",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  domain other = box(0, 1, 0, 1);
  domain wall = boundary(other, axis = 0, side = lower);
  instance c: BoundaryState(body = fluid, interface = wall);
}
"#,
            "is not BoundaryOf its exact bound parent slot `body`",
        ),
        (
            "ambient-dimension",
            r#"
model M() {
  domain line = box(0, 1);
  domain point = boundary(line, axis = 0, side = lower);
  instance c: BoundaryState(body = line, interface = point);
}
"#,
            "requires ambient dimension 2",
        ),
        (
            "boundary-of-boundary",
            r#"
model M() {
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  domain edge = boundary(wall, axis = 1, side = lower);
  instance c: BoundaryState(body = fluid, interface = edge);
}
"#,
            "Cartesian boundary parent must be a Cartesian box Domain",
        ),
    ];

    for (name, model, expected) in cases {
        let source = format!("{component}{model}");
        let diagnostics = compile(&format!("{name}.eqi"), &source).expect_err(name);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{name}: {diagnostics:?}"
        );
    }

    let coordinate = r#"
component C(
  support body: volume(ambient_dimension = 2),
) {
  relation law on body { coordinate(2) = 0; }
}
model M() {
  domain fluid = box(0, 1, 0, 1);
  instance c: C(body = fluid);
}
"#;
    let diagnostics = compile("coordinate.eqi", coordinate).expect_err("axis two is outside 2D");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("coordinate axis 2 is outside Domain dimension 2")
    }));

    for (name, source, expected) in [
        (
            "private-support.eqi",
            "component C() { support body: volume(ambient_dimension = 2); } model M() {}",
            "expected `product`",
        ),
        (
            "zero-dimension-support.eqi",
            "component C(support body: volume(ambient_dimension = 0)) {} model M() {}",
            "requires a positive ambient dimension",
        ),
        (
            "unknown-parent-support.eqi",
            "component C(support wall: boundary(parent = body)) {} model M() {}",
            "refers to unknown parent slot `body`",
        ),
    ] {
        let diagnostics = compile(name, source).expect_err(name);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{name}: {diagnostics:?}"
        );
    }
}

#[test]
fn exact_package_alias_spelling_preserves_the_flattened_support_model() {
    let components = component_release();
    let short = packaged_model(&components, "interfaces");
    let long = packaged_model(&components, "spatial_support_components");

    assert_eq!(
        short.model().canonical_json().expect("short-alias Model"),
        long.model().canonical_json().expect("long-alias Model")
    );
    assert_eq!(
        short.model().digest().expect("short-alias digest"),
        long.model().digest().expect("long-alias digest")
    );
    assert_eq!(short.model().aliases().get("probe.body"), None);
    assert_eq!(short.model().aliases().get("probe.interface"), None);
    assert_eq!(short.model().aliases().get("probe.inner.body"), None);
    assert_eq!(short.model().aliases().get("probe.inner.interface"), None);

    let aliases = short.model().aliases();
    let fluid = aliases["fluid"];
    let wall = aliases["wall"];
    let state = aliases["probe.inner.state"];
    let volume_law = aliases["probe.inner.volume_law"];
    let interface_law = aliases["probe.inner.interface_law"];
    let program = short.model().program();

    assert!(has_edge(program, EdgeKind::BoundaryOf, wall, fluid));
    assert!(has_edge(program, EdgeKind::DefinedOn, state, fluid));
    assert!(has_edge(program, EdgeKind::AppliesOn, volume_law, fluid));
    assert!(has_edge(program, EdgeKind::AppliesOn, interface_law, wall));
}
