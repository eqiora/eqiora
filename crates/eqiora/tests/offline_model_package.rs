use std::num::NonZeroUsize;
use std::path::PathBuf;

#[path = "offline_model_package/directory_input.rs"]
mod directory_input;
#[path = "offline_model_package/store_input.rs"]
mod store_input;

use eqiora::Id;
use eqiora::artifact::{ArtifactDigest, RunManifestV1};
use eqiora::entity::kinds;
use eqiora::graph::EdgeKind;
use eqiora::kernel::KernelNode;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageDirectory, PackageManifestV1, PackagePreparationError,
    PackageReleaseV1, PackageRunBindingV1, PackageSourcesV1, PackageStore, PackagedModelDocument,
    QualifiedName, ResolutionRecordV1, SourceFileV1, prepare_package_release_v1,
};
use eqiora::sem::PhysicalUnknown;
use eqiora::solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{scalar::lower_scalar_physical_affine, scalar::solve_scalar_physical_affine};

const LIBRARY_SOURCE: &str =
    include_str!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi");
const LIBRARY_README: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/README.md");
const LIBRARY_MANIFEST: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/package.json");
const ROOT_SOURCE: &str =
    include_str!("../../../crates/eqiora-api/packages/org.example.parallel/src/main.eqi");
const ROOT_README: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.parallel/README.md");
const ROOT_MANIFEST: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.parallel/package.json");
const SOURCE_PATH: &str = "src/basic.eqi";
const ROOT_SOURCE_PATH: &str = "src/main.eqi";
const VALUE_TOLERANCE: f64 = 2.0e-11;

fn library_sources() -> PackageSourcesV1 {
    directory_sources("Eqiora.Electrical.Basic")
}

fn package_root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../eqiora-api/packages")
        .join(name)
}

fn directory_sources(name: &str) -> PackageSourcesV1 {
    PackageDirectory::open_ambient(package_root(name))
        .expect("open explicit package root")
        .read_sources()
        .expect("read exact package inventory")
}

fn library_sources_in_memory() -> PackageSourcesV1 {
    let manifest = PackageManifestV1::from_json(LIBRARY_MANIFEST).expect("library manifest");
    assert_eq!(
        manifest.canonical_json().expect("canonical manifest"),
        LIBRARY_MANIFEST
            .strip_suffix(b"\n")
            .unwrap_or(LIBRARY_MANIFEST)
    );
    PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                eqiora::package::NormalizedRelativePath::parse("README.md").expect("path"),
                BundleRoleV1::Documentation,
                LIBRARY_README.to_vec(),
            ),
            SourceFileV1::new(
                eqiora::package::NormalizedRelativePath::parse(SOURCE_PATH).expect("path"),
                BundleRoleV1::ModelSource,
                LIBRARY_SOURCE.as_bytes().to_vec(),
            ),
        ],
    )
    .expect("admitted library sources")
}

fn library_release() -> PackageReleaseV1 {
    prepare_package_release_v1(library_sources(), &[]).expect("compiler-derived library release")
}

fn root_sources(library: &PackageReleaseV1) -> PackageSourcesV1 {
    let sources = directory_sources("org.example.parallel");
    assert_eq!(
        sources.manifest().dependencies()[0].target(),
        &library.package_identity().expect("library identity")
    );
    sources
}

fn root_sources_in_memory(library: &PackageReleaseV1) -> PackageSourcesV1 {
    let library_identity = library.package_identity().expect("library identity");
    let manifest = PackageManifestV1::from_json(ROOT_MANIFEST).expect("root manifest");
    assert_eq!(
        manifest.canonical_json().expect("canonical manifest"),
        ROOT_MANIFEST.strip_suffix(b"\n").unwrap_or(ROOT_MANIFEST)
    );
    assert_eq!(manifest.dependencies()[0].target(), &library_identity);
    PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                eqiora::package::NormalizedRelativePath::parse("README.md").expect("path"),
                BundleRoleV1::Documentation,
                ROOT_README.to_vec(),
            ),
            SourceFileV1::new(
                eqiora::package::NormalizedRelativePath::parse(ROOT_SOURCE_PATH).expect("path"),
                BundleRoleV1::ModelSource,
                ROOT_SOURCE.as_bytes().to_vec(),
            ),
        ],
    )
    .expect("admitted root sources")
}

#[test]
fn directory_admission_is_identity_transparent_to_in_memory_preparation() {
    let directory_library = library_sources();
    let in_memory_library = library_sources_in_memory();
    assert_eq!(directory_library, in_memory_library);
    let directory_library_release =
        prepare_package_release_v1(directory_library, &[]).expect("directory library release");
    let in_memory_library_release =
        prepare_package_release_v1(in_memory_library, &[]).expect("in-memory library release");
    assert_eq!(
        directory_library_release
            .canonical_json()
            .expect("directory library bytes"),
        in_memory_library_release
            .canonical_json()
            .expect("in-memory library bytes")
    );

    let directory_root = root_sources(&directory_library_release);
    let in_memory_root = root_sources_in_memory(&directory_library_release);
    assert_eq!(directory_root, in_memory_root);
    let directory_root_release = prepare_package_release_v1(
        directory_root,
        std::slice::from_ref(&directory_library_release),
    )
    .expect("directory root release");
    let in_memory_root_release = prepare_package_release_v1(
        in_memory_root,
        std::slice::from_ref(&directory_library_release),
    )
    .expect("in-memory root release");
    assert_eq!(
        directory_root_release
            .canonical_json()
            .expect("directory root bytes"),
        in_memory_root_release
            .canonical_json()
            .expect("in-memory root bytes")
    );
}

fn root_release(library: &PackageReleaseV1) -> PackageReleaseV1 {
    prepare_package_release_v1(root_sources(library), std::slice::from_ref(library))
        .expect("compiler-derived root release")
}

fn inline_sources(
    name: &str,
    source: &str,
    dependencies: &[(&str, &PackageReleaseV1)],
) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse("src/main.eqi").expect("inline path");
    let requirements = dependencies
        .iter()
        .map(|(_, release)| {
            PackageDependencyV1::new(
                release
                    .package_identity()
                    .expect("inline dependency identity"),
            )
        })
        .collect();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).expect("inline package name"),
        ExactVersion::parse("1.0.0").expect("inline version"),
        requirements,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("inline manifest");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("inline author sources")
}

fn inline_release(
    name: &str,
    source: &str,
    direct_dependencies: &[(&str, &PackageReleaseV1)],
    closure: &[PackageReleaseV1],
) -> PackageReleaseV1 {
    prepare_package_release_v1(inline_sources(name, source, direct_dependencies), closure)
        .expect("inline compiler-derived release")
}

#[test]
fn package_preparation_replays_transitive_closure_independent_of_input_order() {
    let leaf = inline_release(
        "org.example.Leaf",
        "public component Resistor() {}",
        &[],
        &[],
    );
    let middle = inline_release(
        "org.example.Middle",
        "import org.example.Leaf.main as leaf; public component Branch() { instance load: leaf.Resistor(); }",
        &[("leaf", &leaf)],
        std::slice::from_ref(&leaf),
    );
    let root_source = "import org.example.Middle.main as middle; model Main() { instance branch: middle.Branch(); }";
    let first = prepare_package_release_v1(
        inline_sources("org.example.Root", root_source, &[("middle", &middle)]),
        &[leaf.clone(), middle.clone()],
    )
    .expect("leaf-first closure");
    let second = prepare_package_release_v1(
        inline_sources("org.example.Root", root_source, &[("middle", &middle)]),
        &[middle.clone(), leaf.clone()],
    )
    .expect("middle-first closure");
    assert_eq!(
        first.canonical_json().expect("first release"),
        second.canonical_json().expect("second release")
    );
    let resolution = ResolutionRecordV1::from_exact_releases(&first, &[middle, leaf])
        .expect("transitive exact lock");
    assert_eq!(resolution.nodes().len(), 3);
    assert_eq!(resolution.edges().len(), 2);
}

#[test]
fn package_preparation_rejects_unclosed_or_dishonest_release_inputs() {
    let library = library_release();
    assert!(matches!(
        prepare_package_release_v1(root_sources(&library), &[]),
        Err(PackagePreparationError::MissingDependency { .. })
    ));
    assert!(matches!(
        prepare_package_release_v1(root_sources(&library), &[library.clone(), library.clone()]),
        Err(PackagePreparationError::DuplicateDependency(_))
    ));

    let extra = inline_release("org.example.Extra", "public component Extra() {}", &[], &[]);
    assert!(matches!(
        prepare_package_release_v1(library_sources(), std::slice::from_ref(&extra)),
        Err(PackagePreparationError::Contract(_))
    ));

    let dishonest_source = LIBRARY_SOURCE.replacen("relation law", "relation altered", 1);
    let dishonest = PackageReleaseV1::new(
        library.manifest().clone(),
        library.semantic().clone(),
        vec![
            SourceFileV1::new(
                NormalizedRelativePath::parse("README.md").expect("path"),
                BundleRoleV1::Documentation,
                LIBRARY_README.to_vec(),
            ),
            SourceFileV1::new(
                NormalizedRelativePath::parse(SOURCE_PATH).expect("path"),
                BundleRoleV1::ModelSource,
                dishonest_source.into_bytes(),
            ),
        ],
    )
    .expect("locally well-formed dishonest release");
    assert_eq!(
        dishonest.package_identity().expect("dishonest identity"),
        library.package_identity().expect("library identity")
    );
    assert!(matches!(
        prepare_package_release_v1(root_sources(&library), &[dishonest]),
        Err(PackagePreparationError::SemanticContentMismatch { .. })
    ));
}

fn selected_connection(
    packaged: &PackagedModelDocument,
    member: Id<kinds::Port>,
) -> Id<kinds::Connection> {
    packaged
        .model()
        .program()
        .nodes()
        .find_map(|node| {
            let KernelNode::Connection(connection) = node else {
                return None;
            };
            packaged
                .model()
                .program()
                .edges()
                .iter()
                .any(|edge| {
                    edge.kind() == EdgeKind::Connects
                        && edge.from() == connection.id().erase()
                        && edge.to() == member.erase()
                })
                .then_some(connection.id())
        })
        .expect("selected Port belongs to one Connection")
}

#[test]
fn exact_offline_packages_compile_and_solve_through_one_path() {
    let library = library_release();
    let root = root_release(&library);
    let library_identity = library.package_identity().expect("library identity");
    let root_identity = root.package_identity().expect("root identity");
    let mut store = InMemoryPackageStore::default();
    let library_source = store.insert(&library).expect("store library");
    let root_source = store.insert(&root).expect("store root");
    let resolution = ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&library))
        .expect("derived exact resolution");
    assert_eq!(
        resolution
            .nodes()
            .iter()
            .find(|node| node.identity() == &library_identity)
            .expect("library node")
            .source_digest(),
        library_source
    );
    assert_eq!(
        resolution
            .nodes()
            .iter()
            .find(|node| node.identity() == &root_identity)
            .expect("root node")
            .source_digest(),
        root_source
    );

    assert_package_semantics(&store, &resolution);
}

fn assert_package_semantics(store: &impl PackageStore, resolution: &ResolutionRecordV1) {
    let root_node = resolution
        .nodes()
        .iter()
        .find(|node| node.identity() == resolution.root())
        .expect("root lock node");
    let root_identity = root_node.identity();

    let packaged = PackagedModelDocument::compile_locked(store, resolution, "Main")
        .expect("locked package compilation");
    packaged
        .compilation()
        .validate_against(resolution)
        .expect("compilation provenance");
    assert_eq!(packaged.compilation().root(), root_identity);
    assert_eq!(packaged.compilation().packages().len(), 2);

    let positive: Id<kinds::Port> = packaged.model().aliases()["resistor_two.positive"]
        .downcast()
        .expect("resistor Port");
    let connection = selected_connection(&packaged, positive);
    let problem = lower_scalar_physical_affine(packaged.model().program(), connection, None)
        .expect("complete affine physical closure");
    assert_eq!(
        (
            problem.canonical_system().rows(),
            problem.canonical_system().columns()
        ),
        (14, 14)
    );
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(100).expect("nonzero iterations"),
    )
    .expect("solver plan")
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solution = solve_scalar_physical_affine(
        &problem,
        &vec![1.0; problem.canonical_system().columns()],
        LinearSolveRequest::new(&FaerLinearSolver, plan.clone()),
    )
    .expect("physical solution");
    let value = |name: &str, unknown: fn(Id<kinds::Port>) -> PhysicalUnknown| {
        let port = packaged.model().aliases()[name]
            .downcast()
            .expect("fixture Port");
        solution.value(unknown(port)).expect("physical value")
    };
    for (name, across, through) in [
        ("source.positive", 12.0, -9.0),
        ("resistor_two.positive", 12.0, 6.0),
        ("resistor_four.positive", 12.0, 3.0),
        ("ground.terminal", 0.0, 0.0),
    ] {
        assert!((value(name, PhysicalUnknown::Across) - across).abs() <= VALUE_TOLERANCE);
        assert!((value(name, PhysicalUnknown::Through) - through).abs() <= VALUE_TOLERANCE);
    }
    assert!(solution.reference_residual_norm() <= 1.2e-11);

    let provenance = packaged
        .provenance()
        .get_by_graph_id(positive.erase())
        .expect("imported provenance");
    assert!(
        provenance
            .definition_span()
            .file
            .contains("Eqiora.Electrical.Basic")
    );
    assert!(
        provenance
            .instance_span()
            .file
            .contains("org.example.parallel")
    );
    assert_eq!(provenance.binding_spans().len(), 1);
    assert!(
        provenance.binding_spans()[0]
            .file
            .contains("org.example.parallel")
    );

    // Bind lineage only after the analytic values, original-DAG residual, and
    // package-qualified provenance have accepted this exact solve. Run v1 has
    // no appropriate general numerical-result artifact yet, so this evidence
    // deliberately emits an output-less Run rather than misusing an
    // import-provenance DTO.
    let run = RunManifestV1::new(
        ArtifactDigest::from_hex(packaged.model().digest().expect("model digest"))
            .expect("artifact digest"),
        packaged.model().program().revision().0,
        "eqiora-backend-faer",
        env!("CARGO_PKG_VERSION"),
    )
    .expect("run manifest")
    .with_numerical_setting("execution.topology", "one-host-process-one-worker")
    .expect("execution topology")
    .with_numerical_setting("solver.absolute-tolerance", "1e-14")
    .expect("absolute tolerance")
    .with_numerical_setting("solver.initial-guess", "uniform-f64:1")
    .expect("initial guess")
    .with_numerical_setting("solver.maximum-iterations", "100")
    .expect("maximum iterations")
    .with_numerical_setting("solver.method", "bicgstab")
    .expect("solver method")
    .with_numerical_setting("solver.preconditioner", "identity")
    .expect("solver preconditioner")
    .with_numerical_setting("solver.reduction", "fast")
    .expect("solver reduction")
    .with_numerical_setting("solver.relative-tolerance", "1e-12")
    .expect("relative tolerance");

    // The registered emission path is gated by a successful solve. A
    // dimensionally invalid initial state must fail before the binding
    // constructor becomes reachable.
    let failed_lineage = solve_scalar_physical_affine(
        &problem,
        &[],
        LinearSolveRequest::new(&FaerLinearSolver, plan),
    )
    .ok()
    .and_then(|_| packaged.bind_run_v1(&run).ok());
    assert!(failed_lineage.is_none());

    let binding = packaged.bind_run_v1(&run).expect("package run binding");
    packaged
        .validate_run_v1_binding(&binding, &run, resolution)
        .expect("exact package run replay");
    let replayed_binding = PackageRunBindingV1::from_json(
        &binding
            .canonical_json()
            .expect("canonical package run binding"),
    )
    .expect("replayed package run binding");
    packaged
        .validate_run_v1_binding(&replayed_binding, &run, resolution)
        .expect("canonical package run replay");
}
