use eqiora::artifact::{ArtifactDigest, RunManifestV1};
use eqiora::compiler::{
    ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit, analyze_resolved_hierarchy,
};
use eqiora::entity::kinds;
use eqiora::graph::EdgeKind;
use eqiora::kernel::KernelNode;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageRunBindingV1,
    PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionEdgeV1, ResolutionNodeV1,
    ResolutionRecordV1, SourceFileV1, prepare_package_release_v1,
};
use eqiora::sem::{Interpreter, PhysicalUnknown, ReferenceConfig, Trajectory};
use eqiora::{Diagnostic, DimExponents, Id};
use eqiora_numerics::scalar::lower_scalar_physical_affine;

mod support;

use support::exact_package::{canonical_manifest, namespace, provisional_namespace};

const VERSION: &str = "0.1.0";
const ELECTRICAL_PATH: &str = "src/basic.eqi";
const DRIVE_PATH: &str = "src/drive.eqi";
const ROOT_PATH: &str = "src/main.eqi";

const ELECTRICAL_SOURCE: &str =
    include_str!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi");
const ELECTRICAL_README: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/README.md");
const ELECTRICAL_MANIFEST: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/package.json");
const DRIVE_SOURCE: &str = include_str!(
    "../../../crates/eqiora-api/packages/Eqiora.Electromechanical.DcDrive/src/drive.eqi"
);
const DRIVE_README: &[u8] = include_bytes!(
    "../../../crates/eqiora-api/packages/Eqiora.Electromechanical.DcDrive/README.md"
);
const DRIVE_MANIFEST: &[u8] = include_bytes!(
    "../../../crates/eqiora-api/packages/Eqiora.Electromechanical.DcDrive/package.json"
);
const ROOT_SOURCE: &str =
    include_str!("../../../crates/eqiora-api/packages/org.example.dc-motor-control/src/main.eqi");
const ROOT_README: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.dc-motor-control/README.md");
const ROOT_MANIFEST: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.dc-motor-control/package.json");

const SAMPLE_PERIOD: f64 = 0.01;
const RESISTANCE: f64 = 2.0;
const INDUCTANCE: f64 = 0.5;
const MOTOR_CONSTANT: f64 = 0.1;
const INERTIA: f64 = 0.05;
const DAMPING: f64 = 0.1;
const SETPOINT: f64 = 10.0;
const GAIN: f64 = 1.0;
const END_TIME_S: f64 = 0.1;
const COARSE_MAXIMUM_STEP_S: f64 = 0.002;
const ACCEPTED_MAXIMUM_STEP_S: f64 = 0.001;
const NONLINEAR_ABSOLUTE_TOLERANCE: f64 = 1.0e-10;
const NONLINEAR_RELATIVE_TOLERANCE: f64 = 1.0e-10;
const MAXIMUM_NONLINEAR_ITERATIONS: usize = 32;
const MAXIMUM_SEMANTIC_STEPS: usize = 1_000_000;
const VOLTAGE_DIMENSION: DimExponents =
    DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension");
const CURRENT_DIMENSION: DimExponents =
    DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension");
const ANGULAR_SPEED_DIMENSION: DimExponents =
    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
const TORQUE_DIMENSION: DimExponents =
    DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).expect("bounded dimension");
const VOLTAGE_RESIDUAL_TOLERANCE_V: f64 = 2.0e-8;
const CURRENT_RESIDUAL_TOLERANCE_A: f64 = 2.0e-8;
const ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S: f64 = 2.0e-8;
const TORQUE_RESIDUAL_TOLERANCE_N_M: f64 = 2.0e-8;
const POWER_BALANCE_TOLERANCE_W: f64 = 3.0e-8;

#[derive(Clone, Copy)]
struct PackageSpelling<'a> {
    drive_path: &'a str,
    root_path: &'a str,
    electrical_in_drive: &'a str,
    electrical_in_root: &'a str,
    drive_in_root: &'a str,
    reverse_dependency_releases: bool,
}

const CANONICAL_SPELLING: PackageSpelling<'static> = PackageSpelling {
    drive_path: DRIVE_PATH,
    root_path: ROOT_PATH,
    electrical_in_drive: "electrical",
    electrical_in_root: "electrical",
    drive_in_root: "drive",
    reverse_dependency_releases: false,
};

fn source_file(path: &str, role: BundleRoleV1, bytes: &[u8]) -> SourceFileV1 {
    SourceFileV1::new(
        NormalizedRelativePath::parse(path).expect("normalized package path"),
        role,
        bytes.to_vec(),
    )
}

fn package_sources(manifest: PackageManifestV1, files: Vec<SourceFileV1>) -> PackageSourcesV1 {
    PackageSourcesV1::new(manifest, files).expect("admitted package sources")
}

fn electrical_release() -> PackageReleaseV1 {
    let manifest = canonical_manifest(ELECTRICAL_MANIFEST);
    let sources = package_sources(
        manifest,
        vec![
            source_file("README.md", BundleRoleV1::Documentation, ELECTRICAL_README),
            source_file(
                ELECTRICAL_PATH,
                BundleRoleV1::ModelSource,
                ELECTRICAL_SOURCE.as_bytes(),
            ),
        ],
    );
    prepare_package_release_v1(sources, &[]).expect("compiler-derived electrical release")
}

fn drive_release_with_spelling(
    electrical: &PackageReleaseV1,
    spelling: PackageSpelling<'_>,
) -> PackageReleaseV1 {
    let electrical_identity = electrical.package_identity().expect("electrical identity");
    let manifest = PackageManifestV1::new(
        &spelling
            .drive_path
            .strip_prefix("src/")
            .unwrap()
            .strip_suffix(".eqi")
            .unwrap()
            .replace('/', "."),
        QualifiedName::parse("Eqiora.Electromechanical.DcDrive").expect("drive name"),
        ExactVersion::parse(VERSION).expect("drive version"),
        vec![PackageDependencyV1::new(electrical_identity.clone())],
        vec![
            BundleEntryV1::new(
                NormalizedRelativePath::parse("README.md").expect("drive README path"),
                BundleRoleV1::Documentation,
            ),
            BundleEntryV1::new(
                NormalizedRelativePath::parse(spelling.drive_path).expect("drive source path"),
                BundleRoleV1::ModelSource,
            ),
        ],
    )
    .expect("drive manifest");
    let drive_source = DRIVE_SOURCE
        .replace(
            "as electrical;",
            &format!("as {};", spelling.electrical_in_drive),
        )
        .replace("electrical.", &format!("{}.", spelling.electrical_in_drive));
    let sources = package_sources(
        manifest,
        vec![
            source_file("README.md", BundleRoleV1::Documentation, DRIVE_README),
            source_file(
                spelling.drive_path,
                BundleRoleV1::ModelSource,
                drive_source.as_bytes(),
            ),
        ],
    );
    prepare_package_release_v1(sources, std::slice::from_ref(electrical))
        .expect("compiler-derived drive release")
}

fn drive_release(electrical: &PackageReleaseV1) -> PackageReleaseV1 {
    let release = drive_release_with_spelling(electrical, CANONICAL_SPELLING);
    assert_eq!(
        release.manifest(),
        &canonical_manifest(DRIVE_MANIFEST),
        "checked-in manifest is the canonical spelling"
    );
    release
}

fn root_release_with_spelling(
    electrical: &PackageReleaseV1,
    drive: &PackageReleaseV1,
    root_source: &str,
    spelling: PackageSpelling<'_>,
) -> PackageReleaseV1 {
    let electrical_identity = electrical.package_identity().expect("electrical identity");
    let drive_identity = drive.package_identity().expect("drive identity");
    let manifest = PackageManifestV1::new(
        &spelling
            .root_path
            .strip_prefix("src/")
            .unwrap()
            .strip_suffix(".eqi")
            .unwrap()
            .replace('/', "."),
        QualifiedName::parse("org.example.dc_motor_control").expect("root name"),
        ExactVersion::parse(VERSION).expect("root version"),
        vec![
            PackageDependencyV1::new(electrical_identity.clone()),
            PackageDependencyV1::new(drive_identity.clone()),
        ],
        vec![
            BundleEntryV1::new(
                NormalizedRelativePath::parse("README.md").expect("root README path"),
                BundleRoleV1::Documentation,
            ),
            BundleEntryV1::new(
                NormalizedRelativePath::parse(spelling.root_path).expect("root source path"),
                BundleRoleV1::ModelSource,
            ),
        ],
    )
    .expect("root manifest");

    let root_source = root_source
        .replace(
            "as electrical;",
            &format!("as {};", spelling.electrical_in_root),
        )
        .replace("as drive;", &format!("as {};", spelling.drive_in_root))
        .replace("electrical.", &format!("{}.", spelling.electrical_in_root))
        .replace("drive.", &format!("{}.", spelling.drive_in_root));
    let sources = package_sources(
        manifest,
        vec![
            source_file("README.md", BundleRoleV1::Documentation, ROOT_README),
            source_file(
                spelling.root_path,
                BundleRoleV1::ModelSource,
                root_source.as_bytes(),
            ),
        ],
    );
    let dependencies = if spelling.reverse_dependency_releases {
        vec![drive.clone(), electrical.clone()]
    } else {
        vec![electrical.clone(), drive.clone()]
    };
    prepare_package_release_v1(sources, &dependencies).expect("compiler-derived root release")
}

fn root_release(
    electrical: &PackageReleaseV1,
    drive: &PackageReleaseV1,
    root_source: &str,
) -> PackageReleaseV1 {
    let release = root_release_with_spelling(electrical, drive, root_source, CANONICAL_SPELLING);
    assert_eq!(
        release.manifest(),
        &canonical_manifest(ROOT_MANIFEST),
        "checked-in manifest is the canonical spelling"
    );
    release
}

struct PackageFixture {
    packaged: PackagedModelDocument,
    resolution: ResolutionRecordV1,
    electrical_semantic: String,
    electrical_source: String,
    drive_semantic: String,
    drive_source: String,
    root_semantic: String,
    root_source: String,
}

fn packaged_model_from_releases(
    electrical: PackageReleaseV1,
    drive: PackageReleaseV1,
    root: PackageReleaseV1,
) -> PackageFixture {
    let electrical_identity = electrical.package_identity().expect("electrical identity");
    let drive_identity = drive.package_identity().expect("drive identity");
    let root_identity = root.package_identity().expect("root identity");
    let mut store = InMemoryPackageStore::default();
    let electrical_source = store.insert(&electrical).expect("store electrical package");
    let drive_source = store.insert(&drive).expect("store drive package");
    let root_source = store.insert(&root).expect("store root package");
    let resolution = ResolutionRecordV1::new(
        root_identity.clone(),
        vec![
            ResolutionNodeV1::new(electrical_identity.clone(), electrical_source),
            ResolutionNodeV1::new(drive_identity.clone(), drive_source),
            ResolutionNodeV1::new(root_identity.clone(), root_source),
        ],
        vec![
            ResolutionEdgeV1::new(drive_identity.clone(), electrical_identity.clone()),
            ResolutionEdgeV1::new(root_identity.clone(), electrical_identity.clone()),
            ResolutionEdgeV1::new(root_identity.clone(), drive_identity.clone()),
        ],
    )
    .expect("exact resolution");
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("locked package compilation");
    PackageFixture {
        packaged,
        resolution,
        electrical_semantic: electrical_identity.semantic_digest.to_hex(),
        electrical_source: electrical_source.to_hex(),
        drive_semantic: drive_identity.semantic_digest.to_hex(),
        drive_source: drive_source.to_hex(),
        root_semantic: root_identity.semantic_digest.to_hex(),
        root_source: root_source.to_hex(),
    }
}

fn packaged_model_with_root(root_source_text: &str) -> PackageFixture {
    let electrical = electrical_release();
    let drive = drive_release(&electrical);
    let root = root_release(&electrical, &drive, root_source_text);
    packaged_model_from_releases(electrical, drive, root)
}

fn packaged_model_with_spelling(
    root_source_text: &str,
    spelling: PackageSpelling<'_>,
) -> PackageFixture {
    let electrical = electrical_release();
    let drive = drive_release_with_spelling(&electrical, spelling);
    let root = root_release_with_spelling(&electrical, &drive, root_source_text, spelling);
    packaged_model_from_releases(electrical, drive, root)
}

fn packaged_model() -> PackageFixture {
    packaged_model_with_root(ROOT_SOURCE)
}

fn reference_config(end_time: f64, max_step: f64) -> ReferenceConfig {
    ReferenceConfig::new(end_time, max_step)
        .expect("reference config")
        .with_nonlinear_tolerances(NONLINEAR_ABSOLUTE_TOLERANCE, NONLINEAR_RELATIVE_TOLERANCE)
        .expect("nonlinear tolerances")
        .with_limits(MAXIMUM_NONLINEAR_ITERATIONS, MAXIMUM_SEMANTIC_STEPS)
        .expect("execution limits")
}

fn execute(fixture: &PackageFixture, end_time: f64, max_step: f64) -> Trajectory {
    Interpreter::new()
        .run(
            fixture.packaged.model().program(),
            reference_config(end_time, max_step),
        )
        .expect("joint reference execution")
}

fn field_at(trajectory: &Trajectory, field: eqiora::RawId, time: f64) -> f64 {
    trajectory
        .samples()
        .iter()
        .find(|sample| sample.coordinate() == field && (sample.time() - time).abs() <= 2.0e-12)
        .unwrap_or_else(|| panic!("missing Field {field} sample at {time}"))
        .value()
        .value()
}

fn physical_at(trajectory: &Trajectory, unknown: PhysicalUnknown, time: f64) -> f64 {
    trajectory
        .physical_samples()
        .iter()
        .find(|sample| sample.coordinate() == unknown && (sample.time() - time).abs() <= 2.0e-12)
        .unwrap_or_else(|| panic!("missing {unknown:?} sample at {time}"))
        .value()
        .value()
}

fn assert_physical_frame_contract(
    trajectory: &Trajectory,
    time: f64,
    electrical_ports: &[Id<kinds::Port>],
    rotational_ports: &[Id<kinds::Port>],
) {
    let mut ports = electrical_ports
        .iter()
        .chain(rotational_ports)
        .copied()
        .collect::<Vec<_>>();
    ports.sort_by_key(|port| port.erase());
    let expected_unknowns = ports
        .into_iter()
        .flat_map(|port| {
            [
                PhysicalUnknown::Across(port),
                PhysicalUnknown::Through(port),
            ]
        })
        .collect::<Vec<_>>();
    let frame = trajectory
        .physical_samples()
        .iter()
        .filter(|sample| (sample.time() - time).abs() <= 2.0e-12)
        .collect::<Vec<_>>();
    assert_eq!(
        frame
            .iter()
            .map(|sample| sample.coordinate())
            .collect::<Vec<_>>(),
        expected_unknowns,
        "physical frame must be complete and canonically ordered at {time}"
    );
    for sample in frame {
        let unknown = sample.coordinate();
        let expected_dimension = if electrical_ports.contains(&unknown.port()) {
            match unknown {
                PhysicalUnknown::Across(_) => VOLTAGE_DIMENSION,
                PhysicalUnknown::Through(_) => CURRENT_DIMENSION,
            }
        } else {
            assert!(rotational_ports.contains(&unknown.port()));
            match unknown {
                PhysicalUnknown::Across(_) => ANGULAR_SPEED_DIMENSION,
                PhysicalUnknown::Through(_) => TORQUE_DIMENSION,
            }
        };
        assert_eq!(
            sample.value().dim(),
            expected_dimension,
            "physical dimension for {unknown:?} at {time}"
        );
    }
}

fn field(packaged: &PackagedModelDocument, alias: &str) -> eqiora::RawId {
    packaged.model().aliases()[alias]
}

fn port(packaged: &PackagedModelDocument, alias: &str) -> Id<kinds::Port> {
    packaged.model().aliases()[alias]
        .downcast()
        .unwrap_or_else(|| panic!("{alias} is not a Port"))
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

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.16e}, received {actual:.16e}, tolerance {tolerance:.3e}"
    );
}

fn advance_exact(mut state: [f64; 2], command: f64, duration: f64) -> [f64; 2] {
    let half_trace = -3.0;
    let discriminant = 0.6_f64.sqrt();
    let scale = (half_trace * duration).exp();
    let cosine = (discriminant * duration).cosh();
    let sine = (discriminant * duration).sinh() / discriminant;
    let equilibrium = command / 2.1;
    state[0] -= equilibrium;
    state[1] -= equilibrium;
    let current = scale * (cosine * state[0] + sine * (-state[0] - 0.2 * state[1]));
    let speed = scale * (cosine * state[1] + sine * (2.0 * state[0] + state[1]));
    [current + equilibrium, speed + equilibrium]
}

fn exact_sampled_state(time: f64) -> ([f64; 2], f64) {
    let completed_intervals = ((time + 1.0e-13) / SAMPLE_PERIOD).floor() as usize;
    let mut state = [0.0, 0.0];
    let mut command = GAIN * (SETPOINT - state[1]);
    for _ in 0..completed_intervals {
        state = advance_exact(state, command, SAMPLE_PERIOD);
        command = GAIN * (SETPOINT - state[1]);
    }
    let remainder = time - completed_intervals as f64 * SAMPLE_PERIOD;
    if remainder > 1.0e-13 {
        state = advance_exact(state, command, remainder);
    }
    (state, command)
}

fn reference_run(packaged: &PackagedModelDocument, topology: &str) -> RunManifestV1 {
    RunManifestV1::new(
        ArtifactDigest::from_hex(packaged.model().digest().expect("model digest"))
            .expect("artifact digest"),
        packaged.model().program().revision().0,
        "eqiora-sem-reference",
        env!("CARGO_PKG_VERSION"),
    )
    .expect("run manifest")
    .with_numerical_setting("execution.topology", topology)
    .expect("execution topology")
    .with_numerical_setting(
        "execution.maximum-semantic-steps",
        MAXIMUM_SEMANTIC_STEPS.to_string(),
    )
    .expect("semantic-step limit")
    .with_numerical_setting("integration.end-time-seconds", END_TIME_S.to_string())
    .expect("end time")
    .with_numerical_setting(
        "integration.maximum-step-seconds",
        ACCEPTED_MAXIMUM_STEP_S.to_string(),
    )
    .expect("maximum step")
    .with_numerical_setting("integration.method", "backward-euler")
    .expect("integration method")
    .with_numerical_setting("scalar.type", "f64")
    .expect("scalar type")
    .with_numerical_setting(
        "solver.absolute-tolerance",
        NONLINEAR_ABSOLUTE_TOLERANCE.to_string(),
    )
    .expect("absolute tolerance")
    .with_numerical_setting(
        "solver.initial-guess",
        "zero-initial-consistency-then-forward-euler-state-and-accepted-algebraics",
    )
    .expect("initial guess")
    .with_numerical_setting(
        "solver.maximum-iterations",
        MAXIMUM_NONLINEAR_ITERATIONS.to_string(),
    )
    .expect("maximum iterations")
    .with_numerical_setting("solver.method", "dense-finite-difference-newton")
    .expect("solver method")
    .with_numerical_setting(
        "solver.relative-tolerance",
        NONLINEAR_RELATIVE_TOLERANCE.to_string(),
    )
    .expect("relative tolerance")
}

fn invalid_root_diagnostics(root_source: &str) -> Vec<Diagnostic> {
    let electrical = electrical_release();
    let drive = drive_release(&electrical);
    let electrical_identity = electrical.package_identity().expect("electrical identity");
    let drive_identity = drive.package_identity().expect("drive identity");
    let selected = provisional_namespace("org.example.dc_motor_control", VERSION);
    let electrical_namespace = namespace(&electrical_identity);
    let drive_namespace = namespace(&drive_identity);
    let input = ResolvedHierarchyInput::new(
        selected.clone(),
        vec![
            ResolvedSourceUnit::new(selected.clone(), ROOT_PATH, root_source)
                .expect("root source path"),
            ResolvedSourceUnit::new(drive_namespace.clone(), DRIVE_PATH, DRIVE_SOURCE)
                .expect("drive source path"),
            ResolvedSourceUnit::new(
                electrical_namespace.clone(),
                ELECTRICAL_PATH,
                ELECTRICAL_SOURCE,
            )
            .expect("electrical source path"),
        ],
        vec![
            ResolvedDependency::new(selected.clone(), drive_namespace.clone()),
            ResolvedDependency::new(selected, electrical_namespace.clone()),
            ResolvedDependency::new(drive_namespace, electrical_namespace),
        ],
    );
    match analyze_resolved_hierarchy(input) {
        Ok(analysis) => analysis
            .validate_definitions()
            .and_then(|validated| validated.compile_root("Main"))
            .expect_err("invalid root must not expose a compiled model"),
        Err(diagnostics) => diagnostics,
    }
}

#[test]
fn incompatible_connector_families_fail_before_model_exposure() {
    let same_dimension_connector = r#"
connector OtherFlange {
  across potential: 1 / s;
  through flow: kg * m ^ 2 / s ^ 2;
}

component OtherAnchor(
  port shaft: OtherFlange
) {
  relation law { shaft.flow = 0; }
}
"#;
    let nominal_mismatch = ROOT_SOURCE
        .replacen("\n\n", &format!("\n\n{same_dimension_connector}\n"), 1)
        .replace(
            "  instance ground: electrical.Ground();",
            "  instance ground: electrical.Ground();\n  instance other: OtherAnchor();",
        )
        .replace(
            "connect motor.shaft, load.shaft, sensor.shaft;",
            "connect motor.shaft, load.shaft, sensor.shaft, other.shaft;",
        );
    let diagnostics = invalid_root_diagnostics(&nominal_mismatch);
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("nominal Connector or Domain")),
        "{diagnostics:#?}"
    );

    let causal_as_conserving = ROOT_SOURCE.replace(
        "connect controller.command -> source.command;",
        "connect controller.command, source.command;",
    );
    let diagnostics = invalid_root_diagnostics(&causal_as_conserving);
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.message().contains("signal") || diagnostic.message().contains("conserving")
        }),
        "{diagnostics:#?}"
    );

    let conserving_as_causal = ROOT_SOURCE.replace(
        "connect source.positive, motor.positive;",
        "connect source.positive -> motor.positive;",
    );
    let diagnostics = invalid_root_diagnostics(&conserving_as_causal);
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.message().contains("signal") || diagnostic.message().contains("conserving")
        }),
        "{diagnostics:#?}"
    );
}

#[test]
fn exact_packages_execute_and_accept_one_sampled_acausal_drive() {
    let fixture = packaged_model();
    let packaged = &fixture.packaged;
    assert_eq!(
        packaged
            .model()
            .program()
            .nodes()
            .filter(|node| matches!(node, KernelNode::Parameter(_)))
            .count(),
        0,
        "literal Component bindings specialize expressions, not Kernel Parameters"
    );
    for alias in [
        "motor.resistance",
        "motor.inductance",
        "motor.motor_constant",
        "load.inertia",
        "load.damping",
        "controller.setpoint",
        "controller.gain",
    ] {
        assert_eq!(packaged.model().aliases().get(alias), None);
    }
    packaged
        .compilation()
        .validate_against(&fixture.resolution)
        .expect("exact package compilation replay");
    let permuted_resolution = ResolutionRecordV1::new(
        fixture.resolution.root().clone(),
        fixture.resolution.nodes().iter().rev().cloned().collect(),
        fixture.resolution.edges().iter().rev().cloned().collect(),
    )
    .expect("permuted exact resolution");
    assert_eq!(
        fixture
            .resolution
            .canonical_json()
            .expect("canonical resolution"),
        permuted_resolution
            .canonical_json()
            .expect("permuted canonical resolution")
    );

    let current = field(packaged, "motor.current");
    let load_speed = field(packaged, "load.speed");
    let motor_speed = field(packaged, "motor.speed");
    let motor_torque = field(packaged, "motor.electromagnetic_torque");
    let load_torque = field(packaged, "load.reaction_torque");
    let applied_voltage = field(packaged, "source.applied_voltage");
    let held_voltage = field(packaged, "controller.held_voltage");

    let coarse = execute(&fixture, END_TIME_S, COARSE_MAXIMUM_STEP_S);
    let trajectory = execute(&fixture, END_TIME_S, ACCEPTED_MAXIMUM_STEP_S);
    let (exact_terminal, exact_command) = exact_sampled_state(END_TIME_S);
    let coarse_current_error_a = (field_at(&coarse, current, END_TIME_S) - exact_terminal[0]).abs();
    let fine_current_error_a =
        (field_at(&trajectory, current, END_TIME_S) - exact_terminal[0]).abs();
    let coarse_speed_error_per_s =
        (field_at(&coarse, load_speed, END_TIME_S) - exact_terminal[1]).abs();
    let fine_speed_error_per_s =
        (field_at(&trajectory, load_speed, END_TIME_S) - exact_terminal[1]).abs();
    assert!(
        fine_current_error_a < 0.62 * coarse_current_error_a,
        "coarse current error {coarse_current_error_a} A, fine current error {fine_current_error_a} A"
    );
    assert!(
        fine_speed_error_per_s < 0.62 * coarse_speed_error_per_s,
        "coarse speed error {coarse_speed_error_per_s} 1/s, fine speed error {fine_speed_error_per_s} 1/s"
    );
    assert!(
        fine_current_error_a < 3.0e-3,
        "fine current error {fine_current_error_a} A"
    );
    assert!(
        fine_speed_error_per_s < 3.0e-3,
        "fine speed error {fine_speed_error_per_s} 1/s"
    );
    assert_close(
        field_at(&trajectory, held_voltage, END_TIME_S),
        exact_command,
        3.0e-3,
    );

    assert_close(field_at(&trajectory, held_voltage, 0.0), 10.0, 2.0e-10);
    assert_close(field_at(&trajectory, held_voltage, 0.005), 10.0, 2.0e-10);
    let speed_at_first_tick = field_at(&trajectory, load_speed, SAMPLE_PERIOD);
    assert_close(
        field_at(&trajectory, held_voltage, SAMPLE_PERIOD),
        GAIN * (SETPOINT - speed_at_first_tick),
        2.0e-9,
    );
    assert_close(
        field_at(&trajectory, applied_voltage, SAMPLE_PERIOD),
        field_at(&trajectory, held_voltage, SAMPLE_PERIOD),
        2.0e-9,
    );

    let source_positive = port(packaged, "source.positive");
    let source_negative = port(packaged, "source.negative");
    let motor_positive = port(packaged, "motor.positive");
    let motor_negative = port(packaged, "motor.negative");
    let motor_shaft = port(packaged, "motor.shaft");
    let load_shaft = port(packaged, "load.shaft");
    let sensor_shaft = port(packaged, "sensor.shaft");
    let ground = port(packaged, "ground.terminal");
    let electrical_ports = [
        source_positive,
        source_negative,
        motor_positive,
        motor_negative,
        ground,
    ];
    let rotational_ports = [motor_shaft, load_shaft, sensor_shaft];

    // A phase-zero controller tick is followed by physical restoration before
    // the first observation: the states remain zero while the accepted
    // electrical across values already reflect the held 10 V command.
    assert_close(field_at(&trajectory, current, 0.0), 0.0, 1.0e-12);
    assert_close(field_at(&trajectory, load_speed, 0.0), 0.0, 1.0e-12);
    assert_physical_frame_contract(&trajectory, 0.0, &electrical_ports, &rotational_ports);
    assert_close(
        physical_at(&trajectory, PhysicalUnknown::Across(source_positive), 0.0)
            - physical_at(&trajectory, PhysicalUnknown::Across(source_negative), 0.0),
        10.0,
        2.0e-9,
    );

    let time = 0.099;
    let previous_time = 0.098;
    let step = time - previous_time;
    let i = field_at(&trajectory, current, time);
    let previous_i = field_at(&trajectory, current, previous_time);
    let omega = field_at(&trajectory, load_speed, time);
    let previous_omega = field_at(&trajectory, load_speed, previous_time);
    let di = (i - previous_i) / step;
    let domega = (omega - previous_omega) / step;
    let voltage = field_at(&trajectory, applied_voltage, time);
    let torque = field_at(&trajectory, motor_torque, time);
    let reaction = field_at(&trajectory, load_torque, time);
    assert_physical_frame_contract(&trajectory, time, &electrical_ports, &rotational_ports);

    let across = |port| physical_at(&trajectory, PhysicalUnknown::Across(port), time);
    let through = |port| physical_at(&trajectory, PhysicalUnknown::Through(port), time);

    assert_close(
        across(source_positive),
        across(motor_positive),
        VOLTAGE_RESIDUAL_TOLERANCE_V,
    );
    assert_close(
        across(source_negative),
        across(motor_negative),
        VOLTAGE_RESIDUAL_TOLERANCE_V,
    );
    assert_close(
        across(source_negative),
        across(ground),
        VOLTAGE_RESIDUAL_TOLERANCE_V,
    );
    assert_close(
        across(motor_shaft),
        across(load_shaft),
        ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S,
    );
    assert_close(
        across(motor_shaft),
        across(sensor_shaft),
        ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S,
    );
    assert_close(
        through(source_positive) + through(motor_positive),
        0.0,
        CURRENT_RESIDUAL_TOLERANCE_A,
    );
    assert_close(
        through(source_negative) + through(motor_negative) + through(ground),
        0.0,
        CURRENT_RESIDUAL_TOLERANCE_A,
    );
    assert_close(
        through(motor_shaft) + through(load_shaft) + through(sensor_shaft),
        0.0,
        TORQUE_RESIDUAL_TOLERANCE_N_M,
    );

    assert_close(
        across(source_positive) - across(source_negative) - voltage,
        0.0,
        VOLTAGE_RESIDUAL_TOLERANCE_V,
    );
    assert_close(
        across(motor_positive)
            - across(motor_negative)
            - RESISTANCE * i
            - INDUCTANCE * di
            - MOTOR_CONSTANT * omega,
        0.0,
        VOLTAGE_RESIDUAL_TOLERANCE_V,
    );
    assert_close(
        through(motor_positive) - i,
        0.0,
        CURRENT_RESIDUAL_TOLERANCE_A,
    );
    assert_close(
        through(motor_positive) + through(motor_negative),
        0.0,
        CURRENT_RESIDUAL_TOLERANCE_A,
    );
    assert_close(
        across(motor_shaft) - omega,
        0.0,
        ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S,
    );
    assert_close(
        torque - MOTOR_CONSTANT * i,
        0.0,
        TORQUE_RESIDUAL_TOLERANCE_N_M,
    );
    assert_close(
        through(motor_shaft) + torque,
        0.0,
        TORQUE_RESIDUAL_TOLERANCE_N_M,
    );
    assert_close(
        across(load_shaft) - omega,
        0.0,
        ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S,
    );
    assert_close(
        reaction - INERTIA * domega - DAMPING * omega,
        0.0,
        TORQUE_RESIDUAL_TOLERANCE_N_M,
    );
    assert_close(
        through(load_shaft) - reaction,
        0.0,
        TORQUE_RESIDUAL_TOLERANCE_N_M,
    );
    assert_close(through(sensor_shaft), 0.0, TORQUE_RESIDUAL_TOLERANCE_N_M);
    assert_close(across(ground), 0.0, VOLTAGE_RESIDUAL_TOLERANCE_V);
    assert_close(
        field_at(&trajectory, motor_speed, time),
        omega,
        ANGULAR_SPEED_RESIDUAL_TOLERANCE_PER_S,
    );

    let source_power = voltage * i;
    let resistive_loss = RESISTANCE * i * i;
    let load_loss = DAMPING * omega * omega;
    let electrical_storage_rate = INDUCTANCE * i * di;
    let mechanical_storage_rate = INERTIA * omega * domega;
    let transduced_power = MOTOR_CONSTANT * i * omega;
    assert_close(
        source_power,
        resistive_loss + electrical_storage_rate + transduced_power,
        POWER_BALANCE_TOLERANCE_W,
    );
    assert_close(
        transduced_power,
        load_loss + mechanical_storage_rate,
        POWER_BALANCE_TOLERANCE_W,
    );
    let stored_energy_change = (0.5 * INDUCTANCE * (i * i - previous_i * previous_i)
        + 0.5 * INERTIA * (omega * omega - previous_omega * previous_omega))
        / step;
    let numerical_dissipation = (0.5 * INDUCTANCE * (i - previous_i).powi(2)
        + 0.5 * INERTIA * (omega - previous_omega).powi(2))
        / step;
    assert!(resistive_loss >= 0.0 && load_loss >= 0.0 && numerical_dissipation >= 0.0);
    assert_close(
        source_power,
        resistive_loss + load_loss + stored_energy_change + numerical_dissipation,
        POWER_BALANCE_TOLERANCE_W,
    );
    assert_close(
        across(source_positive) * through(source_positive)
            + across(motor_positive) * through(motor_positive),
        0.0,
        POWER_BALANCE_TOLERANCE_W,
    );
    assert_close(
        across(motor_shaft) * through(motor_shaft)
            + across(load_shaft) * through(load_shaft)
            + across(sensor_shaft) * through(sensor_shaft),
        0.0,
        POWER_BALANCE_TOLERANCE_W,
    );

    let connection = selected_connection(packaged, motor_positive);
    let static_error = lower_scalar_physical_affine(packaged.model().program(), connection, None)
        .expect_err("dynamic physical model must not enter the static affine specialization");
    assert!(static_error.message().contains("does not admit state"));

    // Lineage is deliberately unreachable when the requested trajectory
    // exceeds an explicit semantic-step safety limit after the phase-zero
    // tick. This failure is deterministic and independent of Newton details.
    let failed_execution = Interpreter::new().run(
        packaged.model().program(),
        ReferenceConfig::new(0.01, 0.001)
            .and_then(|config| config.with_limits(MAXIMUM_NONLINEAR_ITERATIONS, 1))
            .expect("bounded failing config"),
    );
    assert!(
        failed_execution.as_ref().is_err_and(|diagnostics| {
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains("accepted-step budget"))
        }),
        "{failed_execution:?}"
    );
    let failed_lineage = failed_execution.ok().and_then(|_| {
        packaged
            .bind_run_v1(&reference_run(packaged, "one-host-one-worker"))
            .ok()
    });
    assert!(failed_lineage.is_none());

    // Construct identity lineage only after all package, trajectory,
    // constitutive, junction, convergence, and power evidence above accepts.
    let run = reference_run(packaged, "one-host-one-worker");
    let binding = packaged.bind_run_v1(&run).expect("package run binding");
    packaged
        .validate_run_v1_binding(&binding, &run, &fixture.resolution)
        .expect("exact package/run replay");
    let replayed_binding = PackageRunBindingV1::from_json(
        &binding
            .canonical_json()
            .expect("canonical package run binding"),
    )
    .expect("replayed package run binding");
    packaged
        .validate_run_v1_binding(&replayed_binding, &run, &fixture.resolution)
        .expect("canonical package/run replay");

    let schedule_variant = reference_run(packaged, "one-host-two-workers");
    assert_ne!(
        run.digest().expect("run digest"),
        schedule_variant.digest().expect("schedule run digest")
    );
    assert_eq!(run.model(), schedule_variant.model());

    let permuted_root_source = ROOT_SOURCE
        .replace(
            "  instance source: drive.ControlledVoltageSource();",
            "  instance __permutation_slot: drive.ControlledVoltageSource();",
        )
        .replace(
            "  instance sensor: drive.SpeedSensor();",
            "  instance source: drive.ControlledVoltageSource();",
        )
        .replace(
            "  instance __permutation_slot: drive.ControlledVoltageSource();",
            "  instance sensor: drive.SpeedSensor();",
        );
    let permuted_root = packaged_model_with_root(&permuted_root_source);
    assert_eq!(fixture.root_semantic, permuted_root.root_semantic);
    assert_ne!(fixture.root_source, permuted_root.root_source);
    assert_eq!(
        packaged.model().digest().expect("model digest"),
        permuted_root
            .packaged
            .model()
            .digest()
            .expect("permuted model digest")
    );

    let respelled = packaged_model_with_spelling(
        ROOT_SOURCE,
        PackageSpelling {
            drive_path: DRIVE_PATH,
            root_path: ROOT_PATH,
            electrical_in_drive: "circuit",
            electrical_in_root: "foundation",
            drive_in_root: "machines",
            reverse_dependency_releases: true,
        },
    );
    respelled
        .packaged
        .compilation()
        .validate_against(&respelled.resolution)
        .expect("respelled exact package compilation replay");
    assert_eq!(fixture.electrical_semantic, respelled.electrical_semantic);
    assert_eq!(fixture.drive_semantic, respelled.drive_semantic);
    assert_eq!(fixture.root_semantic, respelled.root_semantic);
    assert_eq!(fixture.electrical_source, respelled.electrical_source);
    assert_ne!(fixture.drive_source, respelled.drive_source);
    assert_ne!(fixture.root_source, respelled.root_source);
    assert_eq!(
        packaged.model().digest().expect("model digest"),
        respelled
            .packaged
            .model()
            .digest()
            .expect("respelled model digest"),
        "source import aliases and source-unit insertion order are not model meaning"
    );

    let changed_clock_source = ROOT_SOURCE.replacen("periodic(1[s] / 100", "periodic(1[s] / 50", 1);
    assert_ne!(
        changed_clock_source, ROOT_SOURCE,
        "clock mutant must change the authored period"
    );
    let changed_clock = packaged_model_with_root(&changed_clock_source);
    assert_ne!(
        packaged.model().digest().expect("model digest"),
        changed_clock
            .packaged
            .model()
            .digest()
            .expect("changed-clock model digest")
    );
    assert_ne!(fixture.root_semantic, changed_clock.root_semantic);

    let multiple_clock_source = ROOT_SOURCE.replace(
        "  clock sample = periodic(1[s] / 100, phase = 0[s] / 1);",
        "  clock sample = periodic(1[s] / 100, phase = 0[s] / 1);\n\
         \n\
           state secondary_hold: kg * m ^ 2 / (s ^ 3 * A) at secondary; initial { secondary_hold = 0; }\n\
           clock secondary = periodic(1[s] / 50, phase = 0[s] / 1);\n\
           relation secondary_update at secondary {\n\
             next(secondary_hold) - pre(secondary_hold) = 0;\n\
           }",
    );
    let multiple_clock = packaged_model_with_root(&multiple_clock_source);
    let multiple_clock_diagnostics = Interpreter::new()
        .run(
            multiple_clock.packaged.model().program(),
            reference_config(0.01, ACCEPTED_MAXIMUM_STEP_S),
        )
        .expect_err("joint physical execution must fail closed on multiple clocks");
    assert!(multiple_clock_diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("at most one periodic ClockDomain")
    }));
}
