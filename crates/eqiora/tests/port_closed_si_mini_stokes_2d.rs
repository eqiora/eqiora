use std::num::NonZeroUsize;

use eqiora::artifact::SimplicialMeshEnvelopeV1;
use eqiora::kernel::BoundarySide;
use eqiora::meshing::{MeshQualityGate, SimplicialMesh};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::realization::{
    FieldwiseRealizationRequest, MeshArtifactReference, RealizationCapabilities,
    RealizationRevision, SemanticRevision, resolve_fieldwise,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    CanonicalCsrSystemView, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    SolverPlan,
};
use eqiora::{DimExponents, DynQuantity};
use eqiora_numerics::{
    common::PhysicalBoundaryDisposition, common::ScalarSpatialExpression,
    fluid::SteadyStokesMiniSolution2d, fluid::SteadyStokesScaleProfile2d,
    fluid::finalize_resolved_steady_stokes_mini_2d,
    fluid::lower_steady_incompressible_stokes_cartesian_2d,
    fluid::steady_stokes_fieldwise_requirements_2d, fluid::steady_stokes_mini_plan_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT: &str =
    include_str!("../../../verify/fluid/port-closed-si-mini-stokes-2d/models/direct.eqi");
const PACKAGED: &str =
    include_str!("../../../verify/fluid/port-closed-si-mini-stokes-2d/models/packaged.eqi");
const MECHANICS_SOURCE: &str = include_str!(
    "../../../crates/eqiora-api/packages/Eqiora.Mechanics.Interfaces/src/interfaces.eqi"
);
const FLUID_SOURCE: &str = include_str!(
    "../../../crates/eqiora-api/packages/Eqiora.Fluid.Incompressible/src/incompressible.eqi"
);

const ROOT_PACKAGE: &str = "org.eqiora.verify.port_closed_si_mini_stokes_2d";
const VERSION: &str = "0.1.0";

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

#[derive(Debug)]
struct Observation {
    system: CanonicalCsrSystemView,
    solution: SteadyStokesMiniSolution2d,
}

#[test]
fn exact_three_release_chain_reaches_the_unchanged_si_mini_path() {
    let mechanics = public_mechanics_release();
    let fluid = public_fluid_release(&mechanics);
    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct zero-trace Stokes Model compiles");
    let packaged = compile_root(&fluid, &mechanics, "fluid", "mechanics", PACKAGED);

    let alias_source = alias_and_reverse_connection_endpoints(PACKAGED);
    let permuted = compile_root(
        &fluid,
        &mechanics,
        "liquid",
        "boundary_physics",
        &alias_source,
    );

    let direct_model = lower_steady_incompressible_stokes_cartesian_2d(direct.program())
        .expect("direct Stokes Model lowers");
    let packaged_model =
        lower_steady_incompressible_stokes_cartesian_2d(packaged.model().program())
            .expect("packaged Stokes Model lowers");
    assert_eq!(
        packaged.model().aliases()["dynamic_viscosity"],
        packaged.model().aliases()["law.dynamic_viscosity"]
    );
    assert_eq!(
        packaged.model().aliases()["dynamic_viscosity"],
        packaged.model().aliases()["boundary.dynamic_viscosity"]
    );
    assert_eq!(
        coefficient_derivatives(direct_model.dynamic_viscosity_expression()),
        coefficient_derivatives(packaged_model.dynamic_viscosity_expression())
    );

    assert_all_boundaries(direct.program(), PhysicalBoundaryDisposition::TraceZero);
    assert_all_boundaries(
        packaged.model().program(),
        PhysicalBoundaryDisposition::TraceZero,
    );
    assert_all_boundaries(
        permuted.model().program(),
        PhysicalBoundaryDisposition::TraceZero,
    );

    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).expect("physical mesh");
    let direct = observe(direct.program(), &mesh, 1);
    let packaged = observe(packaged.model().program(), &mesh, 2);
    let permuted = observe(permuted.model().program(), &mesh, 3);

    assert_eq!(direct.system, packaged.system);
    assert_eq!(packaged.system, permuted.system);
    assert_physical_equivalence(&direct.solution, &packaged.solution);
    assert_physical_equivalence(&packaged.solution, &permuted.solution);
}

#[test]
fn flux_zero_selects_boundary_pressure_while_live_port_fails_before_mesh_access() {
    let mechanics = public_mechanics_release();
    let fluid = public_fluid_release(&mechanics);

    let flux_source = PACKAGED.replacen(
        "instance x_lower_zero: mechanics.ZeroVelocity2d",
        "instance x_lower_zero: mechanics.ZeroTraction2d",
        1,
    );
    let flux = compile_root(&fluid, &mechanics, "fluid", "mechanics", &flux_source);
    let flux_model = lower_steady_incompressible_stokes_cartesian_2d(flux.model().program())
        .expect("zero traction remains valid canonical boundary meaning");
    assert_eq!(
        flux_model
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("x-lower boundary")
            .disposition(),
        PhysicalBoundaryDisposition::FluxZero
    );
    assert_other_sides_are_trace_zero(&flux_model, (0, BoundarySide::Lower));
    let deliberately_unresolved_mesh = MeshArtifactReference::from_sha256([0xA5; 32]);
    let flux_plan = steady_stokes_mini_plan_2d(
        &flux_model,
        deliberately_unresolved_mesh,
        scale_profile(),
        solver(),
    )
    .expect("zero traction is a zero normal-pressure boundary");
    assert!(flux_plan.spatial().constraints().is_empty());
    assert_eq!(flux_plan.scaling().block_scales().len(), 2);

    let live_source = transparent_open_terminal_source(PACKAGED);
    let live = compile_root(&fluid, &mechanics, "fluid", "mechanics", &live_source);
    let live_model = lower_steady_incompressible_stokes_cartesian_2d(live.model().program())
        .expect("a compatible unclosed Port remains valid canonical boundary meaning");
    assert!(matches!(
        live_model
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("x-lower boundary")
            .disposition(),
        PhysicalBoundaryDisposition::PortBinding { .. }
    ));
    assert_other_sides_are_trace_zero(&live_model, (0, BoundarySide::Lower));
    assert_plan_rejects_before_mesh(
        &live_model,
        live_model
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("x-lower boundary")
            .disposition(),
        "live",
    );
}

#[test]
fn wrong_newtonian_stress_and_nominal_connector_near_misses_fail_closed() {
    let mechanics = public_mechanics_release();

    let fluid = public_fluid_release(&mechanics);
    let independent_equal_coefficient = PACKAGED
        .replace(
            "  parameter dynamic_viscosity: kg / (m * s) = 6;",
            "  parameter dynamic_viscosity: kg / (m * s) = 6;\n  parameter boundary_dynamic_viscosity: kg / (m * s) = 6;",
        )
        .replace(
            "instance boundary: fluid.NewtonianMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    velocity = velocity,\n    pressure = pressure,\n    dynamic_viscosity = dynamic_viscosity",
            "instance boundary: fluid.NewtonianMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    velocity = velocity,\n    pressure = pressure,\n    dynamic_viscosity = boundary_dynamic_viscosity",
        );
    assert_ne!(independent_equal_coefficient, PACKAGED);
    let independent = compile_root(
        &fluid,
        &mechanics,
        "fluid",
        "mechanics",
        &independent_equal_coefficient,
    );
    let diagnostic = lower_steady_incompressible_stokes_cartesian_2d(independent.model().program())
        .expect_err("equal values cannot merge independent viscosity directions");
    assert!(
        diagnostic
            .message()
            .contains("viscosity coefficients differ")
    );

    let (volume_source, boundary_source) = FLUID_SOURCE
        .split_once("public component NewtonianMechanicalInterface2d")
        .expect("public fluid source owns a separate boundary Component");
    let wrong_stress_source = format!(
        "{volume_source}public component NewtonianMechanicalInterface2d{}",
        boundary_source.replace("- isotropic_lift(pressure)", "+ isotropic_lift(pressure)")
    );
    let wrong_stress_fluid = inline_fluid_release(&mechanics, &wrong_stress_source);
    let wrong_stress = compile_root(
        &wrong_stress_fluid,
        &mechanics,
        "fluid",
        "mechanics",
        PACKAGED,
    );
    let diagnostic =
        lower_steady_incompressible_stokes_cartesian_2d(wrong_stress.model().program())
            .expect_err("a pressure-sign near miss must not be recognized as Newtonian traction");
    assert!(
        diagnostic.message().contains("stress")
            || diagnostic.message().contains("traction")
            || diagnostic.message().contains("boundary")
            || diagnostic.message().contains("momentum balance"),
        "unexpected stress diagnostic: {}",
        diagnostic.message()
    );

    let distinct_connector = r#"

public connector OtherVelocityTractionBoundary {
  trace velocity: m / s;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
"#;
    let other_mechanics =
        inline_mechanics_release(&format!("{MECHANICS_SOURCE}{distinct_connector}"));
    let other_fluid_source = FLUID_SOURCE.replace(
        "mechanics.VelocityTractionBoundary",
        "mechanics.OtherVelocityTractionBoundary",
    );
    let other_fluid = inline_fluid_release(&other_mechanics, &other_fluid_source);
    let dependencies = vec![
        PackageDependencyV1::new(other_fluid.package_identity().unwrap()),
        PackageDependencyV1::new(other_mechanics.package_identity().unwrap()),
    ];
    let source = format!(
        "import Eqiora.Fluid.Incompressible.incompressible as fluid;\n\
         import Eqiora.Mechanics.Interfaces.interfaces as mechanics;\n{PACKAGED}"
    );
    let error = prepare_package_release_v1(
        inline_sources(ROOT_PACKAGE, VERSION, dependencies, "src/main.eqi", &source),
        &[other_fluid, other_mechanics],
    )
    .expect_err("distinct nominal Connectors must fail before a root release exists");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("Connector")
            || rendered.contains("connector")
            || rendered.contains("nominal"),
        "unexpected nominal-identity diagnostic: {rendered}"
    );
}

fn coefficient_derivatives(expression: &ScalarSpatialExpression<f64>) -> (u64, Vec<u64>, Vec<u64>) {
    let coordinates = vec![0.0; expression.coordinate_dimension()];
    let mut jvps = Vec::new();
    let mut vjps = Vec::new();
    let cotangent = 2.25;
    let primal = expression.evaluate(&coordinates).unwrap();
    for parameter in 0..expression.parameter_fields().len() {
        let mut tangent = vec![0.0; expression.parameter_fields().len()];
        tangent[parameter] = 1.0;
        let (observed_primal, jvp) = expression
            .evaluate_parameter_jvp(&coordinates, &tangent)
            .unwrap();
        let (vjp_primal, vjp) = expression
            .evaluate_parameter_vjp(&coordinates, cotangent)
            .unwrap();
        assert_eq!(observed_primal.to_bits(), primal.to_bits());
        assert_eq!(vjp_primal.to_bits(), primal.to_bits());
        jvps.push(jvp.to_bits());
        vjps.push(vjp[parameter].to_bits());
        assert_eq!(vjp[parameter].to_bits(), (cotangent * jvp).to_bits());
    }
    (primal.to_bits(), jvps, vjps)
}

fn observe(
    program: &KernelProgram,
    mesh: &SimplicialMeshEnvelopeV1,
    realization_revision: u64,
) -> Observation {
    let lowered = lower_steady_incompressible_stokes_cartesian_2d(program)
        .expect("canonical Stokes meaning lowers");
    let mesh_reference = mesh.artifact_reference().expect("mesh identity");
    let plan = steady_stokes_mini_plan_2d(&lowered, mesh_reference, scale_profile(), solver())
        .expect("all-trace boundary admits the existing MINI plan");
    let resolved = resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(realization_revision),
            plan,
        ),
        steady_stokes_fieldwise_requirements_2d(&lowered),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("reference mixed capability resolves the exact plan");
    let (_, finalized) =
        finalize_resolved_steady_stokes_mini_2d(program, &resolved, mesh_reference, mesh.mesh())
            .expect("port-closed Stokes reaches the unchanged SI finalizer");
    let system = finalized.canonical_csr_system_view().clone();
    let linear_solution = REFERENCE_LINEAR_SOLVER
        .solve(
            &finalized.linear_problem().expect("linear problem"),
            solver(),
        )
        .expect("reference MINRES solves the dimensionless system");
    let solution = finalized
        .finish(linear_solution)
        .expect("physical solution is reaccepted");
    assert_physical_solution(&solution);
    Observation { system, solution }
}

fn assert_all_boundaries(program: &KernelProgram, expected: PhysicalBoundaryDisposition) {
    let model = lower_steady_incompressible_stokes_cartesian_2d(program)
        .expect("canonical Stokes boundary inventory");
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            assert_eq!(
                model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete Cartesian side")
                    .disposition(),
                expected
            );
        }
    }
}

fn assert_other_sides_are_trace_zero(
    model: &eqiora_numerics::fluid::SteadyIncompressibleStokesCartesianModel2d,
    excluded: (usize, BoundarySide),
) {
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            if (axis, side) == excluded {
                continue;
            }
            assert_eq!(
                model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete Cartesian side")
                    .disposition(),
                PhysicalBoundaryDisposition::TraceZero
            );
        }
    }
}

fn assert_plan_rejects_before_mesh(
    model: &eqiora_numerics::fluid::SteadyIncompressibleStokesCartesianModel2d,
    disposition: PhysicalBoundaryDisposition,
    message_fragment: &str,
) {
    let deliberately_unresolved_mesh = MeshArtifactReference::from_sha256([0xA5; 32]);
    let diagnostic = steady_stokes_mini_plan_2d(
        model,
        deliberately_unresolved_mesh,
        scale_profile(),
        solver(),
    )
    .expect_err("unsupported boundary meaning must fail at plan construction");
    assert!(
        diagnostic.message().contains(message_fragment)
            || diagnostic.message().contains("trace-space"),
        "{disposition:?} lost to a later mesh diagnostic: {}",
        diagnostic.message()
    );
}

fn assert_physical_solution(solution: &SteadyStokesMiniSolution2d) {
    for value in solution
        .velocity()
        .vertex_values()
        .iter()
        .chain(solution.velocity().cell_bubble_values())
        .flatten()
    {
        assert_close(*value, 0.0, 2.0e-10);
    }
    for (coordinates, pressure) in solution
        .pressure()
        .mesh()
        .vertices()
        .iter()
        .zip(solution.pressure().vertex_values())
    {
        assert_close(*pressure, 0.75 * (coordinates[0] - 2.0), 2.0e-10);
    }
    assert_close(
        solution
            .gauge_multiplier()
            .expect("all-essential pressure uses a zero-integral gauge"),
        0.0,
        2.0e-10,
    );
    assert_close(solution.pressure_integral(), 0.0, 2.0e-10);
    assert_close(solution.integrated_body_force()[0], 6.0, 2.0e-10);
    assert_close(solution.integrated_body_force()[1], 0.0, 2.0e-10);
    assert_close(solution.boundary_reaction()[0], -6.0, 2.0e-9);
    assert_close(solution.boundary_reaction()[1], 0.0, 2.0e-9);
}

fn assert_physical_equivalence(
    left: &SteadyStokesMiniSolution2d,
    right: &SteadyStokesMiniSolution2d,
) {
    assert_eq!(
        left.velocity().vertex_values(),
        right.velocity().vertex_values()
    );
    assert_eq!(
        left.velocity().cell_bubble_values(),
        right.velocity().cell_bubble_values()
    );
    assert_eq!(
        left.pressure().vertex_values(),
        right.pressure().vertex_values()
    );
    assert_eq!(left.gauge_multiplier(), right.gauge_multiplier());
    assert_eq!(left.pressure_integral(), right.pressure_integral());
    assert_eq!(left.integrated_body_force(), right.integrated_body_force());
    assert_eq!(left.boundary_reaction(), right.boundary_reaction());
}

fn public_mechanics_release() -> PackageReleaseV1 {
    public_release("Eqiora.Mechanics.Interfaces", &[])
}

fn public_fluid_release(mechanics: &PackageReleaseV1) -> PackageReleaseV1 {
    public_release(
        "Eqiora.Fluid.Incompressible",
        std::slice::from_ref(mechanics),
    )
}

fn public_release(package: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources(package);
    prepare_package_release_v1(sources, dependencies)
        .unwrap_or_else(|error| panic!("prepare public package {package}: {error:?}"))
}

fn inline_mechanics_release(source: &str) -> PackageReleaseV1 {
    prepare_package_release_v1(
        inline_sources(
            "Eqiora.Mechanics.Interfaces",
            VERSION,
            vec![],
            "src/interfaces.eqi",
            source,
        ),
        &[],
    )
    .expect("prepare synthetic mechanics near miss")
}

fn inline_fluid_release(mechanics: &PackageReleaseV1, source: &str) -> PackageReleaseV1 {
    let dependency = PackageDependencyV1::new(mechanics.package_identity().unwrap());
    prepare_package_release_v1(
        inline_sources(
            "Eqiora.Fluid.Incompressible",
            "0.2.0",
            vec![dependency],
            "src/incompressible.eqi",
            source,
        ),
        std::slice::from_ref(mechanics),
    )
    .expect("prepare synthetic fluid near miss")
}

fn compile_root(
    fluid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    fluid_alias: &str,
    mechanics_alias: &str,
    source: &str,
) -> PackagedModelDocument {
    let root = root_release(fluid, mechanics, fluid_alias, mechanics_alias, source);
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, &[fluid.clone(), mechanics.clone()])
            .expect("exact three-release resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(mechanics).expect("install mechanics package");
    store.insert(fluid).expect("install fluid package");
    store.insert(&root).expect("install verification root");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("exact three-release chain compiles offline")
}

fn root_release(
    fluid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    fluid_alias: &str,
    mechanics_alias: &str,
    source: &str,
) -> PackageReleaseV1 {
    let dependencies = vec![
        PackageDependencyV1::new(fluid.package_identity().unwrap()),
        PackageDependencyV1::new(mechanics.package_identity().unwrap()),
    ];
    let fluid_name = fluid.package_identity().expect("fluid identity").name;
    let mechanics_name = mechanics
        .package_identity()
        .expect("mechanics identity")
        .name;
    let source = format!(
        "import {fluid_name}.incompressible as {fluid_alias};\n\
         import {mechanics_name}.interfaces as {mechanics_alias};\n{source}"
    );
    prepare_package_release_v1(
        inline_sources(ROOT_PACKAGE, VERSION, dependencies, "src/main.eqi", &source),
        &[fluid.clone(), mechanics.clone()],
    )
    .expect("prepare exact verification root")
}

fn inline_sources(
    name: &str,
    version: &str,
    dependencies: Vec<PackageDependencyV1>,
    model_path: &str,
    model_source: &str,
) -> PackageSourcesV1 {
    let readme = NormalizedRelativePath::parse("README.md").unwrap();
    let model = NormalizedRelativePath::parse(model_path).unwrap();
    let manifest = PackageManifestV1::new(
        &model_path
            .strip_prefix("src/")
            .unwrap()
            .strip_suffix(".eqi")
            .unwrap()
            .replace('/', "."),
        QualifiedName::parse(name).unwrap(),
        ExactVersion::parse(version).unwrap(),
        dependencies,
        vec![
            BundleEntryV1::new(readme.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .unwrap();
    PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                readme,
                BundleRoleV1::Documentation,
                format!("Exact test package {name}.\n").into_bytes(),
            ),
            SourceFileV1::new(
                model,
                BundleRoleV1::ModelSource,
                model_source.as_bytes().to_vec(),
            ),
        ],
    )
    .unwrap()
}

fn alias_and_reverse_connection_endpoints(source: &str) -> String {
    let mut source = source
        .replace("fluid.", "liquid.")
        .replace("mechanics.", "boundary_physics.");
    for (side, terminal) in [
        ("x_lower", "x_lower_zero"),
        ("x_upper", "x_upper_zero"),
        ("y_lower", "y_lower_zero"),
        ("y_upper", "y_upper_zero"),
    ] {
        source = source.replace(
            &format!("boundary.mechanical[boundary = {side}], {terminal}.mechanical"),
            &format!("{terminal}.mechanical, boundary.mechanical[boundary = {side}]"),
        );
    }
    source
}

fn transparent_open_terminal_source(source: &str) -> String {
    let terminal = r#"
public component CompatibleOpenVelocityTerminal2d(
  support body: volume(ambient_dimension = 2),
  support face: boundary(parent = body),
  port mechanical: mechanics.VelocityTractionBoundary over face
) {

  relation transparent_carrier on face {
    mechanical.velocity - mechanical.velocity = 0;
    mechanical.traction - mechanical.traction = 0;
  }
}

"#;
    format!(
        "{terminal}{}",
        source.replacen(
            "instance x_lower_zero: mechanics.ZeroVelocity2d",
            "instance x_lower_zero: CompatibleOpenVelocityTerminal2d",
            1,
        )
    )
}

fn scale_profile() -> SteadyStokesScaleProfile2d {
    SteadyStokesScaleProfile2d::new(
        DynQuantity::new(4.0, LENGTH),
        DynQuantity::new(0.5, VELOCITY),
        DynQuantity::new(0.75, PRESSURE),
    )
    .expect("coherent-SI scale profile")
}

fn solver() -> SolverPlan {
    SolverPlan::new(
        eqiora::solver::LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn physical_mesh() -> SimplicialMesh {
    let nx = 4;
    let ny = 2;
    let width = nx + 1;
    let vertices = (0..=ny)
        .flat_map(|j| (0..=nx).map(move |i| vec![i as f64, j as f64]))
        .collect::<Vec<_>>();
    let mut cells = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let lower_left = j * width + i;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.5).unwrap())
        .expect("connected physical triangle mesh")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.17e}, got {actual:.17e}, tolerance {tolerance:.3e}"
    );
}
