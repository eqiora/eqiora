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
    FieldwiseRealizationPlan, FieldwiseRealizationRequest, MeshArtifactReference,
    RealizationCapabilities, RealizationRevision, SemanticRevision, resolve_fieldwise,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    AlgebraicBlock, CanonicalCsrSystemView, LinearSolverBackend, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity};
use eqiora_numerics::{
    common::PhysicalBoundaryDisposition, common::PhysicalBoundaryQuantity,
    fluid::SteadyStokesMiniSolution2d, fluid::SteadyStokesPressureReference2d,
    fluid::SteadyStokesScaleProfile2d, fluid::finalize_resolved_steady_stokes_mini_2d,
    fluid::lower_steady_incompressible_stokes_cartesian_2d,
    fluid::steady_stokes_fieldwise_requirements_2d, fluid::steady_stokes_mini_plan_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT: &str =
    include_str!("../../../verify/fluid/mixed-static-pressure-mini-stokes-2d/models/direct.eqi");
const PACKAGED: &str =
    include_str!("../../../verify/fluid/mixed-static-pressure-mini-stokes-2d/models/packaged.eqi");

const ROOT_PACKAGE: &str = "org.eqiora.verify.mixed_static_pressure_mini_stokes_2d";
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
    plan: FieldwiseRealizationPlan,
    solution: SteadyStokesMiniSolution2d,
}

#[test]
fn direct_and_exact_packages_share_one_boundary_determined_pressure_path() {
    let mechanics = public_mechanics_release();
    let fluid = public_fluid_release(&mechanics);
    let loads = public_loads_release(&mechanics);
    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct mixed-boundary Stokes Model compiles");
    let packaged = compile_root(
        &fluid,
        &mechanics,
        &loads,
        ["fluid", "mechanics", "loads"],
        PACKAGED,
        false,
    );
    let aliased_source = PACKAGED
        .replace("fluid.", "liquid.")
        .replace("mechanics.", "boundary_physics.")
        .replace("loads.", "pressure_loads.");
    let permuted = compile_root(
        &fluid,
        &mechanics,
        &loads,
        ["liquid", "boundary_physics", "pressure_loads"],
        &aliased_source,
        true,
    );

    assert_mixed_boundary_meaning(direct.program());
    assert_mixed_boundary_meaning(packaged.model().program());
    assert_mixed_boundary_meaning(permuted.model().program());

    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).expect("physical mesh");
    let direct = observe(direct.program(), &mesh, profile_a(), 1);
    let packaged = observe(packaged.model().program(), &mesh, profile_a(), 2);
    let permuted = observe(permuted.model().program(), &mesh, profile_a(), 3);

    assert_eq!(direct.system, packaged.system);
    assert_eq!(packaged.system, permuted.system);
    assert_eq!(direct.plan.spatial().constraints(), &[]);
    assert_eq!(direct.plan.scaling().block_scales().len(), 2);
    assert!(
        direct
            .plan
            .scaling()
            .block_scales()
            .iter()
            .all(|entry| { !matches!(entry.block(), AlgebraicBlock::ConstraintMultiplier { .. }) })
    );
    assert_solution(&direct.solution);
    assert_solution(&packaged.solution);
    assert_solution(&permuted.solution);
    assert_solution_equivalent(&direct.solution, &packaged.solution, 2.0e-9);
    assert_solution_equivalent(&packaged.solution, &permuted.solution, 2.0e-9);
}

#[test]
fn congruent_profiles_change_scaled_rhs_but_reconstruct_the_same_physics() {
    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct mixed-boundary Stokes Model compiles");
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).expect("physical mesh");
    let first = observe(direct.program(), &mesh, profile_a(), 10);
    let second = observe(direct.program(), &mesh, profile_b(), 11);

    assert_eq!(first.system.row_offsets(), second.system.row_offsets());
    assert_eq!(
        first.system.column_indices(),
        second.system.column_indices()
    );
    assert_eq!(first.system.values(), second.system.values());
    assert_ne!(
        first.system.right_hand_side(),
        second.system.right_hand_side()
    );
    assert_ne!(first.plan, second.plan);
    assert_solution(&first.solution);
    assert_solution(&second.solution);
    assert_solution_equivalent(&first.solution, &second.solution, 2.0e-8);
}

#[test]
fn all_pressure_and_coordinate_varying_pressure_fail_before_local_assembly() {
    let all_pressure = DIRECT
        .replace(
            "relation x_lower_value on x_lower { trace(velocity) = 0; }",
            "relation x_lower_value on x_lower { normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) + normal(isotropic_lift(ambient_pressure)) = 0; }",
        )
        .replace(
            "relation y_lower_value on y_lower { trace(velocity) = 0; }",
            "relation y_lower_value on y_lower { normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) + normal(isotropic_lift(ambient_pressure)) = 0; }",
        )
        .replace(
            "relation y_upper_value on y_upper { trace(velocity) = 0; }",
            "relation y_upper_value on y_upper { normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) + normal(isotropic_lift(ambient_pressure)) = 0; }",
        );
    let all_pressure = eqiora::api::ModelDocument::compile("all-pressure.eqi", &all_pressure)
        .expect("pure pressure closure is valid Model meaning");
    let lowered = lower_steady_incompressible_stokes_cartesian_2d(all_pressure.program())
        .expect("pure pressure closure lowers before realization admission");
    let diagnostic = steady_stokes_mini_plan_2d(
        &lowered,
        MeshArtifactReference::from_sha256([0xA5; 32]),
        profile_a(),
        solver(),
    )
    .expect_err("pure traction must fail before mesh access");
    assert!(diagnostic.message().contains("pure traction"));

    let varying = DIRECT.replace(
        "ambient_pressure - ambient_pressure_value = 0;",
        "ambient_pressure - (ambient_pressure_value + force_gradient * coordinate(0)) = 0;",
    );
    let varying = eqiora::api::ModelDocument::compile("varying-pressure.eqi", &varying)
        .expect("coordinate-varying pressure is valid Model meaning");
    let lowered = lower_steady_incompressible_stokes_cartesian_2d(varying.program())
        .expect("coordinate-varying pressure tape lowers");
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).expect("physical mesh");
    let mesh_reference = mesh.artifact_reference().expect("mesh identity");
    let plan = steady_stokes_mini_plan_2d(&lowered, mesh_reference, profile_a(), solver())
        .expect("constant-varying distinction is a finalization concern");
    let resolved = resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            varying.program().model(),
            SemanticRevision::new(varying.program().revision().0),
            RealizationRevision::new(20),
            plan,
        ),
        steady_stokes_fieldwise_requirements_2d(&lowered),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("field-wise plan resolves independently of coefficient shape");
    let diagnostic = finalize_resolved_steady_stokes_mini_2d(
        varying.program(),
        &resolved,
        mesh_reference,
        mesh.mesh(),
    )
    .expect_err("coordinate-varying pressure requires an explicit facet quadrature contract");
    assert!(diagnostic.message().contains("coordinate-dependent"));
}

fn observe(
    program: &KernelProgram,
    mesh: &SimplicialMeshEnvelopeV1,
    profile: SteadyStokesScaleProfile2d,
    realization_revision: u64,
) -> Observation {
    let lowered = lower_steady_incompressible_stokes_cartesian_2d(program)
        .expect("canonical mixed Stokes meaning lowers");
    let mesh_reference = mesh.artifact_reference().expect("mesh identity");
    let plan = steady_stokes_mini_plan_2d(&lowered, mesh_reference, profile, solver())
        .expect("mixed pressure boundary admits the MINI plan");
    let resolved = resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(realization_revision),
            plan.clone(),
        ),
        steady_stokes_fieldwise_requirements_2d(&lowered),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("reference mixed capability resolves the exact plan");
    let (_, finalized) =
        finalize_resolved_steady_stokes_mini_2d(program, &resolved, mesh_reference, mesh.mesh())
            .expect("mixed-boundary Stokes finalizes");
    let system = finalized.canonical_csr_system_view().clone();
    let solved = REFERENCE_LINEAR_SOLVER
        .solve(
            &finalized.linear_problem().expect("linear problem"),
            solver(),
        )
        .expect("reference MINRES solves the mixed system");
    let solution = finalized.finish(solved).expect("SI solution reconstructs");
    Observation {
        system,
        plan,
        solution,
    }
}

fn assert_mixed_boundary_meaning(program: &KernelProgram) {
    let model = lower_steady_incompressible_stokes_cartesian_2d(program)
        .expect("canonical mixed boundary inventory");
    for (axis, side) in [
        (0, BoundarySide::Lower),
        (1, BoundarySide::Lower),
        (1, BoundarySide::Upper),
    ] {
        assert_eq!(
            model
                .boundary_inventory()
                .boundary(axis, side)
                .expect("complete Cartesian side")
                .disposition(),
            PhysicalBoundaryDisposition::TraceZero
        );
        assert!(model.normal_pressure(axis, side).is_none());
    }
    let disposition = model
        .boundary_inventory()
        .boundary(0, BoundarySide::Upper)
        .expect("right side")
        .disposition();
    assert!(matches!(
        disposition,
        PhysicalBoundaryDisposition::Prescribed(law)
            if law.quantity() == PhysicalBoundaryQuantity::Flux
    ));
    assert_close(
        model
            .normal_pressure(0, BoundarySide::Upper)
            .expect("right normal pressure")
            .expression()
            .constant_value()
            .expect("constant ambient pressure"),
        4.5,
        0.0,
    );
}

fn assert_solution(solution: &SteadyStokesMiniSolution2d) {
    assert_eq!(
        solution.pressure_reference(),
        SteadyStokesPressureReference2d::BoundaryTraction
    );
    assert_eq!(solution.gauge_multiplier(), None);
    for value in solution
        .velocity()
        .vertex_values()
        .iter()
        .chain(solution.velocity().cell_bubble_values())
        .flatten()
    {
        assert_close(*value, 0.0, 2.0e-8);
    }
    for (coordinates, pressure) in solution
        .pressure()
        .mesh()
        .vertices()
        .iter()
        .zip(solution.pressure().vertex_values())
    {
        assert_close(*pressure, 0.75 * coordinates[0] + 1.5, 2.0e-8);
    }
    assert_close(solution.pressure_integral(), 24.0, 2.0e-8);
    assert_vector_close(solution.integrated_body_force(), [6.0, 0.0], 2.0e-10);
    assert_vector_close(
        solution.integrated_boundary_traction(),
        [-9.0, 0.0],
        2.0e-10,
    );
    assert_vector_close(solution.boundary_reaction(), [3.0, 0.0], 2.0e-8);
    let total = std::array::from_fn(|axis| {
        solution.integrated_body_force()[axis]
            + solution.integrated_boundary_traction()[axis]
            + solution.boundary_reaction()[axis]
    });
    assert_vector_close(total, [0.0, 0.0], 2.0e-8);

    let dimensionless = solution.dimensionless_solution();
    assert_eq!(
        dimensionless.full_system().matrix(),
        dimensionless.volume_only_system().matrix()
    );
    assert_eq!(dimensionless.assembly_report().target_count(), 3);
    let midpoint_x_velocity = 2 * 9;
    let dimensionless_action = dimensionless.full_system().rhs()[midpoint_x_velocity]
        - dimensionless.volume_only_system().rhs()[midpoint_x_velocity];
    let force_scale = solution.scales().pressure().value() * solution.scales().length().value();
    let physical_action = dimensionless_action * force_scale;
    assert_close(physical_action, -4.5, 2.0e-12);
    for row in 0..dimensionless.full_system().rhs().len() {
        let expected = match row {
            8 | 28 => -2.25,
            18 => -4.5,
            _ => 0.0,
        };
        let action = (dimensionless.full_system().rhs()[row]
            - dimensionless.volume_only_system().rhs()[row])
            * force_scale;
        assert_close(action, expected, 2.0e-12);
    }
}

fn assert_solution_equivalent(
    left: &SteadyStokesMiniSolution2d,
    right: &SteadyStokesMiniSolution2d,
    tolerance: f64,
) {
    for (left, right) in left
        .velocity()
        .vertex_values()
        .iter()
        .chain(left.velocity().cell_bubble_values())
        .flatten()
        .zip(
            right
                .velocity()
                .vertex_values()
                .iter()
                .chain(right.velocity().cell_bubble_values())
                .flatten(),
        )
    {
        assert_close(*left, *right, tolerance);
    }
    for (left, right) in left
        .pressure()
        .vertex_values()
        .iter()
        .zip(right.pressure().vertex_values())
    {
        assert_close(*left, *right, tolerance);
    }
    assert_close(
        left.pressure_integral(),
        right.pressure_integral(),
        tolerance,
    );
    assert_vector_close(
        left.integrated_body_force(),
        right.integrated_body_force(),
        tolerance,
    );
    assert_vector_close(
        left.integrated_boundary_traction(),
        right.integrated_boundary_traction(),
        tolerance,
    );
    assert_vector_close(
        left.boundary_reaction(),
        right.boundary_reaction(),
        tolerance,
    );
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

fn public_loads_release(mechanics: &PackageReleaseV1) -> PackageReleaseV1 {
    public_release(
        "Eqiora.Mechanics.BoundaryLoads",
        std::slice::from_ref(mechanics),
    )
}

fn public_release(package: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources(package);
    prepare_package_release_v1(sources, dependencies)
        .unwrap_or_else(|error| panic!("prepare public package {package}: {error:?}"))
}

fn compile_root(
    fluid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    loads: &PackageReleaseV1,
    aliases: [&str; 3],
    source: &str,
    reverse_closure: bool,
) -> PackagedModelDocument {
    let dependencies = [
        (aliases[0], fluid.package_identity().unwrap()),
        (aliases[1], mechanics.package_identity().unwrap()),
        (aliases[2], loads.package_identity().unwrap()),
    ]
    .into_iter()
    .map(|(_, identity)| PackageDependencyV1::new(identity))
    .collect();
    let closure = if reverse_closure {
        vec![loads.clone(), mechanics.clone(), fluid.clone()]
    } else {
        vec![fluid.clone(), mechanics.clone(), loads.clone()]
    };
    let source = format!(
        "import Eqiora.Fluid.Incompressible.incompressible as {};\n\
         import Eqiora.Mechanics.Interfaces.interfaces as {};\n\
         import Eqiora.Mechanics.BoundaryLoads.boundary_loads as {};\n{source}",
        aliases[0], aliases[1], aliases[2]
    );
    let root = prepare_package_release_v1(
        inline_sources(ROOT_PACKAGE, VERSION, dependencies, "src/main.eqi", &source),
        &closure,
    )
    .expect("prepare exact verification root");
    let resolution = ResolutionRecordV1::from_exact_releases(&root, &closure)
        .expect("exact four-release resolution");
    let mut store = InMemoryPackageStore::default();
    for release in [&root, fluid, mechanics, loads] {
        store.insert(release).expect("install exact package");
    }
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("exact four-release chain compiles offline")
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

fn profile_a() -> SteadyStokesScaleProfile2d {
    profile(0.5, 0.75)
}

fn profile_b() -> SteadyStokesScaleProfile2d {
    profile(1.0, 1.5)
}

fn profile(velocity: f64, pressure: f64) -> SteadyStokesScaleProfile2d {
    SteadyStokesScaleProfile2d::new(
        DynQuantity::new(4.0, LENGTH),
        DynQuantity::new(velocity, VELOCITY),
        DynQuantity::new(pressure, PRESSURE),
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

fn assert_vector_close(actual: [f64; 2], expected: [f64; 2], tolerance: f64) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.17e}, got {actual:.17e}, tolerance {tolerance:.3e}"
    );
}
