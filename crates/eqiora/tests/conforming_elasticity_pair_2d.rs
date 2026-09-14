use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::Mutex;

use eqiora::assembly::{
    AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem,
    REFERENCE_ASSEMBLY_BACKEND,
};
use eqiora::diagnostic::codes;
use eqiora::kernel::BoundarySide;
use eqiora::meshing::{MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, Target, VectorLayoutKind,
    resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, SolverPlan};
use eqiora_numerics::{
    common::DiscreteSpace, common::HypercubeQ1Space, common::PhysicalBoundaryDisposition,
    solid::CartesianQ1VectorField2d,
    solid::finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly,
    solid::lower_conforming_isotropic_elasticity_cartesian_pair_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT_SOURCE: &str =
    include_str!("../../../verify/solid/conforming-elasticity-pair-2d/models/direct.eqi");
const PACKAGED_SOURCE: &str =
    include_str!("../../../verify/solid/conforming-elasticity-pair-2d/models/packaged.eqi");

fn mechanics_package() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources("Eqiora.Mechanics.Interfaces"),
        &[],
    )
    .expect("prepare current mechanics dependency")
}

fn elasticity_package() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources("Eqiora.Solid.LinearElasticity"),
        &[mechanics_package()],
    )
    .expect("prepare current elasticity dependency")
}

fn compile_packaged(dependency: &PackageReleaseV1) -> PackagedModelDocument {
    compile_packaged_as(dependency, PACKAGED_SOURCE, "solid")
}

fn compile_packaged_as(
    dependency: &PackageReleaseV1,
    source: &str,
    alias: &str,
) -> PackagedModelDocument {
    let model_path = NormalizedRelativePath::parse("src/main.eqi").expect("root model path");
    let requirement = PackageDependencyV1::new(
        dependency
            .package_identity()
            .expect("elasticity package identity"),
    );
    let package_name = dependency
        .package_identity()
        .expect("elasticity package identity")
        .name;
    let source = format!("import {package_name}.linear_elasticity as {alias};\n{source}");
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.eqiora.verify.conforming_elasticity_pair_2d")
            .expect("root package name"),
        ExactVersion::parse("0.1.0").expect("root version"),
        vec![requirement],
        vec![BundleEntryV1::new(
            model_path.clone(),
            BundleRoleV1::ModelSource,
        )],
    )
    .expect("root package manifest");
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            model_path,
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        )],
    )
    .expect("closed root sources");
    let dependencies = [dependency.clone(), mechanics_package()];
    let root =
        prepare_package_release_v1(sources, &dependencies).expect("prepare exact coupled root");
    let resolution = ResolutionRecordV1::from_exact_releases(&root, &dependencies)
        .expect("exact package resolution");
    let mut store = InMemoryPackageStore::default();
    for release in &dependencies {
        store.insert(release).expect("insert dependency");
    }
    store.insert(&root).expect("insert root");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact packaged pair")
}

fn resolved(program: &KernelProgram, cells: usize, revision: u64) -> ResolvedRealization {
    resolved_with_points(program, cells, revision, 2)
}

fn resolved_with_points(
    program: &KernelProgram,
    cells: usize,
    revision: u64,
    points_per_axis: usize,
) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).expect("positive refinement"),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(points_per_axis)
                    .expect("positive quadrature point count"),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(10_000).expect("finite iteration limit"),
        )
        .expect("coercive-system solver plan"),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .expect("Q1 realization plan");
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(revision),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).expect("two dimensions"),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::isotropic_elasticity_2d_reference(),
    )
    .expect("exact elasticity capability admits Q1")
}

fn permuted_direct_source() -> String {
    let left_domains = "  domain left = box(0, 0.5, 0, 1);\n  domain left_x_lower = boundary(left, axis = 0, side = lower);\n  domain left_x_upper = boundary(left, axis = 0, side = upper);\n  domain left_y_lower = boundary(left, axis = 1, side = lower);\n  domain left_y_upper = boundary(left, axis = 1, side = upper);\n";
    let right_domains = "  domain right = box(0.5, 1, 0, 1);\n  domain right_x_lower = boundary(right, axis = 0, side = lower);\n  domain right_x_upper = boundary(right, axis = 0, side = upper);\n  domain right_y_lower = boundary(right, axis = 1, side = lower);\n  domain right_y_upper = boundary(right, axis = 1, side = upper);\n";
    let source = DIRECT_SOURCE.replace(
        &format!("{left_domains}{right_domains}"),
        &format!("{right_domains}{left_domains}"),
    );
    assert_ne!(source, DIRECT_SOURCE);
    source
        .replace("left_boundary", "negative_body_law")
        .replace("right_boundary", "positive_body_law")
        .replace(
            "connect negative_body_law.mechanical[boundary = left_x_upper],\n    positive_body_law.mechanical[boundary = right_x_lower];",
            "connect positive_body_law.mechanical[boundary = right_x_lower],\n    negative_body_law.mechanical[boundary = left_x_upper];",
        )
}

#[derive(Debug, Default)]
struct CapturingAssemblyBackend {
    systems: Mutex<Vec<LinearSystem>>,
}

impl CapturingAssemblyBackend {
    fn systems(&self) -> Vec<LinearSystem> {
        self.systems
            .lock()
            .expect("capture mutex remains available")
            .clone()
    }
}

impl AssemblyBackend for CapturingAssemblyBackend {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, eqiora::Diagnostic> {
        let result = REFERENCE_ASSEMBLY_BACKEND.assemble(plan, work)?;
        *self
            .systems
            .lock()
            .expect("capture mutex remains available") = result.systems().to_vec();
        Ok(result)
    }
}

fn exact(subdomain: usize, point: &[f64]) -> ([f64; 2], [[f64; 2]; 2]) {
    let x = point[0];
    if subdomain == 0 {
        ([x - 0.5 * x * x, 0.0], [[1.0 - x, 0.0], [0.0, 0.0]])
    } else {
        (
            [3.0 / 16.0 + 0.5 * x - 0.25 * x * x, 0.0],
            [[0.5 - 0.5 * x, 0.0], [0.0, 0.0]],
        )
    }
}

fn recovered_interface_traction(
    field: &CartesianQ1VectorField2d,
    shear_modulus: f64,
    side: BoundarySide,
) -> [f64; 2] {
    let mesh = field.mesh();
    let space = HypercubeQ1Space::new(2).expect("Q1 space");
    let rule = QuadratureRule::gauss_legendre(2).expect("facet rule");
    let normal_cell = if side == BoundarySide::Lower {
        0
    } else {
        mesh.axis_cell_count(0).expect("x cells") - 1
    };
    let normal = if side == BoundarySide::Lower {
        -1.0
    } else {
        1.0
    };
    let mut resultant = [0.0; 2];
    for tangent_cell in 0..mesh.axis_cell_count(1).expect("y cells") {
        let cell = mesh
            .cell_at(&[normal_cell, tangent_cell])
            .expect("boundary-adjacent cell");
        let geometry = mesh.geometry_map(cell).expect("affine geometry");
        let inverse = geometry.inverse_jacobian().expect("invertible geometry");
        let surface_scale =
            (geometry.jacobian()[1].powi(2) + geometry.jacobian()[3].powi(2)).sqrt();
        let vertices = mesh.entity_vertices(cell).expect("cell vertices");
        for point in rule.points() {
            let basis = space
                .tabulate(&[normal, point.coordinates[0]])
                .expect("facet Q1 tabulation");
            let mut gradient = [[0.0; 2]; 2];
            for (local_vertex, vertex) in vertices.iter().enumerate() {
                let nodal = field
                    .vertex_values(vertex.index())
                    .expect("field owns vertex");
                let reference_gradient = basis.gradient(local_vertex).expect("basis owns gradient");
                for component in 0..2 {
                    for physical in 0..2 {
                        gradient[component][physical] += nodal[component]
                            * (0..2)
                                .map(|reference_axis| {
                                    inverse[reference_axis * 2 + physical]
                                        * reference_gradient[reference_axis]
                                })
                                .sum::<f64>();
                    }
                }
            }
            let stress_xx = 2.0 * shear_modulus * gradient[0][0];
            let stress_yx = shear_modulus * (gradient[1][0] + gradient[0][1]);
            resultant[0] += point.weight * surface_scale * normal * stress_xx;
            resultant[1] += point.weight * surface_scale * normal * stress_yx;
        }
    }
    resultant
}

#[test]
fn direct_two_body_model_lowers_to_one_exact_conforming_interface() {
    let document = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT_SOURCE)
        .expect("direct conforming pair compiles");
    let pair = lower_conforming_isotropic_elasticity_cartesian_pair_2d(document.program())
        .expect("direct pair lowers");
    let dependency = elasticity_package();
    let packaged = compile_packaged(&dependency);
    let packaged_pair =
        lower_conforming_isotropic_elasticity_cartesian_pair_2d(packaged.model().program())
            .expect("packaged pair lowers");

    assert_eq!(pair.interface().axis(), 0);
    assert_eq!(pair.interface().negative().side(), BoundarySide::Upper);
    assert_eq!(pair.interface().positive().side(), BoundarySide::Lower);
    assert_eq!(pair.subdomains()[0].bounds(), &[[0.0, 0.5], [0.0, 1.0]]);
    assert_eq!(pair.subdomains()[1].bounds(), &[[0.5, 1.0], [0.0, 1.0]]);
    assert_eq!(pair.subdomains()[0].shear_modulus(), 3.0);
    assert_eq!(pair.subdomains()[1].shear_modulus(), 6.0);
    for (direct, packaged) in pair.subdomains().iter().zip(packaged_pair.subdomains()) {
        assert_eq!(direct.bounds(), packaged.bounds());
        assert_eq!(direct.shear_modulus(), packaged.shear_modulus());
        assert_eq!(
            direct.first_lame_parameter(),
            packaged.first_lame_parameter()
        );
    }
    assert_eq!(pair.interface().axis(), packaged_pair.interface().axis());

    for (index, model) in pair.subdomains().iter().enumerate() {
        for axis in 0..2 {
            for side in [BoundarySide::Lower, BoundarySide::Upper] {
                let disposition = model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete side inventory")
                    .disposition();
                let is_interface = axis == 0
                    && ((index == 0 && side == BoundarySide::Upper)
                        || (index == 1 && side == BoundarySide::Lower));
                assert_eq!(
                    matches!(disposition, PhysicalBoundaryDisposition::PortBinding { .. }),
                    is_interface
                );
            }
        }
    }
}

#[test]
fn direct_and_packaged_pairs_share_one_monolithic_q1_system_and_interface_oracle() {
    let direct_source = permuted_direct_source();
    let direct = eqiora::api::ModelDocument::compile("permuted-direct.eqi", &direct_source)
        .expect("permuted direct conforming pair compiles");
    let dependency = elasticity_package();
    let packaged_source = PACKAGED_SOURCE.replace("solid.", "mechanics.");
    let packaged = compile_packaged_as(&dependency, &packaged_source, "mechanics");
    let error_rule =
        QuadratureRule::tensor_product_gauss_legendre(2, 4).expect("independent norm rule");

    for (revision, cells) in [2, 4, 8].into_iter().enumerate() {
        let direct_assembly = CapturingAssemblyBackend::default();
        let packaged_assembly = CapturingAssemblyBackend::default();
        let (_, direct_problem) =
            finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
                direct.program(),
                &resolved(direct.program(), cells, revision as u64 + 1),
                &direct_assembly,
            )
            .expect("direct pair finalizes");
        let (_, packaged_problem) =
            finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
                packaged.model().program(),
                &resolved(packaged.model().program(), cells, revision as u64 + 1),
                &packaged_assembly,
            )
            .expect("packaged pair finalizes");

        assert_eq!(direct_assembly.systems().len(), 4);
        assert_eq!(direct_assembly.systems(), packaged_assembly.systems());
        assert_eq!(
            direct_problem.canonical_csr_system_view(),
            packaged_problem.canonical_csr_system_view()
        );
        assert_eq!(
            direct_problem.canonical_csr_system_view().rows(),
            4 * cells * (cells + 1)
        );
        assert_eq!(
            direct_problem.assembly_report(),
            packaged_problem.assembly_report()
        );

        let direct_solved = REFERENCE_LINEAR_SOLVER
            .solve(
                &direct_problem
                    .linear_problem()
                    .expect("direct linear problem"),
                direct_problem.solver_plan(),
            )
            .unwrap_or_else(|diagnostic| panic!("direct reference solve at {cells}: {diagnostic}"));
        let packaged_solved = REFERENCE_LINEAR_SOLVER
            .solve(
                &packaged_problem
                    .linear_problem()
                    .expect("packaged linear problem"),
                packaged_problem.solver_plan(),
            )
            .unwrap_or_else(|diagnostic| {
                panic!("packaged reference solve at {cells}: {diagnostic}")
            });
        let direct_solution = direct_problem
            .finish(direct_solved)
            .expect("direct field pair");
        let packaged_solution = packaged_problem
            .finish(packaged_solved)
            .expect("packaged field pair");
        assert_eq!(
            direct_solution.algebraic_values(),
            packaged_solution.algebraic_values()
        );
        assert_eq!(
            direct_solution.displacement(),
            packaged_solution.displacement()
        );
        assert_eq!(
            direct_solution.interface_map(),
            packaged_solution.interface_map()
        );
        assert_eq!(direct_solution.interface_map().axis(), 0);
        assert_eq!(
            direct_solution.interface_map().interface_vertices().len(),
            cells + 1
        );
        assert_eq!(
            direct_solution.interface_map().global_vertex_count(),
            (cells + 1) * (2 * cells + 1)
        );

        for [negative, positive] in direct_solution.interface_map().interface_vertices() {
            assert_eq!(
                direct_solution.displacement()[0].vertex_values(*negative),
                direct_solution.displacement()[1].vertex_values(*positive),
                "quotient identity must reconstruct bit-identical traces"
            );
        }
        let h = 0.5 / cells as f64;
        let mut l2_squared = 0.0;
        let mut h1_seminorm_squared = 0.0;
        for (subdomain, field) in direct_solution.displacement().iter().enumerate() {
            let norms = field
                .error_norms(&|point| exact(subdomain, point), &error_rule)
                .expect("continuous piecewise error evidence");
            l2_squared += norms.l2().powi(2);
            h1_seminorm_squared += norms.h1_seminorm().powi(2);
            for vertex in 0..field.mesh().entity_count(0).expect("mesh vertices") {
                let coordinates = field
                    .mesh()
                    .vertex_coordinates(MeshEntity::new(0, vertex))
                    .expect("vertex coordinates");
                let (expected, _) = exact(subdomain, &coordinates);
                for (actual, expected) in field
                    .vertex_values(vertex)
                    .expect("field vertex")
                    .iter()
                    .zip(expected)
                {
                    assert!((actual - expected).abs() < 2.0e-11);
                }
            }
        }
        assert!((l2_squared - h.powi(4) / 192.0).abs() < 2.0e-12);
        assert!((h1_seminorm_squared - 5.0 * h.powi(2) / 96.0).abs() < 2.0e-12);

        let action = direct_solution.interface_action();
        assert!(action.free_mask().iter().all(|free| *free));
        let negative_resultant = action.negative_free_resultant();
        let positive_resultant = action.positive_free_resultant();
        assert!((negative_resultant[0] - 3.0).abs() < 2.0e-11);
        assert!(negative_resultant[1].abs() < 2.0e-11);
        assert!((positive_resultant[0] + 3.0).abs() < 2.0e-11);
        assert!(positive_resultant[1].abs() < 2.0e-11);
        assert!(action.free_equilibrium_residual().iter().all(|value| {
            value.is_some_and(|value| value[0].abs() < 2.0e-11 && value[1].abs() < 2.0e-11)
        }));

        assert_eq!(direct_solution.integrated_body_force().len(), 2);
        for body_force in direct_solution.integrated_body_force() {
            assert!((body_force[0] - 3.0).abs() < 2.0e-13);
            assert!(body_force[1].abs() < 2.0e-13);
        }
        assert!((direct_solution.boundary_reaction()[0] + 6.0).abs() < 2.0e-11);
        assert!(direct_solution.boundary_reaction()[1].abs() < 2.0e-11);

        let negative_traction = recovered_interface_traction(
            &direct_solution.displacement()[0],
            3.0,
            BoundarySide::Upper,
        );
        let positive_traction = recovered_interface_traction(
            &direct_solution.displacement()[1],
            6.0,
            BoundarySide::Lower,
        );
        assert!((negative_traction[0] - (3.0 + 3.0 * h)).abs() < 2.0e-11);
        assert!((positive_traction[0] - (-3.0 + 3.0 * h)).abs() < 2.0e-11);
        assert!((negative_traction[0] + positive_traction[0] - 6.0 * h).abs() < 2.0e-11);
        assert!(negative_traction[1].abs() < 2.0e-11);
        assert!(positive_traction[1].abs() < 2.0e-11);
        assert!(
            direct_solution.solve_report().true_residual_norm()
                <= direct_solution.solve_report().residual_target()
        );
    }
}

#[test]
fn pair_rejects_same_side_and_non_binary_interface_connections() {
    let same_side = DIRECT_SOURCE
        .replace("domain right = box(0.5, 1, 0, 1);", "domain right = box(0, 0.5, 0, 1);")
        .replace("face = right_x_upper", "face = right_x_lower")
        .replace(
            "right_boundary.mechanical[boundary = right_x_upper],\n    right_x_upper_free.mechanical;",
            "right_boundary.mechanical[boundary = right_x_lower],\n    right_x_upper_free.mechanical;",
        )
        .replace(
            "right_boundary.mechanical[boundary = right_x_lower];",
            "right_boundary.mechanical[boundary = right_x_upper];",
        );
    let same_side = eqiora::api::ModelDocument::compile("same-side.eqi", &same_side)
        .expect("coincident same-side model remains semantically well typed");
    let diagnostic =
        lower_conforming_isotropic_elasticity_cartesian_pair_2d(same_side.program()).unwrap_err();
    assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
    assert!(diagnostic.message().contains("opposite side"));

    let three_port = DIRECT_SOURCE
        .replace(
            "  connect left_boundary.mechanical[boundary = left_x_lower],",
            "  instance interface_terminal: ZeroTraction2d(\n    body = right,\n    face = right_x_lower\n  );\n\n  connect left_boundary.mechanical[boundary = left_x_lower],",
        )
        .replace(
            "connect left_boundary.mechanical[boundary = left_x_upper],\n    right_boundary.mechanical[boundary = right_x_lower];",
            "connect left_boundary.mechanical[boundary = left_x_upper],\n    right_boundary.mechanical[boundary = right_x_lower],\n    interface_terminal.mechanical;",
        );
    let three_port = eqiora::api::ModelDocument::compile("three-port.eqi", &three_port)
        .expect("three-Port conserving junction remains kernel-valid");
    let diagnostic =
        lower_conforming_isotropic_elasticity_cartesian_pair_2d(three_port.program()).unwrap_err();
    assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
    assert!(diagnostic.message().contains("does not interpret"));
}

#[test]
fn pure_natural_pair_fails_the_coupled_realization_gate() {
    let source = DIRECT_SOURCE.replace(
        "instance fixed: FixedDisplacement2d(",
        "instance fixed: ZeroTraction2d(",
    );
    let document = eqiora::api::ModelDocument::compile("unanchored.eqi", &source)
        .expect("pure-natural pair remains valid model meaning");
    let diagnostic =
        finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
            document.program(),
            &resolved(document.program(), 2, 1),
            &REFERENCE_ASSEMBLY_BACKEND,
        )
        .unwrap_err();
    assert_eq!(diagnostic.code(), codes::INVALID_REALIZATION);
    assert!(diagnostic.message().contains("global rigid modes"));
}

#[test]
fn reduced_integration_fails_before_an_spd_system_is_declared() {
    let document = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT_SOURCE)
        .expect("direct conforming pair compiles");
    let diagnostic =
        finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
            document.program(),
            &resolved_with_points(document.program(), 2, 1, 1),
            &REFERENCE_ASSEMBLY_BACKEND,
        )
        .unwrap_err();

    assert_eq!(diagnostic.code(), codes::INVALID_REALIZATION);
    assert!(diagnostic.message().contains("exactly two Gauss-Legendre"));
}

#[test]
fn additional_live_port_relation_fails_before_pair_realization() {
    let source = DIRECT_SOURCE
        .replace(
            "  connect left_boundary.mechanical[boundary = left_x_lower],",
            "  instance interface_terminal: ZeroTraction2d(\n    body = right,\n    face = right_x_lower\n  );\n\n  connect left_boundary.mechanical[boundary = left_x_lower],",
        )
        .replace(
            "connect left_boundary.mechanical[boundary = left_x_upper],\n    right_boundary.mechanical[boundary = right_x_lower];",
            "connect left_boundary.mechanical[boundary = left_x_upper],\n    right_boundary.mechanical[boundary = right_x_lower],\n    interface_terminal.mechanical;",
        );
    assert_ne!(source, DIRECT_SOURCE);
    let document = eqiora::api::ModelDocument::compile("additional-live-relation.eqi", &source)
        .expect("a third typed terminal Relation remains valid semantic meaning");
    let diagnostic =
        lower_conforming_isotropic_elasticity_cartesian_pair_2d(document.program()).unwrap_err();

    assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
    assert!(diagnostic.message().contains("does not interpret"));
}

#[test]
fn constrained_interface_endpoint_is_not_mislabeled_as_coupling_equilibrium() {
    let source = DIRECT_SOURCE
        .replace("face = left_x_lower", "face = swap_face")
        .replace("face = left_y_lower", "face = left_x_lower")
        .replace("face = swap_face", "face = left_y_lower")
        .replace(
            "connect left_boundary.mechanical[boundary = left_x_lower],\n    fixed.mechanical;",
            "connect left_boundary.mechanical[boundary = left_x_lower],\n    left_y_lower_free.mechanical;",
        )
        .replace(
            "connect left_boundary.mechanical[boundary = left_y_lower],\n    left_y_lower_free.mechanical;",
            "connect left_boundary.mechanical[boundary = left_y_lower],\n    fixed.mechanical;",
        );
    let document = eqiora::api::ModelDocument::compile("interface-endpoint-support.eqi", &source)
        .expect("supported interface endpoint remains valid model meaning");
    let (_, problem) =
        finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
            document.program(),
            &resolved(document.program(), 2, 1),
            &REFERENCE_ASSEMBLY_BACKEND,
        )
        .expect("supported pair finalizes");
    let solved = REFERENCE_LINEAR_SOLVER
        .solve(
            &problem.linear_problem().expect("supported linear problem"),
            problem.solver_plan(),
        )
        .expect("supported pair solves");
    let solution = problem.finish(solved).expect("supported pair reconstructs");

    assert_eq!(
        solution.interface_action().free_mask(),
        &[false, true, true]
    );
    let residual = solution.interface_action().free_equilibrium_residual();
    assert!(residual[0].is_none());
    assert!(residual[1..].iter().all(|value| value.is_some()));
}
