use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::Mutex;

use eqiora::assembly::{
    AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem,
    REFERENCE_ASSEMBLY_BACKEND,
};
use eqiora::diagnostic::codes;
use eqiora::kernel::{BoundarySide, DomainKind, KernelNode};
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
    common::DiscreteSpace,
    common::HypercubeQ1Space,
    common::PhysicalBoundaryDisposition,
    common::ScalarSpatialExpression,
    solid::CartesianLinearElasticity2dSolution,
    solid::finalize_resolved_isotropic_elasticity_cartesian_2d,
    solid::finalize_resolved_isotropic_elasticity_cartesian_2d_with_assembly,
    solid::lower_isotropic_elasticity_cartesian_2d,
    solid::{
        ElasticityIntegrationMeasure, IsotropicElasticityContinuum, IsotropicElasticityReduction,
    },
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT_SOURCE: &str =
    include_str!("../../../verify/solid/mixed-boundary-elasticity-2d/models/direct.eqi");
const PACKAGED_SOURCE: &str =
    include_str!("../../../verify/solid/mixed-boundary-elasticity-2d/models/packaged.eqi");
const LIVE_PACKAGE_SOURCE: &str = include_str!(
    "../../../crates/eqiora-api/packages/Eqiora.Solid.LinearElasticity/src/linear_elasticity.eqi"
);
const ROOT_NAME: &str = "org.eqiora.verify.mixed_boundary_elasticity_2d";
const ROOT_VERSION: &str = "0.1.0";

fn mechanics_package() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources("Eqiora.Mechanics.Interfaces"),
        &[],
    )
    .expect("prepare current mechanics package")
}

fn elasticity_package() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources("Eqiora.Solid.LinearElasticity"),
        &[mechanics_package()],
    )
    .expect("prepare current elasticity package")
}

fn elasticity_package_with_source(source: &str) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources("Eqiora.Solid.LinearElasticity");
    let (manifest, files) = sources.into_parts();
    let files = files
        .into_iter()
        .map(|file| {
            if file.path().as_str() == "src/linear_elasticity.eqi" {
                SourceFileV1::new(file.path().clone(), file.role(), source.as_bytes().to_vec())
            } else {
                file
            }
        })
        .collect();
    let sources = PackageSourcesV1::new(manifest, files).expect("modified exact package sources");
    prepare_package_release_v1(sources, &[mechanics_package()])
        .expect("modified package remains valid meaning")
}

fn root_release(dependency: &PackageReleaseV1, source: &str) -> PackageReleaseV1 {
    let model_path = NormalizedRelativePath::parse("src/main.eqi").expect("root model path");
    let requirement = PackageDependencyV1::new(
        dependency
            .package_identity()
            .expect("elasticity package identity"),
    );
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(ROOT_NAME).expect("root package name"),
        ExactVersion::parse(ROOT_VERSION).expect("root package version"),
        vec![requirement],
        vec![BundleEntryV1::new(
            model_path.clone(),
            BundleRoleV1::ModelSource,
        )],
    )
    .expect("root package manifest");
    let dependency_name = dependency
        .package_identity()
        .expect("elasticity package identity")
        .name;
    let source = format!("import {dependency_name}.linear_elasticity as solid;\n{source}");
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            model_path,
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        )],
    )
    .expect("closed root sources");
    prepare_package_release_v1(sources, &[dependency.clone(), mechanics_package()])
        .expect("prepare exact mixed-boundary root")
}

fn compile_packaged(dependency: &PackageReleaseV1, source: &str) -> PackagedModelDocument {
    let root = root_release(dependency, source);
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, &[dependency.clone(), mechanics_package()])
            .expect("exact package resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(dependency).expect("insert dependency release");
    store
        .insert(&mechanics_package())
        .expect("insert mechanics release");
    store.insert(&root).expect("insert root release");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact packaged mixed-boundary model")
}

fn resolved(program: &KernelProgram, cells: usize, revision: u64) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).expect("positive refinement"),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).expect("two-point assembly rule"),
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

fn assert_inventory(program: &KernelProgram, model: &IsotropicElasticityContinuum<2>) {
    assert_eq!(
        model.reduction(),
        IsotropicElasticityReduction::IntrinsicTwoDimensional
    );
    assert_eq!(
        model.integration_measure(),
        ElasticityIntegrationMeasure::PerUnitOutOfPlaneThickness
    );
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let entry = model
                .boundary_inventory()
                .boundary(axis, side)
                .expect("complete side inventory");
            let expected = if axis == 0 && side == BoundarySide::Lower {
                PhysicalBoundaryDisposition::TraceZero
            } else {
                PhysicalBoundaryDisposition::FluxZero
            };
            assert_eq!(entry.disposition(), expected);
            let KernelNode::Domain(boundary) = program
                .node(entry.boundary())
                .expect("inventory retains exact Boundary identity")
            else {
                panic!("inventory identity is not a Domain");
            };
            assert_eq!(
                boundary.kind(),
                &DomainKind::CartesianBoundary { axis, side }
            );
        }
    }
}

fn solve_finalized(
    problem: eqiora_numerics::solid::FinalizedIsotropicElasticityCartesian2dProblem,
) -> CartesianLinearElasticity2dSolution {
    let solution = REFERENCE_LINEAR_SOLVER
        .solve(
            &problem.linear_problem().expect("finalized linear problem"),
            problem.solver_plan(),
        )
        .expect("reference CG solve");
    problem.finish(solution).expect("accepted Q1 field")
}

fn exact(point: &[f64]) -> ([f64; 2], [[f64; 2]; 2]) {
    let x = point[0];
    ([x - 0.5 * x * x, 0.0], [[1.0 - x, 0.0], [0.0, 0.0]])
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

fn recovered_traction_resultants(
    solution: &CartesianLinearElasticity2dSolution,
    shear_modulus: f64,
    first_lame_parameter: f64,
) -> [[[f64; 2]; 2]; 2] {
    let mesh = solution.displacement().mesh();
    let space = HypercubeQ1Space::new(2).expect("Q1 space");
    let rule = QuadratureRule::gauss_legendre(2).expect("facet rule");
    let mut resultants = [[[0.0; 2]; 2]; 2];

    for axis in 0..2 {
        let tangent = 1 - axis;
        let normal_cell_count = mesh.axis_cell_count(axis).expect("normal cell count");
        let tangent_cell_count = mesh
            .axis_cell_count(tangent)
            .expect("tangential cell count");
        for (side_index, side) in [BoundarySide::Lower, BoundarySide::Upper]
            .into_iter()
            .enumerate()
        {
            let normal = if side == BoundarySide::Lower {
                -1.0
            } else {
                1.0
            };
            for tangent_cell in 0..tangent_cell_count {
                let mut cell_index = [0; 2];
                cell_index[axis] = if side == BoundarySide::Lower {
                    0
                } else {
                    normal_cell_count - 1
                };
                cell_index[tangent] = tangent_cell;
                let cell = mesh.cell_at(&cell_index).expect("boundary-adjacent cell");
                let geometry = mesh.geometry_map(cell).expect("affine cell geometry");
                let inverse = geometry.inverse_jacobian().expect("invertible cell map");
                let surface_scale = (0..2)
                    .map(|physical| geometry.jacobian()[physical * 2 + tangent].powi(2))
                    .sum::<f64>()
                    .sqrt();
                let vertices = mesh.entity_vertices(cell).expect("cell vertex closure");

                for point in rule.points() {
                    let mut reference = [0.0; 2];
                    reference[axis] = normal;
                    reference[tangent] = point.coordinates[0];
                    let basis = space.tabulate(&reference).expect("facet Q1 tabulation");
                    let mut gradient = [[0.0; 2]; 2];
                    for (local_vertex, vertex) in vertices.iter().enumerate() {
                        let nodal = solution
                            .displacement()
                            .vertex_values(vertex.index())
                            .expect("solution owns cell vertex");
                        let reference_gradient = basis
                            .gradient(local_vertex)
                            .expect("Q1 basis owns local gradient");
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
                    let divergence = gradient[0][0] + gradient[1][1];
                    for component in 0..2 {
                        let strain = 0.5 * (gradient[component][axis] + gradient[axis][component]);
                        let stress = 2.0 * shear_modulus * strain
                            + if component == axis {
                                first_lame_parameter * divergence
                            } else {
                                0.0
                            };
                        resultants[axis][side_index][component] +=
                            point.weight * surface_scale * normal * stress;
                    }
                }
            }
        }
    }
    resultants
}

#[test]
fn direct_and_packaged_boundaries_finalize_to_one_q1_problem_and_convergence_oracle() {
    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT_SOURCE)
        .expect("direct mixed-boundary model compiles");
    let dependency = elasticity_package();
    let packaged = compile_packaged(&dependency, PACKAGED_SOURCE);
    let direct_model =
        lower_isotropic_elasticity_cartesian_2d(direct.program()).expect("direct model lowers");
    let packaged_model = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect("packaged model lowers");
    assert_eq!(
        packaged.model().aliases()["mu"],
        packaged.model().aliases()["balance_law.mu"]
    );
    assert_eq!(
        packaged.model().aliases()["mu"],
        packaged.model().aliases()["boundary_law.mu"]
    );
    assert_eq!(
        packaged.model().aliases()["lambda"],
        packaged.model().aliases()["balance_law.lambda"]
    );
    assert_eq!(
        packaged.model().aliases()["lambda"],
        packaged.model().aliases()["boundary_law.lambda"]
    );
    assert_eq!(
        coefficient_derivatives(direct_model.shear_modulus_expression()),
        coefficient_derivatives(packaged_model.shear_modulus_expression())
    );
    assert_eq!(
        coefficient_derivatives(direct_model.first_lame_parameter_expression()),
        coefficient_derivatives(packaged_model.first_lame_parameter_expression())
    );
    assert_inventory(direct.program(), &direct_model);
    assert_inventory(packaged.model().program(), &packaged_model);

    let error_rule =
        QuadratureRule::tensor_product_gauss_legendre(2, 4).expect("independent norm rule");
    let mut l2_errors = Vec::new();
    let mut h1_errors = Vec::new();
    let mut free_boundary_traction_errors = Vec::new();
    for (revision, cells) in [4, 8, 16, 32].into_iter().enumerate() {
        let direct_assembly = CapturingAssemblyBackend::default();
        let packaged_assembly = CapturingAssemblyBackend::default();
        let (_, direct_problem) =
            finalize_resolved_isotropic_elasticity_cartesian_2d_with_assembly(
                direct.program(),
                &resolved(direct.program(), cells, revision as u64 + 1),
                &direct_assembly,
            )
            .expect("direct model finalizes");
        let (_, packaged_problem) =
            finalize_resolved_isotropic_elasticity_cartesian_2d_with_assembly(
                packaged.model().program(),
                &resolved(packaged.model().program(), cells, revision as u64 + 1),
                &packaged_assembly,
            )
            .expect("packaged model finalizes");
        assert_eq!(
            direct_assembly.systems(),
            packaged_assembly.systems(),
            "reduced and full CSR/RHS targets must both ignore authoring form"
        );
        assert_eq!(
            direct_problem.canonical_csr_system_view(),
            packaged_problem.canonical_csr_system_view(),
            "source spelling and package hierarchy cannot enter finalized algebra"
        );
        assert_eq!(
            direct_problem.assembly_report(),
            packaged_problem.assembly_report()
        );
        assert_eq!(
            direct_problem.canonical_csr_system_view().rows(),
            2 * cells * (cells + 1)
        );

        let direct_solution = solve_finalized(direct_problem);
        let packaged_solution = solve_finalized(packaged_problem);
        assert_eq!(
            direct_solution.displacement().values(),
            packaged_solution.displacement().values()
        );
        assert_eq!(
            direct_solution.algebraic_values(),
            packaged_solution.algebraic_values()
        );
        assert_eq!(
            direct_solution.boundary_reaction(),
            packaged_solution.boundary_reaction()
        );
        let direct_tractions = recovered_traction_resultants(&direct_solution, 3.0, 0.0);
        let packaged_tractions = recovered_traction_resultants(&packaged_solution, 3.0, 0.0);
        assert_eq!(direct_tractions, packaged_tractions);

        for vertex in 0..direct_solution
            .displacement()
            .mesh()
            .entity_count(0)
            .expect("mesh owns vertices")
        {
            let entity = MeshEntity::new(0, vertex);
            let coordinates = direct_solution
                .displacement()
                .mesh()
                .vertex_coordinates(entity)
                .expect("vertex coordinates");
            let (expected, _) = exact(&coordinates);
            for (actual, expected) in direct_solution
                .displacement()
                .vertex_values(vertex)
                .expect("nodal vector")
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).abs() <= 2.0e-10);
            }
        }

        let h = 1.0 / cells as f64;
        let expected_left_recovered_traction = -6.0 + 3.0 * h;
        let expected_right_recovered_traction = 3.0 * h;
        assert!((direct_tractions[0][0][0] - expected_left_recovered_traction).abs() <= 2.0e-11);
        assert!(direct_tractions[0][0][1].abs() <= 2.0e-11);
        assert!((direct_tractions[0][1][0] - expected_right_recovered_traction).abs() <= 2.0e-11);
        assert!(direct_tractions[0][1][1].abs() <= 2.0e-11);
        for traction in &direct_tractions[1] {
            assert!(traction.iter().all(|value| value.abs() <= 2.0e-11));
        }
        free_boundary_traction_errors.push(direct_tractions[0][1][0].abs());

        let norms = direct_solution
            .displacement()
            .error_norms(&exact, &error_rule)
            .expect("continuous error evidence");
        let expected_l2 = h * h / 120.0_f64.sqrt();
        let expected_h1 = h / 12.0_f64.sqrt();
        assert!((norms.l2() - expected_l2).abs() <= 2.0e-11 + 5.0e-6 * expected_l2);
        assert!((norms.h1_seminorm() - expected_h1).abs() <= 2.0e-11 + 5.0e-6 * expected_h1);
        l2_errors.push(norms.l2());
        h1_errors.push(norms.h1_seminorm());

        let body_force = direct_solution.integrated_body_force();
        let reaction = direct_solution.boundary_reaction();
        assert!((body_force[0] - 6.0).abs() <= 2.0e-13);
        assert!(body_force[1].abs() <= 2.0e-13);
        assert!((reaction[0] + 6.0).abs() <= 2.0e-11);
        assert!(reaction[1].abs() <= 2.0e-11);
        for component in 0..2 {
            assert!((reaction[component] + body_force[component]).abs() <= 2.0e-11);
        }
    }
    assert!(
        l2_errors
            .windows(2)
            .all(|errors| (errors[0] / errors[1]).log2() >= 1.99)
    );
    assert!(
        h1_errors
            .windows(2)
            .all(|errors| (errors[0] / errors[1]).log2() >= 0.99)
    );
    assert!(
        free_boundary_traction_errors
            .windows(2)
            .all(|errors| (errors[0] / errors[1]).log2() >= 0.99)
    );
}

#[test]
fn live_multiport_binding_is_retained_then_rejected_by_the_q1_realization() {
    let dependency = elasticity_package();
    let source = PACKAGED_SOURCE
        .replace(
            "  instance y_lower_free: solid.ZeroTraction2d(",
            "  instance x_upper_free_peer: solid.ZeroTraction2d(\n    body = body,\n    face = x_upper\n  );\n  instance y_lower_free: solid.ZeroTraction2d(",
        )
        .replace(
            "connect boundary_law.mechanical[boundary = x_upper], x_upper_free.mechanical;",
            "connect boundary_law.mechanical[boundary = x_upper], x_upper_free.mechanical, x_upper_free_peer.mechanical;",
        );
    let packaged = compile_packaged(&dependency, &source);
    let lowered = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect("live connection remains valid semantic meaning");
    assert!(matches!(
        lowered
            .boundary_inventory()
            .boundary(0, BoundarySide::Upper)
            .expect("x-upper inventory")
            .disposition(),
        PhysicalBoundaryDisposition::PortBinding { .. }
    ));

    let diagnostic = finalize_resolved_isotropic_elasticity_cartesian_2d(
        packaged.model().program(),
        &resolved(packaged.model().program(), 4, 1),
    )
    .unwrap_err();
    assert_eq!(diagnostic.code(), codes::INVALID_REALIZATION);
    assert!(diagnostic.message().contains("trace-space Realization"));
}

#[test]
fn boundary_normalization_rejects_near_miss_semantics() {
    let dependency = elasticity_package();

    let independent_equal_coefficient = PACKAGED_SOURCE
        .replace(
            "  parameter mu: kg / (m * s ^ 2) = 3;",
            "  parameter mu: kg / (m * s ^ 2) = 3;\n  parameter boundary_mu: kg / (m * s ^ 2) = 3;",
        )
        .replace(
            "instance boundary_law: solid.IsotropicMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    displacement = displacement,\n    mu = mu,",
            "instance boundary_law: solid.IsotropicMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    displacement = displacement,\n    mu = boundary_mu,",
        );
    assert_ne!(independent_equal_coefficient, PACKAGED_SOURCE);
    let packaged = compile_packaged(&dependency, &independent_equal_coefficient);
    let diagnostic = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect_err("equal values cannot merge independent coefficient directions");
    assert!(diagnostic.message().contains("stress coefficients differ"));

    let mismatched_stress = PACKAGED_SOURCE.replace(
        "instance boundary_law: solid.IsotropicMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    displacement = displacement,\n    mu = mu,",
        "instance boundary_law: solid.IsotropicMechanicalInterface2d(\n    body = body,\n    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),\n    displacement = displacement,\n    mu = 4[kg / (m * s ^ 2)],",
    );
    assert_ne!(mismatched_stress, PACKAGED_SOURCE);
    let packaged = compile_packaged(&dependency, &mismatched_stress);
    let diagnostic = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect_err("boundary and volume stress must agree exactly");
    assert!(diagnostic.message().contains("stress coefficients differ"));

    let mut additional_direct_relation = DIRECT_SOURCE
        .strip_suffix("}\n")
        .expect("fixture ends with its Model delimiter")
        .to_owned();
    additional_direct_relation.push_str(
        "  relation conflicting_trace on x_upper {\n    trace(displacement) = 0;\n  }\n}\n",
    );
    let direct = eqiora::api::ModelDocument::compile(
        "additional-direct-relation.eqi",
        &additional_direct_relation,
    )
    .expect("near-miss direct model is valid semantic input");
    let diagnostic = lower_isotropic_elasticity_cartesian_2d(direct.program())
        .expect_err("a recognized direct law cannot hide an additional Relation");
    assert!(
        diagnostic
            .message()
            .contains("ambiguous with additional Relations")
    );

    let duplicate_boundary = DIRECT_SOURCE.replace(
        "  domain x_lower = boundary(body, axis = 0, side = lower);",
        "  domain x_lower = boundary(body, axis = 0, side = lower);\n  domain x_lower_peer = boundary(body, axis = 0, side = lower);",
    );
    let direct = eqiora::api::ModelDocument::compile("duplicate-boundary.eqi", &duplicate_boundary)
        .expect("duplicate geometric side remains distinguishable semantic identity");
    let diagnostic = lower_isotropic_elasticity_cartesian_2d(direct.program())
        .expect_err("the Cartesian side inventory must be a bijection");
    assert!(diagnostic.message().contains("boundary side is duplicated"));

    let simultaneous_terminal = LIVE_PACKAGE_SOURCE.replace(
        "  relation prescribed_traction on face {\n    mechanical.traction = 0;\n  }",
        "  relation prescribed_traction on face {\n    mechanical.displacement = 0;\n    mechanical.traction = 0;\n  }",
    );
    assert_ne!(simultaneous_terminal, LIVE_PACKAGE_SOURCE);
    let dependency = elasticity_package_with_source(&simultaneous_terminal);
    let packaged = compile_packaged(&dependency, PACKAGED_SOURCE);
    let diagnostic = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect_err("one physical terminal cannot prescribe conjugate variables together");
    assert!(
        diagnostic
            .message()
            .contains("cannot prescribe zero trace and zero flux simultaneously")
    );
}

fn coefficient_derivatives(expression: &ScalarSpatialExpression) -> (u64, Vec<u64>, Vec<u64>) {
    let coordinates = vec![0.0; expression.coordinate_dimension()];
    let mut jvps = Vec::new();
    let mut vjps = Vec::new();
    let cotangent = 1.75;
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
