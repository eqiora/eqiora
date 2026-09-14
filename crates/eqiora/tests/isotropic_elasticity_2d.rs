use eqiora_core::ScalarType;
use std::f64::consts::PI;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, RealizationEnvelopeV1,
    RunManifestV2,
};
use eqiora::assembly::{AssemblyMap, CooAssembler, DofId, LocalContribution, LocalUnknown};
use eqiora::compiler::compile;
use eqiora::diagnostic::codes;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::{MeshEntity, MeshTopology, QuadratureRule};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationCapability, RealizationCapabilityContext, RealizationPlan,
    RealizationRequest, RealizationRequirements, RealizationRevision, ScheduleCapability,
    SemanticRevision, Space, SpatialCapability, SpatialDimensionSupport, Target, TargetCapability,
    VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverCapability, SolverPlan,
};
use eqiora_meshing::CartesianMesh;
use eqiora_numerics::{
    solid::CartesianQ1VectorField2d, solid::lower_cartesian_q1_linear_elasticity_local_action_2d,
    solid::lower_isotropic_elasticity_cartesian_2d,
    solid::solve_resolved_isotropic_elasticity_cartesian_2d,
};

const MANUFACTURED: &str =
    include_str!("../../../verify/solid/isotropic-elasticity-2d/models/manufactured.eqi");
const LINEAR_LOAD: &str =
    include_str!("../../../verify/solid/isotropic-elasticity-2d/models/linear-load.eqi");
const COMPONENTS: usize = 2;

fn compile_program() -> KernelProgram {
    compile_program_from(MANUFACTURED)
}

fn compile_program_from(source: &str) -> KernelProgram {
    let mut compiled = compile("manufactured.eqi", source).expect("source must compile");
    assert_eq!(compiled.len(), 1);
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction must commit");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("model must be admitted")
}

fn assert_lowering_rejects(source: &str) {
    assert_eq!(
        lower_isotropic_elasticity_cartesian_2d(&compile_program_from(source))
            .unwrap_err()
            .code(),
        codes::INVALID_SPATIAL_LOWERING
    );
}

fn resolved(
    program: &KernelProgram,
    cells: usize,
    revision: u64,
) -> eqiora::realization::ResolvedRealization {
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
    let request = RealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(revision),
        plan,
    );
    resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::new(2).expect("two dimensions"),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::isotropic_elasticity_2d_reference(),
    )
    .expect("exact elasticity capability must admit the plan")
}

fn exact_displacement_and_gradient(point: &[f64]) -> ([f64; 2], [[f64; 2]; 2]) {
    let x = PI * point[0];
    let y = PI * point[1];
    let constitutive = 8.0;
    let amplitude = 1.0 / (2.0 * PI * constitutive);
    let sx = x.sin();
    let sy = y.sin();
    let sin_2x = (2.0 * x).sin();
    let sin_2y = (2.0 * y).sin();
    let mixed = -sin_2x * sin_2y / (2.0 * constitutive);
    (
        [
            -amplitude * sin_2x * sy.powi(2),
            -amplitude * sx.powi(2) * sin_2y,
        ],
        [
            [-((2.0 * x).cos()) * sy.powi(2) / constitutive, mixed],
            [mixed, -sx.powi(2) * (2.0 * y).cos() / constitutive],
        ],
    )
}

#[test]
fn canonical_model_realization_and_q1_convergence_form_one_closed_slice() {
    let program = compile_program();
    let lowered = lower_isotropic_elasticity_cartesian_2d(&program)
        .expect("canonical elasticity must lower without a numerical method");
    assert_eq!(lowered.bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(lowered.shear_modulus(), 3.0);
    assert_eq!(lowered.first_lame_parameter(), 2.0);
    assert_eq!(
        lowered.shear_modulus_expression().parameter_fields().len(),
        1
    );
    assert_eq!(
        lowered
            .first_lame_parameter_expression()
            .parameter_fields()
            .len(),
        1
    );

    let error_rule =
        QuadratureRule::tensor_product_gauss_legendre(2, 4).expect("independent error quadrature");
    let mut l2_errors = Vec::new();
    let mut h1_errors = Vec::new();
    for (revision, cells) in [4, 8, 16, 32].into_iter().enumerate() {
        let (_, solution) = solve_resolved_isotropic_elasticity_cartesian_2d(
            &program,
            &resolved(&program, cells, revision as u64 + 1),
            &REFERENCE_LINEAR_SOLVER,
        )
        .expect("canonical model must execute through its resolved Q1 plan");
        let norms = solution
            .displacement()
            .error_norms(&exact_displacement_and_gradient, &error_rule)
            .expect("continuous vector error evidence");
        l2_errors.push(norms.l2());
        h1_errors.push(norms.h1_seminorm());
        for component in 0..COMPONENTS {
            let scale = solution.boundary_reaction()[component]
                .abs()
                .max(solution.integrated_body_force()[component].abs())
                .max(1.0);
            assert!(
                (solution.boundary_reaction()[component]
                    + solution.integrated_body_force()[component])
                    .abs()
                    <= 2.0e-11 * scale
            );
        }
        assert!(
            solution.solve_report().true_residual_norm()
                <= solution.solve_report().residual_target()
        );
    }
    assert!(
        l2_errors.windows(2).all(|errors| errors[1] < errors[0]),
        "Q1 displacement L2 error was not strictly decreasing: {l2_errors:?}"
    );
    for errors in l2_errors.windows(2).skip(1) {
        assert!(
            (errors[0] / errors[1]).log2() >= 1.9,
            "Q1 displacement L2 convergence fell below second order: {l2_errors:?}"
        );
    }
    for errors in h1_errors.windows(2).skip(1) {
        assert!(
            (errors[0] / errors[1]).log2() >= 0.9,
            "Q1 displacement H1 convergence fell below first order: {h1_errors:?}"
        );
    }
}

#[test]
fn elasticity_finalization_requires_an_admitted_spd_operator() {
    let program = compile_program();
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(2).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        SolverPlan::new(
            LinearSolver::MinimumResidual,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let dimension = NonZeroUsize::new(2).unwrap();
    let capability = RealizationCapability::new(
        RealizationCapabilityContext::new(
            SpatialCapability::new(
                DiscretizationMethod::ContinuousGalerkin,
                eqiora::realization::MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(dimension),
            ),
            VectorLayoutKind::Replicated,
            TargetCapability::HostCpu {
                maximum_threads: NonZeroUsize::MIN,
            },
            ScheduleCapability::Offline,
        ),
        SolverCapability {
            algorithm: LinearSolver::MinimumResidual,
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        },
    )
    .unwrap();
    let resolved = resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(50),
            plan,
        ),
        RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated),
        &RealizationCapabilities::exact([capability]).unwrap(),
    )
    .expect("legacy resolution retains the sole symmetric-indefinite candidate");

    let error = solve_resolved_isotropic_elasticity_cartesian_2d(
        &program,
        &resolved,
        &REFERENCE_LINEAR_SOLVER,
    )
    .expect_err("elasticity must seal its SPD property before assembly");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(
        error
            .message()
            .contains("not admitted for operator properties")
    );
}

#[test]
fn canonical_lowering_fails_closed_at_physical_identity_and_model_boundaries() {
    let wrong_dimensions = MANUFACTURED
        .replace(
            "variable displacement: vector<m, 2> on body;",
            "variable displacement: vector<1, 2> on body;",
        )
        .replace(
            "variable load_potential: kg / (m * s ^ 2) on body;",
            "variable load_potential: 1 on body;",
        )
        .replace(
            "parameter mu: kg / (m * s ^ 2) = 3;",
            "parameter mu: m = 3;",
        )
        .replace(
            "parameter lambda: kg / (m * s ^ 2) = 2;",
            "parameter lambda: m = 2;",
        )
        .replace(
            "parameter load_scale: kg / (m * s ^ 2) = 1;",
            "parameter load_scale: 1 = 1;",
        );
    assert_lowering_rejects(&wrong_dimensions);

    let retired_representation = MANUFACTURED.replace(
        "model manufactured_isotropic_elasticity_plane() {",
        "model manufactured_isotropic_elasticity_plane() { representation separate = continuum;",
    );
    assert!(
        eqiora::language::parse("retired.eqi", &retired_representation)
            .into_document()
            .is_err()
    );

    let ignored_parameter = MANUFACTURED.replace(
        "parameter mu: kg / (m * s ^ 2) = 3;",
        "parameter unused: 1 = 7;\n  parameter mu: kg / (m * s ^ 2) = 3;",
    );
    assert_lowering_rejects(&ignored_parameter);

    let periodic_load = MANUFACTURED
        .replace(
            "model manufactured_isotropic_elasticity_plane() {",
            "model manufactured_isotropic_elasticity_plane() { clock tick = periodic(1[s] / 1, phase = 0[s] / 1);",
        )
        .replace(
            "relation load on body",
            "relation load on body at tick",
        );
    assert_lowering_rejects(&periodic_load);

    let overflow = MANUFACTURED
        .replace(
            "parameter mu: kg / (m * s ^ 2) = 3;",
            "parameter mu: kg / (m * s ^ 2) = 1e308;",
        )
        .replace(
            "parameter lambda: kg / (m * s ^ 2) = 2;",
            "parameter lambda: kg / (m * s ^ 2) = 1e308;",
        );
    assert_lowering_rejects(&overflow);
}

#[test]
fn canonical_nonzero_potential_has_componentwise_force_and_reaction_balance() {
    let program = compile_program_from(LINEAR_LOAD);
    let (_, solution) = solve_resolved_isotropic_elasticity_cartesian_2d(
        &program,
        &resolved(&program, 8, 1),
        &REFERENCE_LINEAR_SOLVER,
    )
    .expect("linear canonical potential must execute");
    for (actual, exact) in solution.integrated_body_force().into_iter().zip([1.0, 0.0]) {
        assert!((actual - exact).abs() < 3.0e-14);
    }
    for component in 0..COMPONENTS {
        assert!(
            (solution.boundary_reaction()[component] + solution.integrated_body_force()[component])
                .abs()
                < 2.0e-11
        );
    }
}

#[test]
fn current_model_realization_v1_and_run_v2_replay_exact_lineage() {
    let document = eqiora::api::ModelDocument::compile("manufactured.eqi", MANUFACTURED)
        .expect("tensor Model uses the current wire");
    let model_bytes = document.canonical_json().unwrap();
    let model_replay = eqiora::api::ModelDocument::replay(&model_bytes).unwrap();
    assert_eq!(model_replay.canonical_json().unwrap(), model_bytes);
    let model_reference = document.artifact_reference().unwrap();

    let coarse_resolved = resolved(document.program(), 4, 1);
    let fine_resolved = resolved(document.program(), 8, 2);
    let coarse = RealizationEnvelopeV1::from_resolved(
        &model_reference,
        &coarse_resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    let fine = RealizationEnvelopeV1::from_resolved(
        &model_reference,
        &fine_resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    assert_ne!(coarse.digest().unwrap(), fine.digest().unwrap());
    assert_eq!(document.canonical_json().unwrap(), model_bytes);

    let realization_bytes = fine.canonical_json().unwrap();
    let realization_replay =
        RealizationEnvelopeV1::from_json(&realization_bytes, Default::default()).unwrap();
    assert_eq!(
        realization_replay.canonical_json().unwrap(),
        realization_bytes
    );
    assert_eq!(realization_replay.digest().unwrap(), fine.digest().unwrap());
    realization_replay
        .validate_model_artifact(&model_replay.artifact_reference().unwrap())
        .unwrap();
    let mut revision_drift: serde_json::Value = serde_json::from_slice(&realization_bytes).unwrap();
    revision_drift["semantic_revision"] =
        serde_json::Value::from(document.program().revision().0 + 1);
    let revision_drift = RealizationEnvelopeV1::from_json(
        &serde_json::to_vec(&revision_drift).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        revision_drift
            .validate_model_artifact(&model_reference)
            .is_err()
    );

    let execution = ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        env!("CARGO_PKG_VERSION"),
        "eqiora.reference",
        env!("CARGO_PKG_VERSION"),
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap();
    let run = RunManifestV2::new(&fine, execution).unwrap();
    run.validate_against(&fine).unwrap();
    let run_bytes = run.canonical_json().unwrap();
    let run_replay = RunManifestV2::from_json(&run_bytes, Default::default()).unwrap();
    assert_eq!(run_replay.canonical_json().unwrap(), run_bytes);
    assert_eq!(run_replay.digest().unwrap(), run.digest().unwrap());
    run_replay.validate_against(&realization_replay).unwrap();
    let mut realization_digest_drift: serde_json::Value =
        serde_json::from_slice(&run_bytes).unwrap();
    realization_digest_drift["realization_sha256"] = serde_json::Value::String("00".repeat(32));
    let realization_digest_drift = RunManifestV2::from_json(
        &serde_json::to_vec(&realization_digest_drift).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(realization_digest_drift.validate_against(&fine).is_err());

    let changed = eqiora::api::ModelDocument::compile(
        "manufactured.eqi",
        &MANUFACTURED.replace(
            "parameter mu: kg / (m * s ^ 2) = 3;",
            "parameter mu: kg / (m * s ^ 2) = 4;",
        ),
    )
    .unwrap();
    assert!(
        fine.validate_model_artifact(&changed.artifact_reference().unwrap())
            .is_err()
    );
}

#[test]
fn affine_patch_proves_tensor_coupling_and_exact_boundary_equilibrium() {
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 2]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    for (mu, lambda) in [(0.0, 2.0), (3.0, -3.0), (f64::INFINITY, 2.0)] {
        assert!(
            lower_cartesian_q1_linear_elasticity_local_action_2d(&mesh, mu, lambda, &quadrature,)
                .is_err()
        );
    }
    let action = lower_cartesian_q1_linear_elasticity_local_action_2d(&mesh, 3.0, 2.0, &quadrature)
        .expect("isotropic Q1 local actions");
    let vertex_count = mesh.entity_count(0).unwrap();
    let mut assembler = CooAssembler::new(vertex_count * COMPONENTS).unwrap();
    let local_width = action.rows();
    for cell_index in 0..mesh.entity_count(2).unwrap() {
        let offset = cell_index * local_width * local_width;
        let local = LocalContribution::new(
            local_width,
            local_width,
            action.coefficients()[offset..offset + local_width * local_width].to_vec(),
            vec![0.0; local_width],
        )
        .unwrap();
        let vertices = mesh
            .entity_vertices(MeshEntity::new(2, cell_index))
            .unwrap();
        let global = vertices
            .iter()
            .flat_map(|vertex| {
                (0..COMPONENTS).map(move |component| vertex.index() * COMPONENTS + component)
            })
            .collect::<Vec<_>>();
        let map = AssemblyMap::new(
            global
                .iter()
                .map(|index| Some(DofId::new(*index)))
                .collect(),
            global
                .iter()
                .map(|index| LocalUnknown::Free(DofId::new(*index)))
                .collect(),
        )
        .unwrap();
        assembler.scatter(&map, &local).unwrap();
    }
    let system = assembler.finish().unwrap();
    let displacement = (0..vertex_count)
        .flat_map(|vertex| {
            let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
            [
                2.0 * point[0] + 3.0 * point[1] + 1.0,
                5.0 * point[0] + 7.0 * point[1] - 2.0,
            ]
        })
        .collect::<Vec<_>>();
    let patch_error = CartesianQ1VectorField2d::new(mesh.clone(), displacement.clone())
        .unwrap()
        .error_norms(
            &|point| {
                (
                    [
                        2.0 * point[0] + 3.0 * point[1] + 1.0,
                        5.0 * point[0] + 7.0 * point[1] - 2.0,
                    ],
                    [[2.0, 3.0], [5.0, 7.0]],
                )
            },
            &quadrature,
        )
        .unwrap();
    assert!(patch_error.l2() < 5.0e-15);
    assert!(patch_error.h1_seminorm() < 3.0e-14);
    let reactions = system.matrix().multiply(&displacement).unwrap();
    let mut resultant = [0.0; COMPONENTS];
    let mut moment = 0.0;
    for vertex in 0..vertex_count {
        let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
        let reaction = &reactions[vertex * COMPONENTS..(vertex + 1) * COMPONENTS];
        if point == [0.5, 0.5] {
            assert_eq!(
                &displacement[vertex * COMPONENTS..(vertex + 1) * COMPONENTS],
                [3.5, 4.0]
            );
            assert!(reaction.iter().all(|value| value.abs() < 2.0e-14));
            continue;
        }
        let expected = match (point[0], point[1]) {
            (0.0, 0.0) => [-13.5, -21.0],
            (0.5, 0.0) => [-12.0, -30.0],
            (1.0, 0.0) => [1.5, -9.0],
            (0.0, 0.5) => [-15.0, -12.0],
            (1.0, 0.5) => [15.0, 12.0],
            (0.0, 1.0) => [-1.5, 9.0],
            (0.5, 1.0) => [12.0, 30.0],
            (1.0, 1.0) => [13.5, 21.0],
            _ => panic!("unexpected patch vertex {point:?}"),
        };
        for component in 0..COMPONENTS {
            assert!((reaction[component] - expected[component]).abs() < 3.0e-14);
            resultant[component] += reaction[component];
        }
        moment += point[0] * reaction[1] - point[1] * reaction[0];
    }
    assert!(resultant.iter().all(|value| value.abs() < 6.0e-14));
    assert!(moment.abs() < 6.0e-14);
}
