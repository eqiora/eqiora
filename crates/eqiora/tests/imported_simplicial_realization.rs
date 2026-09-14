use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::artifact::{
    LayoutArtifacts, MeshDecoderLimits, ModelEnvelope, RealizationEnvelopeV1,
    SimplicialMeshEnvelopeV1,
};
use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::{MeshQualityGate, SimplicialMesh};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshArtifactReference, MeshKind,
    MeshPolicy, QuadraturePolicy, RealizationCapabilities, RealizationPlan, RealizationRequest,
    RealizationRequirements, RealizationRevision, SemanticRevision, Space, SpatialDimensionSupport,
    Target, TargetCapabilities, VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    ExecutionReport, LinearOperatorProperties, LinearSolver, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolveReport, SolverCapabilities, SolverCapability,
    SolverPlan,
};
use eqiora_backend_rayon::{CpuThreadPool, RAYON_EXECUTION};
use eqiora_numerics::{
    scalar::solve_resolved_scalar_elliptic_simplicial,
    scalar::solve_resolved_scalar_elliptic_simplicial_with_assembly,
};

const SOURCE: &str =
    include_str!("../../../verify/artifacts/imported-simplicial-realization/models/poisson.eqi");

#[test]
fn imported_mesh_identity_round_trips_from_realization_to_assembly_evidence() {
    let program = compile_program();
    let model_artifact = ModelEnvelope::from_program(&program).unwrap();
    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&cross_mesh()).unwrap();
    let mesh_bytes = mesh_artifact.canonical_json().unwrap();
    let mesh_artifact =
        SimplicialMeshEnvelopeV1::from_json(&mesh_bytes, Default::default()).unwrap();
    let resolved = resolve_imported(&program, mesh_artifact.artifact_reference().unwrap(), 2);
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_artifact,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    realization.validate_mesh_artifact(&mesh_artifact).unwrap();
    assert_eq!(
        realization.mesh_artifact().unwrap().unwrap(),
        mesh_artifact.digest().unwrap(),
    );

    let realization_bytes = realization.canonical_json().unwrap();
    let realization =
        RealizationEnvelopeV1::from_json(&realization_bytes, Default::default()).unwrap();
    realization.validate_mesh_artifact(&mesh_artifact).unwrap();
    let (_, solution) = solve_resolved_scalar_elliptic_simplicial(
        &program,
        &resolved,
        mesh_artifact.artifact_reference().unwrap(),
        mesh_artifact.mesh(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();

    assert_eq!(solution.algebraic_values().len(), 1);
    assert!((solution.algebraic_values()[0] - 1.0 / 12.0).abs() < 2.0e-15);
    assert!((solution.integrated_source() - 1.0).abs() < 2.0e-15);
    assert!((solution.boundary_reaction_sum() + 1.0).abs() < 2.0e-15);
    assert!(solution.solve_report().true_residual_norm() < 1.0e-14);
}

#[test]
fn imported_mesh_p1_assembly_is_exact_for_one_and_four_workers() {
    let program = compile_program();
    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&cross_mesh()).unwrap();
    let reference = mesh_artifact.artifact_reference().unwrap();
    let serial = resolve_imported(&program, reference, 2);
    let (_, serial_solution) = solve_resolved_scalar_elliptic_simplicial(
        &program,
        &serial,
        reference,
        mesh_artifact.mesh(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();

    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        scalar_elliptic_solver_capabilities(),
        pool.target_capabilities(),
    )
    .unwrap();
    let threaded = resolve(
        &imported_request_with_threads(&program, reference, workers),
        requirements(2),
        &capabilities,
    )
    .unwrap();
    let assembly = pool.assembler(threaded.plan().target()).unwrap();
    let solver = pool
        .solver(threaded.plan().target(), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, threaded_solution) = solve_resolved_scalar_elliptic_simplicial_with_assembly(
        &program,
        &threaded,
        reference,
        mesh_artifact.mesh(),
        &assembly,
        &solver,
    )
    .unwrap();

    assert_eq!(threaded_solution.field(), serial_solution.field());
    assert_eq!(
        threaded_solution.algebraic_values(),
        serial_solution.algebraic_values()
    );
    assert_eq!(
        threaded_solution.integrated_source(),
        serial_solution.integrated_source()
    );
    assert_eq!(
        threaded_solution.boundary_reaction_sum(),
        serial_solution.boundary_reaction_sum()
    );
    assert_eq!(
        serial_solution.assembly_report().execution(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        threaded_solution.assembly_report().execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    for report in [
        serial_solution.assembly_report(),
        threaded_solution.assembly_report(),
    ] {
        assert_eq!(report.packet_count(), 4);
        assert_eq!(report.target_count(), 2);
    }
    assert_eq!(
        serial_solution.solve_report().execution(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        threaded_solution.solve_report().execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    assert_same_numerical_report(
        serial_solution.solve_report(),
        threaded_solution.solve_report(),
    );
}

fn assert_same_numerical_report(serial: &SolveReport, threaded: &SolveReport) {
    assert_eq!(threaded.backend(), serial.backend());
    assert_eq!(threaded.orientation(), serial.orientation());
    assert_eq!(threaded.algorithm(), serial.algorithm());
    assert_eq!(threaded.preconditioner(), serial.preconditioner());
    assert_eq!(threaded.reduction(), serial.reduction());
    assert_eq!(threaded.reason(), serial.reason());
    assert_eq!(
        threaded.completed_iterations(),
        serial.completed_iterations()
    );
    assert_eq!(
        threaded.initial_residual_norm(),
        serial.initial_residual_norm()
    );
    assert_eq!(
        threaded.reported_residual_norm(),
        serial.reported_residual_norm()
    );
    assert_eq!(threaded.true_residual_norm(), serial.true_residual_norm());
    assert_eq!(threaded.residual_target(), serial.residual_target());
}

#[test]
fn imported_mesh_capability_identity_and_dimension_mismatches_fail_closed() {
    let program = compile_program();
    let model_artifact = ModelEnvelope::from_program(&program).unwrap();
    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&cross_mesh()).unwrap();
    let reference = mesh_artifact.artifact_reference().unwrap();
    let request = imported_request(&program, reference);
    let requirements_2d = requirements(2);
    let generated_only = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        scalar_elliptic_solver_capabilities(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap();
    assert!(resolve(&request, requirements_2d, &generated_only).is_err());

    let wrong_reference = MeshArtifactReference::from_sha256([7; 32]);
    let wrong = resolve_imported(&program, wrong_reference, 2);
    let wrong_envelope =
        RealizationEnvelopeV1::from_resolved(&model_artifact, &wrong, LayoutArtifacts::Replicated)
            .unwrap();
    assert!(
        wrong_envelope
            .validate_mesh_artifact(&mesh_artifact)
            .is_err()
    );
    assert!(
        solve_resolved_scalar_elliptic_simplicial(
            &program,
            &wrong,
            reference,
            mesh_artifact.mesh(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err()
    );

    assert!(
        resolve(
            &imported_request(&program, reference),
            requirements(3),
            &RealizationCapabilities::scalar_elliptic_reference(),
        )
        .is_err()
    );
    let three_dimensional_mesh = SimplicialMeshEnvelopeV1::from_mesh(&tetrahedron_mesh()).unwrap();
    let wrong_dimension = resolve_imported(
        &program,
        three_dimensional_mesh.artifact_reference().unwrap(),
        2,
    );
    let wrong_dimension_envelope = RealizationEnvelopeV1::from_resolved(
        &model_artifact,
        &wrong_dimension,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    assert!(
        wrong_dimension_envelope
            .validate_mesh_artifact(&three_dimensional_mesh)
            .is_err()
    );
    assert!(
        solve_resolved_scalar_elliptic_simplicial(
            &program,
            &wrong_dimension,
            three_dimensional_mesh.artifact_reference().unwrap(),
            three_dimensional_mesh.mesh(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err()
    );
}

fn scalar_elliptic_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .expect("the imported scalar-elliptic solver tuple is exact")
}

#[test]
fn imported_mesh_wire_rejects_resource_excess_unknown_fields_and_forged_evidence() {
    let artifact = SimplicialMeshEnvelopeV1::from_mesh(&cross_mesh()).unwrap();
    let bytes = artifact.canonical_json().unwrap();
    assert!(
        SimplicialMeshEnvelopeV1::from_json(
            &bytes,
            MeshDecoderLimits {
                max_mesh_cells: 3,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["geometry"]["importer"] = serde_json::json!("implicit-path");
    assert!(
        SimplicialMeshEnvelopeV1::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["evidence"]["minimum_signed_measure_scale"] = serde_json::json!(1.0);
    assert!(
        SimplicialMeshEnvelopeV1::from_json(
            &serde_json::to_vec(&forged).unwrap(),
            Default::default(),
        )
        .is_err()
    );
}

fn resolve_imported(
    program: &KernelProgram,
    reference: MeshArtifactReference,
    dimension: usize,
) -> eqiora::realization::ResolvedRealization {
    resolve(
        &imported_request(program, reference),
        requirements(dimension),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap()
}

fn imported_request(
    program: &KernelProgram,
    reference: MeshArtifactReference,
) -> RealizationRequest {
    imported_request_with_threads(program, reference, NonZeroUsize::MIN)
}

fn imported_request_with_threads(
    program: &KernelProgram,
    reference: MeshArtifactReference,
    threads: NonZeroUsize,
) -> RealizationRequest {
    RealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(1),
        RealizationPlan::new(
            Space::continuous_lagrange(NonZeroU16::MIN),
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: reference,
                },
                QuadraturePolicy::SimplexCentroid,
            ),
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1.0e-13,
                1.0e-14,
                NonZeroUsize::new(128).unwrap(),
            )
            .unwrap(),
            Target::HostCpu { threads },
            ExecutionSchedule::Offline,
        )
        .unwrap(),
    )
}

fn requirements(dimension: usize) -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(dimension).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn compile_program() -> KernelProgram {
    let mut compiled = compile("imported-simplicial-poisson.eqi", SOURCE).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn cross_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ],
        vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]],
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap()
}

fn tetrahedron_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![vec![0, 1, 2, 3]],
        MeshQualityGate::new(0.1).unwrap(),
    )
    .unwrap()
}
