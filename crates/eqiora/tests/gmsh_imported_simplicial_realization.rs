use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::artifact::{
    LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV1, SimplicialMeshEnvelopeV1,
};
use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::io::gmsh::{Msh41Policy, import_msh41};
use eqiora::meshing::MeshQualityGate;
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, SemanticRevision, Space, Target, VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use eqiora_numerics::scalar::solve_resolved_scalar_elliptic_simplicial;

const SOURCE: &str = include_str!(
    "../../../verify/artifacts/gmsh-imported-simplicial-realization/models/poisson.eqi"
);
const MSH: &[u8] = include_bytes!(
    "../../../verify/artifacts/gmsh-imported-simplicial-realization/fixtures/unit-square-cross.msh"
);
const BINARY_MSH: &[u8] = include_bytes!(
    "../../../verify/artifacts/gmsh-imported-simplicial-realization/fixtures/unit-square-cross-binary.msh"
);
const EXPECTED_MESH_DIGEST: &str = include_str!(
    "../../../verify/artifacts/gmsh-imported-simplicial-realization/expected/mesh.sha256"
);

#[test]
fn gmsh_fixture_closes_import_artifact_realization_and_solution_evidence() {
    let policy = Msh41Policy::mesh(2, MeshQualityGate::new(0.5).unwrap()).unwrap();
    let mesh = import_msh41(MSH, policy, |_, _, _| {}).unwrap();
    let binary_mesh = import_msh41(BINARY_MSH, policy, |_, _, _| {}).unwrap();
    assert_eq!(binary_mesh, mesh);
    assert_eq!(mesh.vertices().len(), 5);
    assert_eq!(mesh.cells().len(), 4);

    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap();
    let binary_artifact = SimplicialMeshEnvelopeV1::from_mesh(&binary_mesh).unwrap();
    assert_eq!(
        binary_artifact.canonical_json().unwrap(),
        mesh_artifact.canonical_json().unwrap(),
    );
    assert_eq!(
        binary_artifact.digest().unwrap(),
        mesh_artifact.digest().unwrap()
    );
    assert_eq!(
        mesh_artifact.digest().unwrap().as_str(),
        EXPECTED_MESH_DIGEST.trim(),
    );
    let canonical_bytes = mesh_artifact.canonical_json().unwrap();
    let mesh_artifact =
        SimplicialMeshEnvelopeV1::from_json(&canonical_bytes, Default::default()).unwrap();

    let program = compile_program();
    let model_artifact = ModelEnvelope::from_program(&program).unwrap();
    let request = realization_request(&program, mesh_artifact.artifact_reference().unwrap());
    let requirements = RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let resolved = resolve(
        &request,
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_artifact,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
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
fn every_truncated_official_binary_fixture_fails_through_the_public_facade() {
    let policy = Msh41Policy::mesh(2, MeshQualityGate::new(0.5).unwrap()).unwrap();
    for end in 0..BINARY_MSH.len() {
        assert_eq!(
            import_msh41(&BINARY_MSH[..end], policy, |_, _, _| {})
                .unwrap_err()
                .code(),
            eqiora::diagnostic::codes::INVALID_MESH_IMPORT,
            "official fixture prefix ending at byte {end} was unexpectedly admitted",
        );
    }
}

fn realization_request(
    program: &KernelProgram,
    mesh: eqiora::realization::MeshArtifactReference,
) -> RealizationRequest {
    RealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(1),
        RealizationPlan::new(
            Space::continuous_lagrange(NonZeroU16::MIN),
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial { artifact: mesh },
                QuadraturePolicy::SimplexCentroid,
            ),
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1.0e-13,
                1.0e-14,
                NonZeroUsize::new(128).unwrap(),
            )
            .unwrap(),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap(),
    )
}

fn compile_program() -> KernelProgram {
    let mut compiled = compile("gmsh-imported-simplicial-poisson.eqi", SOURCE).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}
