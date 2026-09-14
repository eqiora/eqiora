use eqiora_core::ScalarType;
use std::num::NonZeroUsize;

use eqiora_artifact::{
    ArtifactDigest, DistributedTransportV1, ExecutionProvenanceV1, ExecutionTopologyV1,
    JsonDecoderLimits, LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV1, RunManifestV2,
};
use eqiora_compiler::compile;
use eqiora_core::OntologyId;
use eqiora_core::diagnostic::codes;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_realization::{
    DefaultPolicyVersion, DiscretizationMethod, RealizationCapabilities, RealizationPlan,
    RealizationRequest, RealizationRequirements, RealizationRevision, SemanticRevision,
    SpatialDimensionSupport, TargetCapabilities, VectorLayoutKind, default_plan_v0, resolve,
};
use eqiora_schema::Model;
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    BackendId, ExecutionId, ExecutionProvider, LinearOperatorProperties, LinearSolver,
    PreconditionerPolicy, ProviderLibrary, ReductionPolicy, SolverCapabilities, SolverCapability,
    SolverPlan, SolverProvider,
};

const POISSON: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");

#[test]
fn realization_and_run_v2_round_trip_with_typed_linkage() {
    let (model_envelope, model) = model_fixture();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let resolved = resolve(
        &RealizationRequest::default(
            model,
            SemanticRevision::new(model_envelope.source_revision()),
            DefaultPolicyVersion::V0,
        ),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_envelope,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    let realization_bytes = realization.canonical_json().unwrap();
    let decoded_realization =
        RealizationEnvelopeV1::from_json(&realization_bytes, Default::default()).unwrap();

    assert_eq!(
        decoded_realization.canonical_json().unwrap(),
        realization_bytes
    );
    assert_eq!(
        decoded_realization.digest().unwrap(),
        realization.digest().unwrap()
    );
    assert_eq!(decoded_realization.model().unwrap(), model);
    assert_eq!(decoded_realization.requirements().unwrap(), requirements);
    assert_eq!(decoded_realization.plan().unwrap(), *resolved.plan());

    let execution = ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap()
    .with_library("rust", "1.85.0")
    .unwrap();
    let run = RunManifestV2::new(&realization, execution)
        .unwrap()
        .with_output(digest(2));
    let run_bytes = run.canonical_json().unwrap();
    let decoded_run = RunManifestV2::from_json(&run_bytes, Default::default()).unwrap();

    decoded_run.validate_against(&decoded_realization).unwrap();
    assert_eq!(decoded_run.canonical_json().unwrap(), run_bytes);
    assert_eq!(decoded_run.digest().unwrap(), run.digest().unwrap());
    assert_eq!(decoded_run.model(), model_envelope.digest().unwrap());
    assert_eq!(decoded_run.realization(), realization.digest().unwrap());
}

#[test]
fn execution_provenance_fingerprint_covers_the_complete_runtime_observation() {
    let with_mpi_rs = |version: &str| {
        ExecutionProvenanceV1::new(
            "eqiora.mpi",
            "0.1.0",
            "eqiora.mpi.krylov",
            "0.1.0",
            ExecutionTopologyV1::Distributed {
                partitions: NonZeroUsize::new(2).unwrap(),
                workers_per_partition: NonZeroUsize::MIN,
                transport: DistributedTransportV1::Mpi {
                    implementation: "mpich".to_owned(),
                    version: "5.0.1".to_owned(),
                    thread_support: eqiora_artifact::MpiThreadSupportV1::Funneled,
                },
            },
            ReductionPolicy::Reproducible,
        )
        .unwrap()
        .with_library("mpi-rs", version)
        .unwrap()
    };
    let observed = with_mpi_rs("0.8.2");
    let same = observed.clone();
    let changed = observed
        .clone()
        .with_library("mpi-standard", "4.1")
        .unwrap();
    let changed_version = with_mpi_rs("0.8.3");

    assert_eq!(
        observed.agreement_fingerprint().unwrap(),
        same.agreement_fingerprint().unwrap()
    );
    assert_ne!(
        observed.agreement_fingerprint().unwrap(),
        changed.agreement_fingerprint().unwrap()
    );
    assert_ne!(
        observed.agreement_fingerprint().unwrap(),
        changed_version.agreement_fingerprint().unwrap()
    );
}

#[test]
fn provider_release_projection_deduplicates_equal_components_and_rejects_conflicts() {
    const SOLVER_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("common-runtime", "1.0.0")];
    const EXECUTION_LIBRARIES: &[ProviderLibrary] = &[
        ProviderLibrary::new("common-runtime", "1.0.0"),
        ProviderLibrary::new("rayon", "1.12.0"),
    ];
    let solver = SolverProvider::new(
        BackendId::new("eqiora.test.solver"),
        "0.1.0",
        SOLVER_LIBRARIES,
    );
    let execution = ExecutionProvider::new(
        ExecutionId::new("eqiora.test.execution"),
        "0.2.0",
        EXECUTION_LIBRARIES,
    );
    let topology = ExecutionTopologyV1::Host {
        workers: NonZeroUsize::MIN,
    };

    let provenance = ExecutionProvenanceV1::from_provider_releases(
        solver,
        execution,
        topology.clone(),
        ReductionPolicy::Reproducible,
        [("common-runtime", "1.0.0"), ("native-runtime", "2.0.0")],
    )
    .unwrap();
    assert_eq!(provenance.adapter(), "eqiora.test.execution");
    assert_eq!(provenance.adapter_version(), "0.2.0");
    assert_eq!(provenance.solver_backend(), "eqiora.test.solver");
    assert_eq!(provenance.solver_backend_version(), "0.1.0");
    assert_eq!(provenance.libraries().len(), 3);
    assert_eq!(
        provenance
            .libraries()
            .get("common-runtime")
            .map(String::as_str),
        Some("1.0.0")
    );

    let conflict = ExecutionProvenanceV1::from_provider_releases(
        solver,
        execution,
        topology,
        ReductionPolicy::Reproducible,
        [("common-runtime", "9.9.9")],
    )
    .unwrap_err();
    assert_eq!(conflict.code(), codes::INVALID_ARTIFACT);
    assert!(conflict.message().contains("contradictory versions"));
}

#[test]
fn realization_v1_golden_bytes_are_frozen() {
    let fixture = include_bytes!(
        "../../../verify/artifacts/realization-run-wire/expected/realization-v1.json"
    );
    let fixture = fixture.strip_suffix(b"\n").unwrap_or(fixture);
    let decoded = RealizationEnvelopeV1::from_json(fixture, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), fixture);
}

#[test]
fn realization_v1_rejects_newer_solver_algorithms_without_retagging() {
    let (model_envelope, model) = model_fixture();
    let baseline = default_plan_v0().unwrap();
    for (algorithm, reduction, expected) in [
        (
            LinearSolver::MinimumResidual,
            ReductionPolicy::Reproducible,
            "realization artifact v1 cannot encode MINRES; a versioned wire extension is required",
        ),
        (
            LinearSolver::SparseLu,
            ReductionPolicy::Fast,
            "realization artifact v1 cannot encode sparse LU; a versioned wire extension is required",
        ),
    ] {
        let solver = SolverPlan::new(algorithm, 0.0, 1.0 / 1_073_741_824.0, NonZeroUsize::MIN)
            .unwrap()
            .with_reduction(reduction);
        let plan = RealizationPlan::new(
            baseline.space(),
            baseline.discretization(),
            solver,
            baseline.target(),
            baseline.schedule(),
        )
        .unwrap();
        let request = RealizationRequest::explicit(
            model,
            SemanticRevision::new(model_envelope.source_revision()),
            RealizationRevision::new(12),
            plan,
        );
        let requirements = RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        );
        let capabilities = RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                eqiora_realization::MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::MIN),
            )],
            [VectorLayoutKind::Replicated],
            SolverCapabilities::exact([SolverCapability {
                algorithm,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction,
                scalar_type: ScalarType::F64,
            }])
            .unwrap(),
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .unwrap();
        let resolved = resolve(&request, requirements, &capabilities).unwrap();
        let error = RealizationEnvelopeV1::from_resolved(
            &model_envelope,
            &resolved,
            LayoutArtifacts::Replicated,
        )
        .unwrap_err();

        assert_eq!(error.code(), codes::INVALID_ARTIFACT);
        assert_eq!(error.message(), expected);
    }
}

#[test]
fn distributed_layout_artifacts_and_loopback_topology_are_explicit() {
    let (model_envelope, model) = model_fixture();
    let plan = default_plan_v0().unwrap();
    let request = RealizationRequest::explicit(
        model,
        SemanticRevision::new(model_envelope.source_revision()),
        RealizationRevision::new(9),
        plan,
    );
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Distributed,
    );
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            eqiora_realization::MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Distributed],
        scalar_elliptic_solver_capabilities(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap();
    let resolved = resolve(&request, requirements, &capabilities).unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_envelope,
        &resolved,
        LayoutArtifacts::Distributed {
            layout: digest(4),
            partition: digest(5),
        },
    )
    .unwrap();
    let execution = ExecutionProvenanceV1::new(
        "eqiora.loopback",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Distributed {
            partitions: NonZeroUsize::new(4).unwrap(),
            workers_per_partition: NonZeroUsize::MIN,
            transport: DistributedTransportV1::Loopback,
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap();

    RunManifestV2::new(&realization, execution).unwrap();
    assert!(
        RealizationEnvelopeV1::from_resolved(
            &model_envelope,
            &resolved,
            LayoutArtifacts::Replicated,
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
    .expect("the scalar-elliptic artifact solver tuple is exact")
}

#[test]
fn run_v2_rejects_policy_and_topology_drift() {
    let (model_envelope, model) = model_fixture();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let resolved = resolve(
        &RealizationRequest::default(
            model,
            SemanticRevision::new(model_envelope.source_revision()),
            DefaultPolicyVersion::V0,
        ),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_envelope,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    let unrelated_source =
        POISSON.replace("model manufactured_poisson(", "model unrelated_poisson(");
    assert_ne!(unrelated_source, POISSON);
    let (unrelated_model, unrelated_id) = model_fixture_from_source(&unrelated_source);
    assert_ne!(unrelated_id, model);
    assert!(
        RealizationEnvelopeV1::from_resolved(
            &unrelated_model,
            &resolved,
            LayoutArtifacts::Replicated,
        )
        .is_err()
    );

    let fast = ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Fast,
    )
    .unwrap();
    assert!(RunManifestV2::new(&realization, fast).is_err());

    let wrong_workers = ExecutionProvenanceV1::new(
        "eqiora.rayon",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::new(2).unwrap(),
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap();
    assert!(RunManifestV2::new(&realization, wrong_workers).is_err());
}

#[test]
fn run_v2_rejects_untrusted_wire_ambiguity_and_resource_excess() {
    let (model_envelope, model) = model_fixture();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let resolved = resolve(
        &RealizationRequest::default(
            model,
            SemanticRevision::new(model_envelope.source_revision()),
            DefaultPolicyVersion::V0,
        ),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        &model_envelope,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    let execution = ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap();
    let bytes = RunManifestV2::new(&realization, execution)
        .unwrap()
        .with_output(digest(8))
        .canonical_json()
        .unwrap();

    let mut duplicate: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let output = duplicate["output_sha256"][0].clone();
    duplicate["output_sha256"]
        .as_array_mut()
        .unwrap()
        .push(output);
    assert!(
        RunManifestV2::from_json(&serde_json::to_vec(&duplicate).unwrap(), Default::default(),)
            .is_err()
    );

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["unexpected"] = serde_json::Value::Bool(true);
    assert!(
        RunManifestV2::from_json(&serde_json::to_vec(&unknown).unwrap(), Default::default(),)
            .is_err()
    );
    assert!(
        RunManifestV2::from_json(
            &bytes,
            JsonDecoderLimits {
                max_bytes: bytes.len() - 1,
                ..Default::default()
            },
        )
        .is_err()
    );
}

fn digest(byte: u8) -> ArtifactDigest {
    ArtifactDigest::from_hex(format!("{byte:02x}").repeat(32)).unwrap()
}

fn model_fixture() -> (ModelEnvelope, OntologyId<Model>) {
    model_fixture_from_source(POISSON)
}

fn model_fixture_from_source(source: &str) -> (ModelEnvelope, OntologyId<Model>) {
    let mut compiled = compile("poisson.eqi", source).unwrap();
    let compiled = compiled.remove(0);
    let model = compiled.model();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (ModelEnvelope::from_program(&program).unwrap(), model)
}
