use eqiora_core::ScalarType;
use std::num::NonZeroUsize;

use eqiora_artifact::{
    DistributedLayoutEnvelopeV1, DistributedTransportV1, ExecutionProvenanceV1,
    ExecutionTopologyV1, LayoutArtifacts, LinearSystemEnvelopeV1, ModelEnvelope,
    PartitionEnvelopeV1, RealizationEnvelopeV1, RunManifestV2, validate_distributed_content_dag,
};
use eqiora_compiler::compile;
use eqiora_distributed::{GlobalVectorSpace, Partition, PartitionId};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_realization::{
    DiscretizationMethod, RealizationCapabilities, RealizationRequest, RealizationRequirements,
    RealizationRevision, SemanticRevision, SpatialDimensionSupport, TargetCapabilities,
    VectorLayoutKind, default_plan_v0, resolve,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
};
use serde_json::Value;

const POISSON: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");

#[derive(Debug)]
struct Tridiagonal {
    right_hand_side: [f64; 3],
}

impl CompleteCsrStorage for Tridiagonal {
    fn rows(&self) -> usize {
        3
    }

    fn columns(&self) -> usize {
        3
    }

    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 5, 7]
    }

    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1, 2, 1, 2]
    }

    fn values(&self) -> &[f64] {
        &[2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

struct ContentDagFixture {
    model: ModelEnvelope,
    realization: RealizationEnvelopeV1,
    run: RunManifestV2,
    system: LinearSystemEnvelopeV1,
    partition: PartitionEnvelopeV1,
    layout: DistributedLayoutEnvelopeV1,
}

impl ContentDagFixture {
    fn new(run_partitions: usize) -> Self {
        let model = model();
        let system = system([1.0, 0.0, 1.0]);
        let partition = partition();
        let layout = DistributedLayoutEnvelopeV1::derive(&system, &partition).unwrap();
        let realization = realization(&model, &layout, &partition);
        let run = run(&realization, run_partitions);
        Self {
            model,
            realization,
            run,
            system,
            partition,
            layout,
        }
    }

    fn validate(
        &self,
    ) -> Result<eqiora_distributed::DistributedLinearSystem, eqiora_core::Diagnostic> {
        validate_distributed_content_dag(
            &self.model,
            &self.realization,
            &self.run,
            &self.system,
            &self.partition,
            &self.layout,
        )
    }
}

#[test]
fn exact_distributed_content_dag_reconstructs_one_validated_system() {
    let fixture = ContentDagFixture::new(2);
    let distributed = fixture.validate().unwrap();

    assert_eq!(distributed.partition().count().get(), 2);
    assert_eq!(distributed.partition().space().dimension().get(), 3);
    assert_eq!(distributed.complete_right_hand_side(), &[1.0, 0.0, 1.0]);
}

#[test]
fn content_dag_rejects_cross_wired_layout_and_system_content() {
    let mut fixture = ContentDagFixture::new(2);
    fixture.realization = mutate_realization(&fixture.realization, |wire| {
        wire["layout_artifacts"]["layout_sha256"] = Value::String("ab".repeat(32));
    });
    fixture.run = run(&fixture.realization, 2);
    assert!(fixture.validate().is_err());

    let mut fixture = ContentDagFixture::new(2);
    fixture.realization = mutate_realization(&fixture.realization, |wire| {
        wire["layout_artifacts"]["partition_sha256"] = Value::String("cd".repeat(32));
    });
    fixture.run = run(&fixture.realization, 2);
    assert!(fixture.validate().is_err());

    let fixture = ContentDagFixture::new(2);
    let other_system = system([2.0, 0.0, 1.0]);
    assert!(
        validate_distributed_content_dag(
            &fixture.model,
            &fixture.realization,
            &fixture.run,
            &other_system,
            &fixture.partition,
            &fixture.layout,
        )
        .is_err()
    );
}

#[test]
fn content_dag_rejects_linked_wrong_model_identity_and_revision() {
    let mut wrong_identity = ContentDagFixture::new(2);
    wrong_identity.realization = mutate_realization(&wrong_identity.realization, |wire| {
        wire["model_ulid"] = Value::String("01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned());
    });
    wrong_identity.run = run(&wrong_identity.realization, 2);
    assert!(wrong_identity.validate().is_err());

    let mut wrong_revision = ContentDagFixture::new(2);
    wrong_revision.model = mutate_model(&wrong_revision.model, |wire| {
        let revision = wire["source_revision"].as_u64().unwrap();
        wire["source_revision"] = Value::from(revision + 1);
    });
    assert_eq!(
        wrong_revision.model.digest().unwrap(),
        wrong_revision.realization.model_artifact(),
        "model content identity deliberately excludes source revision"
    );
    assert!(wrong_revision.validate().is_err());
}

#[test]
fn content_dag_rejects_run_partition_count_drift() {
    let fixture = ContentDagFixture::new(3);

    fixture
        .run
        .validate_against(&fixture.realization)
        .expect("run v2 validates target, worker, and reduction independently");
    assert!(fixture.validate().is_err());
}

fn model() -> ModelEnvelope {
    let mut compiled = compile("poisson.eqi", POISSON).unwrap();
    let compiled = compiled.remove(0);
    let model = compiled.model();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    ModelEnvelope::from_program(&program).unwrap()
}

fn system(right_hand_side: [f64; 3]) -> LinearSystemEnvelopeV1 {
    let complete = CanonicalCsrSystemView::new(
        &Tridiagonal { right_hand_side },
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    LinearSystemEnvelopeV1::from_complete(&complete).unwrap()
}

fn partition() -> PartitionEnvelopeV1 {
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
        [0, 1, 0].into_iter().map(PartitionId::new).collect(),
    )
    .unwrap();
    PartitionEnvelopeV1::from_partition(&partition).unwrap()
}

fn realization(
    model: &ModelEnvelope,
    layout: &DistributedLayoutEnvelopeV1,
    partition: &PartitionEnvelopeV1,
) -> RealizationEnvelopeV1 {
    let request = RealizationRequest::explicit(
        model.model().unwrap(),
        SemanticRevision::new(model.source_revision()),
        RealizationRevision::new(1),
        default_plan_v0().unwrap(),
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
    RealizationEnvelopeV1::from_resolved(
        model,
        &resolved,
        LayoutArtifacts::Distributed {
            layout: layout.digest().unwrap(),
            partition: partition.digest().unwrap(),
        },
    )
    .unwrap()
}

fn scalar_elliptic_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .expect("the distributed artifact solver tuple is exact")
}

fn run(realization: &RealizationEnvelopeV1, partitions: usize) -> RunManifestV2 {
    let execution = ExecutionProvenanceV1::new(
        "eqiora.loopback",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Distributed {
            partitions: NonZeroUsize::new(partitions).unwrap(),
            workers_per_partition: NonZeroUsize::MIN,
            transport: DistributedTransportV1::Loopback,
        },
        ReductionPolicy::Reproducible,
    )
    .unwrap();
    RunManifestV2::new(realization, execution).unwrap()
}

fn mutate_realization(
    realization: &RealizationEnvelopeV1,
    mutate: impl FnOnce(&mut Value),
) -> RealizationEnvelopeV1 {
    let mut wire: Value = serde_json::from_slice(&realization.canonical_json().unwrap()).unwrap();
    mutate(&mut wire);
    RealizationEnvelopeV1::from_json(&serde_json::to_vec(&wire).unwrap(), Default::default())
        .unwrap()
}

fn mutate_model(model: &ModelEnvelope, mutate: impl FnOnce(&mut Value)) -> ModelEnvelope {
    let mut wire: Value = serde_json::from_slice(&model.canonical_json().unwrap()).unwrap();
    mutate(&mut wire);
    ModelEnvelope::from_json(&serde_json::to_vec(&wire).unwrap(), Default::default()).unwrap()
}
