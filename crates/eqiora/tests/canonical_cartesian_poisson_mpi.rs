#![cfg(feature = "mpi")]

#[allow(dead_code)] // Backend-specific integration targets consume different shared helpers.
#[path = "support/canonical_cartesian_poisson.rs"]
mod canonical;

use eqiora_core::ScalarType;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use eqiora::artifact::{
    DistributedLayoutEnvelopeV1, DistributedTransportV1, ExecutionProvenanceV1,
    ExecutionTopologyV1, JsonDecoderLimits, LayoutArtifacts, LinearSystemEnvelopeV1,
    ModelDecoderLimits, ModelEnvelope, MpiThreadSupportV1, PartitionEnvelopeV1,
    RealizationEnvelopeV1, RunManifestV2, validate_distributed_content_dag,
};
use eqiora::backends::mpi::{
    MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER, MPI_EXECUTION,
    MPI_EXECUTION_PROVIDER, MpiAdmittedExecutionAdapter, MpiExecutionGroup, MpiThreadSupport,
};
use eqiora::distributed::{GlobalVectorSpace, Partition, PartitionId};
use eqiora::realization::{
    DiscretizationMethod, MeshKind, RealizationCapabilities, RealizationRequest,
    RealizationRequirements, ResolutionSource, SpatialDimensionSupport, TargetCapabilities,
    VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    ExecutionReport, LinearOperatorOrientation, LinearOperatorProperties, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
};
use eqiora_execution::{
    AdmittedExecution, DeploymentBinding, DistributedExecutorDescriptor, ExecutionReceipt,
    ExecutionStepKind, ProcessGroupSlot,
};
use eqiora_numerics::scalar::finalize_resolved_scalar_elliptic_cartesian;
use mpi::Threading;
use mpi::traits::CommunicatorCollectives;

const SOURCE: &str =
    include_str!("../../../verify/numerics/canonical-cartesian-poisson-mpi/models/poisson.eqi");
const CHILD_ENV: &str = "EQIORA_CANONICAL_POISSON_MPI_CHILD";
const MODEL_PATH_ENV: &str = "EQIORA_CANONICAL_POISSON_MODEL_PATH";
const CHILD_TEST: &str = "canonical_cartesian_poisson_mpi_child";
const CHILD_TIMEOUT: Duration = Duration::from_secs(180);
const CHILD_OUTPUT_LIMIT: usize = 64 * 1024;
const SERIAL_MPI_ABSOLUTE: f64 = 2.0e-12;
const SERIAL_MPI_RELATIVE: f64 = 2.0e-12;

#[test]
fn canonical_cartesian_poisson_mpi_runs_on_one_two_and_four_ranks() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }

    let program =
        canonical::compile_program_from_source("canonical-cartesian-poisson-mpi.eqi", SOURCE)
            .unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let model_bytes = model.canonical_json().unwrap();
    let limits = ModelDecoderLimits::default();
    assert!(model_bytes.len() <= limits.json.max_bytes);
    let decoded = ModelEnvelope::from_json(&model_bytes, limits).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), model_bytes);
    let shared_model = SharedModelFile::create(&model_bytes);

    for ranks in [1, 2, 4] {
        let output = run_mpi_child(ranks, shared_model.path());
        assert_success(ranks, output);
    }
}

#[test]
fn canonical_cartesian_poisson_mpi_child() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }

    let model_path = env::var_os(MODEL_PATH_ENV).expect("the parent supplies a Model artifact");
    let model_bytes = read_bounded(
        Path::new(&model_path),
        JsonDecoderLimits::default().max_bytes,
    );
    let model = ModelEnvelope::from_json(&model_bytes, Default::default()).unwrap();
    assert_eq!(model.canonical_json().unwrap(), model_bytes);
    let program = model
        .to_program()
        .unwrap_or_else(|diagnostics| panic!("shared Model artifact is invalid: {diagnostics:?}"));

    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the child application initializes MPI exactly once");
    let world = universe.world();
    let mut group =
        MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    let partitions = group.partitions();
    let capabilities = distributed_capabilities(group.solver_capabilities());
    let context = CanonicalMpiContext {
        program: &program,
        model: &model,
        capabilities: &capabilities,
    };

    for (revision, method, _) in canonical::METHODS {
        execute_method(context, &world, &mut group, revision, method);
    }

    assert_eq!(partitions, group.partitions());
    world.barrier();
    drop(group);
}

#[derive(Clone, Copy)]
struct CanonicalMpiContext<'a> {
    program: &'a KernelProgram,
    model: &'a ModelEnvelope,
    capabilities: &'a RealizationCapabilities,
}

fn execute_method(
    context: CanonicalMpiContext<'_>,
    world: &impl CommunicatorCollectives,
    group: &mut MpiExecutionGroup,
    revision: u64,
    method: DiscretizationMethod,
) {
    let CanonicalMpiContext {
        program,
        model,
        capabilities,
    } = context;
    let request = canonical::host_request(program, method, revision).unwrap();
    let requirements = distributed_requirements();
    let resolved = resolve(&request, requirements, capabilities).unwrap();
    let (_, recorded_problem) =
        finalize_resolved_scalar_elliptic_cartesian(program, &resolved).unwrap();
    assert_eq!(
        recorded_problem.vector_layout(),
        VectorLayoutKind::Distributed
    );
    assert_eq!(
        recorded_problem.solver_plan().reduction(),
        ReductionPolicy::Reproducible
    );

    let system =
        LinearSystemEnvelopeV1::from_complete(recorded_problem.canonical_csr_system_view())
            .unwrap();
    let partition = rotated_cyclic_partition(&system, group.partitions());
    let layout = DistributedLayoutEnvelopeV1::derive(&system, &partition).unwrap();
    let realization = RealizationEnvelopeV1::from_resolved(
        model,
        &resolved,
        LayoutArtifacts::Distributed {
            layout: layout.digest().unwrap(),
            partition: partition.digest().unwrap(),
        },
    )
    .unwrap();
    let decoded_model = round_trip_model(model);
    let decoded_system = round_trip_system(&system);
    let decoded_partition = round_trip_partition(&partition);
    let decoded_layout = round_trip_layout(&layout);
    let decoded_realization = round_trip_realization(&realization);
    assert_artifact_agreement(
        world,
        [
            decoded_model.digest().unwrap(),
            decoded_realization.digest().unwrap(),
            decoded_system.digest().unwrap(),
            decoded_partition.digest().unwrap(),
            decoded_layout.digest().unwrap(),
        ],
    );

    let replay_request = request_from_artifact(&decoded_realization);
    let replayed = resolve(
        &replay_request,
        decoded_realization.requirements().unwrap(),
        capabilities,
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let (_, replayed_problem) =
        finalize_resolved_scalar_elliptic_cartesian(program, &replayed).unwrap();
    require_semantic_replay(&decoded_system, &replayed_problem).unwrap();

    let distributed = decoded_layout
        .validate_against(&decoded_system, &decoded_partition)
        .unwrap();
    let complete = replayed_problem.canonical_csr_system_view();
    let binding = DeploymentBinding::bind_distributed(
        replayed_problem.portable_realization(),
        DistributedExecutorDescriptor::new(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            ProcessGroupSlot::new(0),
            group.partitions(),
            NonZeroUsize::MIN,
            group.solver_capabilities(),
        ),
    )
    .unwrap();
    let expected_fingerprint = distributed
        .admission_fingerprint(binding.solver_plan())
        .unwrap();
    let admitted = AdmittedExecution::admit_distributed_linear(
        replayed_problem.portable_realization(),
        &distributed,
        complete,
        binding,
    )
    .unwrap();
    assert_eq!(admitted.distributed_admission(), Some(expected_fingerprint));
    let accepted = group.execute_admitted(admitted).unwrap();
    let execution = observed_execution(group, accepted.receipt());
    let run = RunManifestV2::new(&decoded_realization, execution.clone()).unwrap();
    let decoded_run = round_trip_run(&run);
    let validated = validate_distributed_content_dag(
        &decoded_model,
        &decoded_realization,
        &decoded_run,
        &decoded_system,
        &decoded_partition,
        &decoded_layout,
    )
    .unwrap();
    assert_eq!(validated, distributed);
    assert_artifact_agreement(world, [decoded_run.digest().unwrap()]);
    let (linear_solution, receipt) = accepted.into_parts();
    assert_distributed_receipt(world, &receipt, expected_fingerprint);
    assert_solve_report(
        linear_solution.report(),
        replayed_problem.solver_plan(),
        group.partitions(),
    );
    let solution = replayed_problem.finish(linear_solution).unwrap();

    canonical::method_metrics(method, &solution).unwrap();
    let reference = canonical::reference_cpu_solution(program, method, revision + 100).unwrap();
    canonical::reference_conformance(
        &reference,
        &solution,
        SERIAL_MPI_ABSOLUTE,
        SERIAL_MPI_RELATIVE,
        "serial/MPI",
    )
    .unwrap();

    if method == DiscretizationMethod::ContinuousGalerkin {
        assert_content_linkage_does_not_claim_semantic_derivation(
            model,
            &resolved,
            &execution,
            &system,
            &partition,
            &recorded_problem,
        );
    }
}

fn assert_distributed_receipt(
    world: &impl CommunicatorCollectives,
    receipt: &ExecutionReceipt,
    admission: eqiora::distributed::DistributedAdmissionFingerprintV1,
) {
    let trace = receipt
        .distributed_trace()
        .expect("the graph-bound MPI run retains its distributed trace");
    assert_eq!(trace.system(), receipt.operator());
    assert_eq!(trace.admission(), admission);
    assert_eq!(trace.owner_gather_dimension(), receipt.dimension());
    assert_eq!(
        trace.partitions().get(),
        usize::try_from(world.size()).unwrap()
    );
    assert_eq!(trace.workers_per_partition(), NonZeroUsize::MIN);
    assert!(!trace.steps().is_empty());
    assert!(trace.steps().len() <= trace.trace_capacity());
    assert!(
        trace
            .steps()
            .iter()
            .enumerate()
            .all(|(ordinal, step)| step.ordinal() == ordinal)
    );
    assert_eq!(
        receipt.dag().steps(),
        &[
            ExecutionStepKind::AgreeDistributedAdmission,
            ExecutionStepKind::SolveDistributedKrylov,
            ExecutionStepKind::AgreeDistributedProducerReport,
            ExecutionStepKind::GatherDistributedOwnedCandidate,
            ExecutionStepKind::AcceptWithNativeHostVerification,
            ExecutionStepKind::AgreeDistributedAcceptedResult,
            ExecutionStepKind::ReplayTrueResidualOnHost,
            ExecutionStepKind::AgreeDistributedReceipt,
            ExecutionStepKind::AcceptHostComplete,
        ]
    );

    let local = [
        receipt.operator().as_bytes(),
        receipt.output().as_bytes(),
        trace.partition().as_bytes(),
        trace.layout().as_bytes(),
        trace.admission().as_bytes(),
    ]
    .concat();
    let ranks = usize::try_from(world.size()).unwrap();
    let mut gathered = vec![0_u8; local.len() * ranks];
    world.all_gather_into(&local[..], &mut gathered[..]);
    assert!(
        gathered
            .chunks_exact(local.len())
            .all(|candidate| candidate == local)
    );
}

fn assert_artifact_agreement<const N: usize>(
    world: &impl CommunicatorCollectives,
    digests: [eqiora::artifact::ArtifactDigest; N],
) {
    const DIGEST_BYTES: usize = 64;
    let block_bytes = DIGEST_BYTES * N;

    let mut local = vec![0_u8; block_bytes];
    for (slot, digest) in local
        .as_chunks_mut::<DIGEST_BYTES>()
        .0
        .iter_mut()
        .zip(digests)
    {
        slot.copy_from_slice(digest.as_str().as_bytes());
    }
    let ranks = usize::try_from(world.size()).expect("MPI size is a nonnegative usize");
    let mut gathered = vec![0_u8; ranks * block_bytes];
    world.all_gather_into(&local[..], &mut gathered[..]);
    assert!(
        gathered
            .chunks_exact(block_bytes)
            .all(|block| block == &local[..])
    );
}

fn distributed_requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(2).expect("two is nonzero"),
        ScalarType::F64,
        VectorLayoutKind::Distributed,
    )
}

fn distributed_capabilities(solver: eqiora::solver::SolverCapabilities) -> RealizationCapabilities {
    let plan = canonical::solver_plan(ReductionPolicy::Reproducible);
    solver
        .require_problem(
            plan,
            ScalarType::F64,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .expect("the MPI group implements the exact Poisson solver tuple");
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: plan.algorithm(),
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: plan.preconditioner(),
        reduction: plan.reduction(),
        scalar_type: ScalarType::F64,
    }])
    .expect("the selected MPI Poisson solver tuple is exact");
    RealizationCapabilities::cartesian_product(
        [
            DiscretizationMethod::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume,
        ],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is nonzero")),
        )],
        [VectorLayoutKind::Distributed],
        solver,
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .expect("the MPI vertical capability axes are exact and nonempty")
}

fn rotated_cyclic_partition(
    system: &LinearSystemEnvelopeV1,
    partitions: NonZeroUsize,
) -> PartitionEnvelopeV1 {
    let complete = system.to_complete().unwrap();
    let dimension = NonZeroUsize::new(complete.rows()).expect("Cartesian algebra is nonempty");
    let owners = (0..dimension.get())
        .map(|global| PartitionId::new((global + 1) % partitions.get()))
        .collect();
    let partition = Partition::new(
        GlobalVectorSpace::new(dimension, ScalarType::F64),
        partitions,
        owners,
    )
    .unwrap();
    PartitionEnvelopeV1::from_partition(&partition).unwrap()
}

fn request_from_artifact(realization: &RealizationEnvelopeV1) -> RealizationRequest {
    let ResolutionSource::Explicit(revision) = realization.source() else {
        panic!("the evidence fixture uses an explicit Realization revision");
    };
    RealizationRequest::explicit(
        realization.model().unwrap(),
        realization.semantic_revision(),
        revision,
        realization.plan().unwrap(),
    )
}

fn require_semantic_replay(
    recorded: &LinearSystemEnvelopeV1,
    fresh: &eqiora_numerics::scalar::FinalizedScalarEllipticCartesianProblem,
) -> Result<(), String> {
    let fresh = LinearSystemEnvelopeV1::from_complete(fresh.canonical_csr_system_view())
        .map_err(|diagnostic| diagnostic.to_string())?;
    let recorded_bytes = recorded
        .canonical_json()
        .map_err(|diagnostic| diagnostic.to_string())?;
    let fresh_bytes = fresh
        .canonical_json()
        .map_err(|diagnostic| diagnostic.to_string())?;
    let recorded_digest = recorded
        .digest()
        .map_err(|diagnostic| diagnostic.to_string())?;
    let fresh_digest = fresh
        .digest()
        .map_err(|diagnostic| diagnostic.to_string())?;
    if recorded_digest != fresh_digest || recorded_bytes != fresh_bytes {
        return Err("recorded complete system differs from fresh semantic derivation".to_owned());
    }
    Ok(())
}

fn assert_content_linkage_does_not_claim_semantic_derivation(
    model: &ModelEnvelope,
    resolved: &eqiora::realization::ResolvedRealization,
    execution: &ExecutionProvenanceV1,
    original: &LinearSystemEnvelopeV1,
    partition: &PartitionEnvelopeV1,
    fresh_problem: &eqiora_numerics::scalar::FinalizedScalarEllipticCartesianProblem,
) {
    let mut wire: serde_json::Value =
        serde_json::from_slice(&original.canonical_json().unwrap()).unwrap();
    let right_hand_side = wire["right_hand_side"]
        .as_array_mut()
        .expect("the closed system DTO has an RHS array");
    let first = right_hand_side[0]
        .as_f64()
        .expect("the closed system DTO has finite f64 RHS values");
    right_hand_side[0] = serde_json::Value::from(first + 0.25);
    let changed =
        LinearSystemEnvelopeV1::from_json(&serde_json::to_vec(&wire).unwrap(), Default::default())
            .unwrap();
    assert_ne!(changed.digest().unwrap(), original.digest().unwrap());

    let changed_layout = DistributedLayoutEnvelopeV1::derive(&changed, partition).unwrap();
    let changed_realization = RealizationEnvelopeV1::from_resolved(
        model,
        resolved,
        LayoutArtifacts::Distributed {
            layout: changed_layout.digest().unwrap(),
            partition: partition.digest().unwrap(),
        },
    )
    .unwrap();
    let changed_run = RunManifestV2::new(&changed_realization, execution.clone()).unwrap();
    validate_distributed_content_dag(
        model,
        &changed_realization,
        &changed_run,
        &changed,
        partition,
        &changed_layout,
    )
    .expect("the changed system is an honestly linked but semantically foreign content DAG");
    assert_eq!(
        require_semantic_replay(&changed, fresh_problem).unwrap_err(),
        "recorded complete system differs from fresh semantic derivation"
    );
}

fn assert_solve_report(
    report: &eqiora::solver::SolveReport,
    plan: eqiora::solver::SolverPlan,
    partitions: NonZeroUsize,
) {
    assert_eq!(report.backend(), MPI_DISTRIBUTED_KRYLOV_BACKEND);
    assert_eq!(
        report.execution(),
        ExecutionReport::distributed(MPI_EXECUTION, partitions)
    );
    assert_eq!(report.verification(), ExecutionReport::host_serial());
    assert_eq!(report.orientation(), LinearOperatorOrientation::Normal);
    assert_eq!(report.solver_plan(), plan);
    assert_eq!(report.algorithm(), LinearSolver::ConjugateGradient);
    assert_eq!(report.preconditioner(), PreconditionerPolicy::Jacobi);
    assert_eq!(report.reduction(), ReductionPolicy::Reproducible);
}

fn observed_execution(
    group: &MpiExecutionGroup,
    receipt: &ExecutionReceipt,
) -> ExecutionProvenanceV1 {
    let raw_library = mpi::environment::library_version()
        .expect("MPI implementation reports a UTF-8 library version");
    let implementation = mpi_implementation(&raw_library);
    let version = normalize_mpi_library_version(&raw_library);
    assert!(!version.is_empty());
    let (standard_major, standard_minor) = mpi::environment::version();
    ExecutionProvenanceV1::from_provider_releases(
        receipt.solver_provider(),
        receipt.execution_provider(),
        ExecutionTopologyV1::Distributed {
            partitions: group.partitions(),
            workers_per_partition: NonZeroUsize::MIN,
            transport: DistributedTransportV1::Mpi {
                implementation: implementation.to_owned(),
                version,
                thread_support: artifact_thread_support(group.thread_support()),
            },
        },
        ReductionPolicy::Reproducible,
        [("mpi-standard", format!("{standard_major}.{standard_minor}"))],
    )
    .unwrap()
}

fn normalize_mpi_library_version(value: &str) -> String {
    value
        .split(|character: char| character.is_whitespace() || character.is_control())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn mpi_implementation(library_version: &str) -> &'static str {
    let lower = library_version.to_ascii_lowercase();
    if lower.contains("open mpi") || lower.contains("open-mpi") {
        "openmpi"
    } else if lower.contains("mpich") {
        "mpich"
    } else {
        "system-mpi"
    }
}

#[test]
fn mpi_library_version_normalization_removes_transport_controls() {
    let raw = "Open MPI v4.1.4, package: runner\nOpen MPI: 4.1.4\t\0\0";
    assert_eq!(
        normalize_mpi_library_version(raw),
        "Open MPI v4.1.4, package: runner Open MPI: 4.1.4"
    );
}

const fn artifact_thread_support(value: MpiThreadSupport) -> MpiThreadSupportV1 {
    match value {
        MpiThreadSupport::Single => MpiThreadSupportV1::Single,
        MpiThreadSupport::Funneled => MpiThreadSupportV1::Funneled,
        MpiThreadSupport::Serialized => MpiThreadSupportV1::Serialized,
        MpiThreadSupport::Multiple => MpiThreadSupportV1::Multiple,
    }
}

fn round_trip_model(value: &ModelEnvelope) -> ModelEnvelope {
    let bytes = value.canonical_json().unwrap();
    let decoded = ModelEnvelope::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

fn round_trip_system(value: &LinearSystemEnvelopeV1) -> LinearSystemEnvelopeV1 {
    let bytes = value.canonical_json().unwrap();
    let decoded = LinearSystemEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

fn round_trip_partition(value: &PartitionEnvelopeV1) -> PartitionEnvelopeV1 {
    let bytes = value.canonical_json().unwrap();
    let decoded = PartitionEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

fn round_trip_layout(value: &DistributedLayoutEnvelopeV1) -> DistributedLayoutEnvelopeV1 {
    let bytes = value.canonical_json().unwrap();
    let decoded = DistributedLayoutEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

fn round_trip_realization(value: &RealizationEnvelopeV1) -> RealizationEnvelopeV1 {
    let bytes = value.canonical_json().unwrap();
    let decoded = RealizationEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

fn round_trip_run(value: &RunManifestV2) -> RunManifestV2 {
    let bytes = value.canonical_json().unwrap();
    let decoded = RunManifestV2::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), value.digest().unwrap());
    decoded
}

struct SharedModelFile {
    path: PathBuf,
}

impl SharedModelFile {
    fn create(bytes: &[u8]) -> Self {
        let digest = ModelEnvelope::from_json(bytes, Default::default())
            .unwrap()
            .digest()
            .unwrap();
        for nonce in 0..16_u8 {
            let path = env::temp_dir().join(format!(
                "eqiora-canonical-poisson-mpi-{}-{}-{nonce}.json",
                std::process::id(),
                digest.as_str()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    file.write_all(bytes).unwrap();
                    file.sync_all().unwrap();
                    return Self { path };
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create shared Model artifact: {error}"),
            }
        }
        panic!("cannot allocate a unique shared Model artifact path");
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SharedModelFile {
    fn drop(&mut self) {
        fs::remove_file(&self.path).expect("the parent removes its exact temporary Model artifact");
    }
}

fn read_bounded(path: &Path, maximum: usize) -> Vec<u8> {
    let metadata = fs::metadata(path).expect("shared Model artifact metadata is readable");
    assert!(metadata.len() <= u64::try_from(maximum).unwrap());
    let mut bytes = Vec::new();
    File::open(path)
        .unwrap()
        .take(u64::try_from(maximum).unwrap() + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= maximum);
    bytes
}

struct ChildOutput {
    status: ExitStatus,
    stdout: BoundedOutput,
    stderr: BoundedOutput,
}

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn run_mpi_child(ranks: usize, model_path: &Path) -> ChildOutput {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    let mut child = command
        .args(["-n", &ranks.to_string()])
        .arg(executable)
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_ENV, "1")
        .env(MODEL_PATH_ENV, model_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("registered MPI evidence requires mpirun on PATH");
    let stdout = child.stdout.take().expect("MPI child stdout is captured");
    let stderr = child.stderr.take().expect("MPI child stderr is captured");
    let stdout_reader = thread::spawn(move || drain_bounded(stdout, CHILD_OUTPUT_LIMIT));
    let stderr_reader = thread::spawn(move || drain_bounded(stderr, CHILD_OUTPUT_LIMIT));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("MPI child status is readable") {
            break (status, false);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            break (
                child.wait().expect("the killed MPI launcher is reaped"),
                true,
            );
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .expect("MPI stdout reader does not panic")
        .expect("MPI child stdout remains readable");
    let stderr = stderr_reader
        .join()
        .expect("MPI stderr reader does not panic")
        .expect("MPI child stderr remains readable");
    if timed_out {
        panic!(
            "{ranks}-rank canonical MPI child exceeded {CHILD_TIMEOUT:?}\nstdout{}:\n{}\nstderr{}:\n{}",
            truncation_marker(&stdout),
            String::from_utf8_lossy(&stdout.bytes),
            truncation_marker(&stderr),
            String::from_utf8_lossy(&stderr.bytes),
        );
    }
    ChildOutput {
        status,
        stdout,
        stderr,
    }
}

fn launcher_accepts_oversubscribe(launcher: &std::ffi::OsStr) -> bool {
    Command::new(launcher)
        .args(["--oversubscribe", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn assert_success(ranks: usize, output: ChildOutput) {
    assert!(
        output.status.success(),
        "{ranks}-rank canonical MPI child failed\nstdout{}:\n{}\nstderr{}:\n{}",
        truncation_marker(&output.stdout),
        String::from_utf8_lossy(&output.stdout.bytes),
        truncation_marker(&output.stderr),
        String::from_utf8_lossy(&output.stderr.bytes),
    );
}

fn drain_bounded(mut reader: impl Read, maximum: usize) -> std::io::Result<BoundedOutput> {
    let mut bytes = Vec::with_capacity(maximum);
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let retained = count.min(maximum.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained != count;
    }
    Ok(BoundedOutput { bytes, truncated })
}

fn truncation_marker(output: &BoundedOutput) -> &'static str {
    if output.truncated { " (truncated)" } else { "" }
}
