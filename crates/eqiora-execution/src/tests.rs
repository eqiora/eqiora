use std::cell::Cell;
use std::num::{NonZeroU64, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Id, OntologyId, ScalarType};
use eqiora_device::{
    BufferId, Completion, DeviceBufferDescriptor, DeviceCapability, DeviceDescriptor, DeviceId,
    Fence, HostBufferDescriptor, MemoryRegion, QueueId, QueueSlot, QueueTimeline, RuntimeId,
    SparseActionPolicy, TransferEvidence, TransferPlan, WaitedCompletion,
};
use eqiora_distributed::{DistributedLinearSystem, GlobalVectorSpace, Partition, PartitionId};
use eqiora_realization::{
    DefaultPolicyVersion, DiscretizationMethod, MeshKind, RealizationCapabilities, RealizationPlan,
    RealizationRequest, RealizationRequirements, RealizationRevision, SemanticRevision,
    SpatialDimensionSupport, Target, TargetCapabilities, VectorLayoutKind, default_plan_v0,
    resolve,
};
use eqiora_solver::{
    BackendId, CanonicalCsrSystemView, CompleteCsrStorage, ConvergenceReason, ExecutionId,
    ExecutionProvider, ExecutionReport, FixedOrderInnerProduct, LinearOperator,
    LinearOperatorProperties, LinearSolver, LinearSolverBackend, PreconditionerPolicy,
    ProviderLibrary, REFERENCE_LINEAR_SOLVER, ReductionPolicy, ReplicatedLinearExecution,
    SERIAL_LINEAR_EXECUTION, SolverCapabilities, SolverCapability, SolverPlan, SolverProvider,
    accept_linear_solution_with_verifier,
};

use super::binding::{DistributedExecutorDescriptor, ProcessGroupSlot};
use super::*;

mod cuda_validation;

const TEST_PROVIDER_VERSION: &str = "0.1.0-test";
const SUBSTITUTED_LIBRARIES: &[ProviderLibrary] =
    &[ProviderLibrary::new("eqiora-test-kernel", "9.9.9")];

const fn solver_provider(id: BackendId) -> SolverProvider {
    SolverProvider::new(id, TEST_PROVIDER_VERSION, &[])
}

const fn execution_provider(id: ExecutionId) -> ExecutionProvider {
    ExecutionProvider::new(id, TEST_PROVIDER_VERSION, &[])
}

fn reference_solver_provider() -> SolverProvider {
    REFERENCE_LINEAR_SOLVER.provider()
}

fn serial_execution_provider() -> ExecutionProvider {
    SERIAL_LINEAR_EXECUTION.provider()
}

#[derive(Debug, Clone, Copy)]
struct SubstitutedSerialVerifier(ExecutionProvider);

impl ReplicatedLinearExecution for SubstitutedSerialVerifier {
    fn provider(&self) -> ExecutionProvider {
        self.0
    }

    fn report(&self) -> ExecutionReport {
        ExecutionReport::host_serial()
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), eqiora_core::Diagnostic> {
        SERIAL_LINEAR_EXECUTION.require_reduction(policy)
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), eqiora_core::Diagnostic> {
        SERIAL_LINEAR_EXECUTION.apply(operator, input, output)
    }

    fn inner_product(
        &self,
        action: FixedOrderInnerProduct<'_>,
    ) -> Result<f64, eqiora_core::Diagnostic> {
        SERIAL_LINEAR_EXECUTION.inner_product(action)
    }
}

#[derive(Debug, Clone, Copy)]
struct TwoByTwo {
    right_hand_side: [f64; 2],
}

impl CompleteCsrStorage for TwoByTwo {
    fn rows(&self) -> usize {
        2
    }

    fn columns(&self) -> usize {
        2
    }

    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }

    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }

    fn values(&self) -> &[f64] {
        &[2.0, -1.0, -1.0, 2.0]
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

#[derive(Debug, Clone, Copy)]
struct TwoByTwoIndefinite {
    right_hand_side: [f64; 2],
}

impl CompleteCsrStorage for TwoByTwoIndefinite {
    fn rows(&self) -> usize {
        2
    }

    fn columns(&self) -> usize {
        2
    }

    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }

    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }

    fn values(&self) -> &[f64] {
        &[1.0, 1.0, 1.0, -1.0]
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

fn portable_graph_with_workers(
    workers: NonZeroUsize,
) -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let request = if workers == NonZeroUsize::MIN {
        RealizationRequest::default(model, semantic_revision, DefaultPolicyVersion::V0)
    } else {
        let default = default_plan_v0().unwrap();
        let plan = RealizationPlan::new(
            default.space(),
            default.discretization(),
            default.solver(),
            Target::HostCpu { threads: workers },
            default.schedule(),
        )
        .unwrap();
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(2), plan)
    };
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        reference_spd_solver_capabilities(),
        TargetCapabilities::none().with_host_cpu(workers),
    )
    .unwrap();
    let resolved = resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .unwrap();
    resolved
        .portable_graph(
            Id::new(),
            Id::new(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
}

fn reference_spd_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .expect("the reference SPD execution tuple is exact")
}

fn portable_graph() -> eqiora_realization::PortableRealizationGraph {
    portable_graph_with_workers(NonZeroUsize::MIN)
}

const TEST_CUDA_RUNTIME: RuntimeId = RuntimeId::new("eqiora.test.cuda");
const TEST_CUDA_BACKEND: BackendId = BackendId::new("eqiora.test.cuda.solver");
const TEST_CUDA_EXECUTION: ExecutionId = ExecutionId::new("eqiora.test.cuda.queue");

fn cuda_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::new(
        [LinearSolver::ConjugateGradient],
        [PreconditionerPolicy::Jacobi],
        [ReductionPolicy::Fast],
        [ScalarType::F64],
    )
    .unwrap()
}

fn cuda_minres_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::MinimumResidual,
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Fast,
        scalar_type: ScalarType::F64,
    }])
    .unwrap()
}

fn portable_cuda_graph() -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let default = default_plan_v0().unwrap();
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        default
            .solver()
            .with_preconditioner(PreconditionerPolicy::Jacobi)
            .with_reduction(ReductionPolicy::Fast),
        Target::CudaGpu { device: 0 },
        default.schedule(),
    )
    .unwrap();
    let request =
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(3), plan);
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        cuda_solver_capabilities(),
        TargetCapabilities::none().with_cuda_device(0),
    )
    .unwrap();
    let resolved = resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .unwrap();
    resolved
        .portable_graph(
            Id::new(),
            Id::new(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
}

fn portable_cuda_minres_graph() -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let default = default_plan_v0().unwrap();
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(64).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        solver,
        Target::CudaGpu { device: 0 },
        default.schedule(),
    )
    .unwrap();
    let request =
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(5), plan);
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        cuda_minres_solver_capabilities(),
        TargetCapabilities::none().with_cuda_device(0),
    )
    .unwrap();
    let resolved = resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .unwrap();
    resolved
        .portable_graph(
            Id::new(),
            Id::new(),
            LinearOperatorProperties::SymmetricIndefinite,
        )
        .unwrap()
}

const TEST_DISTRIBUTED_BACKEND: BackendId = BackendId::new("eqiora.test.distributed.cg");
const TEST_DISTRIBUTED_EXECUTION: ExecutionId = ExecutionId::new("eqiora.test.distributed");

fn distributed_solver_capabilities(reduction: ReductionPolicy) -> SolverCapabilities {
    SolverCapabilities::new(
        [LinearSolver::ConjugateGradient],
        [PreconditionerPolicy::Jacobi],
        [reduction],
        [ScalarType::F64],
    )
    .unwrap()
}

fn distributed_minres_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::MinimumResidual,
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .unwrap()
}

fn portable_distributed_graph_with(
    workers_per_partition: NonZeroUsize,
    reduction: ReductionPolicy,
) -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let default = default_plan_v0().unwrap();
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        default
            .solver()
            .with_preconditioner(PreconditionerPolicy::Jacobi)
            .with_reduction(reduction),
        Target::HostCpu {
            threads: workers_per_partition,
        },
        default.schedule(),
    )
    .unwrap();
    let request =
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(4), plan);
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Distributed],
        distributed_solver_capabilities(reduction),
        TargetCapabilities::none().with_host_cpu(workers_per_partition),
    )
    .unwrap();
    let resolved = resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Distributed,
        ),
        &capabilities,
    )
    .unwrap();
    resolved
        .portable_graph(
            Id::new(),
            Id::new(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
}

fn portable_distributed_graph() -> eqiora_realization::PortableRealizationGraph {
    portable_distributed_graph_with(NonZeroUsize::MIN, ReductionPolicy::Reproducible)
}

fn portable_distributed_minres_graph() -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let default = default_plan_v0().unwrap();
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        default.schedule(),
    )
    .unwrap();
    let request =
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(5), plan);
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Distributed],
        distributed_minres_solver_capabilities(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap();
    resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Distributed,
        ),
        &capabilities,
    )
    .unwrap()
    .portable_graph(
        Id::new(),
        Id::new(),
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap()
}

fn portable_distributed_cuda_minres_graph() -> eqiora_realization::PortableRealizationGraph {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let default = default_plan_v0().unwrap();
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        solver,
        Target::CudaGpu { device: 0 },
        default.schedule(),
    )
    .unwrap();
    let request =
        RealizationRequest::explicit(model, semantic_revision, RealizationRevision::new(6), plan);
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Distributed],
        distributed_minres_solver_capabilities(),
        TargetCapabilities::none().with_cuda_device(0),
    )
    .unwrap();
    resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Distributed,
        ),
        &capabilities,
    )
    .unwrap()
    .portable_graph(
        Id::new(),
        Id::new(),
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap()
}

fn distributed_system(
    complete: &CanonicalCsrSystemView,
    partitions: NonZeroUsize,
) -> DistributedLinearSystem {
    let dimension = NonZeroUsize::new(complete.rows()).unwrap();
    let owners = (0..dimension.get())
        .map(|global| PartitionId::new(global % partitions.get()))
        .collect();
    DistributedLinearSystem::from_complete(
        complete,
        Partition::new(
            GlobalVectorSpace::new(dimension, ScalarType::F64),
            partitions,
            owners,
        )
        .unwrap(),
    )
    .unwrap()
}

fn distributed_binding(
    graph: &eqiora_realization::PortableRealizationGraph,
    partitions: NonZeroUsize,
) -> DeploymentBinding {
    DeploymentBinding::bind_distributed(
        graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            distributed_solver_capabilities(ReductionPolicy::Reproducible),
        ),
    )
    .unwrap()
}

fn distributed_cuda_binding(
    graph: &eqiora_realization::PortableRealizationGraph,
    partitions: NonZeroUsize,
    device: DeviceDescriptor,
) -> DeploymentBinding {
    let queue = QueueSlot::new(device.id(), 0);
    DeploymentBinding::bind_distributed_cuda(
        graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            distributed_minres_solver_capabilities(),
        ),
        CudaPartitionPlacement::new(
            execution_provider(TEST_CUDA_EXECUTION),
            device,
            queue,
            SparseActionPolicy::Deterministic,
        )
        .unwrap(),
        DistributedDeviceTransport::HostStaged,
    )
    .unwrap()
}

fn distributed_trace(
    system: &DistributedLinearSystem,
    complete: &CanonicalCsrSystemView,
    plan: SolverPlan,
) -> DistributedLinearExecutionTrace {
    use DistributedExecutionPhaseV1::{
        AcceptedResultAgreement, AdmissionAgreement, CollectiveReduction, HaloReadiness,
        NativeHostAcceptance, OwnedAction, OwnedVectorUpdate, OwnerGatherPreparation,
        OwnerGatherValidation, ProducerReportAgreement,
    };

    let phases = [
        AdmissionAgreement,
        HaloReadiness,
        OwnedAction,
        CollectiveReduction,
        OwnedVectorUpdate,
        ProducerReportAgreement,
        OwnerGatherPreparation,
        OwnerGatherValidation,
        NativeHostAcceptance,
        AcceptedResultAgreement,
    ];
    let steps = phases
        .into_iter()
        .enumerate()
        .map(|(ordinal, phase)| DistributedCollectiveStepV1::new(phase, 0, ordinal))
        .collect();
    DistributedLinearExecutionTrace::new(
        system.system_identity(),
        system.partition_identity(),
        system.layout_identity(),
        system.admission_fingerprint(plan).unwrap(),
        ProcessGroupSlot::new(0),
        system.partition().count(),
        NonZeroUsize::MIN,
        complete.columns(),
        DistributedLinearExecutionTrace::collective_capacity(plan).unwrap(),
        steps,
        plan,
    )
    .unwrap()
}

fn cuda_device(total_memory_bytes: u64) -> DeviceDescriptor {
    DeviceDescriptor::new(
        DeviceId::new(TEST_CUDA_RUNTIME, 0),
        "test device",
        NonZeroU64::new(total_memory_bytes).unwrap(),
        [
            DeviceCapability::Float64,
            DeviceCapability::CsrMatrixVectorProduct,
            DeviceCapability::DenseVectorLevel1,
            DeviceCapability::AsynchronousQueue,
        ],
    )
    .unwrap()
}

fn cuda_binding(
    graph: &eqiora_realization::PortableRealizationGraph,
    device: DeviceDescriptor,
) -> DeploymentBinding {
    let queue = QueueSlot::new(device.id(), 0);
    DeploymentBinding::bind_cuda(
        graph,
        CudaExecutorDescriptor::new(
            solver_provider(TEST_CUDA_BACKEND),
            execution_provider(TEST_CUDA_EXECUTION),
            device,
            queue,
            cuda_solver_capabilities(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn cuda_minres_binding(
    graph: &eqiora_realization::PortableRealizationGraph,
    device: DeviceDescriptor,
) -> DeploymentBinding {
    let queue = QueueSlot::new(device.id(), 0);
    DeploymentBinding::bind_cuda(
        graph,
        CudaExecutorDescriptor::new(
            solver_provider(TEST_CUDA_BACKEND),
            execution_provider(TEST_CUDA_EXECUTION),
            device,
            queue,
            cuda_minres_solver_capabilities(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn materialized_queue(device: DeviceId, slot: u32, materialization: u64) -> QueueId {
    QueueId::new(
        QueueSlot::new(device, slot),
        NonZeroU64::new(materialization).unwrap(),
    )
}

fn system(right_hand_side: [f64; 2]) -> CanonicalCsrSystemView {
    CanonicalCsrSystemView::new(
        &TwoByTwo { right_hand_side },
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
}

fn symmetric_indefinite_system(right_hand_side: [f64; 2]) -> CanonicalCsrSystemView {
    CanonicalCsrSystemView::new(
        &TwoByTwoIndefinite { right_hand_side },
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap()
}

fn serial_binding(graph: &eqiora_realization::PortableRealizationGraph) -> DeploymentBinding {
    DeploymentBinding::bind_host(
        graph,
        HostExecutorDescriptor::new(
            reference_solver_provider(),
            serial_execution_provider(),
            NonZeroUsize::MIN,
            REFERENCE_LINEAR_SOLVER.capabilities(),
        ),
    )
    .unwrap()
}

fn solve(system: &CanonicalCsrSystemView, plan: SolverPlan) -> eqiora_solver::LinearSolution {
    REFERENCE_LINEAR_SOLVER
        .solve(&system.linear_problem().unwrap(), plan)
        .unwrap()
}

fn solve_with_providers(
    system: &CanonicalCsrSystemView,
    plan: SolverPlan,
    solver: SolverProvider,
    execution: ExecutionProvider,
) -> eqiora_solver::LinearSolution {
    accept_linear_solution_with_verifier(
        &system.linear_problem().unwrap(),
        plan,
        solver,
        execution,
        ExecutionReport::host_serial(),
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        0.0,
        vec![2.0 / 3.0, 1.0 / 3.0],
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap()
}

fn cuda_solution(
    system: &CanonicalCsrSystemView,
    plan: SolverPlan,
) -> eqiora_solver::LinearSolution {
    accept_linear_solution_with_verifier(
        &system.linear_problem().unwrap(),
        plan,
        solver_provider(TEST_CUDA_BACKEND),
        execution_provider(TEST_CUDA_EXECUTION),
        ExecutionReport::cuda(TEST_CUDA_EXECUTION, 0),
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        0.0,
        vec![2.0 / 3.0, 1.0 / 3.0],
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap()
}

fn distributed_solution(
    system: &CanonicalCsrSystemView,
    plan: SolverPlan,
    partitions: NonZeroUsize,
) -> eqiora_solver::LinearSolution {
    accept_linear_solution_with_verifier(
        &system.linear_problem().unwrap(),
        plan,
        solver_provider(TEST_DISTRIBUTED_BACKEND),
        execution_provider(TEST_DISTRIBUTED_EXECUTION),
        ExecutionReport::distributed(TEST_DISTRIBUTED_EXECUTION, partitions),
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        0.0,
        vec![2.0 / 3.0, 1.0 / 3.0],
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap()
}

fn device_buffer<T: eqiora_device::DeviceElement>(
    device: DeviceId,
    allocation: u64,
    elements: usize,
) -> DeviceBufferDescriptor<T> {
    DeviceBufferDescriptor::new(
        BufferId::new(device, NonZeroU64::new(allocation).unwrap()),
        NonZeroUsize::new(elements).unwrap(),
    )
}

fn upload<T: eqiora_device::DeviceElement>(
    buffer: DeviceBufferDescriptor<T>,
    completion: Completion,
) -> TransferEvidence<T> {
    TransferEvidence::new(
        TransferPlan::new(
            MemoryRegion::Host(HostBufferDescriptor::new(buffer.elements())),
            MemoryRegion::Device(buffer),
        )
        .unwrap(),
        completion,
    )
    .unwrap()
}

fn download<T: eqiora_device::DeviceElement>(
    buffer: DeviceBufferDescriptor<T>,
    completion: Completion,
) -> TransferEvidence<T> {
    TransferEvidence::new(
        TransferPlan::new(
            MemoryRegion::Device(buffer),
            MemoryRegion::Host(HostBufferDescriptor::new(buffer.elements())),
        )
        .unwrap(),
        completion,
    )
    .unwrap()
}

#[derive(Debug)]
struct TestFence {
    completion: Completion,
    succeeds: bool,
}

impl Fence for TestFence {
    fn completion(&self) -> Completion {
        self.completion
    }

    fn wait(&self) -> Result<(), eqiora_core::Diagnostic> {
        self.succeeds
            .then_some(())
            .ok_or_else(|| super::binding::invalid("test fence failed"))
    }
}

fn cuda_trace(
    device: DeviceId,
    queue: QueueId,
) -> (CudaLinearExecutionTrace, DeviceValueGeneration) {
    let mut timeline = QueueTimeline::new(queue);
    let row_completion = Completion::new(timeline.next_submission().unwrap());
    let column_completion = Completion::new(timeline.next_submission().unwrap());
    let values_completion = Completion::new(timeline.next_submission().unwrap());
    let right_completion = Completion::new(timeline.next_submission().unwrap());
    let initial_completion = Completion::new(timeline.next_submission().unwrap());
    let diagonal_completion = Completion::new(timeline.next_submission().unwrap());
    let inputs_completion = Completion::new(timeline.next_submission().unwrap());
    let solve_completion = Completion::new(timeline.next_submission().unwrap());
    let output_transfer = Completion::new(timeline.next_submission().unwrap());
    let output_completion = Completion::new(timeline.next_submission().unwrap());
    let inputs_ready = WaitedCompletion::wait(&TestFence {
        completion: inputs_completion,
        succeeds: true,
    })
    .unwrap();
    let solve_visible = WaitedCompletion::wait(&TestFence {
        completion: solve_completion,
        succeeds: true,
    })
    .unwrap();
    let solution_visible = WaitedCompletion::wait(&TestFence {
        completion: output_completion,
        succeeds: true,
    })
    .unwrap();
    let row_offsets = device_buffer(device, 1, 3);
    let column_indices = device_buffer(device, 2, 4);
    let values = device_buffer(device, 3, 4);
    let right = device_buffer(device, 4, 2);
    let solution = device_buffer(device, 5, 2);
    let diagonal = device_buffer(device, 6, 2);
    let initial = DeviceValueGeneration::new(solution.id(), NonZeroU64::MIN);
    let solved = DeviceValueGeneration::new(solution.id(), NonZeroU64::new(2).unwrap());
    (
        CudaLinearExecutionTrace::new(
            CsrDeviceTransferEvidence::new(
                upload(row_offsets, row_completion),
                upload(column_indices, column_completion),
                upload(values, values_completion),
                upload(right, right_completion),
                upload(solution, initial_completion),
                Some(upload(diagonal, diagonal_completion)),
                download(solution, output_transfer),
            ),
            inputs_ready,
            solve_visible,
            solution_visible,
            initial,
            solved,
            solved,
            128,
        )
        .unwrap(),
        initial,
    )
}

#[test]
fn host_receipt_seals_the_fixed_dag_and_existing_identities() {
    let graph = portable_graph();
    let system = system([1.0, 0.0]);
    let fingerprint = system.agreement_fingerprint();
    let binding = serial_binding(&graph);
    let plan = binding.solver_plan();
    let admitted = AdmittedExecution::admit_host_linear(&graph, &system, binding).unwrap();
    let accepted = admitted.accept(solve(&system, plan)).unwrap();
    let receipt = accepted.receipt();

    assert_eq!(receipt.operator(), fingerprint);
    assert_eq!(receipt.dimension(), 2);
    assert_eq!(receipt.solver_plan(), plan);
    assert_eq!(receipt.report().solver_plan(), plan);
    assert_eq!(receipt.solver_provider(), reference_solver_provider());
    assert_eq!(receipt.execution_provider(), serial_execution_provider());
    assert_eq!(receipt.verification_provider(), serial_execution_provider());
    assert_eq!(
        receipt.solver_provider(),
        receipt.report().solver_provider()
    );
    assert_eq!(
        receipt.execution_provider(),
        receipt.report().execution_provider()
    );
    assert_eq!(
        receipt.solver_provider(),
        receipt.binding().solver_provider()
    );
    assert_eq!(
        receipt.execution_provider(),
        receipt.binding().execution_provider()
    );
    assert_eq!(
        receipt.verification_provider(),
        receipt.binding().verification_provider()
    );
    assert_eq!(receipt.report().execution(), ExecutionReport::host_serial());
    assert_eq!(
        receipt.report().verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        receipt.acceptance_verification(),
        ExecutionReport::host_serial()
    );
    assert_ne!(receipt.output().as_bytes(), [0; 32]);
    assert_eq!(receipt.dag().operator(), fingerprint);
    assert_eq!(receipt.dag().solver_plan(), plan);
    assert_eq!(
        receipt.dag().steps(),
        &[
            ExecutionStepKind::SolveWithNativeAcceptance,
            ExecutionStepKind::ReplayTrueResidualOnHost,
            ExecutionStepKind::AcceptHostComplete,
        ]
    );
}

#[test]
fn receipt_binding_sharing_preserves_the_entire_accepted_receipt_and_rejects_foreign_authority() {
    let graph = portable_graph();
    let accept = |graph: &eqiora_realization::PortableRealizationGraph, right| {
        let system = system(right);
        let binding = serial_binding(graph);
        let plan = binding.solver_plan();
        AdmittedExecution::admit_host_linear(graph, &system, binding)
            .unwrap()
            .accept(solve(&system, plan))
            .unwrap()
            .into_parts()
            .1
    };
    let authority = accept(&graph, [1.0, 0.0]);
    let other = accept(&graph, [0.0, 1.0]);
    assert_ne!(authority.output(), other.output());
    assert!(!std::ptr::eq(authority.binding(), other.binding()));
    let shared = other.clone().with_shared_binding(&authority).unwrap();
    assert_eq!(
        shared, other,
        "sharing preserves every report/evidence/operator/output field"
    );
    assert!(std::ptr::eq(authority.binding(), shared.binding()));
    let foreign = accept(&portable_graph(), [0.0, 1.0]);
    let error = other.with_shared_binding(&foreign).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(
        error
            .message()
            .contains("foreign execution receipt deployment binding")
    );
}

#[test]
fn insufficient_worker_capacity_fails_before_system_admission() {
    let graph = portable_graph_with_workers(NonZeroUsize::new(4).unwrap());
    let two_workers = HostExecutorDescriptor::new(
        reference_solver_provider(),
        execution_provider(ExecutionId::new("eqiora.test.host")),
        NonZeroUsize::new(2).unwrap(),
        REFERENCE_LINEAR_SOLVER.capabilities(),
    );
    let mismatch = DeploymentBinding::bind_host(&graph, two_workers).unwrap_err();
    assert_eq!(mismatch.code(), codes::INVALID_REALIZATION);
}

#[test]
fn invalid_provider_provenance_fails_during_pure_deployment_binding() {
    let graph = portable_graph();
    for descriptor in [
        HostExecutorDescriptor::new(
            SolverProvider::new(REFERENCE_LINEAR_SOLVER.id(), "", &[]),
            serial_execution_provider(),
            NonZeroUsize::MIN,
            REFERENCE_LINEAR_SOLVER.capabilities(),
        ),
        HostExecutorDescriptor::new(
            reference_solver_provider(),
            ExecutionProvider::new(ExecutionReport::host_serial().adapter(), "", &[]),
            NonZeroUsize::MIN,
            REFERENCE_LINEAR_SOLVER.capabilities(),
        ),
    ] {
        let rejected = DeploymentBinding::bind_host(&graph, descriptor).unwrap_err();
        assert_eq!(rejected.code(), codes::INVALID_REALIZATION);
        assert!(rejected.message().contains("implementation version"));
    }
}

#[test]
fn distributed_binding_seals_one_transport_neutral_process_group() {
    let graph = portable_distributed_graph();
    let slot = ProcessGroupSlot::new(3);
    let partitions = NonZeroUsize::new(4).unwrap();
    let binding = DeploymentBinding::bind_distributed(
        &graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            slot,
            partitions,
            NonZeroUsize::MIN,
            distributed_solver_capabilities(ReductionPolicy::Reproducible),
        ),
    )
    .unwrap();

    let executor = binding.distributed_executor().unwrap();
    assert_eq!(executor.backend(), TEST_DISTRIBUTED_BACKEND);
    assert_eq!(executor.adapter(), TEST_DISTRIBUTED_EXECUTION);
    assert_eq!(
        executor.solver_provider(),
        solver_provider(TEST_DISTRIBUTED_BACKEND)
    );
    assert_eq!(
        executor.execution_provider(),
        execution_provider(TEST_DISTRIBUTED_EXECUTION)
    );
    assert_eq!(binding.solver_provider(), executor.solver_provider());
    assert_eq!(binding.execution_provider(), executor.execution_provider());
    assert_eq!(executor.process_group(), slot);
    assert_eq!(executor.process_group().ordinal(), 3);
    assert_eq!(executor.partitions(), partitions);
    assert_eq!(executor.workers_per_partition(), NonZeroUsize::MIN);
    assert_eq!(
        executor.solver_capabilities(),
        &distributed_solver_capabilities(ReductionPolicy::Reproducible)
    );
    assert_eq!(binding.realization(), &graph);
    assert_eq!(
        binding.solver_plan().reduction(),
        ReductionPolicy::Reproducible
    );
    assert_eq!(
        binding.execution(),
        ExecutionReport::distributed(TEST_DISTRIBUTED_EXECUTION, partitions)
    );
    assert!(binding.host_executor().is_none());
    assert!(binding.cuda_executor().is_none());
}

#[test]
fn distributed_binding_rejects_layout_policy_and_executor_drift() {
    let descriptor = |capabilities| {
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::MIN,
            capabilities,
        )
    };

    let rejected = DeploymentBinding::bind_distributed(
        &portable_graph(),
        descriptor(distributed_solver_capabilities(
            ReductionPolicy::Reproducible,
        )),
    )
    .unwrap_err();
    assert!(rejected.message().contains("explicitly distributed"));

    let fast_graph = portable_distributed_graph_with(NonZeroUsize::MIN, ReductionPolicy::Fast);
    let rejected = DeploymentBinding::bind_distributed(
        &fast_graph,
        descriptor(distributed_solver_capabilities(ReductionPolicy::Fast)),
    )
    .unwrap_err();
    assert!(rejected.message().contains("reproducible"));

    let multi_worker_graph = portable_distributed_graph_with(
        NonZeroUsize::new(2).unwrap(),
        ReductionPolicy::Reproducible,
    );
    let rejected = DeploymentBinding::bind_distributed(
        &multi_worker_graph,
        descriptor(distributed_solver_capabilities(
            ReductionPolicy::Reproducible,
        )),
    )
    .unwrap_err();
    assert!(rejected.message().contains("exactly one host worker"));

    let graph = portable_distributed_graph();
    let rejected = DeploymentBinding::bind_distributed(
        &graph,
        descriptor(distributed_solver_capabilities(ReductionPolicy::Fast)),
    )
    .unwrap_err();
    assert_eq!(rejected.code(), codes::INVALID_REALIZATION);
}

#[test]
fn distributed_binding_accepts_only_the_exact_symmetric_indefinite_minres_tuple() {
    let graph = portable_distributed_minres_graph();
    let partitions = NonZeroUsize::new(2).unwrap();
    let capabilities = SolverCapabilities::new(
        [LinearSolver::MinimumResidual],
        [PreconditionerPolicy::Identity],
        [ReductionPolicy::Reproducible],
        [ScalarType::F64],
    )
    .unwrap();
    let binding = DeploymentBinding::bind_distributed(
        &graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            capabilities,
        ),
    )
    .unwrap();
    assert_eq!(
        binding.solver_plan().algorithm(),
        LinearSolver::MinimumResidual
    );
    assert_eq!(
        binding.solver_plan().preconditioner(),
        PreconditionerPolicy::Identity
    );

    let rejected = DeploymentBinding::bind_distributed(
        &graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            distributed_solver_capabilities(ReductionPolicy::Reproducible),
        ),
    )
    .unwrap_err();
    assert_eq!(rejected.code(), codes::INVALID_REALIZATION);
}

#[test]
fn distributed_cuda_binding_and_admission_keep_solver_and_local_action_authority_separate() {
    let graph = portable_distributed_cuda_minres_graph();
    let complete = symmetric_indefinite_system([1.0, 0.0]);
    let partitions = NonZeroUsize::new(2).unwrap();
    let distributed = distributed_system(&complete, partitions);
    let device = cuda_device(1 << 30);
    let device_id = device.id();
    let binding = distributed_cuda_binding(&graph, partitions, device);

    assert!(binding.cuda_executor().is_none());
    let placement = binding.cuda_partition_placement().unwrap();
    assert_eq!(placement.device().id(), device_id);
    assert_eq!(placement.action_policy(), SparseActionPolicy::Deterministic);
    assert_eq!(
        placement.execution_provider(),
        execution_provider(TEST_CUDA_EXECUTION)
    );
    assert_eq!(
        binding.solver_provider(),
        solver_provider(TEST_DISTRIBUTED_BACKEND)
    );
    assert_eq!(
        binding.execution_provider(),
        execution_provider(TEST_DISTRIBUTED_EXECUTION)
    );
    assert_eq!(
        binding.distributed_device_transport(),
        Some(DistributedDeviceTransport::HostStaged)
    );
    assert_eq!(
        binding.solver_plan().reduction(),
        ReductionPolicy::Reproducible
    );

    let admission = distributed
        .admission_fingerprint(binding.solver_plan())
        .unwrap();
    let admitted =
        AdmittedExecution::admit_distributed_cuda_linear(&graph, &distributed, &complete, binding)
            .unwrap();
    assert_eq!(admitted.distributed_system(), Some(&distributed));
    assert_eq!(admitted.distributed_admission(), Some(admission));
    assert_eq!(
        admitted.distributed_local_action_system(),
        Some(&distributed)
    );
    assert_eq!(
        admitted.distributed_local_action_admission(),
        Some(admission)
    );
    assert_eq!(admitted.distributed_host_system(), None);
    assert_eq!(admitted.distributed_host_admission(), None);
}

#[test]
fn distributed_cuda_binding_rejects_host_action_and_fast_or_nondeterministic_policy() {
    let graph = portable_distributed_cuda_minres_graph();
    let partitions = NonZeroUsize::new(2).unwrap();
    let device = cuda_device(1 << 30);
    let queue = QueueSlot::new(device.id(), 0);
    let distributed = DistributedExecutorDescriptor::new(
        solver_provider(TEST_DISTRIBUTED_BACKEND),
        execution_provider(TEST_DISTRIBUTED_EXECUTION),
        ProcessGroupSlot::new(0),
        partitions,
        NonZeroUsize::MIN,
        distributed_minres_solver_capabilities(),
    );
    let rejected = DeploymentBinding::bind_distributed_cuda(
        &graph,
        distributed,
        CudaPartitionPlacement::new(
            execution_provider(TEST_CUDA_EXECUTION),
            device,
            queue,
            SparseActionPolicy::BackendNative,
        )
        .unwrap(),
        DistributedDeviceTransport::HostStaged,
    )
    .unwrap_err();
    assert!(rejected.message().contains("deterministic"));

    let foreign_ordinal = DeviceDescriptor::new(
        DeviceId::new(TEST_CUDA_RUNTIME, 1),
        "unmasked test device",
        NonZeroU64::new(1 << 30).unwrap(),
        [
            DeviceCapability::Float64,
            DeviceCapability::CsrMatrixVectorProduct,
            DeviceCapability::AsynchronousQueue,
        ],
    )
    .unwrap();
    let foreign_queue = QueueSlot::new(foreign_ordinal.id(), 0);
    let rejected = DeploymentBinding::bind_distributed_cuda(
        &graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            distributed_minres_solver_capabilities(),
        ),
        CudaPartitionPlacement::new(
            execution_provider(TEST_CUDA_EXECUTION),
            foreign_ordinal,
            foreign_queue,
            SparseActionPolicy::Deterministic,
        )
        .unwrap(),
        DistributedDeviceTransport::HostStaged,
    )
    .unwrap_err();
    assert!(rejected.message().contains("ordinal zero"));

    let host_graph = portable_distributed_minres_graph();
    let device = cuda_device(1 << 30);
    let rejected = DeploymentBinding::bind_distributed_cuda(
        &host_graph,
        DistributedExecutorDescriptor::new(
            solver_provider(TEST_DISTRIBUTED_BACKEND),
            execution_provider(TEST_DISTRIBUTED_EXECUTION),
            ProcessGroupSlot::new(0),
            partitions,
            NonZeroUsize::MIN,
            distributed_minres_solver_capabilities(),
        ),
        CudaPartitionPlacement::new(
            execution_provider(TEST_CUDA_EXECUTION),
            device.clone(),
            QueueSlot::new(device.id(), 0),
            SparseActionPolicy::Deterministic,
        )
        .unwrap(),
        DistributedDeviceTransport::HostStaged,
    )
    .unwrap_err();
    assert!(rejected.message().contains("device placement"));
}

#[test]
fn distributed_receipt_seals_layout_collectives_and_complete_host_output() {
    let graph = portable_distributed_graph();
    let complete = system([1.0, 0.0]);
    let partitions = NonZeroUsize::new(2).unwrap();
    let distributed = distributed_system(&complete, partitions);
    let binding = distributed_binding(&graph, partitions);
    let plan = binding.solver_plan();
    let admission = distributed.admission_fingerprint(plan).unwrap();
    let admitted =
        AdmittedExecution::admit_distributed_linear(&graph, &distributed, &complete, binding)
            .unwrap();
    assert_eq!(admitted.distributed_system(), Some(&distributed));
    assert_eq!(admitted.distributed_admission(), Some(admission));
    assert_eq!(admitted.distributed_host_system(), Some(&distributed));
    assert_eq!(admitted.distributed_host_admission(), Some(admission));
    assert_eq!(admitted.distributed_local_action_system(), None);
    assert_eq!(admitted.distributed_local_action_admission(), None);
    let trace = distributed_trace(&distributed, &complete, plan);
    let accepted = admitted
        .accept_distributed(
            distributed_solution(&complete, plan, partitions),
            trace.clone(),
        )
        .unwrap();
    let receipt = accepted.receipt();

    assert_eq!(receipt.distributed_trace(), Some(&trace));
    assert_eq!(receipt.operator(), distributed.system_identity());
    assert_eq!(receipt.dimension(), complete.columns());
    assert_eq!(receipt.minimum_device_payload_bytes(), None);
    assert_eq!(receipt.cuda_trace(), None);
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
}

#[test]
fn distributed_admission_and_trace_fail_closed_on_cross_wiring() {
    let graph = portable_distributed_graph();
    let complete = system([1.0, 0.0]);
    let partitions = NonZeroUsize::new(2).unwrap();
    let distributed = distributed_system(&complete, partitions);
    let wrong_complete = system([0.0, 1.0]);
    let rejected = AdmittedExecution::admit_distributed_linear(
        &graph,
        &distributed,
        &wrong_complete,
        distributed_binding(&graph, partitions),
    )
    .unwrap_err();
    assert!(rejected.message().contains("identity"));

    let plan = distributed_binding(&graph, partitions).solver_plan();
    let trace = distributed_trace(&distributed, &complete, plan);
    let mut sparse = trace.steps().to_vec();
    sparse[1] = DistributedCollectiveStepV1::new(
        sparse[1].phase(),
        sparse[1].iteration(),
        sparse[1].ordinal() + 1,
    );
    let rejected = DistributedLinearExecutionTrace::new(
        distributed.system_identity(),
        distributed.partition_identity(),
        distributed.layout_identity(),
        distributed.admission_fingerprint(plan).unwrap(),
        ProcessGroupSlot::new(0),
        partitions,
        NonZeroUsize::MIN,
        complete.columns(),
        DistributedLinearExecutionTrace::collective_capacity(plan).unwrap(),
        sparse,
        plan,
    )
    .unwrap_err();
    assert!(rejected.message().contains("ordinals"));
}

#[test]
fn binding_rejects_a_provider_without_the_exact_solver_tuple() {
    let graph = portable_graph();
    let unsupported = SolverCapabilities::new(
        [LinearSolver::BiConjugateGradientStabilized],
        [PreconditionerPolicy::Identity],
        [ReductionPolicy::Reproducible],
        [ScalarType::F64],
    )
    .unwrap();
    let rejected = DeploymentBinding::bind_host(
        &graph,
        HostExecutorDescriptor::new(
            reference_solver_provider(),
            serial_execution_provider(),
            NonZeroUsize::MIN,
            unsupported,
        ),
    )
    .unwrap_err();
    assert_eq!(rejected.code(), codes::INVALID_REALIZATION);
}

#[test]
fn graph_or_operator_drift_cannot_enter_an_admitted_token() {
    let graph = portable_graph();
    let other_graph = portable_graph();
    let binding = serial_binding(&graph);
    let captured = system([1.0, 0.0]);
    let mismatch =
        AdmittedExecution::admit_host_linear(&other_graph, &captured, binding).unwrap_err();
    assert_eq!(mismatch.code(), codes::INVALID_REALIZATION);

    let general = CanonicalCsrSystemView::new(
        &TwoByTwo {
            right_hand_side: [1.0, 0.0],
        },
        LinearOperatorProperties::General,
    )
    .unwrap();
    let mismatch =
        AdmittedExecution::admit_host_linear(&graph, &general, serial_binding(&graph)).unwrap_err();
    assert_eq!(mismatch.code(), codes::INVALID_REALIZATION);
}

#[test]
fn distributed_admission_rejects_sparse_lu_before_fingerprinting() {
    let complete = system([1.0, 0.0]);
    let distributed = distributed_system(&complete, NonZeroUsize::new(2).unwrap());
    let plan = SolverPlan::new(LinearSolver::SparseLu, 0.0, 1.0e-12, NonZeroUsize::MIN).unwrap();
    let error = distributed.admission_fingerprint(plan).unwrap_err();

    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert_eq!(error.message(), "distributed sparse LU is not implemented");
}

#[test]
fn execution_and_plan_substitution_fail_at_the_output_boundary() {
    let graph = portable_graph();
    let system = system([1.0, 0.0]);
    let custom_execution = ExecutionId::new("eqiora.test.alternate-host");
    let binding = DeploymentBinding::bind_host(
        &graph,
        HostExecutorDescriptor::new(
            reference_solver_provider(),
            execution_provider(custom_execution),
            NonZeroUsize::MIN,
            REFERENCE_LINEAR_SOLVER.capabilities(),
        ),
    )
    .unwrap();
    let admitted = AdmittedExecution::admit_host_linear(&graph, &system, binding).unwrap();
    let error = admitted
        .accept(solve(&system, serial_binding(&graph).solver_plan()))
        .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);

    let binding = serial_binding(&graph);
    let admitted = AdmittedExecution::admit_host_linear(&graph, &system, binding).unwrap();
    let substituted = SolverPlan::new(
        eqiora_solver::LinearSolver::ConjugateGradient,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(64).unwrap(),
    )
    .unwrap();
    let error = admitted.accept(solve(&system, substituted)).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}

#[test]
fn same_id_provider_version_and_library_substitution_fail_at_receipt_boundary() {
    let graph = portable_graph();
    let system = system([1.0, 0.0]);
    let plan = serial_binding(&graph).solver_plan();
    let expected_solver = reference_solver_provider();
    let expected_execution = serial_execution_provider();
    let substituted = [
        (
            SolverProvider::new(expected_solver.id(), "9.9.9", expected_solver.libraries()),
            expected_execution,
        ),
        (
            SolverProvider::new(
                expected_solver.id(),
                expected_solver.implementation_version(),
                SUBSTITUTED_LIBRARIES,
            ),
            expected_execution,
        ),
        (
            expected_solver,
            ExecutionProvider::new(
                expected_execution.id(),
                "9.9.9",
                expected_execution.libraries(),
            ),
        ),
        (
            expected_solver,
            ExecutionProvider::new(
                expected_execution.id(),
                expected_execution.implementation_version(),
                SUBSTITUTED_LIBRARIES,
            ),
        ),
    ];

    for (solver, execution) in substituted {
        let admitted =
            AdmittedExecution::admit_host_linear(&graph, &system, serial_binding(&graph)).unwrap();
        let error = admitted
            .accept(solve_with_providers(&system, plan, solver, execution))
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("provider provenance contradicts"));
    }

    for verifier in [
        ExecutionProvider::new(
            expected_execution.id(),
            "9.9.9",
            expected_execution.libraries(),
        ),
        ExecutionProvider::new(
            expected_execution.id(),
            expected_execution.implementation_version(),
            SUBSTITUTED_LIBRARIES,
        ),
    ] {
        let admitted =
            AdmittedExecution::admit_host_linear(&graph, &system, serial_binding(&graph)).unwrap();
        let solution = accept_linear_solution_with_verifier(
            &system.linear_problem().unwrap(),
            plan,
            expected_solver,
            expected_execution,
            ExecutionReport::host_serial(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.0,
            vec![2.0 / 3.0, 1.0 / 3.0],
            &SubstitutedSerialVerifier(verifier),
        )
        .unwrap();
        let error = admitted.accept(solution).unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("verifier provenance contradicts"));
    }
}

#[test]
fn candidate_from_another_operator_is_rejected_by_true_residual_replay() {
    let graph = portable_graph();
    let admitted_system = system([1.0, 0.0]);
    let other_system = system([0.0, 1.0]);
    let binding = serial_binding(&graph);
    let plan = binding.solver_plan();
    let admitted = AdmittedExecution::admit_host_linear(&graph, &admitted_system, binding).unwrap();
    let error = admitted.accept(solve(&other_system, plan)).unwrap_err();

    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(error.message().contains("independent host residual"));
}

#[test]
fn accepted_output_fingerprint_changes_with_the_complete_vector() {
    let graph = portable_graph();
    let first_system = system([1.0, 0.0]);
    let second_system = system([0.0, 1.0]);
    let plan = serial_binding(&graph).solver_plan();

    let first = AdmittedExecution::admit_host_linear(&graph, &first_system, serial_binding(&graph))
        .unwrap()
        .accept(solve(&first_system, plan))
        .unwrap();
    let second =
        AdmittedExecution::admit_host_linear(&graph, &second_system, serial_binding(&graph))
            .unwrap()
            .accept(solve(&second_system, plan))
            .unwrap();

    assert_ne!(first.receipt().output(), second.receipt().output());
}

#[test]
fn cuda_receipt_seals_exact_movement_generation_fences_and_dag() {
    let graph = portable_cuda_graph();
    let system = system([1.0, 0.0]);
    let device = cuda_device(1 << 20);
    let queue = materialized_queue(device.id(), 0, 1);
    let binding = cuda_binding(&graph, device);
    let plan = binding.solver_plan();
    let admitted = AdmittedExecution::admit_cuda_linear(&graph, &system, binding).unwrap();
    assert!(admitted.minimum_device_payload_bytes().unwrap() > 0);
    let (trace, _) = cuda_trace(queue.device(), queue);
    let accepted = admitted
        .accept_cuda(cuda_solution(&system, plan), trace)
        .unwrap();
    let receipt = accepted.receipt();

    assert_eq!(
        receipt.solver_provider(),
        solver_provider(TEST_CUDA_BACKEND)
    );

    assert_eq!(
        receipt.execution_provider(),
        execution_provider(TEST_CUDA_EXECUTION)
    );
    assert_eq!(receipt.verification_provider(), serial_execution_provider());
    assert_eq!(
        receipt.solver_provider(),
        receipt.report().solver_provider()
    );
    assert_eq!(
        receipt.execution_provider(),
        receipt.report().execution_provider()
    );
    assert_eq!(
        receipt.report().execution(),
        ExecutionReport::cuda(TEST_CUDA_EXECUTION, 0)
    );
    assert_eq!(
        receipt.report().verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        receipt.acceptance_verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(receipt.cuda_trace(), Some(trace));
    assert!(receipt.minimum_device_payload_bytes().unwrap() > 0);
    assert_eq!(
        receipt.dag().steps(),
        &[
            ExecutionStepKind::TransferInputsToCuda,
            ExecutionStepKind::AwaitCudaInputsReady,
            ExecutionStepKind::SolveOnCuda,
            ExecutionStepKind::AwaitCudaSolveCompletion,
            ExecutionStepKind::TransferCandidateToHost,
            ExecutionStepKind::AwaitHostVisibility,
            ExecutionStepKind::AcceptWithNativeHostVerification,
            ExecutionStepKind::ReplayTrueResidualOnHost,
            ExecutionStepKind::AcceptHostComplete,
        ]
    );
}

#[test]
fn cuda_minres_admission_accounts_for_exact_resident_workspace() {
    let graph = portable_cuda_minres_graph();
    let system = symmetric_indefinite_system([1.0, 0.0]);
    let device = cuda_device(232);
    let binding = cuda_minres_binding(&graph, device);
    let plan = binding.solver_plan();

    assert_eq!(plan.algorithm(), LinearSolver::MinimumResidual);
    assert_eq!(plan.preconditioner(), PreconditionerPolicy::Identity);
    assert_eq!(plan.reduction(), ReductionPolicy::Fast);
    let admitted = AdmittedExecution::admit_cuda_linear(&graph, &system, binding).unwrap();

    // 7 i64 CSR indices, 6 matrix/RHS scalars, and eight resident vectors:
    // solution + residual + the six-vector MINRES recurrence workspace.
    assert_eq!(admitted.minimum_device_payload_bytes(), Some(232));

    let insufficient = cuda_device(231);
    let rejected = AdmittedExecution::admit_cuda_linear(
        &graph,
        &system,
        cuda_minres_binding(&graph, insufficient),
    )
    .unwrap_err();
    assert!(rejected.message().contains("requires 232 bytes"));
}

#[test]
fn cuda_binding_rejects_capability_and_memory_before_device_allocation() {
    let graph = portable_cuda_graph();
    let device = cuda_device(1 << 20);
    let queue = QueueSlot::new(device.id(), 0);
    let allocation_attempted = Cell::new(false);
    let unsupported = SolverCapabilities::new(
        [LinearSolver::ConjugateGradient],
        [PreconditionerPolicy::Jacobi],
        [ReductionPolicy::Reproducible],
        [ScalarType::F64],
    )
    .unwrap();
    let binding = DeploymentBinding::bind_cuda(
        &graph,
        CudaExecutorDescriptor::new(
            solver_provider(TEST_CUDA_BACKEND),
            execution_provider(TEST_CUDA_EXECUTION),
            device,
            queue,
            unsupported,
        )
        .unwrap(),
    );
    if binding.is_ok() {
        allocation_attempted.set(true);
    }
    assert!(binding.is_err());
    assert!(!allocation_attempted.get());

    let tiny = cuda_device(NonZeroU64::MIN.get());
    let binding = cuda_binding(&graph, tiny);
    let rejected =
        AdmittedExecution::admit_cuda_linear(&graph, &system([1.0, 0.0]), binding).unwrap_err();
    assert!(rejected.message().contains("known device payload"));
    assert!(!allocation_attempted.get());

    let roomy = cuda_device(1 << 20);
    let minimum = AdmittedExecution::admit_cuda_linear(
        &graph,
        &system([1.0, 0.0]),
        cuda_binding(&graph, roomy),
    )
    .unwrap()
    .minimum_device_payload_bytes()
    .unwrap();
    let tight = cuda_device(u64::try_from(minimum).unwrap());
    let tight_queue = materialized_queue(tight.id(), 0, 2);
    let tight_system = system([1.0, 0.0]);
    let tight_binding = cuda_binding(&graph, tight);
    let tight_plan = tight_binding.solver_plan();
    let admitted =
        AdmittedExecution::admit_cuda_linear(&graph, &tight_system, tight_binding).unwrap();
    let (trace, _) = cuda_trace(tight_queue.device(), tight_queue);
    let rejected = admitted
        .accept_cuda(cuda_solution(&tight_system, tight_plan), trace)
        .unwrap_err();
    assert!(rejected.message().contains("external sparse workspace"));
}
