use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_device::{DeviceCapability, DeviceDescriptor, QueueSlot, SparseActionPolicy};
use eqiora_realization::{
    ExecutionSchedule, PlacementRequirementNode, PortableRealizationGraph, SolveRoot,
    VectorLayoutKind,
};

/// Exact device capabilities consumed by the first one-device CSR Krylov
/// operation contract.
pub const CUDA_LINEAR_DEVICE_CAPABILITIES: [DeviceCapability; 4] = [
    DeviceCapability::Float64,
    DeviceCapability::CsrMatrixVectorProduct,
    DeviceCapability::DenseVectorLevel1,
    DeviceCapability::AsynchronousQueue,
];

/// Exact device capabilities consumed by a resident partition-local CSR
/// action whose Krylov vectors and reductions remain host-owned.
pub const CUDA_PARTITION_CSR_DEVICE_CAPABILITIES: [DeviceCapability; 3] = [
    DeviceCapability::Float64,
    DeviceCapability::CsrMatrixVectorProduct,
    DeviceCapability::AsynchronousQueue,
];
use eqiora_solver::{
    BackendId, ExecutionId, ExecutionProvider, ExecutionReport, LinearOperatorProperties,
    LinearSolver, PreconditionerPolicy, ReductionPolicy, SERIAL_EXECUTION_PROVIDER,
    SolverCapabilities, SolverPlan, SolverProvider,
};

/// Logical process-group position selected before transport materialization.
///
/// Like a device [`QueueSlot`], this is deployment-local selection rather
/// than a globally ordered runtime identity. MPI communicators and transport
/// handles remain private to their L3 adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessGroupSlot(u32);

impl ProcessGroupSlot {
    /// Construct a zero-based logical process-group position.
    #[must_use]
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Zero-based logical process-group position.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// One selected host executor and its declared deployment capabilities.
///
/// The descriptor seals the exact declared solver/execution releases before
/// materialization. The later solver report must repeat that complete
/// provenance and remains the authority for the actual worker topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostExecutorDescriptor {
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    maximum_workers: std::num::NonZeroUsize,
    solver_capabilities: SolverCapabilities,
}

impl HostExecutorDescriptor {
    /// Describe a selected host solver/execution pair before a worker pool or
    /// numerical run exists. Binding validates the graph's exact solver tuple
    /// against these capabilities; the later receipt proves actual identity.
    #[must_use]
    pub fn new(
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        maximum_workers: std::num::NonZeroUsize,
        solver_capabilities: SolverCapabilities,
    ) -> Self {
        Self {
            solver_provider,
            execution_provider,
            maximum_workers,
            solver_capabilities,
        }
    }

    /// Exact selected solver provider.
    #[must_use]
    pub const fn backend(&self) -> BackendId {
        self.solver_provider.id()
    }

    /// Exact selected solver implementation and library provenance.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Exact selected execution adapter.
    #[must_use]
    pub const fn adapter(&self) -> ExecutionId {
        self.execution_provider.id()
    }

    /// Exact selected execution implementation and library provenance.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Maximum workers available before pool materialization.
    #[must_use]
    pub const fn maximum_workers(&self) -> std::num::NonZeroUsize {
        self.maximum_workers
    }

    /// Exact solver tuples declared by the selected provider.
    #[must_use]
    pub const fn solver_capabilities(&self) -> &SolverCapabilities {
        &self.solver_capabilities
    }
}

/// One selected device executor and its pre-device-allocation capability snapshot.
///
/// The portable graph requests a device count only. Runtime-local device and
/// logical queue-slot selection enter through this deployment descriptor.
/// Concrete queue materialization belongs to Run evidence and changes neither
/// Model nor portable Realization identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaExecutorDescriptor {
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    device: DeviceDescriptor,
    queue: QueueSlot,
    solver_capabilities: SolverCapabilities,
}

impl CudaExecutorDescriptor {
    /// Describe one selected logical queue slot before a context, concrete
    /// queue, allocation, or vendor-library handle exists.
    ///
    /// # Errors
    /// Returns `EQ0807` when the queue belongs to another device.
    pub fn new(
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        device: DeviceDescriptor,
        queue: QueueSlot,
        solver_capabilities: SolverCapabilities,
    ) -> Result<Self, Diagnostic> {
        if queue.device() != device.id() {
            return Err(invalid(
                "device deployment queue does not belong to the selected device",
            ));
        }
        Ok(Self {
            solver_provider,
            execution_provider,
            device,
            queue,
            solver_capabilities,
        })
    }

    /// Exact selected solver provider.
    #[must_use]
    pub const fn backend(&self) -> BackendId {
        self.solver_provider.id()
    }

    /// Exact selected solver implementation and library provenance.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Exact selected execution adapter.
    #[must_use]
    pub const fn adapter(&self) -> ExecutionId {
        self.execution_provider.id()
    }

    /// Exact selected execution implementation and library provenance.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Exact runtime-local device selected by deployment.
    #[must_use]
    pub const fn device(&self) -> &DeviceDescriptor {
        &self.device
    }

    /// Exact logical queue slot selected before runtime materialization.
    #[must_use]
    pub const fn queue(&self) -> QueueSlot {
        self.queue
    }

    /// Exact solver tuples declared by the selected provider.
    #[must_use]
    pub const fn solver_capabilities(&self) -> &SolverCapabilities {
        &self.solver_capabilities
    }
}

/// One CUDA placement selected for the sparse action inside a distributed
/// partition.
///
/// This descriptor deliberately owns no linear-solver provider or
/// `SolverCapabilities`: the distributed executor owns the Krylov method and
/// collective reduction, while this placement owns only the rank-local
/// device, queue, execution provider, and sparse-action ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPartitionPlacement {
    execution_provider: ExecutionProvider,
    device: DeviceDescriptor,
    queue: QueueSlot,
    action_policy: SparseActionPolicy,
}

impl CudaPartitionPlacement {
    /// Describe one selected partition-local CUDA sparse-action queue before
    /// context creation or allocation.
    ///
    /// # Errors
    /// Returns `EQ0807` when the queue belongs to another device.
    pub fn new(
        execution_provider: ExecutionProvider,
        device: DeviceDescriptor,
        queue: QueueSlot,
        action_policy: SparseActionPolicy,
    ) -> Result<Self, Diagnostic> {
        if queue.device() != device.id() {
            return Err(invalid(
                "partition CUDA queue does not belong to the selected device",
            ));
        }
        Ok(Self {
            execution_provider,
            device,
            queue,
            action_policy,
        })
    }

    /// Exact rank-local sparse-action adapter.
    #[must_use]
    pub const fn adapter(&self) -> ExecutionId {
        self.execution_provider.id()
    }

    /// Exact rank-local execution implementation and library provenance.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Exact runtime-local device selected for this partition.
    #[must_use]
    pub const fn device(&self) -> &DeviceDescriptor {
        &self.device
    }

    /// Logical queue selected before runtime materialization.
    #[must_use]
    pub const fn queue(&self) -> QueueSlot {
        self.queue
    }

    /// Sparse-action ordering promised by the selected adapter.
    #[must_use]
    pub const fn action_policy(&self) -> SparseActionPolicy {
        self.action_policy
    }
}

/// Explicit transport between MPI-owned host vectors and a partition-local
/// device action.
///
/// The first composition has one variant on purpose. GPU-aware transport is
/// added only with independent discovery, synchronization, and evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DistributedDeviceTransport {
    /// Halo exchange and Krylov state remain in distinct host buffers; the
    /// local owned-plus-ghost action is explicitly copied to and from CUDA.
    HostStaged,
}

/// One selected transport-neutral distributed executor.
///
/// This descriptor records only the process-group shape and exact numerical
/// capabilities needed before communication workspace is allocated. Compiled
/// provider release and declared dependency versions are sealed here; a live MPI
/// implementation/version, communicator, provided thread-support level, and
/// rank-local handle remain paired adapter/Run evidence rather than portable
/// Realization fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedExecutorDescriptor {
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    process_group: ProcessGroupSlot,
    partitions: std::num::NonZeroUsize,
    workers_per_partition: std::num::NonZeroUsize,
    solver_capabilities: SolverCapabilities,
}

impl DistributedExecutorDescriptor {
    /// Describe one selected process group before distributed runtime
    /// workspace or numerical communication is materialized.
    #[must_use]
    pub fn new(
        solver_provider: SolverProvider,
        execution_provider: ExecutionProvider,
        process_group: ProcessGroupSlot,
        partitions: std::num::NonZeroUsize,
        workers_per_partition: std::num::NonZeroUsize,
        solver_capabilities: SolverCapabilities,
    ) -> Self {
        Self {
            solver_provider,
            execution_provider,
            process_group,
            partitions,
            workers_per_partition,
            solver_capabilities,
        }
    }

    /// Exact selected solver provider.
    #[must_use]
    pub const fn backend(&self) -> BackendId {
        self.solver_provider.id()
    }

    /// Exact selected solver implementation and library provenance.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Exact selected distributed execution adapter.
    #[must_use]
    pub const fn adapter(&self) -> ExecutionId {
        self.execution_provider.id()
    }

    /// Exact selected execution implementation and library provenance.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Logical process group selected for this deployment.
    #[must_use]
    pub const fn process_group(&self) -> ProcessGroupSlot {
        self.process_group
    }

    /// Exact number of unique-owner partitions in the selected group.
    #[must_use]
    pub const fn partitions(&self) -> std::num::NonZeroUsize {
        self.partitions
    }

    /// Exact admitted host workers inside every partition.
    #[must_use]
    pub const fn workers_per_partition(&self) -> std::num::NonZeroUsize {
        self.workers_per_partition
    }

    /// Exact solver tuples declared by the selected provider.
    #[must_use]
    pub const fn solver_capabilities(&self) -> &SolverCapabilities {
        &self.solver_capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BoundExecutor {
    Host(HostExecutorDescriptor),
    Cuda(CudaExecutorDescriptor),
    Distributed(DistributedExecutorDescriptor),
    DistributedCuda {
        distributed: DistributedExecutorDescriptor,
        cuda: CudaPartitionPlacement,
        transport: DistributedDeviceTransport,
    },
}

/// Immutable binding of one portable compute placement to one selected
/// executor.
#[derive(Debug, Clone, PartialEq)]
pub struct DeploymentBinding {
    realization: PortableRealizationGraph,
    executor: BoundExecutor,
    execution: ExecutionReport,
    verification_provider: ExecutionProvider,
}

impl DeploymentBinding {
    /// Bind the sole host placement of an accepted portable Realization graph.
    ///
    /// This validation runs before any system capture, buffer materialization,
    /// solver call, or trace construction.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the graph has an offline linear root, replicated
    /// `f64` algebra, an exact admitted solver tuple, and a host-worker
    /// requirement within the declared capacity.
    pub fn bind_host(
        realization: &PortableRealizationGraph,
        executor: HostExecutorDescriptor,
    ) -> Result<Self, Diagnostic> {
        executor.solver_provider.validate()?;
        executor.execution_provider.validate()?;
        let contract = host_contract(realization)?;
        executor.solver_capabilities.require_problem(
            contract.plan,
            ScalarType::F64,
            contract.properties,
        )?;
        if contract.workers > executor.maximum_workers {
            return Err(invalid(format!(
                "portable Realization requires {} host worker(s), executor capacity is {}",
                contract.workers, executor.maximum_workers
            )));
        }
        let execution = ExecutionReport::host(executor.adapter(), contract.workers);
        let verification_provider = executor.execution_provider;
        Ok(Self {
            realization: realization.clone(),
            executor: BoundExecutor::Host(executor),
            execution,
            verification_provider,
        })
    }

    /// Bind the sole one-device placement of an accepted portable Realization
    /// graph to one already selected runtime device and logical queue slot.
    ///
    /// This validates the complete solver tuple, scalar/layout/schedule, device
    /// count, and device capability set before context creation or numerical
    /// allocation.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the graph has an offline linear root,
    /// replicated `f64` algebra, exactly one device requirement, an exact
    /// admitted solver tuple, and the four capabilities required by the first
    /// CSR Krylov device slice.
    pub fn bind_cuda(
        realization: &PortableRealizationGraph,
        executor: CudaExecutorDescriptor,
    ) -> Result<Self, Diagnostic> {
        executor.solver_provider.validate()?;
        executor.execution_provider.validate()?;
        let contract = cuda_contract(realization)?;
        executor.solver_capabilities.require_problem(
            contract.plan,
            ScalarType::F64,
            contract.properties,
        )?;
        executor.device.require(CUDA_LINEAR_DEVICE_CAPABILITIES)?;
        let execution = ExecutionReport::cuda(executor.adapter(), executor.device.id().ordinal());
        Ok(Self {
            realization: realization.clone(),
            executor: BoundExecutor::Cuda(executor),
            execution,
            verification_provider: SERIAL_EXECUTION_PROVIDER,
        })
    }

    /// Bind the sole distributed host placement to one selected process
    /// group without exposing a transport-specific target variant.
    ///
    /// This graph-bound distributed slice is deliberately exact: one host
    /// worker per partition, offline `f64` distributed algebra, and either
    /// reproducible Jacobi-preconditioned CG for an asserted SPD operator or
    /// reproducible identity-preconditioned MINRES for an asserted symmetric
    /// indefinite operator.
    /// Process-count agreement with the concrete owner map is checked later
    /// by distributed system admission, before communication begins.
    ///
    /// # Errors
    /// Returns `EQ0807` for any placement, scalar, schedule, operator, solver,
    /// reduction, capability, or per-partition worker contradiction.
    pub fn bind_distributed(
        realization: &PortableRealizationGraph,
        executor: DistributedExecutorDescriptor,
    ) -> Result<Self, Diagnostic> {
        executor.solver_provider.validate()?;
        executor.execution_provider.validate()?;
        let contract = distributed_contract(realization)?;
        executor.solver_capabilities.require_problem(
            contract.plan,
            ScalarType::F64,
            contract.properties,
        )?;
        if executor.workers_per_partition != contract.workers_per_partition {
            return Err(invalid(format!(
                "portable Realization requires {} host worker(s) per partition, distributed executor selects {}",
                contract.workers_per_partition, executor.workers_per_partition
            )));
        }
        let execution = ExecutionReport::distributed(executor.adapter(), executor.partitions);
        Ok(Self {
            realization: realization.clone(),
            executor: BoundExecutor::Distributed(executor),
            execution,
            verification_provider: SERIAL_EXECUTION_PROVIDER,
        })
    }

    /// Bind one distributed algebraic solve to its process group and one
    /// partition-local CUDA CSR action without merging their authorities.
    ///
    /// The distributed executor owns MINRES and collectives. The CUDA
    /// placement owns only a deterministic sparse action on one local device.
    /// Host staging is explicit and no fallback transport exists.
    ///
    /// # Errors
    /// Returns `EQ0807` for layout, placement, device-count, scalar, schedule,
    /// solver, worker, queue, device-capability, or sparse-action-policy drift.
    pub fn bind_distributed_cuda(
        realization: &PortableRealizationGraph,
        distributed: DistributedExecutorDescriptor,
        cuda: CudaPartitionPlacement,
        transport: DistributedDeviceTransport,
    ) -> Result<Self, Diagnostic> {
        distributed.solver_provider.validate()?;
        distributed.execution_provider.validate()?;
        cuda.execution_provider.validate()?;
        let contract = distributed_cuda_contract(realization)?;
        distributed.solver_capabilities.require_problem(
            contract.plan,
            ScalarType::F64,
            contract.properties,
        )?;
        if distributed.workers_per_partition != contract.workers_per_partition {
            return Err(invalid(format!(
                "portable distributed CUDA Realization requires {} host control worker(s) per partition, executor selects {}",
                contract.workers_per_partition, distributed.workers_per_partition
            )));
        }
        cuda.device
            .require(CUDA_PARTITION_CSR_DEVICE_CAPABILITIES)?;
        if cuda.device.id().ordinal() != 0 {
            return Err(invalid(
                "distributed CUDA execution v1 requires the rank-local visible device ordinal zero",
            ));
        }
        if cuda.action_policy != SparseActionPolicy::Deterministic {
            return Err(invalid(
                "distributed CUDA execution v1 requires a deterministic rank-local CSR action",
            ));
        }
        let execution = ExecutionReport::distributed(distributed.adapter(), distributed.partitions);
        Ok(Self {
            realization: realization.clone(),
            executor: BoundExecutor::DistributedCuda {
                distributed,
                cuda,
                transport,
            },
            execution,
            verification_provider: SERIAL_EXECUTION_PROVIDER,
        })
    }

    /// Exact portable Realization sealed by this binding.
    #[must_use]
    pub const fn realization(&self) -> &PortableRealizationGraph {
        &self.realization
    }

    /// Exact selected executor and admitted capacity.
    #[must_use]
    pub const fn host_executor(&self) -> Option<&HostExecutorDescriptor> {
        match &self.executor {
            BoundExecutor::Host(executor) => Some(executor),
            BoundExecutor::Cuda(_)
            | BoundExecutor::Distributed(_)
            | BoundExecutor::DistributedCuda { .. } => None,
        }
    }

    /// Selected device executor, when this is a device binding.
    #[must_use]
    pub const fn cuda_executor(&self) -> Option<&CudaExecutorDescriptor> {
        match &self.executor {
            BoundExecutor::Host(_)
            | BoundExecutor::Distributed(_)
            | BoundExecutor::DistributedCuda { .. } => None,
            BoundExecutor::Cuda(executor) => Some(executor),
        }
    }

    /// Selected CUDA placement inside a distributed partition, when present.
    #[must_use]
    pub const fn cuda_partition_placement(&self) -> Option<&CudaPartitionPlacement> {
        match &self.executor {
            BoundExecutor::DistributedCuda { cuda, .. } => Some(cuda),
            BoundExecutor::Host(_) | BoundExecutor::Cuda(_) | BoundExecutor::Distributed(_) => None,
        }
    }

    /// Explicit distributed/device transport selected by the binding.
    #[must_use]
    pub const fn distributed_device_transport(&self) -> Option<DistributedDeviceTransport> {
        match &self.executor {
            BoundExecutor::DistributedCuda { transport, .. } => Some(*transport),
            BoundExecutor::Host(_) | BoundExecutor::Cuda(_) | BoundExecutor::Distributed(_) => None,
        }
    }

    /// Selected process-group executor, when this is a distributed binding.
    #[must_use]
    pub const fn distributed_executor(&self) -> Option<&DistributedExecutorDescriptor> {
        match &self.executor {
            BoundExecutor::Host(_) | BoundExecutor::Cuda(_) => None,
            BoundExecutor::Distributed(executor) => Some(executor),
            BoundExecutor::DistributedCuda { distributed, .. } => Some(distributed),
        }
    }

    /// Exact execution report expected after selecting the requested workers.
    #[must_use]
    pub const fn execution(&self) -> ExecutionReport {
        self.execution
    }

    /// Sole solver plan selected by the portable root.
    #[must_use]
    pub fn solver_plan(&self) -> SolverPlan {
        linear_contract(&self.realization)
            .expect("DeploymentBinding construction validated the linear contract")
            .plan
    }

    /// Exact selected solver implementation and library provenance.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        match &self.executor {
            BoundExecutor::Host(executor) => executor.solver_provider,
            BoundExecutor::Cuda(executor) => executor.solver_provider,
            BoundExecutor::Distributed(executor) => executor.solver_provider,
            BoundExecutor::DistributedCuda { distributed, .. } => distributed.solver_provider,
        }
    }

    /// Exact selected primary execution implementation and library provenance.
    ///
    /// For distributed CUDA composition this is the distributed solver
    /// provider. The partition-local device-action provider remains available
    /// through [`Self::cuda_partition_placement`].
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        match &self.executor {
            BoundExecutor::Host(executor) => executor.execution_provider,
            BoundExecutor::Cuda(executor) => executor.execution_provider,
            BoundExecutor::Distributed(executor) => executor.execution_provider,
            BoundExecutor::DistributedCuda { distributed, .. } => distributed.execution_provider,
        }
    }

    /// Exact solver-native verifier required by this bounded execution
    /// contract.
    ///
    /// Host execution verifies through its selected execution provider.
    /// Current CUDA and distributed contracts require Eqiora's serial-host
    /// verifier after producer execution.
    #[must_use]
    pub const fn verification_provider(&self) -> ExecutionProvider {
        self.verification_provider
    }

    pub(crate) const fn backend(&self) -> BackendId {
        self.solver_provider().id()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HostContract {
    pub(crate) plan: SolverPlan,
    pub(crate) properties: eqiora_solver::LinearOperatorProperties,
    pub(crate) workers: std::num::NonZeroUsize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaContract {
    pub(crate) plan: SolverPlan,
    pub(crate) properties: eqiora_solver::LinearOperatorProperties,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DistributedContract {
    pub(crate) plan: SolverPlan,
    pub(crate) properties: LinearOperatorProperties,
    pub(crate) workers_per_partition: std::num::NonZeroUsize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DistributedCudaContract {
    pub(crate) plan: SolverPlan,
    pub(crate) properties: LinearOperatorProperties,
    pub(crate) workers_per_partition: std::num::NonZeroUsize,
}

#[derive(Debug, Clone, Copy)]
struct LinearContract {
    plan: SolverPlan,
    properties: eqiora_solver::LinearOperatorProperties,
    scalar_type: ScalarType,
    partition: VectorLayoutKind,
    schedule: ExecutionSchedule,
    placement: PlacementRequirementNode,
}

fn linear_contract(realization: &PortableRealizationGraph) -> Result<LinearContract, Diagnostic> {
    let SolveRoot::Linear(root) = realization.root() else {
        return Err(invalid(
            "the first execution contract admits only a portable linear solve root",
        ));
    };
    let linear = realization
        .linear_solve(root)
        .ok_or_else(|| invalid("portable linear root is absent"))?;
    let system = realization
        .system(linear.system())
        .ok_or_else(|| invalid("portable linear solve references an absent system"))?;
    let placement = realization
        .placement(linear.placement())
        .ok_or_else(|| invalid("portable linear solve references an absent placement"))?;
    Ok(LinearContract {
        plan: linear.plan(),
        properties: system.operator_properties(),
        scalar_type: system.scalar_type(),
        partition: system.partition(),
        schedule: linear.schedule(),
        placement,
    })
}

pub(crate) fn host_contract(
    realization: &PortableRealizationGraph,
) -> Result<HostContract, Diagnostic> {
    let contract = linear_contract(realization)?;
    if contract.partition != VectorLayoutKind::Replicated {
        return Err(invalid("host execution v1 requires replicated algebra"));
    }
    if contract.scalar_type != ScalarType::F64 {
        return Err(invalid("host execution v1 requires f64 algebra"));
    }
    if contract.schedule != ExecutionSchedule::Offline {
        return Err(invalid(
            "host execution v1 does not claim real-time schedule support",
        ));
    }
    let PlacementRequirementNode::HostWorkers {
        workers_per_partition,
    } = contract.placement
    else {
        return Err(invalid(
            "host deployment cannot bind a CUDA placement requirement",
        ));
    };
    Ok(HostContract {
        plan: contract.plan,
        properties: contract.properties,
        workers: workers_per_partition,
    })
}

pub(crate) fn cuda_contract(
    realization: &PortableRealizationGraph,
) -> Result<CudaContract, Diagnostic> {
    let contract = linear_contract(realization)?;
    if contract.partition != VectorLayoutKind::Replicated {
        return Err(invalid(
            "single-device execution v1 requires replicated algebra",
        ));
    }
    if contract.scalar_type != ScalarType::F64 {
        return Err(invalid("single-device execution v1 requires f64 algebra"));
    }
    if contract.schedule != ExecutionSchedule::Offline {
        return Err(invalid(
            "single-device execution v1 does not claim real-time schedule support",
        ));
    }
    let PlacementRequirementNode::CudaDevices {
        devices_per_partition,
    } = contract.placement
    else {
        return Err(invalid(
            "device deployment cannot bind a host placement requirement",
        ));
    };
    if devices_per_partition.get() != 1 {
        return Err(invalid(
            "single-device execution v1 requires exactly one device per partition",
        ));
    }
    Ok(CudaContract {
        plan: contract.plan,
        properties: contract.properties,
    })
}

pub(crate) fn distributed_contract(
    realization: &PortableRealizationGraph,
) -> Result<DistributedContract, Diagnostic> {
    let contract = linear_contract(realization)?;
    if contract.partition != VectorLayoutKind::Distributed {
        return Err(invalid(
            "distributed execution v1 requires explicitly distributed algebra",
        ));
    }
    if contract.scalar_type != ScalarType::F64 {
        return Err(invalid("distributed execution v1 requires f64 algebra"));
    }
    if contract.schedule != ExecutionSchedule::Offline {
        return Err(invalid(
            "distributed execution v1 does not claim real-time schedule support",
        ));
    }
    let PlacementRequirementNode::HostWorkers {
        workers_per_partition,
    } = contract.placement
    else {
        return Err(invalid(
            "distributed host deployment cannot bind a CUDA placement requirement",
        ));
    };
    if workers_per_partition != std::num::NonZeroUsize::MIN {
        return Err(invalid(
            "distributed execution v1 requires exactly one host worker per partition",
        ));
    }
    let solver_is_admitted = matches!(
        (
            contract.properties,
            contract.plan.algorithm(),
            contract.plan.preconditioner(),
            contract.plan.reduction(),
        ),
        (
            LinearOperatorProperties::SymmetricPositiveDefinite,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            ReductionPolicy::Reproducible,
        ) | (
            LinearOperatorProperties::SymmetricIndefinite,
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
        )
    );
    if !solver_is_admitted {
        return Err(invalid(
            "distributed execution v1 requires either asserted SPD algebra with reproducible Jacobi-preconditioned CG or asserted symmetric-indefinite algebra with reproducible identity-preconditioned MINRES",
        ));
    }
    Ok(DistributedContract {
        plan: contract.plan,
        properties: contract.properties,
        workers_per_partition,
    })
}

pub(crate) fn distributed_cuda_contract(
    realization: &PortableRealizationGraph,
) -> Result<DistributedCudaContract, Diagnostic> {
    let contract = linear_contract(realization)?;
    if contract.partition != VectorLayoutKind::Distributed {
        return Err(invalid(
            "distributed CUDA execution v1 requires explicitly distributed algebra",
        ));
    }
    if contract.scalar_type != ScalarType::F64 {
        return Err(invalid(
            "distributed CUDA execution v1 requires f64 algebra",
        ));
    }
    if contract.schedule != ExecutionSchedule::Offline {
        return Err(invalid(
            "distributed CUDA execution v1 does not claim real-time schedule support",
        ));
    }
    let PlacementRequirementNode::CudaDevices {
        devices_per_partition,
    } = contract.placement
    else {
        return Err(invalid(
            "distributed CUDA deployment requires a device placement",
        ));
    };
    if devices_per_partition != std::num::NonZeroUsize::MIN {
        return Err(invalid(
            "distributed CUDA execution v1 requires exactly one device per partition",
        ));
    }
    if !matches!(
        (
            contract.properties,
            contract.plan.algorithm(),
            contract.plan.preconditioner(),
            contract.plan.reduction(),
        ),
        (
            LinearOperatorProperties::SymmetricIndefinite,
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
        )
    ) {
        return Err(invalid(
            "distributed CUDA execution v1 requires asserted symmetric-indefinite algebra with reproducible identity-preconditioned MINRES",
        ));
    }
    Ok(DistributedCudaContract {
        plan: contract.plan,
        properties: contract.properties,
        workers_per_partition: std::num::NonZeroUsize::MIN,
    })
}

pub(crate) fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
