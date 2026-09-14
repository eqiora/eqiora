//! **eqiora-execution** — deployment binding and accepted execution shape.
//!
//! This first bounded slice binds one accepted host placement from a
//! [`eqiora_realization::PortableRealizationGraph`] to an existing
//! [`eqiora_solver::ExecutionReport`]. It then seals one exact canonical CSR
//! system and admits a receipt only after the resulting [`eqiora_solver::SolveReport`]
//! is independently rechecked against that same system. The solver-native
//! verifier remains visible in its report; the receipt records its additional
//! serial-host replay separately.
//!
//! The public lifecycle is intentionally one-way:
//!
//! ```text
//! PortableRealizationGraph -> DeploymentBinding -> AdmittedExecution
//!     -> accepted LinearSolution -> ExecutionReceipt -> ExecutionDagView
//! ```
//!
//! There is no mutable public graph builder, raw node identity, arbitrary
//! action payload, or target cross product. Transfer, halo, collective, and
//! device completion nodes enter only with their first executable consumer.
//! Raw admission is an adapter contract and is intentionally absent from the
//! curated `eqiora` facade; ordinary applications receive only read-only
//! receipts produced after an equation-aware numerical finalizer has replayed
//! the exact portable graph.

mod binding;
mod device;
mod distributed;
mod receipt;

mod telemetry;

pub use binding::{
    CUDA_LINEAR_DEVICE_CAPABILITIES, CUDA_PARTITION_CSR_DEVICE_CAPABILITIES,
    CudaExecutorDescriptor, CudaPartitionPlacement, DeploymentBinding, DistributedDeviceTransport,
    DistributedExecutorDescriptor, HostExecutorDescriptor, ProcessGroupSlot,
};
pub use device::{CsrDeviceTransferEvidence, CudaLinearExecutionTrace, DeviceValueGeneration};
pub use distributed::{
    DistributedCollectiveStepV1, DistributedExecutionPhaseV1, DistributedLinearExecutionTrace,
};
pub use receipt::{
    AcceptedLinearExecution, AcceptedOutputFingerprintV1, AdmittedExecution, ExecutionDagView,
    ExecutionReceipt, ExecutionStepKind,
};

#[cfg(test)]
mod tests;
