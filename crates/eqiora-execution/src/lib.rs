//! **eqiora-execution** — deployment binding and accepted execution shape.
//!
//! This crate binds a placement from an
//! [`eqiora_realization::PortableRealizationGraph`] to a selected executor,
//! seals one canonical CSR system, and issues a receipt only after the
//! resulting [`eqiora_solver::SolveReport`] has been rechecked against that same
//! system. The solver's own verifier stays visible in its report; the receipt
//! records Eqiora's additional serial-host replay separately.
//!
//! The public lifecycle runs in one direction:
//!
//! ```text
//! PortableRealizationGraph -> DeploymentBinding -> AdmittedExecution
//!     -> accepted LinearSolution -> ExecutionReceipt -> ExecutionDagView
//! ```
//!
//! There is no mutable public graph builder, raw node identity, arbitrary
//! action payload, or target cross product. Raw admission is an adapter
//! contract and is not part of the `eqiora` facade; applications see only
//! read-only receipts, produced after an equation-aware finalizer has replayed
//! the portable graph.

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
