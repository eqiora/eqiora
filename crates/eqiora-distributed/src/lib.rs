//! **eqiora-distributed** — backend-neutral distributed algebra contracts.
//!
//! Unique ownership, local/ghost layouts, halo plans, distributed operator
//! shards, and collective-policy admission are independent of transport.
//! `LoopbackExecutor` runs the protocol in one process as a reference; it does
//! not provide MPI or multi-node execution.

mod agreement;
mod allocation;
mod csr;
mod error;
mod layout;
mod partition;
mod system;

pub use agreement::{
    DistributedAdmissionFingerprintV1, DistributedLayoutAgreementIdentityV1,
    PartitionAgreementIdentityV1,
};
pub use csr::{
    DistributedCsr, LocalCsrExecutionCapture, LocalCsrExecutionView, LocalCsrShard,
    LoopbackExecutor, OwnedLinearSystemShard,
};
pub use layout::{HaloExchange, HaloPlan, LocalLayout};
pub use partition::{GlobalVectorSpace, Partition, PartitionId};
pub use system::{DistributedLinearProblem, DistributedLinearSystem, LocalLinearSolution};
