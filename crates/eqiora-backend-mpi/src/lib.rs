//! **eqiora-backend-mpi** — optional MPI transport adapter.
//!
//! The application owns MPI initialization and passes a communicator to a
//! duplicated Eqiora execution group. MPI handles never enter Semantic Model,
//! Realization, distributed algebra, or artifact contracts.

mod protocol;

#[cfg(feature = "mpi-runtime")]
mod runtime;

#[cfg(feature = "mpi-runtime")]
mod spatial_assembly;

pub use protocol::{
    AdmissionRecordV1, CollectivePhaseV1, CollectiveStepV1, DistributedProtocolFailureV1,
    MpiCollectiveTraceV1, OwnedGatherPlanV1, PhaseStatusV1, ProducerReportSummaryV2,
    evaluate_admission, evaluate_phase_statuses,
};

/// Exact Eqiora MPI adapter package version compiled into this binary.
pub const MPI_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exact mpi-rs transport-binding dependency version.
pub const MPI_RS_VERSION: &str = env!("EQIORA_DEPENDENCY_VERSION_MPI");

#[cfg(feature = "mpi-runtime")]
pub use runtime::{
    AdmittedDistributedRun, MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
    MPI_EXECUTION, MPI_EXECUTION_PROVIDER, MpiAdmittedExecutionAdapter, MpiExecutionGroup,
    MpiLinearSolveResult, MpiRankDeviceTopologyV1, MpiRankLocalCsrAction, MpiThreadSupport,
    RankLocalDeviceV1,
};

#[cfg(feature = "mpi-runtime")]
pub use spatial_assembly::{MPI_SPATIAL_ASSEMBLY_EXECUTION, MpiSpatialAssemblyBackend};
