//! **eqiora-backend-cuda** — isolated CUDA execution adapter.
//!
//! CUDA contexts, streams, device pointers, descriptors, workspaces, and
//! errors remain private to this L3 crate. The public surface contains only
//! Eqiora-owned contracts and evidence.

#[cfg(feature = "cuda-runtime")]
#[allow(unsafe_code)]
mod blas;
#[cfg(feature = "cuda-runtime")]
#[allow(unsafe_code)]
mod ffi;
#[cfg(feature = "cuda-runtime")]
mod runtime;
#[cfg(feature = "cuda-runtime")]
mod solver;

#[cfg(feature = "cuda-runtime")]
pub use runtime::{
    CudaComputeCapability, CudaCsrActionEvidence, CudaCsrActionResult, CudaCsrTransferEvidence,
    CudaDeviceObservation, CudaDeviceUuid, CudaLibraryVersions, CudaResidentCsrActionEvidence,
    CudaResidentCsrActionSession, CudaResidentCsrSetupEvidence, CudaRuntime, verify_csr_action,
    verify_csr_action_against,
};
#[cfg(feature = "cuda-runtime")]
pub use solver::{
    AcceptedCudaLinearSolveResult, CUDA_LINEAR_EXECUTION, CUDA_LINEAR_EXECUTION_PROVIDER,
    CUDA_LINEAR_SOLVER_BACKEND, CUDA_LINEAR_SOLVER_PROVIDER, CudaAdmittedExecutionAdapter,
    CudaLinearSolveEvidence, CudaLinearSolveResult, CudaLinearSolver, CudaLinearTransferEvidence,
};

/// Stable runtime identity even when the optional adapter is not compiled.
pub const CUDA_RUNTIME_ID: eqiora_device::RuntimeId =
    eqiora_device::RuntimeId::new("eqiora.cuda.cudarc");

/// Exact Eqiora CUDA adapter package version compiled into this binary.
pub const CUDA_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exact dynamically loaded driver-binding dependency version.
pub const CUDARC_VERSION: &str = "0.19.9";

/// CUDA toolkit ABI selected for the generated driver bindings.
pub const CUDA_BINDING_TOOLKIT: &str = "12.0";
