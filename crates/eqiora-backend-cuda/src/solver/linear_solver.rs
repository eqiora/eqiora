//! Concrete CUDA solver ownership and execution entry points.
use super::selection::SelectedDeployment;
use super::{
    AcceptedCudaLinearSolveResult, CUDA_LINEAR_EXECUTION_PROVIDER, CUDA_LINEAR_SOLVER_PROVIDER,
    CudaAdmittedExecutionAdapter, CudaLinearSolveResult, solve,
};
use crate::runtime::unsupported;
use eqiora_core::Diagnostic;
use eqiora_device::DeviceDescriptor;
use eqiora_execution::{AdmittedExecution, CUDA_LINEAR_DEVICE_CAPABILITIES};
use eqiora_solver::{CanonicalCsrSystemView, SolverPlan};

/// One CUDA device selected for CSR Krylov execution.
#[derive(Debug)]
pub struct CudaLinearSolver {
    pub(super) device_ordinal: u16,
    pub(super) selected: Option<SelectedDeployment>,
}

impl CudaLinearSolver {
    fn require_unselected_execution(&self) -> Result<(), Diagnostic> {
        if self.selected.is_some() {
            return Err(unsupported(
                "selected CUDA solver requires retained decision binding before execution",
            ));
        }
        Ok(())
    }

    /// Select a runtime-visible device without loading CUDA yet.
    #[must_use]
    pub const fn new(device_ordinal: u16) -> Self {
        Self {
            device_ordinal,
            selected: None,
        }
    }

    /// Admit one already discovered device for this solver slice without
    /// creating a context, queue, allocation, or vendor-library handle.
    ///
    /// # Errors
    /// Returns `EQ0807` for a descriptor from another runtime, a descriptor
    /// with a different deployment-visible ordinal, or the first missing
    /// device capability, in that order.
    pub fn admit_device(&self, device: &DeviceDescriptor) -> Result<(), Diagnostic> {
        if device.id().runtime() != crate::CUDA_RUNTIME_ID {
            return Err(unsupported(format!(
                "CUDA solver requires runtime `{}`, received `{}`",
                crate::CUDA_RUNTIME_ID.as_str(),
                device.id().runtime().as_str(),
            )));
        }
        if device.id().ordinal() != self.device_ordinal {
            return Err(unsupported(format!(
                "CUDA solver selected device {}, received descriptor for device {}",
                self.device_ordinal,
                device.id().ordinal(),
            )));
        }
        device.require(CUDA_LINEAR_DEVICE_CAPABILITIES)
    }

    /// Solve one finalized assembled system with a single resident CSR and
    /// resident Krylov vectors, then accept it through the independent serial
    /// host operator and fixed-order reduction oracle.
    ///
    /// # Errors
    /// Returns a stable diagnostic for unsupported policy/property, invalid
    /// CSR or preconditioner, unavailable CUDA capability, library failure,
    /// numerical breakdown/non-convergence, host true-residual rejection, or
    /// an owner retaining a selection that must first be bound.
    pub fn solve(
        &self,
        system: &CanonicalCsrSystemView,
        initial_guess: Option<&[f64]>,
        plan: SolverPlan,
    ) -> Result<CudaLinearSolveResult, Diagnostic> {
        self.require_unselected_execution()?;
        solve(
            self.device_ordinal,
            None,
            None,
            None,
            system,
            initial_guess,
            plan,
        )
    }
}

impl CudaAdmittedExecutionAdapter for CudaLinearSolver {
    fn execute_admitted(
        self,
        admitted: AdmittedExecution<'_>,
    ) -> Result<AcceptedCudaLinearSolveResult, Diagnostic> {
        self.require_unselected_execution()?;
        let executor = admitted
            .binding()
            .cuda_executor()
            .ok_or_else(|| unsupported("CUDA adapter requires a device deployment binding"))?;
        if executor.solver_provider() != CUDA_LINEAR_SOLVER_PROVIDER {
            return Err(unsupported(
                "CUDA deployment selected a different linear-solver backend",
            ));
        }
        if executor.execution_provider() != CUDA_LINEAR_EXECUTION_PROVIDER {
            return Err(unsupported(
                "CUDA deployment selected a different execution adapter",
            ));
        }
        if executor.device().id().runtime() != crate::CUDA_RUNTIME_ID {
            return Err(unsupported(
                "CUDA deployment selected a device from another runtime",
            ));
        }
        if executor.device().id().ordinal() != self.device_ordinal {
            return Err(unsupported(
                "CUDA solver ordinal differs from the admitted device",
            ));
        }
        if executor.queue().ordinal() != 0 {
            return Err(unsupported(
                "single-device CUDA execution v1 admits only queue ordinal zero",
            ));
        }
        let device = executor.device().clone();
        let queue = executor.queue();
        let minimum_device_payload_bytes =
            admitted.minimum_device_payload_bytes().ok_or_else(|| {
                unsupported(
                    "CUDA adapter received an admission without a device-memory lower bound",
                )
            })?;
        let result = solve(
            self.device_ordinal,
            Some(&device),
            Some(queue),
            Some(minimum_device_payload_bytes),
            admitted.system(),
            None,
            admitted.solver_plan(),
        )?;
        let (solution, evidence) = result.into_parts();
        let trace = evidence.execution_trace()?;
        let accepted = admitted.accept_cuda(solution, trace)?;
        Ok(AcceptedCudaLinearSolveResult { accepted, evidence })
    }
}
