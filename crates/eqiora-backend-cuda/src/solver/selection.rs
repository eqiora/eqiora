//! Selection for an explicitly chosen CUDA deployment, without runtime work.
use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_execution::{AdmittedExecution, CudaExecutorDescriptor, DeploymentBinding};
use eqiora_realization::{PortableRealizationGraph, SolveRoot};
use eqiora_solver::{
    AlgebraicStructure, CanonicalCsrSystemView, ExecutionReport, SolverPlanningObjective,
    SolverPlanningProfile, SolverProviderAvailability, SolverSelection,
};

use super::{CUDA_LINEAR_EXECUTION_PROVIDER, CUDA_LINEAR_SOLVER_PROVIDER, CudaLinearSolver};
use crate::runtime::unsupported;

/// One inspected solver decision retained with its explicitly selected device.
///
/// Construction validates a device descriptor without discovering a runtime,
/// creating a context, or allocating device memory. Execution still checks the
/// selected runtime and independently accepts the numerical result.
#[derive(Debug)]
pub struct CudaSolverSelection {
    executor: CudaExecutorDescriptor,
    selection: SolverSelection,
}

impl CudaLinearSolver {
    /// Select an existing CUDA tuple for one explicitly selected deployment.
    /// No host candidate or other device is a fallback.
    ///
    /// # Errors
    /// Rejects a foreign provider, invented capability, unsupported queue/device,
    /// mismatched profile execution, or a profile without an admitted candidate.
    pub fn select_plan(
        self,
        executor: CudaExecutorDescriptor,
        profile: SolverPlanningProfile,
        objective: SolverPlanningObjective,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
    ) -> Result<CudaSolverSelection, Diagnostic> {
        let execution = checked_executor(self, &executor)?;
        if profile.execution_provider() != CUDA_LINEAR_EXECUTION_PROVIDER
            || profile.execution() != execution
        {
            return Err(unsupported(
                "CUDA selection profile differs from the selected deployment",
            ));
        }
        let availability = SolverProviderAvailability::new(
            executor.solver_provider(),
            executor.execution_provider(),
            execution,
            executor.solver_capabilities().clone(),
        )?;
        let selection = SolverSelection::resolve::<f64>(
            profile,
            objective,
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
            &[availability],
        )?;
        Ok(CudaSolverSelection {
            executor,
            selection,
        })
    }
}

impl CudaSolverSelection {
    /// Inspect the exact decision, controls, reasons and unmodeled cost terms.
    #[must_use]
    pub const fn selection(&self) -> &SolverSelection {
        &self.selection
    }

    /// Check the finalized problem and exact graph plan before binding.
    /// The native finalizer still owns its exact target and graph/CSR subject.
    ///
    /// # Errors
    /// Rejects descriptor drift, changed profile facts, structure, resources or
    /// exact solver controls before deployment binding or device allocation.
    pub fn bind(
        self,
        realization: &PortableRealizationGraph,
        system: &CanonicalCsrSystemView,
        structure: Option<&AlgebraicStructure>,
    ) -> Result<DeploymentBinding, Diagnostic> {
        let execution = checked_executor(
            CudaLinearSolver::new(self.executor.device().id().ordinal()),
            &self.executor,
        )?;
        if self.selection.execution_provider() != self.executor.execution_provider()
            || self.selection.execution() != execution
        {
            return Err(unsupported(
                "CUDA selection differs from its retained deployment",
            ));
        }
        self.selection
            .require_problem::<f64>(&system.linear_problem()?, structure)?;
        let SolveRoot::Linear(root) = realization.root() else {
            return Err(unsupported(
                "CUDA selection requires a portable linear solve root",
            ));
        };
        let linear = realization
            .linear_solve(root)
            .ok_or_else(|| unsupported("CUDA selection graph omits its linear root"))?;
        if linear.plan() != self.selection.solver_plan() {
            return Err(unsupported(
                "CUDA native graph solver plan differs from the selected exact plan",
            ));
        }
        DeploymentBinding::bind_cuda(realization, self.executor)
    }

    /// Admit the selected canonical problem through generic CUDA admission.
    /// Native workflows with their own admission method should consume `bind`
    /// through that method to retain its exact target and subject checks.
    ///
    /// # Errors
    /// Preserves selection, deployment and canonical-system admission failures.
    pub fn admit<'system>(
        self,
        realization: &PortableRealizationGraph,
        system: &'system CanonicalCsrSystemView,
        structure: Option<&AlgebraicStructure>,
    ) -> Result<AdmittedExecution<'system>, Diagnostic> {
        let binding = self.bind(realization, system, structure)?;
        AdmittedExecution::admit_cuda_linear(realization, system, binding)
    }
}

fn checked_executor(
    solver: CudaLinearSolver,
    executor: &CudaExecutorDescriptor,
) -> Result<ExecutionReport, Diagnostic> {
    if executor.solver_provider() != CUDA_LINEAR_SOLVER_PROVIDER {
        return Err(unsupported(
            "CUDA selection requires the exact CUDA solver provider",
        ));
    }
    if executor.execution_provider() != CUDA_LINEAR_EXECUTION_PROVIDER {
        return Err(unsupported(
            "CUDA selection requires the exact CUDA execution provider",
        ));
    }
    if executor.queue().ordinal() != 0 {
        return Err(unsupported(
            "single-device CUDA selection admits only queue ordinal zero",
        ));
    }
    solver.admit_device(executor.device())?;
    let execution = ExecutionReport::cuda(
        CUDA_LINEAR_EXECUTION_PROVIDER.id(),
        executor.device().id().ordinal(),
    );
    let implemented = CudaLinearSolver::capabilities();
    if executor
        .solver_capabilities()
        .combinations()
        .iter()
        .any(|capability| !implemented.combinations().contains(capability))
    {
        return Err(unsupported(
            "CUDA selection descriptor advertises an unimplemented solver tuple",
        ));
    }
    Ok(execution)
}

#[cfg(test)]
mod tests;
