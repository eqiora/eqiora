//! Native MPI solver selection and collective pre-execution authentication.
use super::{
    CollectivePhaseV1, MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER, MPI_EXECUTION_PROVIDER,
    MpiExecutionGroup, checked_extent, fixed_all_gather, invalid_realization, zeroed,
};
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_execution::{DeploymentBinding, DistributedExecutorDescriptor};
use eqiora_solver::{
    AlgebraicStructure, CanonicalCsrSystemView, ExecutionReport, LinearOperatorProperties,
    LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
    SolverPlanningObjective, SolverPlanningProfile, SolverSelection,
};
use std::num::NonZeroUsize;

const PLANNING_FINGERPRINT_BYTES: usize = 32;

pub(super) fn agreement_buffer(partitions: NonZeroUsize) -> Result<Vec<u8>, Diagnostic> {
    zeroed(
        checked_extent(
            PLANNING_FINGERPRINT_BYTES,
            partitions.get(),
            "MPI planning agreement",
        )?,
        "MPI planning agreement",
    )
}

impl MpiExecutionGroup {
    /// Resolve an existing native tuple for this exact MPI process group.
    ///
    /// Selection is provisional until `bind_selected` authenticates the complete
    /// global CSR and the native owner admits its original execution subject.
    /// No host provider, other rank count, or local-CSR mathematical claim is a fallback.
    ///
    /// # Errors
    /// Makes rank-local profile, selection and fingerprint failures collective
    /// before comparing complete planning projections on every rank.
    pub fn select_plan(
        &mut self,
        profile: SolverPlanningProfile,
        objective: SolverPlanningObjective,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
    ) -> Result<SolverSelection, Diagnostic> {
        let local = (|| {
            self.require_planning_execution(profile.execution_provider(), profile.execution())?;
            let decision = SolverSelection::resolve_declared::<f64>(
                profile,
                objective,
                relative_tolerance,
                absolute_tolerance,
                maximum_iterations,
                MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
                &self.solver_capabilities(),
            )?;
            let fingerprint = decision.planning_agreement_fingerprint()?;
            Ok((decision, fingerprint))
        })();
        self.agree_local_result(&local, CollectivePhaseV1::Admission)?;
        let (decision, fingerprint) = local.map_err(|_| {
            invalid_realization(
                "local MPI selection failed despite all-rank planning readiness agreement",
            )
        })?;
        self.agree_planning_fingerprint(fingerprint)?;
        Ok(decision)
    }

    /// Reauthenticate an exact selected decision before native subject admission.
    ///
    /// Pass the result of ordinary binding construction without returning a
    /// rank-local error first. The unchanged binding then returns to the native
    /// finalizer, which still owns assembly lineage, target and CSR identity.
    /// `complete` is the canonical global verifier, never an owned-row shard.
    ///
    /// # Errors
    /// Collectively rejects binding construction failure, changed provider or
    /// capability snapshot, deployment, complete mathematical/resource facts,
    /// exact graph plan, or cross-rank planning projection.
    pub fn bind_selected(
        &mut self,
        selection: &SolverSelection,
        binding: Result<DeploymentBinding, Diagnostic>,
        complete: &CanonicalCsrSystemView,
        structure: Option<&AlgebraicStructure>,
    ) -> Result<DeploymentBinding, Diagnostic> {
        let local = (|| {
            let binding = binding?;
            if binding.distributed_device_transport().is_some() {
                return Err(invalid_realization(
                    "selected native MPI binding requires host execution, not a composed device placement",
                ));
            }
            self.require_planning_execution(selection.execution_provider(), selection.execution())?;
            let capabilities = self.solver_capabilities();
            selection
                .require_declared_provider(MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER, &capabilities)?;
            let executor = binding.distributed_executor().ok_or_else(|| {
                invalid_realization(
                    "selected native MPI binding requires a distributed host executor",
                )
            })?;
            require_selected_descriptor(executor, self.partitions, &capabilities)?;
            if binding.execution() != selection.execution()
                || binding.solver_plan() != selection.solver_plan()
            {
                return Err(invalid_realization(
                    "MPI binding differs from the selected execution or exact graph plan",
                ));
            }
            selection.require_problem::<f64>(&complete.linear_problem()?, structure)?;
            capabilities.require_problem(
                selection.solver_plan(),
                eqiora_core::ScalarDomain::Real,
                ScalarType::F64,
                complete.properties(),
            )?;
            let fingerprint = selection.planning_agreement_fingerprint()?;
            Ok((binding, fingerprint))
        })();
        self.agree_local_result(&local, CollectivePhaseV1::Admission)?;
        let (binding, fingerprint) = local.map_err(|_| {
            invalid_realization(
                "local MPI selected binding failed despite all-rank readiness agreement",
            )
        })?;
        self.agree_planning_fingerprint(fingerprint)?;
        Ok(binding)
    }

    fn require_planning_execution(
        &self,
        provider: eqiora_solver::ExecutionProvider,
        execution: ExecutionReport,
    ) -> Result<(), Diagnostic> {
        if provider != MPI_EXECUTION_PROVIDER
            || execution
                != ExecutionReport::distributed(MPI_EXECUTION_PROVIDER.id(), self.partitions)
        {
            return Err(invalid_realization(
                "MPI solver planning differs from the actual provider, rank count or one-worker execution",
            ));
        }
        Ok(())
    }

    fn agree_planning_fingerprint(&mut self, fingerprint: [u8; 32]) -> Result<(), Diagnostic> {
        fixed_all_gather(
            &self.communicator,
            &fingerprint,
            &mut self.planning_agreement_bytes,
        );
        if self
            .planning_agreement_bytes
            .as_chunks::<PLANNING_FINGERPRINT_BYTES>()
            .0
            .iter()
            .all(|other| other == &fingerprint)
        {
            Ok(())
        } else {
            Err(invalid_realization(
                "MPI solver planning projections disagree across partitions",
            ))
        }
    }
    /// Exact numerical policies admitted by the current distributed evidence slice.
    #[must_use]
    pub fn solver_capabilities(&self) -> SolverCapabilities {
        native_capabilities()
    }
}

fn require_selected_descriptor(
    executor: &DistributedExecutorDescriptor,
    partitions: NonZeroUsize,
    capabilities: &SolverCapabilities,
) -> Result<(), Diagnostic> {
    if executor.solver_provider() != MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER
        || executor.execution_provider() != MPI_EXECUTION_PROVIDER
        || executor.process_group().ordinal() != 0
        || executor.partitions() != partitions
        || executor.workers_per_partition() != NonZeroUsize::MIN
        || executor.solver_capabilities() != capabilities
    {
        return Err(invalid_realization(
            "MPI selected binding contradicts actual providers, slot zero, ranks, one worker or exact capabilities",
        ));
    }
    Ok(())
}

fn native_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([
        SolverCapability {
            scalar_domain: eqiora_core::ScalarDomain::Real,
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            scalar_domain: eqiora_core::ScalarDomain::Real,
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            scalar_domain: eqiora_core::ScalarDomain::Real,
            algorithm: LinearSolver::MinimumResidual,
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        },
    ])
    .expect("MPI distributed solver exact capability set is nonempty")
}

#[cfg(test)]
mod tests;
