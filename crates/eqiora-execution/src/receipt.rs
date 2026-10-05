use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_distributed::{DistributedAdmissionFingerprintV1, DistributedLinearSystem};
use eqiora_realization::PortableRealizationGraph;
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, CanonicalCsrSystemView, ExecutionProvider, ExecutionReport,
    ExecutionTopology, FixedOrderInnerProduct, LinearOperator, LinearOperatorOrientation,
    LinearSolution, SolveReport, SolverPlan, SolverProvider,
};
use sha2::{Digest, Sha256};

use crate::binding::{
    DeploymentBinding, cuda_contract, distributed_contract, distributed_cuda_contract,
    host_contract, invalid,
};
use crate::device::CudaLinearExecutionTrace;
use crate::distributed::DistributedLinearExecutionTrace;

mod device_payload;

use device_payload::minimum_device_payload_bytes;

/// Closed operation vocabulary of the first accepted linear-execution DAGs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStepKind {
    /// Produce a solver-native accepted solution and its own verification.
    SolveWithNativeAcceptance,
    /// Upload every canonical CSR, right-hand-side, zero-initial, and optional
    /// preconditioner slot to the selected device.
    TransferInputsToCuda,
    /// Wait successfully until every canonical device input is visible.
    AwaitCudaInputsReady,
    /// Execute the sole admitted Krylov plan on the selected device queue.
    SolveOnCuda,
    /// Wait successfully for the solve completion recorded by the adapter.
    AwaitCudaSolveCompletion,
    /// Copy the complete solved generation from device to host.
    TransferCandidateToHost,
    /// Wait successfully until the complete host candidate is visible.
    AwaitHostVisibility,
    /// Retain the solver-native serial-host acceptance of the device result.
    AcceptWithNativeHostVerification,
    /// Agree the exact system, owner map, layout, plan, and process group.
    AgreeDistributedAdmission,
    /// Execute the bounded repeating halo/action/reduction/update region.
    SolveDistributedKrylov,
    /// Agree the method-native producer report across all partitions.
    AgreeDistributedProducerReport,
    /// Gather explicit owner indices and values into a complete candidate.
    GatherDistributedOwnedCandidate,
    /// Agree the natively accepted complete result across all partitions.
    AgreeDistributedAcceptedResult,
    /// Agree the independently replayed immutable receipt across partitions.
    AgreeDistributedReceipt,
    /// Independently replay the true residual on the complete host system.
    ReplayTrueResidualOnHost,
    /// Expose only the complete host value paired with accepted evidence.
    AcceptHostComplete,
}

impl ExecutionStepKind {
    /// Stable kebab-case name used by execution evidence projections.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::SolveWithNativeAcceptance => "solve-with-native-acceptance",
            Self::TransferInputsToCuda => "transfer-inputs-to-cuda",
            Self::AwaitCudaInputsReady => "await-cuda-inputs-ready",
            Self::SolveOnCuda => "solve-on-cuda",
            Self::AwaitCudaSolveCompletion => "await-cuda-solve-completion",
            Self::TransferCandidateToHost => "transfer-candidate-to-host",
            Self::AwaitHostVisibility => "await-host-visibility",
            Self::AcceptWithNativeHostVerification => "accept-with-native-host-verification",
            Self::AgreeDistributedAdmission => "agree-distributed-admission",
            Self::SolveDistributedKrylov => "solve-distributed-krylov",
            Self::AgreeDistributedProducerReport => "agree-distributed-producer-report",
            Self::GatherDistributedOwnedCandidate => "gather-distributed-owned-candidate",
            Self::AgreeDistributedAcceptedResult => "agree-distributed-accepted-result",
            Self::AgreeDistributedReceipt => "agree-distributed-receipt",
            Self::ReplayTrueResidualOnHost => "replay-true-residual-on-host",
            Self::AcceptHostComplete => "accept-host-complete",
        }
    }
}

const HOST_LINEAR_STEPS: [ExecutionStepKind; 3] = [
    ExecutionStepKind::SolveWithNativeAcceptance,
    ExecutionStepKind::ReplayTrueResidualOnHost,
    ExecutionStepKind::AcceptHostComplete,
];

const CUDA_LINEAR_STEPS: [ExecutionStepKind; 9] = [
    ExecutionStepKind::TransferInputsToCuda,
    ExecutionStepKind::AwaitCudaInputsReady,
    ExecutionStepKind::SolveOnCuda,
    ExecutionStepKind::AwaitCudaSolveCompletion,
    ExecutionStepKind::TransferCandidateToHost,
    ExecutionStepKind::AwaitHostVisibility,
    ExecutionStepKind::AcceptWithNativeHostVerification,
    ExecutionStepKind::ReplayTrueResidualOnHost,
    ExecutionStepKind::AcceptHostComplete,
];

const DISTRIBUTED_LINEAR_STEPS: [ExecutionStepKind; 9] = [
    ExecutionStepKind::AgreeDistributedAdmission,
    ExecutionStepKind::SolveDistributedKrylov,
    ExecutionStepKind::AgreeDistributedProducerReport,
    ExecutionStepKind::GatherDistributedOwnedCandidate,
    ExecutionStepKind::AcceptWithNativeHostVerification,
    ExecutionStepKind::AgreeDistributedAcceptedResult,
    ExecutionStepKind::ReplayTrueResidualOnHost,
    ExecutionStepKind::AgreeDistributedReceipt,
    ExecutionStepKind::AcceptHostComplete,
];

const ACCEPTED_OUTPUT_DOMAIN_V1: &[u8] = b"eqiora.accepted-host-output/v1\0";

/// Exact normalized `f64` identity of one accepted complete host output.
///
/// This L2 identity is domain-separated from durable artifact digests. It
/// binds an immutable execution receipt to the vector it accepted even when a
/// method-native consumer subsequently separates reconstruction from receipt
/// storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AcceptedOutputFingerprintV1([u8; 32]);

impl AcceptedOutputFingerprintV1 {
    /// Raw SHA-256 bytes for fixed-size agreement and later artifact projection.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Opaque, preflighted execution token.
///
/// Construction seals the exact portable graph, canonical system fingerprint,
/// solver plan, executor binding, and verifier workspace. It exposes no mutable
/// graph or way to replace any sealed input.
#[derive(Debug)]
pub struct AdmittedExecution<'system> {
    binding: DeploymentBinding,
    system: &'system CanonicalCsrSystemView,
    applied: Vec<f64>,
    residual: Vec<f64>,
    partials: Vec<f64>,
    kind: AdmittedExecutionKind<'system>,
}

#[derive(Debug)]
enum AdmittedExecutionKind<'system> {
    Host,
    Cuda {
        minimum_device_payload_bytes: usize,
    },
    DistributedHost {
        system: &'system DistributedLinearSystem,
        admission: DistributedAdmissionFingerprintV1,
    },
    DistributedLocalAction {
        system: &'system DistributedLinearSystem,
        admission: DistributedAdmissionFingerprintV1,
    },
}

impl<'system> AdmittedExecution<'system> {
    /// Seal one exact canonical host-linear execution after deployment binding.
    ///
    /// This is a low-level adapter boundary: the caller must supply the graph
    /// independently regenerated by the equation-aware finalizer that owns
    /// `system`. This crate rechecks graph/binding equality, mathematical
    /// properties, shape, and every later report, but does not infer Semantic
    /// identity from CSR coefficients.
    ///
    /// Verification workspace and the maximum fixed-order partial count are
    /// reserved here. Acceptance performs no further vector allocation.
    ///
    /// # Errors
    /// Returns `EQ0807` for graph/binding drift, operator-property drift, or a
    /// verifier workspace that cannot be reserved.
    pub fn admit_host_linear(
        realization: &PortableRealizationGraph,
        system: &'system CanonicalCsrSystemView,
        binding: DeploymentBinding,
    ) -> Result<Self, Diagnostic> {
        if binding.host_executor().is_none() {
            return Err(invalid("host admission requires a host deployment binding"));
        }
        if binding.realization() != realization {
            return Err(invalid(
                "deployment binding does not belong to the supplied portable Realization graph",
            ));
        }
        let contract = host_contract(realization)?;
        Self::admit_linear(
            realization,
            system,
            binding,
            contract.properties,
            AdmittedExecutionKind::Host,
        )
    }

    /// Seal one exact canonical one-device linear execution after allocation-
    /// free deployment binding.
    ///
    /// The first device slice fixes the initial solution to implicit zero. The
    /// canonical CSR fingerprint therefore names every admitted numerical
    /// input; arbitrary initial vectors remain outside this contract.
    ///
    /// # Errors
    /// Returns `EQ0807` for graph/binding/property drift or when the known
    /// resident payload alone exceeds total device memory. Vendor-specific
    /// external workspace remains a later runtime observation.
    pub fn admit_cuda_linear(
        realization: &PortableRealizationGraph,
        system: &'system CanonicalCsrSystemView,
        binding: DeploymentBinding,
    ) -> Result<Self, Diagnostic> {
        let Some(executor) = binding.cuda_executor() else {
            return Err(invalid(
                "device admission requires a CUDA deployment binding",
            ));
        };
        let contract = cuda_contract(realization)?;
        let minimum_device_payload_bytes = minimum_device_payload_bytes(system, contract.plan)?;
        let available =
            usize::try_from(executor.device().total_memory_bytes().get()).unwrap_or(usize::MAX);
        if minimum_device_payload_bytes > available {
            return Err(invalid(format!(
                "known device payload requires {minimum_device_payload_bytes} bytes, selected device reports {available} total bytes",
            )));
        }
        Self::admit_linear(
            realization,
            system,
            binding,
            contract.properties,
            AdmittedExecutionKind::Cuda {
                minimum_device_payload_bytes,
            },
        )
    }

    /// Seal one exact distributed algebra and its complete verifier after a
    /// transport-neutral process-group binding.
    ///
    /// The derived owner map, local layouts, halo plan, and complete CSR must
    /// share one identity. This admission runs before the L3 adapter reserves
    /// numerical communication workspace or enters its first collective.
    ///
    /// # Errors
    /// Returns `EQ0807` for graph, placement, complete-system, owner/layout,
    /// partition-count, property, or sole-plan drift.
    pub fn admit_distributed_linear(
        realization: &PortableRealizationGraph,
        system: &'system DistributedLinearSystem,
        complete: &'system CanonicalCsrSystemView,
        binding: DeploymentBinding,
    ) -> Result<Self, Diagnostic> {
        let Some(executor) = binding.distributed_executor() else {
            return Err(invalid(
                "distributed admission requires a process-group deployment binding",
            ));
        };
        let contract = distributed_contract(realization)?;
        if system.partition().count() != executor.partitions() {
            return Err(invalid(format!(
                "distributed owner map declares {} partitions, selected process group declares {}",
                system.partition().count(),
                executor.partitions()
            )));
        }
        if !system.matches_complete(complete) || system.properties() != contract.properties {
            return Err(invalid(
                "distributed system identity or properties contradict the complete canonical system",
            ));
        }
        let admission = system.admission_fingerprint(contract.plan)?;
        Self::admit_linear(
            realization,
            complete,
            binding,
            contract.properties,
            AdmittedExecutionKind::DistributedHost { system, admission },
        )
    }

    /// Seal one exact distributed algebra whose rank-local CSR action is
    /// placed on CUDA while host MPI retains Krylov and reduction ownership.
    ///
    /// This admits the same distributed system and complete host verifier as
    /// [`Self::admit_distributed_linear`]. The distinct entry point prevents a
    /// composite deployment from being accepted through the host-action
    /// contract and rechecks the exact reproducible MINRES tuple selected by
    /// the portable graph.
    ///
    /// # Errors
    /// Returns `EQ0807` for graph, composite placement, complete-system,
    /// owner/layout, partition-count, property, or sole-plan drift.
    pub fn admit_distributed_cuda_linear(
        realization: &PortableRealizationGraph,
        system: &'system DistributedLinearSystem,
        complete: &'system CanonicalCsrSystemView,
        binding: DeploymentBinding,
    ) -> Result<Self, Diagnostic> {
        let Some(executor) = binding.distributed_executor() else {
            return Err(invalid(
                "distributed CUDA admission requires a process-group deployment binding",
            ));
        };
        if binding.cuda_partition_placement().is_none()
            || binding.distributed_device_transport().is_none()
        {
            return Err(invalid(
                "distributed CUDA admission requires a partition-local device placement and explicit transport",
            ));
        }
        let contract = distributed_cuda_contract(realization)?;
        if system.partition().count() != executor.partitions() {
            return Err(invalid(format!(
                "distributed owner map declares {} partitions, selected process group declares {}",
                system.partition().count(),
                executor.partitions()
            )));
        }
        if !system.matches_complete(complete) || system.properties() != contract.properties {
            return Err(invalid(
                "distributed CUDA system identity or properties contradict the complete canonical system",
            ));
        }
        let admission = system.admission_fingerprint(contract.plan)?;
        Self::admit_linear(
            realization,
            complete,
            binding,
            contract.properties,
            AdmittedExecutionKind::DistributedLocalAction { system, admission },
        )
    }

    fn admit_linear(
        realization: &PortableRealizationGraph,
        system: &'system CanonicalCsrSystemView,
        binding: DeploymentBinding,
        properties: eqiora_solver::LinearOperatorProperties,
        kind: AdmittedExecutionKind<'system>,
    ) -> Result<Self, Diagnostic> {
        if binding.realization() != realization {
            return Err(invalid(
                "deployment binding does not belong to the supplied portable Realization graph",
            ));
        }
        if system.properties() != properties {
            return Err(invalid(
                "canonical system properties contradict the portable algebraic system",
            ));
        }
        if system.rows() != system.columns() {
            return Err(invalid(
                "linear execution requires a complete square canonical system",
            ));
        }
        let dimension = system.rows();
        let mut applied = Vec::new();
        applied
            .try_reserve_exact(dimension)
            .map_err(|_| invalid("could not reserve host verifier action workspace"))?;
        applied.resize(dimension, 0.0);
        let mut residual = Vec::new();
        residual
            .try_reserve_exact(dimension)
            .map_err(|_| invalid("could not reserve host verifier residual workspace"))?;
        residual.resize(dimension, 0.0);
        let partial_count = dimension.div_ceil(eqiora_solver::FixedOrderInnerProduct::CHUNK_LENGTH);
        let mut partials = Vec::new();
        partials
            .try_reserve_exact(partial_count)
            .map_err(|_| invalid("could not reserve host verifier reduction workspace"))?;
        partials.resize(partial_count, 0.0);
        Ok(Self {
            binding,
            system,
            applied,
            residual,
            partials,
            kind,
        })
    }

    /// Exact known resident payload admitted before any device allocation.
    ///
    /// This excludes vendor-reported external sparse workspace and is not a
    /// free-memory reservation or total-run memory claim.
    #[must_use]
    pub const fn minimum_device_payload_bytes(&self) -> Option<usize> {
        match &self.kind {
            AdmittedExecutionKind::Cuda {
                minimum_device_payload_bytes,
            } => Some(*minimum_device_payload_bytes),
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::DistributedHost { .. }
            | AdmittedExecutionKind::DistributedLocalAction { .. } => None,
        }
    }

    /// Exact immutable deployment selected before this token was constructed.
    ///
    /// This read-only seam exists for isolated L3 adapters; it does not allow
    /// rebinding or replacing the sealed graph.
    #[must_use]
    pub const fn binding(&self) -> &DeploymentBinding {
        &self.binding
    }

    /// Exact finalized canonical system borrowed by this token.
    #[must_use]
    pub const fn system(&self) -> &'system CanonicalCsrSystemView {
        self.system
    }

    /// Exact derived distributed system sealed beside the complete verifier.
    #[must_use]
    pub const fn distributed_system(&self) -> Option<&'system DistributedLinearSystem> {
        match &self.kind {
            AdmittedExecutionKind::DistributedHost { system, .. }
            | AdmittedExecutionKind::DistributedLocalAction { system, .. } => Some(*system),
            AdmittedExecutionKind::Host | AdmittedExecutionKind::Cuda { .. } => None,
        }
    }

    /// Exact distributed system whose rank-local action remains host-owned.
    #[must_use]
    pub const fn distributed_host_system(&self) -> Option<&'system DistributedLinearSystem> {
        match &self.kind {
            AdmittedExecutionKind::DistributedHost { system, .. } => Some(*system),
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::Cuda { .. }
            | AdmittedExecutionKind::DistributedLocalAction { .. } => None,
        }
    }

    /// Exact distributed system whose rank-local action is delegated.
    #[must_use]
    pub const fn distributed_local_action_system(
        &self,
    ) -> Option<&'system DistributedLinearSystem> {
        match &self.kind {
            AdmittedExecutionKind::DistributedLocalAction { system, .. } => Some(*system),
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::Cuda { .. }
            | AdmittedExecutionKind::DistributedHost { .. } => None,
        }
    }

    /// Exact distributed admission identity, when this is a process-group run.
    #[must_use]
    pub const fn distributed_admission(&self) -> Option<DistributedAdmissionFingerprintV1> {
        match &self.kind {
            AdmittedExecutionKind::DistributedHost { admission, .. }
            | AdmittedExecutionKind::DistributedLocalAction { admission, .. } => Some(*admission),
            AdmittedExecutionKind::Host | AdmittedExecutionKind::Cuda { .. } => None,
        }
    }

    /// Admission identity for a host-owned distributed action.
    #[must_use]
    pub const fn distributed_host_admission(&self) -> Option<DistributedAdmissionFingerprintV1> {
        match &self.kind {
            AdmittedExecutionKind::DistributedHost { admission, .. } => Some(*admission),
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::Cuda { .. }
            | AdmittedExecutionKind::DistributedLocalAction { .. } => None,
        }
    }

    /// Admission identity for a delegated rank-local distributed action.
    #[must_use]
    pub const fn distributed_local_action_admission(
        &self,
    ) -> Option<DistributedAdmissionFingerprintV1> {
        match &self.kind {
            AdmittedExecutionKind::DistributedLocalAction { admission, .. } => Some(*admission),
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::Cuda { .. }
            | AdmittedExecutionKind::DistributedHost { .. } => None,
        }
    }

    /// Sole solver plan inherited from the portable root.
    #[must_use]
    pub fn solver_plan(&self) -> SolverPlan {
        self.binding.solver_plan()
    }

    /// Seal an immutable receipt after exact report and residual reacceptance.
    ///
    /// # Errors
    /// Returns `EQ0807` if the candidate shape, plan, production execution,
    /// host verification topology, orientation, output identity, or
    /// independently recomputed true residual contradicts the admitted graph,
    /// system, or deployment.
    pub fn accept(
        mut self,
        solution: LinearSolution,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        if !matches!(&self.kind, AdmittedExecutionKind::Host)
            || self.binding.host_executor().is_none()
        {
            return Err(invalid(
                "host acceptance requires a host deployment binding",
            ));
        }
        if !matches!(
            solution.report().verification().topology(),
            ExecutionTopology::Host { .. }
        ) {
            return Err(invalid(
                "host execution requires a solver-native host verification placement",
            ));
        }
        let output = self.replay_solution(&solution)?;
        Ok(self.finish_acceptance(solution, output, AcceptedExecutionEvidence::Host))
    }

    /// Seal an immutable receipt for a device-produced solution after exact
    /// transfer, generation, fence, report, and residual reacceptance.
    ///
    /// # Errors
    /// Returns `EQ0807` for any substitution in the selected device/queue,
    /// transfer slots, logical solution generation, completion order, native
    /// serial verifier, provider report, or independently replayed residual.
    pub fn accept_cuda(
        mut self,
        solution: LinearSolution,
        trace: CudaLinearExecutionTrace,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        let Some(executor) = self.binding.cuda_executor() else {
            return Err(invalid(
                "device acceptance requires a CUDA deployment binding",
            ));
        };
        trace.validate_against(
            self.system,
            self.binding.solver_plan(),
            executor.device().id(),
            executor.queue(),
        )?;
        let minimum_payload = match &self.kind {
            AdmittedExecutionKind::Cuda {
                minimum_device_payload_bytes,
            } => *minimum_device_payload_bytes,
            AdmittedExecutionKind::Host
            | AdmittedExecutionKind::DistributedHost { .. }
            | AdmittedExecutionKind::DistributedLocalAction { .. } => {
                return Err(invalid(
                    "CUDA acceptance lost its admitted resident-payload lower bound",
                ));
            }
        };
        let observed_lower_bound = minimum_payload
            .checked_add(trace.external_sparse_workspace_bytes().max(1))
            .ok_or_else(|| invalid("observed CUDA device-memory lower bound overflowed"))?;
        let device_total =
            usize::try_from(executor.device().total_memory_bytes().get()).unwrap_or(usize::MAX);
        if observed_lower_bound > device_total {
            return Err(invalid(format!(
                "known device payload plus observed external sparse workspace requires at least {observed_lower_bound} bytes, selected device reports {device_total} total bytes",
            )));
        }
        if solution.report().verification() != ExecutionReport::host_serial() {
            return Err(invalid(
                "single-device execution v1 requires solver-native serial-host verification",
            ));
        }
        let output = self.replay_solution(&solution)?;
        Ok(self.finish_acceptance(
            solution,
            output,
            AcceptedExecutionEvidence::Cuda {
                minimum_device_payload_bytes: minimum_payload,
                trace: Box::new(trace),
            },
        ))
    }

    /// Seal an immutable distributed receipt after exact graph/system/layout,
    /// actual collective-trace, native host acceptance, result agreement, and
    /// independent complete-host residual replay.
    ///
    /// # Errors
    /// Returns `EQ0807` for any process-group, system, owner map, halo layout,
    /// admission, collective order, gather extent, provider report, native
    /// verifier, or independent residual contradiction.
    pub fn accept_distributed(
        mut self,
        solution: LinearSolution,
        trace: DistributedLinearExecutionTrace,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        let Some(executor) = self.binding.distributed_executor() else {
            return Err(invalid(
                "distributed acceptance requires a process-group deployment binding",
            ));
        };
        let (system, admission) = match &self.kind {
            AdmittedExecutionKind::DistributedHost { system, admission }
            | AdmittedExecutionKind::DistributedLocalAction { system, admission } => {
                (*system, *admission)
            }
            AdmittedExecutionKind::Host | AdmittedExecutionKind::Cuda { .. } => {
                return Err(invalid(
                    "distributed acceptance requires a distributed system admission",
                ));
            }
        };
        if trace.system() != self.system.agreement_fingerprint()
            || trace.partition() != system.partition_identity()
            || trace.layout() != system.layout_identity()
            || trace.admission() != admission
            || trace.process_group() != executor.process_group()
            || trace.partitions() != executor.partitions()
            || trace.workers_per_partition() != executor.workers_per_partition()
            || trace.owner_gather_dimension() != self.system.columns()
        {
            return Err(invalid(
                "distributed execution trace contradicts its sealed system, layout, process group, or owner gather",
            ));
        }
        if solution.report().verification() != ExecutionReport::host_serial() {
            return Err(invalid(
                "distributed execution v1 requires native serial-host verification on every partition",
            ));
        }
        let output = self.replay_solution(&solution)?;
        Ok(self.finish_acceptance(
            solution,
            output,
            AcceptedExecutionEvidence::Distributed {
                trace: Box::new(trace),
            },
        ))
    }

    fn replay_solution(
        &mut self,
        solution: &LinearSolution,
    ) -> Result<AcceptedOutputFingerprintV1, Diagnostic> {
        let values = solution.values();
        let report = solution.report();
        let expected_execution = self.binding.execution();
        if report.solver_provider() != self.binding.solver_provider()
            || report.execution_provider() != self.binding.execution_provider()
        {
            return Err(invalid(
                "solve report provider provenance contradicts the selected deployment binding",
            ));
        }
        if report.verification_provider() != self.binding.verification_provider() {
            return Err(invalid(
                "solve report verifier provenance contradicts the admitted execution contract",
            ));
        }
        if report.backend() != self.binding.backend() {
            return Err(invalid(
                "solve report backend contradicts the admitted provider",
            ));
        }
        if report.solver_plan() != self.binding.solver_plan() {
            return Err(invalid(
                "solve report substituted a solver plan after execution admission",
            ));
        }
        if report.execution() != expected_execution {
            return Err(invalid(
                "solve report execution contradicts the selected deployment binding",
            ));
        }
        if report.orientation() != LinearOperatorOrientation::Normal {
            return Err(invalid(
                "linear execution receipt requires a normally oriented canonical operator",
            ));
        }
        if values.len() != self.system.columns() {
            return Err(invalid(format!(
                "accepted candidate has {} value(s) for canonical dimension {}",
                values.len(),
                self.system.columns()
            )));
        }
        self.system.apply(values, &mut self.applied)?;
        for ((residual, right), applied) in self
            .residual
            .iter_mut()
            .zip(self.system.right_hand_side())
            .zip(&self.applied)
        {
            *residual = right - applied;
        }
        let action = FixedOrderInnerProduct::new(&self.residual, &self.residual)?;
        if action.partial_count() != self.partials.len() {
            return Err(invalid(
                "admitted verifier partial count changed after system capture",
            ));
        }
        for (index, partial) in self.partials.iter_mut().enumerate() {
            *partial = action.evaluate_partial(index)?;
        }
        let true_residual = action.finish(&self.partials)?.sqrt();
        if true_residual.to_bits() != report.true_residual_norm().to_bits() {
            return Err(invalid(format!(
                "independent host residual {true_residual:e} contradicts reported residual {:e}",
                report.true_residual_norm()
            )));
        }
        let right_hand_side = self.system.right_hand_side();
        let right_norm_action = FixedOrderInnerProduct::new(right_hand_side, right_hand_side)?;
        if right_norm_action.partial_count() != self.partials.len() {
            return Err(invalid(
                "admitted right-hand-side reduction shape changed after system capture",
            ));
        }
        for (index, partial) in self.partials.iter_mut().enumerate() {
            *partial = right_norm_action.evaluate_partial(index)?;
        }
        let residual_target = self
            .binding
            .solver_plan()
            .residual_target(right_norm_action.finish(&self.partials)?.sqrt())?;
        if residual_target.to_bits() != report.residual_target().to_bits() {
            return Err(invalid(format!(
                "independent host residual target {residual_target:e} contradicts reported target {:e}",
                report.residual_target()
            )));
        }
        if true_residual > residual_target {
            return Err(invalid(
                "independent host residual exceeds the accepted report target",
            ));
        }
        accepted_output_fingerprint(values)
    }

    fn finish_acceptance(
        self,
        solution: LinearSolution,
        output: AcceptedOutputFingerprintV1,
        evidence: AcceptedExecutionEvidence,
    ) -> AcceptedLinearExecution {
        let report = solution.report().clone();
        let receipt = ExecutionReceipt {
            operator: self.system.agreement_fingerprint(),
            output,
            dimension: self.system.columns(),
            plan: self.binding.solver_plan(),
            binding: Arc::new(self.binding),
            report,
            acceptance_verification: ExecutionReport::host_serial(),
            evidence,
        };
        AcceptedLinearExecution { solution, receipt }
    }
}

/// Accepted solution retained with its immutable execution receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedLinearExecution {
    solution: LinearSolution,
    receipt: ExecutionReceipt,
}

impl AcceptedLinearExecution {
    /// Accepted numerical solution without copying.
    #[must_use]
    pub const fn solution(&self) -> &LinearSolution {
        &self.solution
    }

    /// Exact deployment, operator, plan, and DAG receipt.
    #[must_use]
    pub const fn receipt(&self) -> &ExecutionReceipt {
        &self.receipt
    }

    /// Consume the outcome without cloning the method-native solution.
    #[must_use]
    pub fn into_parts(self) -> (LinearSolution, ExecutionReceipt) {
        (self.solution, self.receipt)
    }
}

/// Immutable accepted linear-execution evidence with a complete host output.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    binding: Arc<DeploymentBinding>,
    operator: CanonicalCsrAgreementFingerprintV1,
    output: AcceptedOutputFingerprintV1,
    dimension: usize,
    plan: SolverPlan,
    report: SolveReport,
    acceptance_verification: ExecutionReport,
    evidence: AcceptedExecutionEvidence,
}

#[derive(Debug, Clone, PartialEq)]
enum AcceptedExecutionEvidence {
    Host,
    Cuda {
        minimum_device_payload_bytes: usize,
        trace: Box<CudaLinearExecutionTrace>,
    },
    Distributed {
        trace: Box<DistributedLinearExecutionTrace>,
    },
}

impl ExecutionReceipt {
    /// Exact deployment and portable Realization binding.
    #[must_use]
    pub fn binding(&self) -> &DeploymentBinding {
        &self.binding
    }

    /// Share an exactly equal immutable deployment authority without changing
    /// this receipt's operator, output, report or acceptance evidence.
    ///
    /// # Errors
    /// Rejects a foreign deployment, including any changed portable graph,
    /// provider, execution placement or verification authority.
    pub fn with_shared_binding(mut self, authority: &Self) -> Result<Self, Diagnostic> {
        if self.binding != authority.binding {
            return Err(invalid(
                "cannot share a foreign execution receipt deployment binding",
            ));
        }
        self.binding = Arc::clone(&authority.binding);
        Ok(self)
    }

    /// Exact solver implementation and libraries matched at acceptance.
    #[must_use]
    pub fn solver_provider(&self) -> SolverProvider {
        self.binding.solver_provider()
    }

    /// Exact primary execution implementation and libraries matched at
    /// acceptance.
    #[must_use]
    pub fn execution_provider(&self) -> ExecutionProvider {
        self.binding.execution_provider()
    }

    /// Exact solver-native verification implementation and libraries matched
    /// at acceptance.
    #[must_use]
    pub const fn verification_provider(&self) -> ExecutionProvider {
        self.report.verification_provider()
    }

    /// Exact canonical CSR system fingerprint.
    #[must_use]
    pub const fn operator(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.operator
    }

    /// Exact complete vector accepted by this receipt.
    #[must_use]
    pub const fn output(&self) -> AcceptedOutputFingerprintV1 {
        self.output
    }

    /// Complete accepted host-vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Sole solver plan inherited from the portable graph.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.plan
    }

    /// Solver-native accepted report retained without rewriting its verifier.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }

    /// Placement of the Eqiora-owned receipt replay, distinct from the
    /// solver-native verifier recorded by [`SolveReport::verification`].
    ///
    /// This is algorithm placement evidence, not a dynamically selected
    /// provider role.
    #[must_use]
    pub const fn acceptance_verification(&self) -> ExecutionReport {
        self.acceptance_verification
    }

    /// Known resident payload admitted before device allocation, when this was
    /// a device execution. Vendor external sparse workspace is separate.
    #[must_use]
    pub const fn minimum_device_payload_bytes(&self) -> Option<usize> {
        match &self.evidence {
            AcceptedExecutionEvidence::Cuda {
                minimum_device_payload_bytes,
                ..
            } => Some(*minimum_device_payload_bytes),
            AcceptedExecutionEvidence::Host | AcceptedExecutionEvidence::Distributed { .. } => None,
        }
    }

    /// Exact movement/generation/fence trace for a CUDA execution.
    #[must_use]
    pub const fn cuda_trace(&self) -> Option<CudaLinearExecutionTrace> {
        match &self.evidence {
            AcceptedExecutionEvidence::Cuda { trace, .. } => Some(**trace),
            AcceptedExecutionEvidence::Host | AcceptedExecutionEvidence::Distributed { .. } => None,
        }
    }

    /// Exact system/layout/process-group/collective trace for a distributed
    /// execution.
    #[must_use]
    pub fn distributed_trace(&self) -> Option<&DistributedLinearExecutionTrace> {
        match &self.evidence {
            AcceptedExecutionEvidence::Distributed { trace } => Some(trace.as_ref()),
            AcceptedExecutionEvidence::Host | AcceptedExecutionEvidence::Cuda { .. } => None,
        }
    }

    /// Read-only fixed execution DAG view.
    #[must_use]
    pub const fn dag(&self) -> ExecutionDagView<'_> {
        ExecutionDagView { receipt: self }
    }
}

/// Read-only view of the placement-specific execution DAG and its sealed subject.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionDagView<'a> {
    receipt: &'a ExecutionReceipt,
}

impl ExecutionDagView<'_> {
    /// Canonical topological order. Each step consumes the preceding step's
    /// candidate or completion; callers cannot edit these dependencies.
    #[must_use]
    pub const fn steps(self) -> &'static [ExecutionStepKind] {
        match &self.receipt.evidence {
            AcceptedExecutionEvidence::Host => &HOST_LINEAR_STEPS,
            AcceptedExecutionEvidence::Cuda { .. } => &CUDA_LINEAR_STEPS,
            AcceptedExecutionEvidence::Distributed { .. } => &DISTRIBUTED_LINEAR_STEPS,
        }
    }

    /// Exact canonical operator executed by this DAG.
    #[must_use]
    pub const fn operator(self) -> CanonicalCsrAgreementFingerprintV1 {
        self.receipt.operator
    }

    /// Exact solver plan consumed by the solve step.
    #[must_use]
    pub const fn solver_plan(self) -> SolverPlan {
        self.receipt.plan
    }
}

fn accepted_output_fingerprint(values: &[f64]) -> Result<AcceptedOutputFingerprintV1, Diagnostic> {
    let length = u64::try_from(values.len())
        .map_err(|_| invalid("accepted host output length exceeds portable u64 identity"))?;
    let mut hash = Sha256::new();
    hash.update(ACCEPTED_OUTPUT_DOMAIN_V1);
    hash.update(length.to_le_bytes());
    for value in values {
        if !value.is_finite() {
            return Err(invalid("accepted host output contains a non-finite value"));
        }
        let normalized = if *value == 0.0 { 0.0 } else { *value };
        hash.update(normalized.to_bits().to_le_bytes());
    }
    Ok(AcceptedOutputFingerprintV1(hash.finalize().into()))
}
