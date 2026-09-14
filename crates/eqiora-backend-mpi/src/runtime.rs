use std::fmt;
use std::mem;
use std::num::NonZeroUsize;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_distributed::{
    DistributedAdmissionFingerprintV1, DistributedLinearProblem, DistributedLinearSystem,
    LocalCsrShard, LocalLinearSolution, Partition, PartitionId,
};
use eqiora_execution::{
    AcceptedLinearExecution, AdmittedExecution, DistributedCollectiveStepV1,
    DistributedExecutionPhaseV1, DistributedLinearExecutionTrace,
};
use eqiora_solver::{
    BackendId, CanonicalCsrSystemView, ConvergenceReason, DiagonalAvailability, ExecutionId,
    ExecutionProvider, ExecutionReport, LinearAcceptanceWorkspace, LinearOperatorOrientation,
    LinearOperatorProperties, LinearSolution, LinearSolver, PreconditionerPolicy, ProviderLibrary,
    ReductionPolicy, SERIAL_LINEAR_EXECUTION, SolveReport, SolverCapabilities, SolverCapability,
    SolverPlan, SolverProvider,
};
use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::{Communicator, SimpleCommunicator};
use mpi::traits::{CommunicatorCollectives, Destination, Source};
use sha2::{Digest, Sha256};

#[cfg(feature = "mpi-test-hooks")]
use crate::protocol::ProviderSummarySubstitution;
use crate::{
    AdmissionRecordV1, CollectivePhaseV1, CollectiveStepV1, DistributedProtocolFailureV1,
    MPI_ADAPTER_VERSION, MPI_RS_VERSION, MpiCollectiveTraceV1, OwnedGatherPlanV1, PhaseStatusV1,
    ProducerReportSummaryV2, evaluate_admission, evaluate_phase_statuses,
};

mod krylov_workspace;

/// Stable backend identity for Eqiora's distributed Krylov solvers over MPI.
pub const MPI_DISTRIBUTED_KRYLOV_BACKEND: BackendId = BackendId::new("eqiora.mpi.krylov");

/// Stable execution identity for MPI communicator actions.
pub const MPI_EXECUTION: ExecutionId = ExecutionId::new("eqiora.mpi");

const MPI_EXECUTION_LIBRARIES: &[ProviderLibrary] =
    &[ProviderLibrary::new("mpi-rs", MPI_RS_VERSION)];

/// Exact declared distributed Krylov solver release.
pub const MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER: SolverProvider =
    SolverProvider::new(MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_ADAPTER_VERSION, &[]);

/// Exact declared MPI execution release and transport binding.
pub const MPI_EXECUTION_PROVIDER: ExecutionProvider =
    ExecutionProvider::new(MPI_EXECUTION, MPI_ADAPTER_VERSION, MPI_EXECUTION_LIBRARIES);

const ACCEPTED_SOLUTION_DOMAIN_V2: &[u8] = b"eqiora.mpi-accepted-solution/v2\0";
const COMPOSED_EXECUTION_SUMMARY_DOMAIN_V1: &[u8] = b"eqiora.mpi-composed-execution-summary/v1\0";
const EXECUTION_RECEIPT_AGREEMENT_DOMAIN_V2: &[u8] = b"eqiora.mpi-execution-receipt-agreement/v2\0";
const EXECUTION_RECEIPT_SUMMARY_BYTES: usize = 32;
const RANK_DEVICE_TOPOLOGY_DOMAIN_V1: &[u8] = b"eqiora.mpi-rank-device-topology/v1\0";
const RANK_DEVICE_RECORD_BYTES: usize = 18;

/// One rank-local physical device observation supplied by a composed adapter.
///
/// The ordinal is the process-visible resource ordinal. The physical identity
/// is an opaque fixed-width identity obtained from the device runtime; this
/// MPI adapter assigns it no vendor-specific meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankLocalDeviceV1 {
    ordinal: u16,
    physical_identity: [u8; 16],
}

impl RankLocalDeviceV1 {
    /// Construct one local observation for collective topology agreement.
    #[must_use]
    pub const fn new(ordinal: u16, physical_identity: [u8; 16]) -> Self {
        Self {
            ordinal,
            physical_identity,
        }
    }

    /// Process-visible device ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u16 {
        self.ordinal
    }

    /// Opaque physical device identity.
    #[must_use]
    pub const fn physical_identity(self) -> [u8; 16] {
        self.physical_identity
    }

    const fn encode(self) -> [u8; RANK_DEVICE_RECORD_BYTES] {
        let ordinal = self.ordinal.to_be_bytes();
        let mut bytes = [0_u8; RANK_DEVICE_RECORD_BYTES];
        bytes[0] = ordinal[0];
        bytes[1] = ordinal[1];
        let mut index = 0;
        while index < self.physical_identity.len() {
            bytes[index + 2] = self.physical_identity[index];
            index += 1;
        }
        bytes
    }

    fn decode(bytes: [u8; RANK_DEVICE_RECORD_BYTES]) -> Self {
        let mut physical_identity = [0_u8; 16];
        physical_identity.copy_from_slice(&bytes[2..]);
        Self {
            ordinal: u16::from_be_bytes([bytes[0], bytes[1]]),
            physical_identity,
        }
    }
}

/// Common rank-ordered device topology admitted by one MPI execution group.
///
/// V1 deliberately admits exactly one process-visible device at ordinal zero
/// per rank and requires distinct, nonzero physical identities. This evidence
/// is transport-owned and never enters the portable Realization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpiRankDeviceTopologyV1 {
    devices: Vec<RankLocalDeviceV1>,
    fingerprint: [u8; 32],
}

impl MpiRankDeviceTopologyV1 {
    /// Rank-ordered device observations, one per execution partition.
    #[must_use]
    pub fn devices(&self) -> &[RankLocalDeviceV1] {
        &self.devices
    }

    /// Content fingerprint over the rank count and ordered observations.
    #[must_use]
    pub const fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
}

/// Eqiora-owned mirror of MPI's provided thread-support level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MpiThreadSupport {
    /// Every rank is single-threaded.
    Single,
    /// MPI calls occur only on the initializing thread.
    Funneled,
    /// Calls from multiple threads are externally serialized.
    Serialized,
    /// Concurrent MPI calls are admitted.
    Multiple,
}

impl MpiThreadSupport {
    const fn from_mpi(value: Threading) -> Self {
        match value {
            Threading::Single => Self::Single,
            Threading::Funneled => Self::Funneled,
            Threading::Serialized => Self::Serialized,
            Threading::Multiple => Self::Multiple,
        }
    }
}

/// One application-owned MPI communicator duplicated for Eqiora admissions.
pub struct MpiExecutionGroup {
    communicator: SimpleCommunicator,
    partitions: NonZeroUsize,
    partition: PartitionId,
    thread_support: MpiThreadSupport,
    preparation_statuses: Vec<u8>,
    receipt_summary_bytes: Vec<u8>,
    rank_device_bytes: Vec<u8>,
    #[cfg(feature = "mpi-test-hooks")]
    fault: Option<TestFault>,
}

impl MpiExecutionGroup {
    pub(crate) const fn communicator(&self) -> &SimpleCommunicator {
        &self.communicator
    }

    /// Duplicate an initialized communicator and validate thread support.
    ///
    /// MPI initialization/finalization remains entirely with the application.
    /// `provided` must be the level returned by `initialize_with_threading`.
    ///
    /// # Errors
    /// Returns `EQ0807` if rank/size do not fit the Eqiora partition contract
    /// or the provided threading level is below `required`.
    pub fn duplicate<C: Communicator>(
        communicator: &C,
        provided: Threading,
        required: MpiThreadSupport,
    ) -> Result<Self, Diagnostic> {
        let provided = MpiThreadSupport::from_mpi(provided);
        if provided < required {
            return Err(invalid_realization(format!(
                "MPI provided {provided:?} thread support, below required {required:?}"
            )));
        }
        let size = usize::try_from(communicator.size())
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| invalid_realization("MPI communicator size is not a positive usize"))?;
        let rank = usize::try_from(communicator.rank())
            .map_err(|_| invalid_realization("MPI rank is negative or exceeds usize"))?;
        if rank >= size.get() {
            return Err(invalid_realization(
                "MPI rank lies outside its communicator size",
            ));
        }
        let preparation_statuses = zeroed(size.get(), "MPI preparation statuses")?;
        let receipt_summary_bytes = zeroed(
            checked_extent(
                EXECUTION_RECEIPT_SUMMARY_BYTES,
                size.get(),
                "MPI execution receipt summaries",
            )?,
            "MPI execution receipt summaries",
        )?;
        let rank_device_bytes = zeroed(
            checked_extent(
                RANK_DEVICE_RECORD_BYTES,
                size.get(),
                "MPI rank-device records",
            )?,
            "MPI rank-device records",
        )?;
        Ok(Self {
            communicator: communicator.duplicate(),
            partitions: size,
            partition: PartitionId::new(rank),
            thread_support: provided,
            preparation_statuses,
            receipt_summary_bytes,
            rank_device_bytes,
            #[cfg(feature = "mpi-test-hooks")]
            fault: TestFault::from_environment(rank),
        })
    }

    /// Number of ranks/partitions in this execution group.
    #[must_use]
    pub const fn partitions(&self) -> NonZeroUsize {
        self.partitions
    }

    /// Local rank represented as an Eqiora partition identity.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Thread-support level actually provided at MPI initialization.
    #[must_use]
    pub const fn thread_support(&self) -> MpiThreadSupport {
        self.thread_support
    }

    /// Exact numerical policies admitted by the current distributed evidence
    /// slice.
    #[must_use]
    pub fn solver_capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ])
        .expect("MPI distributed solver exact capability set is nonempty")
    }

    /// Collectively bind one distinct physical device to every MPI rank.
    ///
    /// V1 requires the launcher to expose exactly one device to each process,
    /// making the local ordinal zero on every rank. The fixed-width physical
    /// identity prevents two ranks on the same host from silently selecting
    /// the same device. The returned rank-ordered evidence and fingerprint are
    /// identical on every rank.
    ///
    /// # Errors
    /// Returns a common `EQ0807` diagnostic when an ordinal is not zero, a
    /// physical identity is zero or duplicated, allocation fails, or the
    /// gathered record count contradicts the execution group.
    pub fn agree_rank_device_topology(
        &mut self,
        local: RankLocalDeviceV1,
    ) -> Result<MpiRankDeviceTopologyV1, Diagnostic> {
        let prepared = reserved(self.partitions.get(), "MPI rank-device topology");
        let local_validation = if prepared.is_err() {
            Err(invalid_realization(
                "could not reserve MPI rank-device topology",
            ))
        } else if local.ordinal == 0 && local.physical_identity.iter().any(|byte| *byte != 0) {
            Ok(())
        } else {
            Err(invalid_realization(
                "MPI rank-device topology requires local ordinal zero and a nonzero physical identity",
            ))
        };
        self.agree_local_result(&local_validation, CollectivePhaseV1::Admission)?;
        let mut devices = prepared.map_err(|_| {
            invalid_realization(
                "local rank-device preparation failed despite all-rank readiness agreement",
            )
        })?;

        fixed_all_gather(
            &self.communicator,
            &local.encode(),
            &mut self.rank_device_bytes,
        );
        let rank_device_bytes = self.rank_device_bytes.as_slice();
        let gathered = (|| {
            for bytes in rank_device_bytes.as_chunks::<RANK_DEVICE_RECORD_BYTES>().0 {
                let encoded = <[u8; RANK_DEVICE_RECORD_BYTES]>::try_from(bytes.as_slice())
                    .map_err(|_| {
                        invalid_realization("MPI rank-device record extent is inconsistent")
                    })?;
                devices.push(RankLocalDeviceV1::decode(encoded));
            }
            if devices.len() != self.partitions.get()
                || devices.iter().any(|device| {
                    device.ordinal != 0 || device.physical_identity.iter().all(|byte| *byte == 0)
                })
            {
                return Err(invalid_realization(
                    "MPI rank-device records contradict the execution-group topology",
                ));
            }
            for (rank, device) in devices.iter().enumerate() {
                if devices[..rank]
                    .iter()
                    .any(|prior| prior.physical_identity == device.physical_identity)
                {
                    return Err(invalid_realization(
                        "MPI ranks must bind distinct physical device identities",
                    ));
                }
            }
            Ok(())
        })();
        self.agree_local_result(&gathered, CollectivePhaseV1::Admission)?;
        gathered.map_err(|_| {
            invalid_realization(
                "local rank-device validation failed despite all-rank topology agreement",
            )
        })?;

        let rank_count = u64::try_from(devices.len())
            .map_err(|_| invalid_realization("MPI rank-device count exceeds portable u64"))?;
        let mut hash = Sha256::new();
        hash.update(RANK_DEVICE_TOPOLOGY_DOMAIN_V1);
        hash.update(rank_count.to_be_bytes());
        for device in &devices {
            hash.update(device.encode());
        }
        Ok(MpiRankDeviceTopologyV1 {
            devices,
            fingerprint: hash.finalize().into(),
        })
    }

    /// Agree readiness of fallible rank-local work in a composed adapter.
    ///
    /// A composed adapter calls this before entering its next MPI collective:
    /// after local capture/allocation/session preparation and again after any
    /// fallible post-execution evidence validation or summary construction.
    /// This prevents a ready rank from entering a collective while another
    /// rank returns from local work. No payload or backend-specific identity
    /// crosses this seam.
    ///
    /// # Errors
    /// Returns a common admission-phase diagnostic when any rank reports local
    /// readiness failure.
    pub fn agree_composed_local_readiness(
        &mut self,
        local: &Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        self.agree_local_result(local, CollectivePhaseV1::Admission)
    }

    /// Require one fixed-width composed-execution summary on every rank.
    ///
    /// A composition adapter calls this after the ordinary MPI execution
    /// receipt has been independently agreed. The adapter's summary can bind
    /// topology, transport, local-action counts, and that receipt without
    /// exposing a general byte registry or MPI communicator. This seam applies
    /// its own domain separation before comparison.
    ///
    /// # Errors
    /// Returns a common `EQ0802` diagnostic when rank summaries differ.
    pub fn agree_composed_execution_summary(
        &mut self,
        local_summary: [u8; 32],
    ) -> Result<(), Diagnostic> {
        let mut hash = Sha256::new();
        hash.update(COMPOSED_EXECUTION_SUMMARY_DOMAIN_V1);
        hash.update(local_summary);
        let agreed: [u8; 32] = hash.finalize().into();
        fixed_all_gather(&self.communicator, &agreed, &mut self.receipt_summary_bytes);
        let bytes = self.receipt_summary_bytes.as_slice();
        let summaries = bytes.as_chunks::<EXECUTION_RECEIPT_SUMMARY_BYTES>().0;
        if summaries.iter().all(|candidate| candidate == &agreed[..]) {
            Ok(())
        } else {
            Err(solve_failed(
                "composed MPI execution summaries disagree across partitions",
            ))
        }
    }

    /// Collectively seal one system, complete verifier, and sole solver plan.
    ///
    /// All dynamically sized communication and verification storage is
    /// fallibly reserved before the first admission record is exchanged. A
    /// scalar readiness reduction makes allocation/validation failure common
    /// before any rank can enter the record gather.
    ///
    /// # Errors
    /// Returns the same stable diagnostic on every participating rank for a
    /// system, layout, plan, verifier, count, or workspace contradiction.
    pub fn admit<'group, 'model>(
        &'group mut self,
        system: &'model DistributedLinearSystem,
        complete: &'model CanonicalCsrSystemView,
        plan: SolverPlan,
    ) -> Result<AdmittedDistributedRun<'group, 'model>, Diagnostic> {
        let prepared = PreparedRun::new(self, system, complete, plan);
        let failure = prepared
            .as_ref()
            .err()
            .map(DistributedProtocolFailureV1::from_diagnostic);
        agree_preparation(
            &self.communicator,
            self.partition,
            failure,
            CollectivePhaseV1::Admission,
            &mut self.preparation_statuses,
        )?;
        let mut prepared = prepared.map_err(|_| {
            invalid_realization("local admission failed despite all-rank readiness agreement")
        })?;

        let record = AdmissionRecordV1::new(
            self.partitions,
            self.partition,
            prepared.fingerprint,
            DistributedProtocolFailureV1::Ready,
        )?;
        fixed_all_gather(
            &self.communicator,
            &record.encode(),
            &mut prepared.buffers.fixed_bytes,
        );
        prepared.buffers.admission_records.clear();
        for bytes in prepared
            .buffers
            .fixed_bytes
            .as_chunks::<{ AdmissionRecordV1::ENCODED_LEN }>()
            .0
        {
            let encoded = <[u8; AdmissionRecordV1::ENCODED_LEN]>::try_from(bytes.as_slice())
                .map_err(|_| invalid_realization("admission record extent is inconsistent"))?;
            prepared
                .buffers
                .admission_records
                .push(AdmissionRecordV1::decode(encoded)?);
        }
        evaluate_admission(&prepared.buffers.admission_records, self.partitions)?;

        Ok(AdmittedDistributedRun {
            group: self,
            system,
            complete,
            problem: prepared.problem,
            complete_problem: prepared.complete_problem,
            plan,
            fingerprint: prepared.fingerprint,
            gather_plan: prepared.gather_plan,
            buffers: prepared.buffers,
        })
    }
}

impl fmt::Debug for MpiExecutionGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MpiExecutionGroup")
            .field("partitions", &self.partitions)
            .field("partition", &self.partition)
            .field("thread_support", &self.thread_support)
            .finish_non_exhaustive()
    }
}

/// L3 adapter seam consuming one exact L2 distributed execution admission.
///
/// The process group cannot reselect the portable graph, complete system,
/// owner/halo layout, solver provider, process count, or solver plan. MPI
/// communicator and thread-support observations remain private runtime/Run
/// evidence rather than becoming portable Realization fields.
pub trait MpiAdmittedExecutionAdapter {
    /// Execute one graph-bound distributed solve and return its common L2
    /// receipt with a transport-normalized actual collective trace.
    ///
    /// # Errors
    /// Returns a stable diagnostic for any binding/runtime substitution,
    /// collective admission or numerical failure, trace contradiction, or
    /// independent complete-host receipt replay failure.
    fn execute_admitted(
        &mut self,
        admitted: AdmittedExecution<'_>,
    ) -> Result<AcceptedLinearExecution, Diagnostic>;

    /// Execute one graph-bound distributed solve while delegating only the
    /// rank-local owned-row CSR action.
    ///
    /// MPI retains ownership of admission, halo exchange, Krylov state,
    /// reductions, gather, and host acceptance. The injected action is called
    /// only after the local halo is complete; its result is agreed across all
    /// ranks at [`CollectivePhaseV1::LocalAction`] before execution continues.
    ///
    /// # Errors
    /// Returns the same stable diagnostics as [`Self::execute_admitted`], plus
    /// any action failure normalized through the collective local-action phase.
    fn execute_admitted_with_local_action(
        &mut self,
        admitted: AdmittedExecution<'_>,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<AcceptedLinearExecution, Diagnostic>;
}

/// Narrow rank-local sparse-action seam for composed execution adapters.
///
/// Implementations must perform no MPI communication. The shard and vectors
/// are the exact objects sealed by distributed admission: `owned_input` and
/// `ghosts` follow the shard layout's ascending orders, and `owned_output`
/// contains one result for each owned row. MPI synchronizes the returned
/// status before any subsequent collective phase.
pub trait MpiRankLocalCsrAction {
    /// Apply the admitted owned-row CSR shard.
    ///
    /// # Errors
    /// Returns a diagnostic when the local backend cannot produce the exact
    /// finite owned-row action. MPI converts it into a common-rank failure at
    /// the existing local-action boundary.
    fn apply_owned_rows(
        &mut self,
        shard: LocalCsrShard<'_>,
        owned_input: &[f64],
        ghosts: &[f64],
        owned_output: &mut [f64],
    ) -> Result<(), Diagnostic>;
}

struct HostLocalCsrAction;

impl MpiRankLocalCsrAction for HostLocalCsrAction {
    fn apply_owned_rows(
        &mut self,
        shard: LocalCsrShard<'_>,
        owned_input: &[f64],
        ghosts: &[f64],
        owned_output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        shard.apply(owned_input, ghosts, owned_output)
    }
}

#[derive(Debug, Clone, Copy)]
enum LocalActionAuthority {
    Host,
    Delegated,
}

impl MpiAdmittedExecutionAdapter for MpiExecutionGroup {
    fn execute_admitted(
        &mut self,
        admitted: AdmittedExecution<'_>,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        self.execute_admitted_with_action(
            admitted,
            LocalActionAuthority::Host,
            &mut HostLocalCsrAction,
        )
    }

    fn execute_admitted_with_local_action(
        &mut self,
        admitted: AdmittedExecution<'_>,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        self.execute_admitted_with_action(admitted, LocalActionAuthority::Delegated, action)
    }
}

impl MpiExecutionGroup {
    fn execute_admitted_with_action(
        &mut self,
        admitted: AdmittedExecution<'_>,
        authority: LocalActionAuthority,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<AcceptedLinearExecution, Diagnostic> {
        let preflight: Result<_, Diagnostic> = (|| {
            let executor = admitted
                .binding()
                .distributed_executor()
                .ok_or_else(|| invalid_realization("MPI adapter requires a distributed binding"))?;
            if executor.solver_provider() != MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER
                || executor.execution_provider() != MPI_EXECUTION_PROVIDER
                || executor.process_group().ordinal() != 0
                || executor.partitions() != self.partitions
                || executor.workers_per_partition() != NonZeroUsize::MIN
            {
                return Err(invalid_realization(
                    "MPI execution group contradicts the selected distributed provider, logical slot, partition count, or worker shape",
                ));
            }
            let (distributed, expected_admission) = match authority {
                LocalActionAuthority::Host => (
                    admitted.distributed_host_system(),
                    admitted.distributed_host_admission(),
                ),
                LocalActionAuthority::Delegated => (
                    admitted.distributed_local_action_system(),
                    admitted.distributed_local_action_admission(),
                ),
            };
            let distributed = distributed.ok_or_else(|| {
                invalid_realization(
                    "MPI adapter received a distributed token with different local-action authority",
                )
            })?;
            let expected_admission = expected_admission.ok_or_else(|| {
                invalid_realization(
                    "MPI adapter received no admission identity for the selected local-action authority",
                )
            })?;
            Ok((
                executor.process_group(),
                executor.partitions(),
                executor.workers_per_partition(),
                distributed,
                expected_admission,
                admitted.system(),
                admitted.solver_plan(),
            ))
        })();
        self.agree_local_result(&preflight, CollectivePhaseV1::Admission)?;
        let (
            process_group,
            partitions,
            workers_per_partition,
            distributed,
            expected_admission,
            complete,
            plan,
        ) = preflight.map_err(|_| {
            invalid_realization(
                "local MPI binding failed despite all-rank preflight readiness agreement",
            )
        })?;
        let result = self
            .admit(distributed, complete, plan)?
            .solve_and_replicate_with_trace_using(action)?;
        let accepted = (|| {
            if result.trace().admission_fingerprint() != expected_admission {
                return Err(invalid_realization(
                    "MPI runtime admission identity contradicts the L2 execution token",
                ));
            }
            let trace = normalize_execution_trace(
                result.trace(),
                distributed,
                complete.columns(),
                process_group,
                partitions,
                workers_per_partition,
                plan,
            )?;
            let (solution, _) = result.into_parts();
            admitted.accept_distributed(solution, trace)
        })();
        self.agree_local_result(&accepted, CollectivePhaseV1::ResultAgreement)?;
        let accepted = accepted.map_err(|_| {
            solve_failed(
                "local post-solve validation failed despite all-rank replay readiness agreement",
            )
        })?;
        self.agree_execution_receipt(&accepted)?;
        Ok(accepted)
    }
}

impl MpiExecutionGroup {
    fn agree_local_result<T>(
        &mut self,
        local: &Result<T, Diagnostic>,
        phase: CollectivePhaseV1,
    ) -> Result<(), Diagnostic> {
        let failure = local
            .as_ref()
            .err()
            .map(DistributedProtocolFailureV1::from_diagnostic);
        agree_preparation(
            &self.communicator,
            self.partition,
            failure,
            phase,
            &mut self.preparation_statuses,
        )
    }

    fn agree_execution_receipt(
        &mut self,
        accepted: &AcceptedLinearExecution,
    ) -> Result<(), Diagnostic> {
        let summary = execution_receipt_summary(accepted);
        self.agree_local_result(&summary, CollectivePhaseV1::ResultAgreement)?;
        let summary = summary.map_err(|_| {
            solve_failed(
                "local receipt summary failed despite all-rank summary readiness agreement",
            )
        })?;
        fixed_all_gather(
            &self.communicator,
            &summary,
            &mut self.receipt_summary_bytes,
        );
        let bytes = self.receipt_summary_bytes.as_slice();
        let summaries = bytes.as_chunks::<EXECUTION_RECEIPT_SUMMARY_BYTES>().0;
        if summaries.iter().all(|candidate| candidate == &summary[..]) {
            Ok(())
        } else {
            Err(solve_failed(
                "independently replayed MPI execution receipts disagree across partitions",
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn normalize_execution_trace(
    trace: &MpiCollectiveTraceV1,
    system: &DistributedLinearSystem,
    complete_dimension: usize,
    process_group: eqiora_execution::ProcessGroupSlot,
    partitions: NonZeroUsize,
    workers_per_partition: NonZeroUsize,
    plan: SolverPlan,
) -> Result<DistributedLinearExecutionTrace, Diagnostic> {
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(trace.steps().len())
        .map_err(|_| invalid_realization("could not reserve normalized MPI collective trace"))?;
    for step in trace.steps() {
        let phase = match step.phase() {
            CollectivePhaseV1::Admission => DistributedExecutionPhaseV1::AdmissionAgreement,
            CollectivePhaseV1::Halo => DistributedExecutionPhaseV1::HaloReadiness,
            CollectivePhaseV1::LocalAction => DistributedExecutionPhaseV1::OwnedAction,
            CollectivePhaseV1::VectorUpdate => DistributedExecutionPhaseV1::OwnedVectorUpdate,
            CollectivePhaseV1::Reduction => DistributedExecutionPhaseV1::CollectiveReduction,
            CollectivePhaseV1::ProducerReport => {
                DistributedExecutionPhaseV1::ProducerReportAgreement
            }
            CollectivePhaseV1::GatherPreparation => {
                DistributedExecutionPhaseV1::OwnerGatherPreparation
            }
            CollectivePhaseV1::GatherValidation => {
                DistributedExecutionPhaseV1::OwnerGatherValidation
            }
            CollectivePhaseV1::HostAcceptance => DistributedExecutionPhaseV1::NativeHostAcceptance,
            CollectivePhaseV1::ResultAgreement => {
                DistributedExecutionPhaseV1::AcceptedResultAgreement
            }
        };
        steps.push(DistributedCollectiveStepV1::new(
            phase,
            step.iteration(),
            step.ordinal(),
        ));
    }
    DistributedLinearExecutionTrace::new(
        system.system_identity(),
        system.partition_identity(),
        system.layout_identity(),
        trace.admission_fingerprint(),
        process_group,
        partitions,
        workers_per_partition,
        complete_dimension,
        trace.capacity(),
        steps,
        plan,
    )
}

/// A collectively admitted, plan-sealed distributed execution.
///
/// This token is the only public route to MPI numerical communication. It
/// borrows the execution group mutably, preventing overlapping collective
/// streams, and owns every dynamically sized workspace required through host
/// reacceptance.
pub struct AdmittedDistributedRun<'group, 'model> {
    group: &'group mut MpiExecutionGroup,
    system: &'model DistributedLinearSystem,
    complete: &'model CanonicalCsrSystemView,
    problem: DistributedLinearProblem<'model>,
    complete_problem: eqiora_solver::LinearProblem<'model>,
    plan: SolverPlan,
    fingerprint: DistributedAdmissionFingerprintV1,
    gather_plan: OwnedGatherPlanV1,
    buffers: RunBuffers,
}

/// Complete host solution paired with its exact accepted MPI collective trace.
#[derive(Debug, Clone, PartialEq)]
pub struct MpiLinearSolveResult {
    solution: LinearSolution,
    trace: MpiCollectiveTraceV1,
}

impl MpiLinearSolveResult {
    /// Complete independently accepted host solution.
    #[must_use]
    pub const fn solution(&self) -> &LinearSolution {
        &self.solution
    }

    /// Admission and every successfully synchronized collective boundary.
    #[must_use]
    pub const fn trace(&self) -> &MpiCollectiveTraceV1 {
        &self.trace
    }

    /// Consume the result without copying its complete vector or trace.
    #[must_use]
    pub fn into_parts(self) -> (LinearSolution, MpiCollectiveTraceV1) {
        (self.solution, self.trace)
    }

    /// Consume the result while deliberately discarding the trace.
    #[must_use]
    pub fn into_solution(self) -> LinearSolution {
        self.solution
    }
}

impl AdmittedDistributedRun<'_, '_> {
    /// Exact distributed algebra object sealed by admission.
    #[must_use]
    pub const fn system(&self) -> &DistributedLinearSystem {
        self.system
    }

    /// Exact complete host verifier sealed by admission.
    #[must_use]
    pub const fn complete(&self) -> &CanonicalCsrSystemView {
        self.complete
    }

    /// The sole solver plan sealed by collective admission.
    #[must_use]
    pub const fn plan(&self) -> SolverPlan {
        self.plan
    }

    /// Exact system/layout/plan identity agreed by every rank.
    #[must_use]
    pub const fn admission_fingerprint(&self) -> DistributedAdmissionFingerprintV1 {
        self.fingerprint
    }

    /// Solve owned rows, reconstruct by explicit global indices on every rank,
    /// and independently reaccept the replicated vector through host CSR.
    ///
    /// # Errors
    /// Returns a common stable diagnostic for synchronized action,
    /// preconditioner, reduction, producer, gather, or host-verifier failure.
    pub fn solve_and_replicate(self) -> Result<LinearSolution, Diagnostic> {
        self.solve_and_replicate_with_trace()
            .map(MpiLinearSolveResult::into_solution)
    }

    /// Solve and return the complete accepted vector with the exact collective
    /// sequence that produced and agreed it.
    ///
    /// The trace storage is reserved during admission from a checked bound on
    /// the maximum Krylov iteration count. Recording a collective boundary cannot
    /// allocate after the admitted run exists.
    ///
    /// # Errors
    /// Returns a common stable diagnostic for synchronized action,
    /// preconditioner, reduction, producer, gather, or host-verifier failure,
    /// or `EQ0807` if the runtime exceeds its pre-admitted trace bound.
    pub fn solve_and_replicate_with_trace(self) -> Result<MpiLinearSolveResult, Diagnostic> {
        self.solve_and_replicate_with_trace_using(&mut HostLocalCsrAction)
    }

    /// Solve with a caller-owned rank-local CSR action and return the complete
    /// accepted vector plus the exact MPI collective trace.
    ///
    /// Only owned-row sparse action is delegated. MPI continues to own halo
    /// exchange, Krylov vectors, reductions, gather, and host reacceptance.
    ///
    /// # Errors
    /// Returns a common stable diagnostic for any synchronized execution
    /// failure, including an injected action failure.
    pub fn solve_and_replicate_with_local_action(
        self,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<MpiLinearSolveResult, Diagnostic> {
        self.solve_and_replicate_with_trace_using(action)
    }

    fn solve_and_replicate_with_trace_using(
        mut self,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<MpiLinearSolveResult, Diagnostic> {
        let local_solution = self.solve_local(action)?;
        self.agree_producer_report(local_solution.report())?;
        let complete_values = self.gather_owned(&local_solution)?;
        let report = local_solution.report();

        let mut complete_values = complete_values;
        if self.take_fault(FaultPoint::HostVerifier) && !complete_values.is_empty() {
            complete_values[0] = f64::NAN;
        }
        let accepted = self.buffers.acceptance.accept_with_verifier(
            &self.complete_problem,
            self.plan,
            report.solver_provider(),
            report.execution_provider(),
            report.execution(),
            report.reason(),
            report.completed_iterations(),
            report.reported_residual_norm(),
            complete_values,
            &SERIAL_LINEAR_EXECUTION,
        );
        let accepted = self.synchronize(
            CollectivePhaseV1::HostAcceptance,
            report.completed_iterations(),
            accepted,
        )?;
        self.agree_accepted_solution(&accepted)?;
        let completed_iterations = accepted.report().completed_iterations();
        let trace_capacity = self.buffers.collective_step_limit;
        let trace = MpiCollectiveTraceV1::accepted(
            self.fingerprint,
            trace_capacity,
            mem::take(&mut self.buffers.collective_steps),
            completed_iterations,
        )?;
        Ok(MpiLinearSolveResult {
            solution: accepted,
            trace,
        })
    }

    fn solve_local(
        &mut self,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<LocalLinearSolution, Diagnostic> {
        let workspace = self.buffers.krylov.take();
        let plan_check = if self.take_fault(FaultPoint::Plan) {
            Err(invalid_realization("injected post-admission plan failure"))
        } else {
            match (
                workspace,
                self.problem.properties(),
                self.plan.algorithm(),
                self.plan.preconditioner(),
                self.plan.reduction(),
            ) {
                (
                    Some(KrylovWorkspace::Cg(workspace)),
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                    LinearSolver::ConjugateGradient,
                    PreconditionerPolicy::Jacobi,
                    ReductionPolicy::Reproducible | ReductionPolicy::Fast,
                ) => Ok(KrylovWorkspace::Cg(workspace)),
                (
                    Some(KrylovWorkspace::Minres(workspace)),
                    LinearOperatorProperties::SymmetricIndefinite,
                    LinearSolver::MinimumResidual,
                    PreconditionerPolicy::Identity,
                    ReductionPolicy::Reproducible,
                ) => Ok(KrylovWorkspace::Minres(workspace)),
                _ => Err(invalid_realization(
                    "admitted distributed run contradicts its sealed Krylov solver tuple",
                )),
            }
        };
        match self.synchronize(CollectivePhaseV1::LocalAction, 0, plan_check)? {
            KrylovWorkspace::Cg(workspace) => self.solve_conjugate_gradient(workspace, action),
            KrylovWorkspace::Minres(workspace) => self.solve_minimum_residual(workspace, action),
        }
    }

    fn solve_conjugate_gradient(
        &mut self,
        workspace: CgWorkspace,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<LocalLinearSolution, Diagnostic> {
        let CgWorkspace {
            mut solution,
            mut applied,
            mut residual,
            mut preconditioned,
            mut direction,
            mut inverse_diagonal,
        } = workspace;
        let jacobi = (|| {
            let shard = self
                .problem
                .operator()
                .shard(self.group.partition)
                .ok_or_else(|| invalid_realization("MPI rank has no local CSR shard"))?;
            if shard.diagonal(&mut inverse_diagonal)? == DiagonalAvailability::Unavailable {
                return Err(invalid_realization(
                    "distributed Jacobi requires every admitted local diagonal",
                ));
            }
            if self.take_fault(FaultPoint::Jacobi) {
                return Err(solve_failed("injected post-admission Jacobi failure"));
            }
            for value in &mut inverse_diagonal {
                if !value.is_finite() || *value <= 0.0 {
                    return Err(solve_failed(
                        "Jacobi-preconditioned distributed CG requires a finite positive diagonal",
                    ));
                }
                *value = 1.0 / *value;
            }
            Ok(())
        })();
        self.synchronize(CollectivePhaseV1::LocalAction, 0, jacobi)?;

        self.apply_into(&solution, &mut applied, 0, action)?;
        for ((value, right), applied) in residual
            .iter_mut()
            .zip(self.problem.right_hand_side())
            .zip(&applied)
        {
            *value = right - applied;
        }
        self.synchronize(
            CollectivePhaseV1::VectorUpdate,
            0,
            require_finite(&residual, "initial distributed residual"),
        )?;

        let right_hand_side = self.problem.right_hand_side();
        let right_hand_side_norm = self.norm_owned(right_hand_side, 0)?;
        let target_result = self.plan.residual_target(right_hand_side_norm);
        let target = self.synchronize(CollectivePhaseV1::VectorUpdate, 0, target_result)?;
        let initial_residual_norm = self.norm_owned(&residual, 0)?;
        if initial_residual_norm <= target {
            let accepted = self.accept_local_solution(
                ConvergenceReason::InitialResidualSatisfied,
                0,
                initial_residual_norm,
                initial_residual_norm,
                initial_residual_norm,
                target,
                solution,
            );
            return self.synchronize(CollectivePhaseV1::ProducerReport, 0, accepted);
        }

        apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
        direction.copy_from_slice(&preconditioned);
        let mut residual_product = self.dot_owned(&residual, &preconditioned, 0)?;
        self.synchronize(
            CollectivePhaseV1::VectorUpdate,
            0,
            positive(residual_product, "preconditioned residual curvature"),
        )?;

        let mut reported_residual_norm = initial_residual_norm;
        for iteration in 1..=self.plan.maximum_iterations().get() {
            self.apply_into(&direction, &mut applied, iteration, action)?;
            let curvature = self.dot_owned(&direction, &applied, iteration)?;
            self.synchronize(
                CollectivePhaseV1::VectorUpdate,
                iteration,
                positive(curvature, "operator curvature"),
            )?;
            let step = residual_product / curvature;
            let update = if step.is_finite() {
                for index in 0..solution.len() {
                    solution[index] += step * direction[index];
                    residual[index] -= step * applied[index];
                }
                require_finite(&solution, "distributed CG solution")
                    .and_then(|()| require_finite(&residual, "distributed CG residual"))
            } else {
                Err(solve_failed("distributed CG step is non-finite"))
            };
            self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, update)?;
            reported_residual_norm = self.norm_owned(&residual, iteration)?;
            if reported_residual_norm <= target {
                let true_residual_norm = self.true_residual_norm(
                    &solution,
                    &mut applied,
                    &mut residual,
                    iteration,
                    action,
                )?;
                if true_residual_norm <= target {
                    let accepted = self.accept_local_solution(
                        ConvergenceReason::ResidualToleranceSatisfied,
                        iteration,
                        initial_residual_norm,
                        reported_residual_norm,
                        true_residual_norm,
                        target,
                        solution,
                    );
                    return self.synchronize(
                        CollectivePhaseV1::ProducerReport,
                        iteration,
                        accepted,
                    );
                }
                apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
                residual_product = self.dot_owned(&residual, &preconditioned, iteration)?;
                self.synchronize(
                    CollectivePhaseV1::VectorUpdate,
                    iteration,
                    positive(residual_product, "restarted residual curvature"),
                )?;
                direction.copy_from_slice(&preconditioned);
                continue;
            }

            apply_preconditioner(&inverse_diagonal, &residual, &mut preconditioned);
            let next_product = self.dot_owned(&residual, &preconditioned, iteration)?;
            self.synchronize(
                CollectivePhaseV1::VectorUpdate,
                iteration,
                positive(next_product, "preconditioned residual curvature"),
            )?;
            let beta = next_product / residual_product;
            let update = if beta.is_finite() {
                for index in 0..direction.len() {
                    direction[index] = preconditioned[index] + beta * direction[index];
                }
                require_finite(&direction, "distributed CG direction")
            } else {
                Err(solve_failed("distributed CG beta is non-finite"))
            };
            self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, update)?;
            residual_product = next_product;
        }

        let maximum_iterations = self.plan.maximum_iterations().get();
        let true_residual_norm = self.true_residual_norm(
            &solution,
            &mut applied,
            &mut residual,
            maximum_iterations,
            action,
        )?;
        self.synchronize(
            CollectivePhaseV1::ProducerReport,
            maximum_iterations,
            Err(solve_failed(format!(
                "distributed CG reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
                self.plan.maximum_iterations()
            ))),
        )
    }

    fn solve_minimum_residual(
        &mut self,
        workspace: MinresWorkspace,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<LocalLinearSolution, Diagnostic> {
        let MinresWorkspace {
            mut solution,
            mut applied,
            mut previous_residual,
            mut current_residual,
            mut lanczos_image,
            mut basis,
            mut direction,
            mut previous_direction,
            mut older_direction,
        } = workspace;

        self.apply_into(&solution, &mut applied, 0, action)?;
        for ((value, right), applied) in previous_residual
            .iter_mut()
            .zip(self.problem.right_hand_side())
            .zip(&applied)
        {
            *value = right - applied;
        }
        self.synchronize(
            CollectivePhaseV1::VectorUpdate,
            0,
            require_finite(&previous_residual, "initial distributed MINRES residual"),
        )?;

        let right_hand_side_norm = self.norm_owned(self.problem.right_hand_side(), 0)?;
        let target_result = self.plan.residual_target(right_hand_side_norm);
        let target = self.synchronize(CollectivePhaseV1::VectorUpdate, 0, target_result)?;
        let initial_residual_norm = self.norm_owned(&previous_residual, 0)?;
        if initial_residual_norm <= target {
            let accepted = self.accept_local_solution(
                ConvergenceReason::InitialResidualSatisfied,
                0,
                initial_residual_norm,
                initial_residual_norm,
                initial_residual_norm,
                target,
                solution,
            );
            return self.synchronize(CollectivePhaseV1::ProducerReport, 0, accepted);
        }

        current_residual.copy_from_slice(&previous_residual);
        lanczos_image.copy_from_slice(&previous_residual);
        let mut beta = initial_residual_norm;
        let mut previous_beta = 0.0;
        let mut diagonal_bar = 0.0;
        let mut epsilon = 0.0;
        let mut residual_projection = initial_residual_norm;
        let mut cosine = -1.0;
        let mut sine = 0.0;
        let mut reported_residual_norm = initial_residual_norm;

        for iteration in 1..=self.plan.maximum_iterations().get() {
            let normalization = if beta.is_finite() && beta > 0.0 {
                for (basis, image) in basis.iter_mut().zip(&lanczos_image) {
                    *basis = image / beta;
                }
                require_finite(&basis, "distributed MINRES Lanczos basis")
            } else {
                Err(solve_failed(
                    "distributed MINRES Lanczos normalization broke down before convergence",
                ))
            };
            self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, normalization)?;

            self.apply_into(&basis, &mut applied, iteration, action)?;
            let recurrence_update = if iteration < 2 {
                Ok(())
            } else {
                let recurrence = beta / previous_beta;
                if recurrence.is_finite() {
                    for (value, previous) in applied.iter_mut().zip(&previous_residual) {
                        *value -= recurrence * previous;
                    }
                    require_finite(&applied, "distributed MINRES Lanczos recurrence")
                } else {
                    Err(solve_failed(
                        "distributed MINRES Lanczos recurrence is non-finite",
                    ))
                }
            };
            self.synchronize(
                CollectivePhaseV1::VectorUpdate,
                iteration,
                recurrence_update,
            )?;

            let diagonal = self.dot_owned(&basis, &applied, iteration)?;
            let residual_update = {
                let recurrence = diagonal / beta;
                if recurrence.is_finite() {
                    for (value, current) in applied.iter_mut().zip(&current_residual) {
                        *value -= recurrence * current;
                    }
                    require_finite(&applied, "distributed MINRES residual recurrence")
                } else {
                    Err(solve_failed(
                        "distributed MINRES residual recurrence is non-finite",
                    ))
                }
            };
            self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, residual_update)?;
            previous_residual.copy_from_slice(&current_residual);
            current_residual.copy_from_slice(&applied);
            lanczos_image.copy_from_slice(&current_residual);
            previous_beta = beta;
            beta = self.norm_owned(&current_residual, iteration)?;

            let previous_epsilon = epsilon;
            let rotation = (|| {
                let delta = cosine * diagonal_bar + sine * diagonal;
                let diagonal_rotated = sine * diagonal_bar - cosine * diagonal;
                epsilon = sine * beta;
                diagonal_bar = -cosine * beta;
                let rotation_norm = diagonal_rotated.hypot(beta);
                if !rotation_norm.is_finite() || rotation_norm <= f64::MIN_POSITIVE {
                    return Err(solve_failed(
                        "distributed MINRES orthogonal rotation broke down before convergence",
                    ));
                }
                cosine = diagonal_rotated / rotation_norm;
                sine = beta / rotation_norm;
                let step_projection = cosine * residual_projection;
                residual_projection *= sine;
                if [
                    delta,
                    epsilon,
                    diagonal_bar,
                    cosine,
                    sine,
                    step_projection,
                    residual_projection,
                ]
                .iter()
                .all(|value| value.is_finite())
                {
                    Ok((delta, rotation_norm, step_projection))
                } else {
                    Err(solve_failed(
                        "distributed MINRES orthogonal rotation is non-finite",
                    ))
                }
            })();
            let (delta, rotation_norm, step_projection) =
                self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, rotation)?;

            older_direction.copy_from_slice(&previous_direction);
            previous_direction.copy_from_slice(&direction);
            for index in 0..direction.len() {
                direction[index] = (basis[index]
                    - previous_epsilon * older_direction[index]
                    - delta * previous_direction[index])
                    / rotation_norm;
                solution[index] += step_projection * direction[index];
            }
            let update = require_finite(&direction, "distributed MINRES direction")
                .and_then(|()| require_finite(&solution, "distributed MINRES solution"));
            self.synchronize(CollectivePhaseV1::VectorUpdate, iteration, update)?;

            reported_residual_norm = residual_projection.abs();
            if reported_residual_norm <= target {
                let true_residual_norm = self.true_residual_norm(
                    &solution,
                    &mut applied,
                    &mut basis,
                    iteration,
                    action,
                )?;
                if true_residual_norm <= target {
                    let accepted = self.accept_local_solution(
                        ConvergenceReason::ResidualToleranceSatisfied,
                        iteration,
                        initial_residual_norm,
                        reported_residual_norm,
                        true_residual_norm,
                        target,
                        solution,
                    );
                    return self.synchronize(
                        CollectivePhaseV1::ProducerReport,
                        iteration,
                        accepted,
                    );
                }
            }
            if beta == 0.0 {
                let true_residual_norm = self.true_residual_norm(
                    &solution,
                    &mut applied,
                    &mut basis,
                    iteration,
                    action,
                )?;
                return self.synchronize(
                    CollectivePhaseV1::ProducerReport,
                    iteration,
                    Err(solve_failed(format!(
                        "distributed MINRES Lanczos space closed with true residual {true_residual_norm:e} above target {target:e}"
                    ))),
                );
            }
        }

        let maximum_iterations = self.plan.maximum_iterations().get();
        let true_residual_norm = self.true_residual_norm(
            &solution,
            &mut applied,
            &mut basis,
            maximum_iterations,
            action,
        )?;
        self.synchronize(
            CollectivePhaseV1::ProducerReport,
            maximum_iterations,
            Err(solve_failed(format!(
                "distributed MINRES reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
                self.plan.maximum_iterations()
            ))),
        )
    }

    fn true_residual_norm(
        &mut self,
        solution: &[f64],
        applied: &mut [f64],
        residual: &mut [f64],
        iteration: usize,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<f64, Diagnostic> {
        self.apply_into(solution, applied, iteration, action)?;
        for ((value, right), applied) in residual
            .iter_mut()
            .zip(self.problem.right_hand_side())
            .zip(applied.iter())
        {
            *value = right - applied;
        }
        self.synchronize(
            CollectivePhaseV1::VectorUpdate,
            iteration,
            require_finite(residual, "true distributed residual"),
        )?;
        self.norm_owned(residual, iteration)
    }

    fn apply_into(
        &mut self,
        owned_input: &[f64],
        output: &mut [f64],
        iteration: usize,
        action: &mut dyn MpiRankLocalCsrAction,
    ) -> Result<(), Diagnostic> {
        let layout = &self.problem.operator().layouts()[self.group.partition.index()];
        let preparation = if owned_input.len() == layout.owned().len()
            && output.len() == layout.owned().len()
            && owned_input.iter().all(|value| value.is_finite())
        {
            Ok(())
        } else {
            Err(solve_failed("admitted MPI action vector shape is invalid"))
        };
        self.synchronize(CollectivePhaseV1::Halo, iteration, preparation)?;

        self.buffers.ghost_seen.fill(false);
        let mut local_failure = None;
        for exchange in self.problem.operator().halo().exchanges() {
            let owner = mpi_rank(exchange.owner())?;
            let receiver = mpi_rank(exchange.receiver())?;
            let payload = &mut self.buffers.halo_payload[..exchange.indices().len()];
            if self.group.partition == exchange.owner() {
                for (slot, global) in payload.iter_mut().zip(exchange.indices()) {
                    match layout.owned().binary_search(global) {
                        Ok(local) => *slot = owned_input[local],
                        Err(_) => {
                            *slot = 0.0;
                            local_failure.get_or_insert_with(|| {
                                solve_failed("declared halo source is not locally owned")
                            });
                        }
                    }
                }
                self.group
                    .communicator
                    .process_at_rank(receiver)
                    .send_with_tag(&payload[..], 0);
            } else if self.group.partition == exchange.receiver() {
                let status = self
                    .group
                    .communicator
                    .process_at_rank(owner)
                    .receive_into_with_tag(&mut payload[..], 0);
                if status.source_rank() != owner || status.tag() != 0 {
                    local_failure
                        .get_or_insert_with(|| solve_failed("MPI halo receive status drifted"));
                }
                for (&global, &value) in exchange.indices().iter().zip(payload.iter()) {
                    match layout.ghosts().binary_search(&global) {
                        Ok(local) if value.is_finite() => {
                            self.buffers.ghosts[local] = value;
                            self.buffers.ghost_seen[local] = true;
                        }
                        _ => {
                            local_failure.get_or_insert_with(|| {
                                solve_failed("MPI halo payload contradicts the admitted layout")
                            });
                        }
                    }
                }
            }
        }
        if self.buffers.ghost_seen.iter().any(|seen| !seen) {
            local_failure.get_or_insert_with(|| solve_failed("MPI halo omitted an admitted ghost"));
        }
        if self.take_fault(FaultPoint::LocalAction) {
            local_failure.get_or_insert_with(|| solve_failed("injected local action failure"));
        }
        if local_failure.is_none() {
            match self.problem.operator().shard(self.group.partition) {
                Some(shard) => {
                    match action.apply_owned_rows(shard, owned_input, &self.buffers.ghosts, output)
                    {
                        Ok(()) if output.iter().all(|value| value.is_finite()) => {}
                        Ok(()) => {
                            local_failure = Some(solve_failed(
                                "rank-local CSR action produced non-finite owned values",
                            ));
                        }
                        Err(diagnostic) => local_failure = Some(diagnostic),
                    }
                }
                None => {
                    local_failure = Some(invalid_realization("MPI rank has no admitted CSR shard"));
                }
            }
        }
        self.synchronize(
            CollectivePhaseV1::LocalAction,
            iteration,
            local_failure.map_or(Ok(()), Err),
        )
    }

    fn dot_owned(
        &mut self,
        left: &[f64],
        right: &[f64],
        iteration: usize,
    ) -> Result<f64, Diagnostic> {
        let expected = self.problem.right_hand_side().len();
        let local = if left.len() == expected
            && right.len() == expected
            && left.iter().chain(right).all(|value| value.is_finite())
        {
            left.iter()
                .zip(right)
                .try_fold(0.0, |sum, (left, right)| finite_sum(sum, left * right))
        } else {
            Err(solve_failed("admitted MPI dot vector shape is invalid"))
        };
        let local = self.synchronize(CollectivePhaseV1::Reduction, iteration, local)?;
        let reduced = match self.plan.reduction() {
            ReductionPolicy::Reproducible => {
                self.group
                    .communicator
                    .all_gather_into(&local, &mut self.buffers.reduction_partials[..]);
                self.buffers
                    .reduction_partials
                    .iter()
                    .copied()
                    .try_fold(0.0, finite_sum)
            }
            ReductionPolicy::Fast => {
                let mut global = 0.0;
                self.group.communicator.all_reduce_into(
                    &local,
                    &mut global,
                    SystemOperation::sum(),
                );
                Ok(global)
            }
        };
        let reduced = reduced.and_then(|value| {
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| solve_failed("MPI reduction produced a non-finite value"))
        });
        self.synchronize(CollectivePhaseV1::Reduction, iteration, reduced)
    }

    fn norm_owned(&mut self, values: &[f64], iteration: usize) -> Result<f64, Diagnostic> {
        let squared = self.dot_owned(values, values, iteration)?;
        self.synchronize(
            CollectivePhaseV1::VectorUpdate,
            iteration,
            if squared >= 0.0 {
                Ok(squared.sqrt())
            } else {
                Err(solve_failed("distributed squared norm is negative"))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_local_solution(
        &self,
        reason: ConvergenceReason,
        completed_iterations: usize,
        initial_residual_norm: f64,
        reported_residual_norm: f64,
        true_residual_norm: f64,
        residual_target: f64,
        values: Vec<f64>,
    ) -> Result<LocalLinearSolution, Diagnostic> {
        let report = SolveReport::accepted(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            ExecutionReport::distributed(MPI_EXECUTION, self.group.partitions),
            LinearOperatorOrientation::Normal,
            self.plan,
            reason,
            completed_iterations,
            initial_residual_norm,
            reported_residual_norm,
            true_residual_norm,
            residual_target,
        )?;
        LocalLinearSolution::new(&self.problem, values, report)
    }

    fn agree_producer_report(&mut self, report: &SolveReport) -> Result<(), Diagnostic> {
        #[cfg(feature = "mpi-test-hooks")]
        let provider_substitution = if self.take_fault(FaultPoint::ProviderVersion) {
            Some(ProviderSummarySubstitution::SolverImplementationVersion)
        } else if self.take_fault(FaultPoint::ProviderLibrary) {
            Some(ProviderSummarySubstitution::ExecutionLibraries)
        } else {
            None
        };
        let local = (|| {
            let expected_execution =
                ExecutionReport::distributed(MPI_EXECUTION, self.group.partitions);
            if report.solver_provider() != MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER
                || report.execution_provider() != MPI_EXECUTION_PROVIDER
                || report.verification_provider() != MPI_EXECUTION_PROVIDER
                || report.execution() != expected_execution
                || report.verification() != expected_execution
                || report.orientation() != LinearOperatorOrientation::Normal
                || report.solver_plan() != self.plan
            {
                return Err(invalid_realization(
                    "local MPI report contradicts the admitted execution",
                ));
            }
            #[cfg(feature = "mpi-test-hooks")]
            if let Some(substitution) = provider_substitution {
                return ProducerReportSummaryV2::from_report_with_substitution(
                    report,
                    substitution,
                );
            }
            ProducerReportSummaryV2::from_report(report)
        })();
        let summary = self.synchronize(
            CollectivePhaseV1::ProducerReport,
            report.completed_iterations(),
            local,
        )?;
        let mut summary_bytes = summary.as_bytes();
        if self.take_fault(FaultPoint::Producer) {
            summary_bytes[0] ^= 0x80;
        }
        fixed_all_gather(
            &self.group.communicator,
            &summary_bytes,
            &mut self.buffers.summary_bytes,
        );
        let expected = summary.as_bytes();
        let agreement = if self
            .buffers
            .summary_bytes
            .as_chunks::<{ ProducerReportSummaryV2::ENCODED_LEN }>()
            .0
            .iter()
            .all(|candidate| candidate.as_slice() == expected)
        {
            Ok(())
        } else {
            Err(invalid_realization(
                "MPI producer-report summaries disagree",
            ))
        };
        self.synchronize(
            CollectivePhaseV1::ProducerReport,
            report.completed_iterations(),
            agreement,
        )
    }

    fn gather_owned(
        &mut self,
        local_solution: &LocalLinearSolution,
    ) -> Result<Vec<f64>, Diagnostic> {
        if self.take_fault(FaultPoint::Gather) && !self.buffers.local_indices.is_empty() {
            self.buffers.local_indices[0] = u64::MAX;
        }
        let local = if local_solution.partition() == self.group.partition
            && local_solution.values().len() == self.buffers.local_indices.len()
            && local_solution
                .values()
                .iter()
                .all(|value| value.is_finite())
        {
            Ok(())
        } else {
            Err(solve_failed("local owner-gather payload is invalid"))
        };
        let step = self.next_step(CollectivePhaseV1::GatherPreparation, 0)?;
        let status = phase_status(step, self.group.partition, &local)?;

        let communicator = &self.group.communicator;
        let partitions = self.group.partitions;
        let counts = self.gather_plan.counts();
        let displacements = self.gather_plan.displacements();
        let RunBuffers {
            status_bytes,
            phase_records,
            local_indices,
            gathered_indices,
            gathered_values,
            ..
        } = &mut self.buffers;
        let mut index_receive = PartitionMut::new(&mut gathered_indices[..], counts, displacements);
        let mut value_receive = PartitionMut::new(&mut gathered_values[..], counts, displacements);
        synchronize_status(
            communicator,
            partitions,
            status_bytes,
            phase_records,
            status,
            step,
        )?;
        // Normative pair: both wrappers exist before readiness and there is no
        // branch, allocation, or return between these two collectives.
        communicator.all_gather_varcount_into(&local_indices[..], &mut index_receive);
        communicator.all_gather_varcount_into(local_solution.values(), &mut value_receive);

        let reconstructed = self.gather_plan.reconstruct_into(
            &self.buffers.gathered_indices,
            &self.buffers.gathered_values,
            &mut self.buffers.complete_values,
        );
        self.synchronize(CollectivePhaseV1::GatherValidation, 0, reconstructed)?;
        Ok(mem::take(&mut self.buffers.complete_values))
    }

    fn agree_accepted_solution(&mut self, solution: &LinearSolution) -> Result<(), Diagnostic> {
        let summary = accepted_solution_summary(solution);
        let summary = self.synchronize(CollectivePhaseV1::ResultAgreement, 0, summary)?;
        fixed_all_gather(
            &self.group.communicator,
            &summary,
            &mut self.buffers.summary_bytes,
        );
        let reference = &self.buffers.summary_bytes[..summary.len()];
        let agreement = if self
            .buffers
            .summary_bytes
            .as_chunks::<32>()
            .0
            .iter()
            .all(|candidate| candidate == reference)
        {
            Ok(())
        } else {
            Err(solve_failed(
                "host-accepted solution vectors or reports disagree across ranks",
            ))
        };
        self.synchronize(CollectivePhaseV1::ResultAgreement, 0, agreement)
    }

    fn synchronize<T>(
        &mut self,
        phase: CollectivePhaseV1,
        iteration: usize,
        local: Result<T, Diagnostic>,
    ) -> Result<T, Diagnostic> {
        let step = self.next_step(phase, iteration)?;
        let status = phase_status(step, self.group.partition, &local)?;
        synchronize_status(
            &self.group.communicator,
            self.group.partitions,
            &mut self.buffers.status_bytes,
            &mut self.buffers.phase_records,
            status,
            step,
        )?;
        local.map_err(|_| {
            solve_failed("local phase failed despite an all-rank ready status agreement")
        })
    }

    fn next_step(
        &mut self,
        phase: CollectivePhaseV1,
        iteration: usize,
    ) -> Result<CollectiveStepV1, Diagnostic> {
        if self.buffers.collective_steps.len() >= self.buffers.collective_step_limit {
            return Err(invalid_realization(
                "MPI execution exceeded its pre-admitted collective trace bound",
            ));
        }
        let ordinal = self.buffers.collective_steps.len();
        let step = CollectiveStepV1::new(phase, iteration, ordinal);
        self.buffers.collective_steps.push(step);
        Ok(step)
    }

    #[cfg(feature = "mpi-test-hooks")]
    fn take_fault(&mut self, point: FaultPoint) -> bool {
        if self.group.fault.is_some_and(|fault| fault.point == point) {
            self.group.fault = None;
            true
        } else {
            false
        }
    }

    #[cfg(not(feature = "mpi-test-hooks"))]
    const fn take_fault(&mut self, _point: FaultPoint) -> bool {
        false
    }
}

impl fmt::Debug for AdmittedDistributedRun<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedDistributedRun")
            .field("partition", &self.group.partition)
            .field("partitions", &self.group.partitions)
            .field("plan", &self.plan)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

struct PreparedRun<'model> {
    problem: DistributedLinearProblem<'model>,
    complete_problem: eqiora_solver::LinearProblem<'model>,
    fingerprint: DistributedAdmissionFingerprintV1,
    gather_plan: OwnedGatherPlanV1,
    buffers: RunBuffers,
}

impl<'model> PreparedRun<'model> {
    fn new(
        group: &MpiExecutionGroup,
        system: &'model DistributedLinearSystem,
        complete: &'model CanonicalCsrSystemView,
        plan: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        require_partition(group, system.partition())?;
        if !system.matches_complete(complete) {
            return Err(invalid_realization(
                "distributed system and complete CSR view have different identities",
            ));
        }
        group.solver_capabilities().require_problem(
            plan,
            system.partition().space().scalar_type(),
            complete.properties(),
        )?;
        let fingerprint = system.admission_fingerprint(plan)?;
        let problem = system.local_problem(group.partition)?;
        let complete_problem = complete.linear_problem()?;
        for exchange in system.operator().halo().exchanges() {
            mpi_rank(exchange.owner())?;
            mpi_rank(exchange.receiver())?;
        }
        let gather_plan = OwnedGatherPlanV1::new(system.partition())?;
        let buffers = RunBuffers::new(group, system, complete, &problem, &complete_problem, plan)?;
        Ok(Self {
            problem,
            complete_problem,
            fingerprint,
            gather_plan,
            buffers,
        })
    }
}

struct RunBuffers {
    fixed_bytes: Vec<u8>,
    status_bytes: Vec<u8>,
    summary_bytes: Vec<u8>,
    admission_records: Vec<AdmissionRecordV1>,
    phase_records: Vec<PhaseStatusV1>,
    collective_steps: Vec<CollectiveStepV1>,
    collective_step_limit: usize,
    halo_payload: Vec<f64>,
    ghosts: Vec<f64>,
    ghost_seen: Vec<bool>,
    reduction_partials: Vec<f64>,
    krylov: Option<KrylovWorkspace>,
    local_indices: Vec<u64>,
    gathered_indices: Vec<u64>,
    gathered_values: Vec<f64>,
    complete_values: Vec<f64>,
    acceptance: LinearAcceptanceWorkspace,
}

enum KrylovWorkspace {
    Cg(CgWorkspace),
    Minres(MinresWorkspace),
}

struct CgWorkspace {
    solution: Vec<f64>,
    applied: Vec<f64>,
    residual: Vec<f64>,
    preconditioned: Vec<f64>,
    direction: Vec<f64>,
    inverse_diagonal: Vec<f64>,
}

struct MinresWorkspace {
    solution: Vec<f64>,
    applied: Vec<f64>,
    previous_residual: Vec<f64>,
    current_residual: Vec<f64>,
    lanczos_image: Vec<f64>,
    basis: Vec<f64>,
    direction: Vec<f64>,
    previous_direction: Vec<f64>,
    older_direction: Vec<f64>,
}

impl RunBuffers {
    fn new(
        group: &MpiExecutionGroup,
        system: &DistributedLinearSystem,
        complete: &CanonicalCsrSystemView,
        problem: &DistributedLinearProblem<'_>,
        complete_problem: &eqiora_solver::LinearProblem<'_>,
        plan: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        let ranks = group.partitions.get();
        let local_dimension = problem.right_hand_side().len();
        let global_dimension = complete.rows();
        let layout = system
            .operator()
            .layouts()
            .get(group.partition.index())
            .ok_or_else(|| invalid_realization("MPI rank has no distributed layout"))?;
        let max_halo_payload = system
            .operator()
            .halo()
            .exchanges()
            .iter()
            .map(|exchange| exchange.indices().len())
            .max()
            .unwrap_or(0);
        let fixed_bytes = zeroed(
            checked_extent(AdmissionRecordV1::ENCODED_LEN, ranks, "admission records")?,
            "admission records",
        )?;
        let status_bytes = zeroed(
            checked_extent(PhaseStatusV1::ENCODED_LEN, ranks, "phase statuses")?,
            "phase statuses",
        )?;
        let summary_bytes = zeroed(
            checked_extent(
                ProducerReportSummaryV2::ENCODED_LEN,
                ranks,
                "report summaries",
            )?,
            "report summaries",
        )?;
        let mut admission_records = reserved(ranks, "decoded admission records")?;
        let phase_records = reserved(ranks, "decoded phase statuses")?;
        let collective_step_limit = DistributedLinearExecutionTrace::collective_capacity(plan)?;
        let mut collective_steps = reserved(collective_step_limit, "MPI collective trace")?;
        collective_steps.push(CollectiveStepV1::new(CollectivePhaseV1::Admission, 0, 0));
        admission_records.clear();
        let krylov = KrylovWorkspace::new(problem, plan)?;
        let mut local_indices = reserved(local_dimension, "local global indices")?;
        for global in system.partition().owned_indices(group.partition) {
            local_indices.push(
                u64::try_from(global)
                    .map_err(|_| invalid_realization("global index exceeds portable u64"))?,
            );
        }
        if local_indices.len() != local_dimension || layout.owned().len() != local_dimension {
            return Err(invalid_realization(
                "local problem, owner map, and layout dimensions disagree",
            ));
        }
        Ok(Self {
            fixed_bytes,
            status_bytes,
            summary_bytes,
            admission_records,
            phase_records,
            collective_steps,
            collective_step_limit,
            halo_payload: zeroed(max_halo_payload, "halo payload")?,
            ghosts: zeroed(layout.ghosts().len(), "ghost values")?,
            ghost_seen: falses(layout.ghosts().len(), "ghost readiness")?,
            reduction_partials: zeroed(ranks, "reduction partials")?,
            krylov: Some(krylov),
            local_indices,
            gathered_indices: zeroed(global_dimension, "gathered global indices")?,
            gathered_values: zeroed(global_dimension, "gathered values")?,
            complete_values: zeroed(global_dimension, "reconstructed values")?,
            acceptance: LinearAcceptanceWorkspace::new(complete_problem)?,
        })
    }
}

fn phase_status<T>(
    step: CollectiveStepV1,
    partition: PartitionId,
    local: &Result<T, Diagnostic>,
) -> Result<PhaseStatusV1, Diagnostic> {
    match local {
        Ok(_) => PhaseStatusV1::ready(step, partition),
        Err(diagnostic) => PhaseStatusV1::rejected(
            step,
            partition,
            DistributedProtocolFailureV1::from_diagnostic(diagnostic),
        ),
    }
}

fn synchronize_status(
    communicator: &SimpleCommunicator,
    partitions: NonZeroUsize,
    bytes: &mut [u8],
    records: &mut Vec<PhaseStatusV1>,
    local: PhaseStatusV1,
    step: CollectiveStepV1,
) -> Result<(), Diagnostic> {
    fixed_all_gather(communicator, &local.encode(), bytes);
    records.clear();
    for bytes in bytes.as_chunks::<{ PhaseStatusV1::ENCODED_LEN }>().0 {
        let encoded = <[u8; PhaseStatusV1::ENCODED_LEN]>::try_from(bytes.as_slice())
            .map_err(|_| invalid_realization("phase-status extent is inconsistent"))?;
        records.push(PhaseStatusV1::decode(encoded)?);
    }
    evaluate_phase_statuses(records, partitions, step)
}

fn fixed_all_gather(communicator: &SimpleCommunicator, local: &[u8], gathered: &mut [u8]) {
    communicator.all_gather_into(local, gathered);
}

fn agree_preparation(
    communicator: &SimpleCommunicator,
    partition: PartitionId,
    failure: Option<DistributedProtocolFailureV1>,
    phase: CollectivePhaseV1,
    statuses: &mut [u8],
) -> Result<(), Diagnostic> {
    let expected = usize::try_from(communicator.size())
        .map_err(|_| invalid_realization("MPI communicator size is negative"))?;
    if statuses.len() != expected || partition.index() >= expected {
        return Err(invalid_realization(
            "MPI preparation status storage contradicts the execution group",
        ));
    }
    let local = failure.map_or(DistributedProtocolFailureV1::Ready as u8, |failure| {
        failure as u8
    });
    communicator.all_gather_into(&local, statuses);
    let Some((rank, encoded)) = statuses
        .iter()
        .copied()
        .enumerate()
        .find(|(_, encoded)| *encoded != DistributedProtocolFailureV1::Ready as u8)
    else {
        return Ok(());
    };
    let failure = DistributedProtocolFailureV1::decode(encoded)
        .unwrap_or(DistributedProtocolFailureV1::ProtocolMismatch);
    Err(failure.diagnostic(rank, phase))
}

fn accepted_solution_summary(solution: &LinearSolution) -> Result<[u8; 32], Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(ACCEPTED_SOLUTION_DOMAIN_V2);
    let length = u64::try_from(solution.values().len())
        .map_err(|_| invalid_realization("accepted solution length exceeds portable u64"))?;
    hash.update(length.to_be_bytes());
    for value in solution.values() {
        hash.update(value.to_bits().to_be_bytes());
    }
    hash.update(ProducerReportSummaryV2::from_report(solution.report())?.as_bytes());
    Ok(hash.finalize().into())
}

fn execution_receipt_summary(
    accepted: &AcceptedLinearExecution,
) -> Result<[u8; EXECUTION_RECEIPT_SUMMARY_BYTES], Diagnostic> {
    let receipt = accepted.receipt();
    let trace = receipt.distributed_trace().ok_or_else(|| {
        invalid_realization("MPI accepted result has no distributed execution trace")
    })?;
    let dimension = u64::try_from(receipt.dimension())
        .map_err(|_| invalid_realization("MPI receipt dimension exceeds portable u64"))?;
    let trace_capacity = u64::try_from(trace.trace_capacity())
        .map_err(|_| invalid_realization("MPI receipt trace capacity exceeds portable u64"))?;
    let step_count = u64::try_from(trace.steps().len())
        .map_err(|_| invalid_realization("MPI receipt trace length exceeds portable u64"))?;
    let partitions = u64::try_from(trace.partitions().get())
        .map_err(|_| invalid_realization("MPI receipt partition count exceeds portable u64"))?;
    let workers = u64::try_from(trace.workers_per_partition().get())
        .map_err(|_| invalid_realization("MPI receipt worker count exceeds portable u64"))?;
    let mut hash = Sha256::new();
    hash.update(EXECUTION_RECEIPT_AGREEMENT_DOMAIN_V2);
    hash.update(receipt.operator().as_bytes());
    hash.update(receipt.output().as_bytes());
    hash.update(dimension.to_be_bytes());
    hash.update(trace.partition().as_bytes());
    hash.update(trace.layout().as_bytes());
    hash.update(trace.admission().as_bytes());
    hash.update(trace.process_group().ordinal().to_be_bytes());
    hash.update(partitions.to_be_bytes());
    hash.update(workers.to_be_bytes());
    hash.update(trace_capacity.to_be_bytes());
    hash.update(step_count.to_be_bytes());
    hash.update(ProducerReportSummaryV2::from_report(receipt.report())?.as_bytes());
    for step in trace.steps() {
        let iteration = u64::try_from(step.iteration())
            .map_err(|_| invalid_realization("MPI receipt iteration exceeds portable u64"))?;
        let ordinal = u64::try_from(step.ordinal())
            .map_err(|_| invalid_realization("MPI receipt ordinal exceeds portable u64"))?;
        hash.update([distributed_phase_tag(step.phase())]);
        hash.update(iteration.to_be_bytes());
        hash.update(ordinal.to_be_bytes());
    }
    Ok(hash.finalize().into())
}

const fn distributed_phase_tag(phase: DistributedExecutionPhaseV1) -> u8 {
    match phase {
        DistributedExecutionPhaseV1::AdmissionAgreement => 0,
        DistributedExecutionPhaseV1::HaloReadiness => 1,
        DistributedExecutionPhaseV1::OwnedAction => 2,
        DistributedExecutionPhaseV1::OwnedVectorUpdate => 3,
        DistributedExecutionPhaseV1::CollectiveReduction => 4,
        DistributedExecutionPhaseV1::ProducerReportAgreement => 5,
        DistributedExecutionPhaseV1::OwnerGatherPreparation => 6,
        DistributedExecutionPhaseV1::OwnerGatherValidation => 7,
        DistributedExecutionPhaseV1::NativeHostAcceptance => 8,
        DistributedExecutionPhaseV1::AcceptedResultAgreement => 9,
    }
}

fn require_partition(group: &MpiExecutionGroup, partition: &Partition) -> Result<(), Diagnostic> {
    if partition.count() != group.partitions {
        return Err(invalid_realization(format!(
            "MPI communicator has {} ranks but partition declares {}",
            group.partitions,
            partition.count()
        )));
    }
    Ok(())
}

fn mpi_rank(partition: PartitionId) -> Result<i32, Diagnostic> {
    i32::try_from(partition.index())
        .map_err(|_| invalid_realization("partition index exceeds MPI rank range"))
}

fn checked_extent(unit: usize, count: usize, name: &'static str) -> Result<usize, Diagnostic> {
    unit.checked_mul(count)
        .ok_or_else(|| invalid_realization(format!("{name} extent overflowed")))
}

fn reserved<T>(capacity: usize, purpose: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_realization(format!("could not reserve {purpose}")))?;
    Ok(values)
}

fn zeroed<T: Clone + Default>(length: usize, purpose: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut values = reserved(length, purpose)?;
    values.resize(length, T::default());
    Ok(values)
}

fn falses(length: usize, purpose: &'static str) -> Result<Vec<bool>, Diagnostic> {
    let mut values = reserved(length, purpose)?;
    values.resize(length, false);
    Ok(values)
}

fn apply_preconditioner(inverse: &[f64], residual: &[f64], output: &mut [f64]) {
    for ((output, inverse), residual) in output.iter_mut().zip(inverse).zip(residual) {
        *output = inverse * residual;
    }
}

fn require_finite(values: &[f64], meaning: &'static str) -> Result<(), Diagnostic> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(solve_failed(format!("{meaning} must be finite")))
    }
}

fn positive(value: f64, meaning: &'static str) -> Result<(), Diagnostic> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(solve_failed(format!(
            "distributed CG requires positive {meaning}"
        )))
    }
}

fn finite_sum(sum: f64, value: f64) -> Result<f64, Diagnostic> {
    let sum = sum + value;
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(solve_failed("distributed reduction overflowed"))
    }
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultPoint {
    LocalAction,
    Jacobi,
    Plan,
    Producer,
    #[cfg(feature = "mpi-test-hooks")]
    ProviderVersion,
    #[cfg(feature = "mpi-test-hooks")]
    ProviderLibrary,
    Gather,
    HostVerifier,
}

#[cfg(feature = "mpi-test-hooks")]
#[derive(Debug, Clone, Copy)]
struct TestFault {
    point: FaultPoint,
}

#[cfg(feature = "mpi-test-hooks")]
impl TestFault {
    fn from_environment(rank: usize) -> Option<Self> {
        let value = std::env::var("EQIORA_MPI_TEST_FAULT").ok()?;
        let (point, selected_rank) = value.split_once(':')?;
        if selected_rank.parse::<usize>().ok()? != rank {
            return None;
        }
        let point = match point {
            "local-action" => FaultPoint::LocalAction,
            "jacobi" => FaultPoint::Jacobi,
            "plan" => FaultPoint::Plan,
            "producer" => FaultPoint::Producer,
            "provider-version" => FaultPoint::ProviderVersion,
            "provider-library" => FaultPoint::ProviderLibrary,
            "gather" => FaultPoint::Gather,
            "host-verifier" => FaultPoint::HostVerifier,
            _ => return None,
        };
        Some(Self { point })
    }
}
