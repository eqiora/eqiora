use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_distributed::{DistributedAdmissionFingerprintV1, Partition, PartitionId};
use eqiora_solver::{
    ConvergenceReason, ExecutionProvider, ExecutionTopology, LinearOperatorOrientation,
    LinearSolver, PreconditionerPolicy, ProviderLibrary, ReductionPolicy, SolveReport,
    SolverProvider,
};
use sha2::{Digest, Sha256};

mod solver_tag;

const ADMISSION_PROTOCOL_VERSION: u8 = 1;
const ADMISSION_RECORD_BYTES: usize = 58;
const PHASE_STATUS_BYTES: usize = 28;
const PRODUCER_REPORT_DOMAIN_V2: &[u8] = b"eqiora.mpi-producer-report-summary/v2\0";
const PROVIDER_RECORD_DOMAIN_V1: &[u8] = b"eqiora.mpi-provider-record/v1\0";

/// Communication boundary synchronized by one fixed-size status collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CollectivePhaseV1 {
    /// System, partition, layout, and plan admission.
    Admission = 0,
    /// Readiness immediately before halo exchange.
    Halo = 1,
    /// Local operator/preconditioner work before a reduction.
    LocalAction = 2,
    /// Local Krylov updates before the next communication.
    VectorUpdate = 3,
    /// Producer report construction and agreement.
    ProducerReport = 4,
    /// Allocation and shape readiness before the two owner gathers.
    GatherPreparation = 5,
    /// Validation after both owner gathers.
    GatherValidation = 6,
    /// Readiness and acceptance around one scalar reduction.
    Reduction = 7,
    /// Independent complete-host residual acceptance.
    HostAcceptance = 8,
    /// Cross-rank equality of accepted vectors and reports.
    ResultAgreement = 9,
}

impl CollectivePhaseV1 {
    fn decode(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Admission,
            1 => Self::Halo,
            2 => Self::LocalAction,
            3 => Self::VectorUpdate,
            4 => Self::ProducerReport,
            5 => Self::GatherPreparation,
            6 => Self::GatherValidation,
            7 => Self::Reduction,
            8 => Self::HostAcceptance,
            9 => Self::ResultAgreement,
            _ => return None,
        })
    }
}

/// Exact identity of one communication boundary in an admitted run.
///
/// `ordinal` is dense and global across the admitted run; phase and iteration
/// retain the mathematical location of that boundary. Peers compare all three
/// fields before proceeding, so a drifted control path fails closed instead of
/// entering a different MPI operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollectiveStepV1 {
    phase: CollectivePhaseV1,
    iteration: usize,
    ordinal: usize,
}

impl CollectiveStepV1 {
    /// Name one exact collective boundary.
    #[must_use]
    pub const fn new(phase: CollectivePhaseV1, iteration: usize, ordinal: usize) -> Self {
        Self {
            phase,
            iteration,
            ordinal,
        }
    }

    /// Communication phase.
    #[must_use]
    pub const fn phase(self) -> CollectivePhaseV1 {
        self.phase
    }

    /// Numerical iteration, with zero reserved for setup/finalization.
    #[must_use]
    pub const fn iteration(self) -> usize {
        self.iteration
    }

    /// Boundary ordinal within a phase and iteration.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
}

#[cfg(any(feature = "mpi-runtime", test))]
const SUCCESSFUL_TERMINAL_PHASES: [CollectivePhaseV1; 8] = [
    CollectivePhaseV1::ProducerReport,
    CollectivePhaseV1::ProducerReport,
    CollectivePhaseV1::ProducerReport,
    CollectivePhaseV1::GatherPreparation,
    CollectivePhaseV1::GatherValidation,
    CollectivePhaseV1::HostAcceptance,
    CollectivePhaseV1::ResultAgreement,
    CollectivePhaseV1::ResultAgreement,
];

/// Immutable ordered collective trace for one successfully accepted MPI run.
///
/// The first logical step covers the collective admission handshake. Every
/// later step is the exact [`CollectiveStepV1`] synchronized by the runtime.
/// Dense ordinals therefore span admission, iterative halo/reduction work, the
/// paired owner gather, complete-host acceptance, and final result agreement
/// without creating a second ordinal namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpiCollectiveTraceV1 {
    admission_fingerprint: DistributedAdmissionFingerprintV1,
    capacity: usize,
    steps: Vec<CollectiveStepV1>,
}

impl MpiCollectiveTraceV1 {
    #[cfg(feature = "mpi-runtime")]
    pub(crate) fn accepted(
        admission_fingerprint: DistributedAdmissionFingerprintV1,
        capacity: usize,
        steps: Vec<CollectiveStepV1>,
        completed_iterations: usize,
    ) -> Result<Self, Diagnostic> {
        if steps.len() > capacity {
            return Err(protocol_error(
                "accepted MPI collective trace exceeds its pre-admitted capacity",
            ));
        }
        validate_successful_collective_trace(&steps, completed_iterations)?;
        Ok(Self {
            admission_fingerprint,
            capacity,
            steps,
        })
    }

    /// Exact system/layout/plan identity agreed during collective admission.
    #[must_use]
    pub const fn admission_fingerprint(&self) -> DistributedAdmissionFingerprintV1 {
        self.admission_fingerprint
    }

    /// Record capacity reserved before collective admission.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Admission followed by every successfully synchronized runtime step.
    #[must_use]
    pub fn steps(&self) -> &[CollectiveStepV1] {
        &self.steps
    }

    /// Completed iteration shared by producer acceptance and host acceptance.
    #[must_use]
    pub fn completed_iterations(&self) -> usize {
        self.steps[self.steps.len() - 3].iteration()
    }
}

#[cfg(any(feature = "mpi-runtime", test))]
fn validate_successful_collective_trace(
    steps: &[CollectiveStepV1],
    completed_iterations: usize,
) -> Result<(), Diagnostic> {
    let Some(admission) = steps.first().copied() else {
        return Err(protocol_error("accepted MPI collective trace is empty"));
    };
    if admission != CollectiveStepV1::new(CollectivePhaseV1::Admission, 0, 0) {
        return Err(protocol_error(
            "accepted MPI collective trace must begin with logical admission ordinal zero",
        ));
    }
    for (ordinal, step) in steps.iter().copied().enumerate() {
        if step.ordinal() != ordinal {
            return Err(protocol_error(
                "accepted MPI collective trace ordinals are not dense",
            ));
        }
        if ordinal > 0 && step.phase() == CollectivePhaseV1::Admission {
            return Err(protocol_error(
                "accepted MPI collective trace repeats logical admission",
            ));
        }
    }

    let terminal = steps
        .len()
        .checked_sub(SUCCESSFUL_TERMINAL_PHASES.len())
        .and_then(|start| steps.get(start..))
        .ok_or_else(|| protocol_error("accepted MPI collective trace has no terminal suffix"))?;
    if !terminal
        .iter()
        .zip(SUCCESSFUL_TERMINAL_PHASES)
        .all(|(step, phase)| step.phase() == phase)
    {
        return Err(protocol_error(
            "accepted MPI collective trace terminal phases are incomplete or reordered",
        ));
    }
    if terminal[..3]
        .iter()
        .any(|step| step.iteration() != completed_iterations)
        || terminal[3].iteration() != 0
        || terminal[4].iteration() != 0
        || terminal[5].iteration() != completed_iterations
        || terminal[6].iteration() != 0
        || terminal[7].iteration() != 0
    {
        return Err(protocol_error(
            "accepted MPI collective trace terminal iterations contradict the solve report",
        ));
    }
    Ok(())
}

/// Stable rank-local failure category exchanged before communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DistributedProtocolFailureV1 {
    /// Local candidate is ready.
    Ready = 0,
    /// Realization, partition, topology, or plan is inconsistent.
    InvalidRealization = 1,
    /// Numerical work failed or produced a non-finite value.
    NumericalFailure = 2,
    /// A portable count/index conversion overflowed.
    CountOverflow = 3,
    /// Fixed protocol records disagree or are malformed.
    ProtocolMismatch = 4,
    /// Gathered owner/index/value content is invalid.
    GatherInvalid = 5,
}

impl DistributedProtocolFailureV1 {
    pub(crate) fn decode(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Ready,
            1 => Self::InvalidRealization,
            2 => Self::NumericalFailure,
            3 => Self::CountOverflow,
            4 => Self::ProtocolMismatch,
            5 => Self::GatherInvalid,
            _ => return None,
        })
    }

    #[cfg(feature = "mpi-runtime")]
    pub(crate) fn from_diagnostic(diagnostic: &Diagnostic) -> Self {
        if diagnostic.code() == codes::INVALID_REALIZATION {
            Self::InvalidRealization
        } else {
            Self::NumericalFailure
        }
    }

    pub(crate) fn diagnostic(self, partition: usize, phase: CollectivePhaseV1) -> Diagnostic {
        let (code, meaning) = match self {
            Self::Ready => (codes::NUMERICAL_SOLVE_FAILED, "unexpected ready status"),
            Self::InvalidRealization => (
                codes::INVALID_REALIZATION,
                "invalid distributed realization",
            ),
            Self::NumericalFailure => (
                codes::NUMERICAL_SOLVE_FAILED,
                "distributed numerical failure",
            ),
            Self::CountOverflow => (codes::INVALID_REALIZATION, "MPI count overflow"),
            Self::ProtocolMismatch => (codes::INVALID_REALIZATION, "collective protocol mismatch"),
            Self::GatherInvalid => (codes::NUMERICAL_SOLVE_FAILED, "invalid gathered owner data"),
        };
        Diagnostic::error(
            code,
            format!("distributed phase {phase:?} rejected partition {partition}: {meaning}"),
        )
    }
}

/// Fixed-size per-rank readiness or rejection record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseStatusV1 {
    step: CollectiveStepV1,
    iteration: u64,
    ordinal: u64,
    partition: u64,
    failure: DistributedProtocolFailureV1,
}

impl PhaseStatusV1 {
    /// Fixed wire extent in bytes.
    pub const ENCODED_LEN: usize = PHASE_STATUS_BYTES;

    /// Construct a ready record.
    ///
    /// # Errors
    /// Returns `EQ0807` when the partition index exceeds portable `u64`.
    pub fn ready(step: CollectiveStepV1, partition: PartitionId) -> Result<Self, Diagnostic> {
        Self::new(step, partition, DistributedProtocolFailureV1::Ready)
    }

    /// Construct a rejected record with a stable failure category.
    ///
    /// # Errors
    /// Returns `EQ0807` when a count exceeds portable `u64`, or when `Ready`
    /// is passed as the rejection category.
    pub fn rejected(
        step: CollectiveStepV1,
        partition: PartitionId,
        failure: DistributedProtocolFailureV1,
    ) -> Result<Self, Diagnostic> {
        if failure == DistributedProtocolFailureV1::Ready {
            return Err(protocol_error("a rejected phase status cannot use Ready"));
        }
        Self::new(step, partition, failure)
    }

    fn new(
        step: CollectiveStepV1,
        partition: PartitionId,
        failure: DistributedProtocolFailureV1,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            step,
            iteration: portable_u64(step.iteration, "phase iteration")?,
            ordinal: portable_u64(step.ordinal, "phase ordinal")?,
            partition: portable_u64(partition.index(), "phase partition")?,
            failure,
        })
    }

    /// Fixed portable bytes exchanged by the transport adapter.
    #[must_use]
    pub fn encode(self) -> [u8; PHASE_STATUS_BYTES] {
        let mut bytes = [0; PHASE_STATUS_BYTES];
        bytes[0] = ADMISSION_PROTOCOL_VERSION;
        bytes[1] = self.step.phase as u8;
        bytes[2] = self.failure as u8;
        bytes[4..12].copy_from_slice(&self.iteration.to_be_bytes());
        bytes[12..20].copy_from_slice(&self.ordinal.to_be_bytes());
        bytes[20..28].copy_from_slice(&self.partition.to_be_bytes());
        bytes
    }

    /// Decode one fixed portable record.
    ///
    /// # Errors
    /// Returns `EQ0807` for a version, reserved-byte, phase, or failure tag
    /// contradiction.
    pub fn decode(bytes: [u8; PHASE_STATUS_BYTES]) -> Result<Self, Diagnostic> {
        if bytes[0] != ADMISSION_PROTOCOL_VERSION || bytes[3] != 0 {
            return Err(protocol_error(
                "phase status version or reserved byte is invalid",
            ));
        }
        Ok(Self {
            step: CollectiveStepV1::new(
                CollectivePhaseV1::decode(bytes[1])
                    .ok_or_else(|| protocol_error("phase status tag is invalid"))?,
                usize::try_from(u64::from_be_bytes(
                    bytes[4..12].try_into().expect("fixed slice"),
                ))
                .map_err(|_| protocol_error("phase iteration exceeds usize"))?,
                usize::try_from(u64::from_be_bytes(
                    bytes[12..20].try_into().expect("fixed slice"),
                ))
                .map_err(|_| protocol_error("phase ordinal exceeds usize"))?,
            ),
            failure: DistributedProtocolFailureV1::decode(bytes[2])
                .ok_or_else(|| protocol_error("phase failure tag is invalid"))?,
            iteration: u64::from_be_bytes(bytes[4..12].try_into().expect("fixed slice")),
            ordinal: u64::from_be_bytes(bytes[12..20].try_into().expect("fixed slice")),
            partition: u64::from_be_bytes(bytes[20..28].try_into().expect("fixed slice")),
        })
    }
}

/// Require one ready status per partition in exact rank order.
///
/// The diagnostic for a common rejection is selected by the lowest rejected
/// partition, so every rank presented with the same records returns the same
/// stable result.
///
/// # Errors
/// Returns `EQ0807` for record-count, phase, iteration, or rank-order drift,
/// or the selected common diagnostic for a rejected rank.
pub fn evaluate_phase_statuses(
    records: &[PhaseStatusV1],
    partitions: NonZeroUsize,
    step: CollectiveStepV1,
) -> Result<(), Diagnostic> {
    let iteration = portable_u64(step.iteration, "expected phase iteration")?;
    let ordinal = portable_u64(step.ordinal, "expected phase ordinal")?;
    if records.len() != partitions.get() {
        return Err(protocol_error(format!(
            "phase status count {} differs from partition count {partitions}",
            records.len()
        )));
    }
    for (rank, record) in records.iter().enumerate() {
        if record.step.phase != step.phase
            || record.iteration != iteration
            || record.ordinal != ordinal
            || record.partition != portable_u64(rank, "expected phase rank")?
        {
            return Err(protocol_error(format!(
                "phase status at rank {rank} contradicts the common phase, iteration, ordinal, or partition order"
            )));
        }
    }
    if let Some(record) = records
        .iter()
        .find(|record| record.failure != DistributedProtocolFailureV1::Ready)
    {
        return Err(record.failure.diagnostic(
            usize::try_from(record.partition).unwrap_or(usize::MAX),
            step.phase,
        ));
    }
    Ok(())
}

/// Fixed-size rank record used before operator-dependent communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionRecordV1 {
    partitions: u64,
    partition: u64,
    failure: DistributedProtocolFailureV1,
    fingerprint: [u8; 32],
}

impl AdmissionRecordV1 {
    /// Fixed wire extent in bytes.
    pub const ENCODED_LEN: usize = ADMISSION_RECORD_BYTES;

    /// Build one admission record from local preparation.
    ///
    /// # Errors
    /// Returns `EQ0807` when a count exceeds portable `u64`.
    pub fn new(
        partitions: NonZeroUsize,
        partition: PartitionId,
        fingerprint: DistributedAdmissionFingerprintV1,
        failure: DistributedProtocolFailureV1,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            partitions: portable_u64(partitions.get(), "admission partition count")?,
            partition: portable_u64(partition.index(), "admission partition")?,
            failure,
            fingerprint: fingerprint.as_bytes(),
        })
    }

    /// Build a rejected record when local preparation could not derive a
    /// meaningful fingerprint.
    ///
    /// Rejection is evaluated before fingerprint equality, so the all-zero
    /// placeholder can never admit work.
    ///
    /// # Errors
    /// Returns `EQ0807` for `Ready` or a count outside portable `u64`.
    pub fn rejected(
        partitions: NonZeroUsize,
        partition: PartitionId,
        failure: DistributedProtocolFailureV1,
    ) -> Result<Self, Diagnostic> {
        if failure == DistributedProtocolFailureV1::Ready {
            return Err(protocol_error(
                "a rejected admission record cannot use Ready",
            ));
        }
        Ok(Self {
            partitions: portable_u64(partitions.get(), "admission partition count")?,
            partition: portable_u64(partition.index(), "admission partition")?,
            failure,
            fingerprint: [0; 32],
        })
    }

    /// Fixed portable bytes exchanged by the transport adapter.
    #[must_use]
    pub fn encode(self) -> [u8; ADMISSION_RECORD_BYTES] {
        let mut bytes = [0; ADMISSION_RECORD_BYTES];
        bytes[0] = ADMISSION_PROTOCOL_VERSION;
        bytes[1] = self.failure as u8;
        bytes[2..10].copy_from_slice(&self.partitions.to_be_bytes());
        bytes[10..18].copy_from_slice(&self.partition.to_be_bytes());
        bytes[18..50].copy_from_slice(&self.fingerprint);
        // Eight reserved bytes make future fixed additions explicit.
        bytes
    }

    /// Decode one fixed portable admission record.
    ///
    /// # Errors
    /// Returns `EQ0807` for an unknown version/tag or nonzero reserved byte.
    pub fn decode(bytes: [u8; ADMISSION_RECORD_BYTES]) -> Result<Self, Diagnostic> {
        if bytes[0] != ADMISSION_PROTOCOL_VERSION || bytes[50..].iter().any(|byte| *byte != 0) {
            return Err(protocol_error(
                "admission record version or reserved bytes are invalid",
            ));
        }
        Ok(Self {
            failure: DistributedProtocolFailureV1::decode(bytes[1])
                .ok_or_else(|| protocol_error("admission failure tag is invalid"))?,
            partitions: u64::from_be_bytes(bytes[2..10].try_into().expect("fixed slice")),
            partition: u64::from_be_bytes(bytes[10..18].try_into().expect("fixed slice")),
            fingerprint: bytes[18..50].try_into().expect("fixed slice"),
        })
    }
}

/// Validate all fixed admission records deterministically.
///
/// # Errors
/// Returns `EQ0807` for communicator/count/rank/fingerprint drift or a common
/// stable rejection selected from the lowest rejected partition.
pub fn evaluate_admission(
    records: &[AdmissionRecordV1],
    communicator_partitions: NonZeroUsize,
) -> Result<(), Diagnostic> {
    if records.len() != communicator_partitions.get() {
        return Err(protocol_error(
            "admission record count differs from communicator size",
        ));
    }
    let expected_count = portable_u64(communicator_partitions.get(), "communicator size")?;
    for (rank, record) in records.iter().enumerate() {
        if record.partitions != expected_count
            || record.partition != portable_u64(rank, "communicator rank")?
        {
            return Err(protocol_error(format!(
                "admission record {rank} contradicts communicator partition order"
            )));
        }
    }
    if let Some(record) = records
        .iter()
        .find(|record| record.failure != DistributedProtocolFailureV1::Ready)
    {
        return Err(record.failure.diagnostic(
            usize::try_from(record.partition).unwrap_or(usize::MAX),
            CollectivePhaseV1::Admission,
        ));
    }
    let expected_fingerprint = records
        .first()
        .expect("nonzero communicator has one record")
        .fingerprint;
    if let Some(rank) = records
        .iter()
        .position(|record| record.fingerprint != expected_fingerprint)
    {
        return Err(protocol_error(format!(
            "admission fingerprint differs at partition {rank}"
        )));
    }
    Ok(())
}

/// MPI-v0 count/displacement plan derived from an arbitrary owner map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedGatherPlanV1 {
    counts: Vec<i32>,
    displacements: Vec<i32>,
    expected_indices: Vec<u64>,
    total: usize,
}

impl OwnedGatherPlanV1 {
    /// Derive checked rank-order counts and prefix-sum displacements.
    ///
    /// # Errors
    /// Returns `EQ0807` when any count, displacement, or total exceeds the
    /// signed 32-bit `mpi::Count` boundary used by `mpi` 0.8.2.
    pub fn new(partition: &Partition) -> Result<Self, Diagnostic> {
        let mut counts = reserved_vector(partition.count().get(), "gather counts")?;
        let mut displacements = reserved_vector(partition.count().get(), "gather displacements")?;
        let mut expected_indices =
            reserved_vector(partition.space().dimension().get(), "gather index blocks")?;
        let mut total = 0_usize;
        for rank in 0..partition.count().get() {
            let count = partition.owned_indices(PartitionId::new(rank)).count();
            counts.push(checked_mpi_count(count, "owned value count")?);
            displacements.push(checked_mpi_count(total, "owned value displacement")?);
            for global in partition.owned_indices(PartitionId::new(rank)) {
                expected_indices.push(
                    u64::try_from(global)
                        .map_err(|_| count_overflow("owned global index exceeds portable u64"))?,
                );
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| count_overflow("owned value total overflowed usize"))?;
        }
        checked_mpi_count(total, "owned value total")?;
        if total != partition.space().dimension().get() {
            return Err(protocol_error(
                "gather counts do not cover the global dimension exactly",
            ));
        }
        Ok(Self {
            counts,
            displacements,
            expected_indices,
            total,
        })
    }

    /// Rank-order signed counts accepted by MPI v0.
    #[must_use]
    #[cfg(any(test, feature = "mpi-runtime"))]
    pub(crate) fn counts(&self) -> &[i32] {
        &self.counts
    }

    /// Rank-order signed displacements accepted by MPI v0.
    #[must_use]
    #[cfg(any(test, feature = "mpi-runtime"))]
    pub(crate) fn displacements(&self) -> &[i32] {
        &self.displacements
    }

    /// Complete receive extent.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.total
    }

    /// Reconstruct and validate one complete vector from explicit indices.
    ///
    /// Receive blocks are interpreted in rank order using this plan, but
    /// global indices are never inferred from counts or displacement.
    ///
    /// # Errors
    /// Returns `EQ0802` for shape, owner, index, duplicate/missing, or finite-
    /// value contradictions.
    pub fn reconstruct(
        &self,
        gathered_indices: &[u64],
        gathered_values: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        let mut complete = Vec::new();
        complete
            .try_reserve_exact(self.total)
            .map_err(|_| gather_invalid("could not reserve complete reconstructed vector"))?;
        complete.resize(self.total, 0.0);
        self.reconstruct_into(gathered_indices, gathered_values, &mut complete)?;
        Ok(complete)
    }

    /// Validate explicit owner blocks and reconstruct into admitted storage.
    ///
    /// # Errors
    /// Returns `EQ0802` for shape, owner, index, duplicate/missing, or finite-
    /// value contradictions. This method performs no allocation.
    pub fn reconstruct_into(
        &self,
        gathered_indices: &[u64],
        gathered_values: &[f64],
        complete: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if gathered_indices.len() != self.total || gathered_values.len() != self.total {
            return Err(gather_invalid(
                "gathered index/value extents are inconsistent",
            ));
        }
        if complete.len() != self.total {
            return Err(gather_invalid(
                "complete reconstruction storage has the wrong extent",
            ));
        }
        if gathered_indices != self.expected_indices {
            let block = self
                .displacements
                .iter()
                .zip(&self.counts)
                .position(|(&start, &count)| {
                    let Ok(start) = usize::try_from(start) else {
                        return true;
                    };
                    let Ok(count) = usize::try_from(count) else {
                        return true;
                    };
                    let Some(end) = start.checked_add(count) else {
                        return true;
                    };
                    gathered_indices.get(start..end) != self.expected_indices.get(start..end)
                })
                .unwrap_or(0);
            return Err(gather_invalid(format!(
                "gathered global-index block for partition {block} contradicts the admitted owner map"
            )));
        }
        complete.fill(f64::NAN);
        for (&global, &value) in gathered_indices.iter().zip(gathered_values) {
            let global = usize::try_from(global)
                .map_err(|_| gather_invalid("gathered global index exceeds usize"))?;
            let Some(slot) = complete.get_mut(global) else {
                return Err(gather_invalid("gathered global index is out of range"));
            };
            if !value.is_finite() || slot.is_finite() {
                return Err(gather_invalid(
                    "gathered values contain a non-finite or duplicate entry",
                ));
            }
            *slot = value;
        }
        if complete.iter().any(|value| !value.is_finite()) {
            return Err(gather_invalid("gather omitted a global index"));
        }
        Ok(())
    }
}

fn reserved_vector<T>(capacity: usize, purpose: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| count_overflow(format!("could not reserve {purpose}")))?;
    Ok(values)
}

/// Domain-separated fixed-size summary of accepted producer evidence.
///
/// V2 adds the complete solver, production-execution, and verification
/// provider descriptors. This is an in-memory MPI agreement protocol, not a
/// durable artifact wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProducerReportSummaryV2([u8; 32]);

impl ProducerReportSummaryV2 {
    /// Fixed summary extent in bytes.
    pub const ENCODED_LEN: usize = 32;

    /// Derive the summary from every producer-evidence field used by the MPI
    /// bridge.
    ///
    /// # Errors
    /// Returns `EQ0807` if a string or count exceeds portable `u64`.
    pub fn from_report(report: &SolveReport) -> Result<Self, Diagnostic> {
        ProducerReportRecordV2::from_report(report).summarize()
    }

    /// Raw summary bytes for one fixed-size all-rank comparison.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[cfg(feature = "mpi-test-hooks")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProviderSummarySubstitution {
    SolverImplementationVersion,
    ExecutionLibraries,
}

#[cfg(feature = "mpi-test-hooks")]
const SUBSTITUTED_PROVIDER_LIBRARIES: &[ProviderLibrary] =
    &[ProviderLibrary::new("eqiora-test-provider", "9.9.9")];

#[cfg(feature = "mpi-test-hooks")]
impl ProducerReportSummaryV2 {
    pub(crate) fn from_report_with_substitution(
        report: &SolveReport,
        substitution: ProviderSummarySubstitution,
    ) -> Result<Self, Diagnostic> {
        let mut record = ProducerReportRecordV2::from_report(report);
        match substitution {
            ProviderSummarySubstitution::SolverImplementationVersion => {
                record.solver_provider = SolverProvider::new(
                    record.solver_provider.id(),
                    "9.9.9-substituted",
                    record.solver_provider.libraries(),
                );
            }
            ProviderSummarySubstitution::ExecutionLibraries => {
                record.execution_provider = ExecutionProvider::new(
                    record.execution_provider.id(),
                    record.execution_provider.implementation_version(),
                    SUBSTITUTED_PROVIDER_LIBRARIES,
                );
            }
        }
        record.summarize()
    }
}

#[derive(Debug, Clone, Copy)]
struct ProducerReportRecordV2 {
    solver_provider: SolverProvider,
    execution_provider: ExecutionProvider,
    verification_provider: ExecutionProvider,
    backend: &'static str,
    execution_adapter: &'static str,
    topology: ExecutionTopology,
    orientation: LinearOperatorOrientation,
    reason: ConvergenceReason,
    completed_iterations: usize,
    initial_residual_norm: f64,
    reported_residual_norm: f64,
    true_residual_norm: f64,
    residual_target: f64,
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
}

impl ProducerReportRecordV2 {
    fn from_report(report: &SolveReport) -> Self {
        let plan = report.solver_plan();
        Self {
            solver_provider: report.solver_provider(),
            execution_provider: report.execution_provider(),
            verification_provider: report.verification_provider(),
            backend: report.backend().as_str(),
            execution_adapter: report.execution().adapter().as_str(),
            topology: report.execution().topology(),
            orientation: report.orientation(),
            reason: report.reason(),
            completed_iterations: report.completed_iterations(),
            initial_residual_norm: report.initial_residual_norm(),
            reported_residual_norm: report.reported_residual_norm(),
            true_residual_norm: report.true_residual_norm(),
            residual_target: report.residual_target(),
            algorithm: plan.algorithm(),
            preconditioner: plan.preconditioner(),
            reduction: plan.reduction(),
            relative_tolerance: plan.relative_tolerance(),
            absolute_tolerance: plan.absolute_tolerance(),
            maximum_iterations: plan.maximum_iterations(),
        }
    }

    fn summarize(self) -> Result<ProducerReportSummaryV2, Diagnostic> {
        self.solver_provider.validate()?;
        self.execution_provider.validate()?;
        self.verification_provider.validate()?;
        let mut hash = Sha256::new();
        hash.update(PRODUCER_REPORT_DOMAIN_V2);
        update_provider(
            &mut hash,
            0,
            self.solver_provider.id().as_str(),
            self.solver_provider.implementation_version(),
            self.solver_provider.libraries(),
        )?;
        update_provider(
            &mut hash,
            1,
            self.execution_provider.id().as_str(),
            self.execution_provider.implementation_version(),
            self.execution_provider.libraries(),
        )?;
        update_provider(
            &mut hash,
            2,
            self.verification_provider.id().as_str(),
            self.verification_provider.implementation_version(),
            self.verification_provider.libraries(),
        )?;
        update_string(&mut hash, self.backend)?;
        update_string(&mut hash, self.execution_adapter)?;
        match self.topology {
            ExecutionTopology::Host { workers } => {
                hash.update([0]);
                update_usize(&mut hash, workers.get(), "host worker count")?;
            }
            ExecutionTopology::Distributed {
                ranks,
                workers_per_partition,
            } => {
                hash.update([1]);
                update_usize(&mut hash, ranks.get(), "distributed rank count")?;
                update_usize(
                    &mut hash,
                    workers_per_partition.get(),
                    "distributed worker count",
                )?;
            }
            ExecutionTopology::Cuda { device } => {
                hash.update([2]);
                hash.update(device.to_be_bytes());
            }
        }
        hash.update([match self.orientation {
            LinearOperatorOrientation::Normal => 0,
            LinearOperatorOrientation::Transposed => 1,
        }]);
        hash.update([match self.reason {
            ConvergenceReason::InitialResidualSatisfied => 0,
            ConvergenceReason::ResidualToleranceSatisfied => 1,
        }]);
        update_usize(&mut hash, self.completed_iterations, "completed iterations")?;
        for value in [
            self.initial_residual_norm,
            self.reported_residual_norm,
            self.true_residual_norm,
            self.residual_target,
        ] {
            hash.update(value.to_bits().to_be_bytes());
        }
        hash.update([solver_tag::linear_solver_tag(self.algorithm)]);
        hash.update([match self.preconditioner {
            PreconditionerPolicy::Identity => 0,
            PreconditionerPolicy::Jacobi => 1,
        }]);
        hash.update([match self.reduction {
            ReductionPolicy::Reproducible => 0,
            ReductionPolicy::Fast => 1,
        }]);
        hash.update(self.relative_tolerance.to_bits().to_be_bytes());
        hash.update(self.absolute_tolerance.to_bits().to_be_bytes());
        update_usize(
            &mut hash,
            self.maximum_iterations.get(),
            "maximum iterations",
        )?;
        Ok(ProducerReportSummaryV2(hash.finalize().into()))
    }
}

fn update_provider(
    hash: &mut Sha256,
    role: u8,
    id: &str,
    implementation_version: &str,
    libraries: &[ProviderLibrary],
) -> Result<(), Diagnostic> {
    hash.update(PROVIDER_RECORD_DOMAIN_V1);
    hash.update([role]);
    update_string(hash, id)?;
    update_string(hash, implementation_version)?;
    update_usize(hash, libraries.len(), "provider dependency-release count")?;
    for library in libraries {
        update_string(hash, library.name())?;
        update_string(hash, library.version())?;
    }
    Ok(())
}

pub(crate) fn checked_mpi_count(value: usize, name: &'static str) -> Result<i32, Diagnostic> {
    i32::try_from(value).map_err(|_| count_overflow(format!("{name} exceeds mpi::Count")))
}

fn update_string(hash: &mut Sha256, value: &str) -> Result<(), Diagnostic> {
    update_usize(hash, value.len(), "producer identity byte length")?;
    hash.update(value.as_bytes());
    Ok(())
}

fn update_usize(hash: &mut Sha256, value: usize, name: &'static str) -> Result<(), Diagnostic> {
    hash.update(portable_u64(value, name)?.to_be_bytes());
    Ok(())
}

fn portable_u64(value: usize, name: &'static str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| count_overflow(format!("{name} exceeds portable u64")))
}

fn protocol_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn count_overflow(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn gather_invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    use eqiora_distributed::GlobalVectorSpace;
    use eqiora_solver::{
        BackendId, ExecutionId, ExecutionProvider, ExecutionReport, LinearOperatorOrientation,
        SolverPlan, SolverProvider,
    };

    #[test]
    fn phase_status_selects_lowest_rejected_partition() {
        let partitions = NonZeroUsize::new(3).unwrap();
        let step = CollectiveStepV1::new(CollectivePhaseV1::LocalAction, 4, 2);
        let records = [
            PhaseStatusV1::ready(step, PartitionId::new(0)).unwrap(),
            PhaseStatusV1::rejected(
                step,
                PartitionId::new(1),
                DistributedProtocolFailureV1::NumericalFailure,
            )
            .unwrap(),
            PhaseStatusV1::rejected(
                step,
                PartitionId::new(2),
                DistributedProtocolFailureV1::InvalidRealization,
            )
            .unwrap(),
        ];
        let decoded = records
            .into_iter()
            .map(PhaseStatusV1::encode)
            .map(PhaseStatusV1::decode)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let error = evaluate_phase_statuses(&decoded, partitions, step).unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert!(error.message().contains("partition 1"));

        let mut wrong_iteration = decoded;
        wrong_iteration[2].iteration = 5;
        assert_eq!(
            evaluate_phase_statuses(&wrong_iteration, partitions, step)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        let mut wrong_ordinal = wrong_iteration;
        wrong_ordinal[2].iteration = 4;
        wrong_ordinal[2].ordinal = 3;
        assert_eq!(
            evaluate_phase_statuses(&wrong_ordinal, partitions, step)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn phase_status_wire_golden_includes_the_exact_collective_step() {
        let encoded = PhaseStatusV1::rejected(
            CollectiveStepV1::new(CollectivePhaseV1::Reduction, 0x0102, 0x0304),
            PartitionId::new(5),
            DistributedProtocolFailureV1::NumericalFailure,
        )
        .unwrap()
        .encode();
        assert_eq!(
            encoded,
            [
                1, 7, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 5,
            ]
        );
        assert_eq!(PhaseStatusV1::decode(encoded).unwrap().encode(), encoded);
    }

    #[test]
    fn accepted_collective_trace_requires_dense_admission_and_terminal_suffix() {
        let completed_iterations = 4;
        let phases = [
            (CollectivePhaseV1::Admission, 0),
            (CollectivePhaseV1::LocalAction, 0),
            (CollectivePhaseV1::ProducerReport, completed_iterations),
            (CollectivePhaseV1::ProducerReport, completed_iterations),
            (CollectivePhaseV1::ProducerReport, completed_iterations),
            (CollectivePhaseV1::GatherPreparation, 0),
            (CollectivePhaseV1::GatherValidation, 0),
            (CollectivePhaseV1::HostAcceptance, completed_iterations),
            (CollectivePhaseV1::ResultAgreement, 0),
            (CollectivePhaseV1::ResultAgreement, 0),
        ];
        let steps = phases
            .into_iter()
            .enumerate()
            .map(|(ordinal, (phase, iteration))| CollectiveStepV1::new(phase, iteration, ordinal))
            .collect::<Vec<_>>();
        validate_successful_collective_trace(&steps, completed_iterations).unwrap();

        let mut sparse = steps.clone();
        sparse[1] = CollectiveStepV1::new(CollectivePhaseV1::LocalAction, 0, 2);
        assert!(validate_successful_collective_trace(&sparse, completed_iterations).is_err());

        let mut repeated_admission = steps.clone();
        repeated_admission[1] = CollectiveStepV1::new(CollectivePhaseV1::Admission, 0, 1);
        assert!(
            validate_successful_collective_trace(&repeated_admission, completed_iterations)
                .is_err()
        );

        let mut reordered = steps.clone();
        reordered.swap(5, 6);
        reordered[5] = CollectiveStepV1::new(reordered[5].phase(), reordered[5].iteration(), 5);
        reordered[6] = CollectiveStepV1::new(reordered[6].phase(), reordered[6].iteration(), 6);
        assert!(validate_successful_collective_trace(&reordered, completed_iterations).is_err());

        assert!(validate_successful_collective_trace(&steps, completed_iterations + 1).is_err());
    }

    #[test]
    fn admission_requires_exact_rank_order_and_fingerprint() {
        let partitions = NonZeroUsize::new(2).unwrap();
        let record = |rank: u64, fingerprint: u8| {
            let mut bytes = [0_u8; ADMISSION_RECORD_BYTES];
            bytes[0] = ADMISSION_PROTOCOL_VERSION;
            bytes[2..10].copy_from_slice(&2_u64.to_be_bytes());
            bytes[10..18].copy_from_slice(&rank.to_be_bytes());
            bytes[18..50].fill(fingerprint);
            AdmissionRecordV1::decode(bytes).unwrap()
        };
        assert!(evaluate_admission(&[record(0, 7), record(1, 7)], partitions).is_ok());
        assert_eq!(
            evaluate_admission(&[record(0, 7), record(1, 8)], partitions)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        assert_eq!(
            evaluate_admission(&[record(1, 7), record(0, 7)], partitions)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn owner_gather_uses_explicit_indices_for_noncontiguous_ownership() {
        let partition = arbitrary_partition();
        let plan = OwnedGatherPlanV1::new(&partition).unwrap();
        assert_eq!(plan.counts(), &[2, 1, 2]);
        assert_eq!(plan.displacements(), &[0, 2, 3]);

        // Rank-order blocks: rank 0 owns [1, 4], rank 1 owns [2], rank 2 owns
        // [0, 3]. Neither counts nor displacements imply these indices.
        let indices = [1_u64, 4, 2, 0, 3];
        let values = [11.0, 44.0, 22.0, 0.0, 33.0];
        assert_eq!(
            plan.reconstruct(&indices, &values).unwrap(),
            [0.0, 11.0, 22.0, 33.0, 44.0]
        );

        let mut wrong_owner = indices;
        wrong_owner[0] = 0;
        assert!(plan.reconstruct(&wrong_owner, &values).is_err());
        let mut nonfinite = values;
        nonfinite[3] = f64::NAN;
        assert!(plan.reconstruct(&indices, &nonfinite).is_err());
    }

    #[test]
    fn mpi_v0_count_conversion_is_checked() {
        assert_eq!(
            checked_mpi_count(i32::MAX as usize, "test").unwrap(),
            i32::MAX
        );
        assert!(checked_mpi_count(i32::MAX as usize + 1, "test").is_err());
    }

    #[test]
    fn producer_summary_covers_every_admitted_report_axis() {
        const CHANGED_LIBRARIES: &[ProviderLibrary] =
            &[ProviderLibrary::new("eqiora-test-provider", "9.9.9")];
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(10).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Jacobi);
        let solver_provider = SolverProvider::new(BackendId::new("eqiora.mpi.krylov"), "test", &[]);
        let execution_provider =
            ExecutionProvider::new(ExecutionId::new("eqiora.mpi"), "test", &[]);
        let report = SolveReport::accepted(
            solver_provider,
            execution_provider,
            ExecutionReport::distributed(
                ExecutionId::new("eqiora.mpi"),
                NonZeroUsize::new(3).unwrap(),
            ),
            LinearOperatorOrientation::Normal,
            plan,
            ConvergenceReason::ResidualToleranceSatisfied,
            2,
            1.0,
            1.0e-15,
            1.0e-15,
            1.0e-12,
        )
        .unwrap();
        let summary = ProducerReportSummaryV2::from_report(&report).unwrap();
        let baseline = ProducerReportRecordV2::from_report(&report);
        assert_eq!(baseline.summarize().unwrap(), summary);

        let cases = [
            (
                "solver provider ID",
                ProducerReportRecordV2 {
                    solver_provider: SolverProvider::new(
                        BackendId::new("eqiora.other.cg"),
                        baseline.solver_provider.implementation_version(),
                        baseline.solver_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "solver provider implementation version",
                ProducerReportRecordV2 {
                    solver_provider: SolverProvider::new(
                        baseline.solver_provider.id(),
                        "9.9.9",
                        baseline.solver_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "solver provider libraries",
                ProducerReportRecordV2 {
                    solver_provider: SolverProvider::new(
                        baseline.solver_provider.id(),
                        baseline.solver_provider.implementation_version(),
                        CHANGED_LIBRARIES,
                    ),
                    ..baseline
                },
            ),
            (
                "execution provider ID",
                ProducerReportRecordV2 {
                    execution_provider: ExecutionProvider::new(
                        ExecutionId::new("eqiora.other.mpi"),
                        baseline.execution_provider.implementation_version(),
                        baseline.execution_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "execution provider implementation version",
                ProducerReportRecordV2 {
                    execution_provider: ExecutionProvider::new(
                        baseline.execution_provider.id(),
                        "9.9.9",
                        baseline.execution_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "execution provider libraries",
                ProducerReportRecordV2 {
                    execution_provider: ExecutionProvider::new(
                        baseline.execution_provider.id(),
                        baseline.execution_provider.implementation_version(),
                        CHANGED_LIBRARIES,
                    ),
                    ..baseline
                },
            ),
            (
                "verification provider ID",
                ProducerReportRecordV2 {
                    verification_provider: ExecutionProvider::new(
                        ExecutionId::new("eqiora.other.verifier"),
                        baseline.verification_provider.implementation_version(),
                        baseline.verification_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "verification provider implementation version",
                ProducerReportRecordV2 {
                    verification_provider: ExecutionProvider::new(
                        baseline.verification_provider.id(),
                        "9.9.9",
                        baseline.verification_provider.libraries(),
                    ),
                    ..baseline
                },
            ),
            (
                "verification provider libraries",
                ProducerReportRecordV2 {
                    verification_provider: ExecutionProvider::new(
                        baseline.verification_provider.id(),
                        baseline.verification_provider.implementation_version(),
                        CHANGED_LIBRARIES,
                    ),
                    ..baseline
                },
            ),
            (
                "backend compatibility projection",
                ProducerReportRecordV2 {
                    backend: "eqiora.other.cg",
                    ..baseline
                },
            ),
            (
                "execution adapter compatibility projection",
                ProducerReportRecordV2 {
                    execution_adapter: "eqiora.other.mpi",
                    ..baseline
                },
            ),
            (
                "distributed ranks",
                ProducerReportRecordV2 {
                    topology: ExecutionTopology::Distributed {
                        ranks: NonZeroUsize::new(4).unwrap(),
                        workers_per_partition: NonZeroUsize::MIN,
                    },
                    ..baseline
                },
            ),
            (
                "workers per partition",
                ProducerReportRecordV2 {
                    topology: ExecutionTopology::Distributed {
                        ranks: NonZeroUsize::new(3).unwrap(),
                        workers_per_partition: NonZeroUsize::new(2).unwrap(),
                    },
                    ..baseline
                },
            ),
            (
                "orientation",
                ProducerReportRecordV2 {
                    orientation: LinearOperatorOrientation::Transposed,
                    ..baseline
                },
            ),
            (
                "reason",
                ProducerReportRecordV2 {
                    reason: ConvergenceReason::InitialResidualSatisfied,
                    ..baseline
                },
            ),
            (
                "iterations",
                ProducerReportRecordV2 {
                    completed_iterations: 3,
                    ..baseline
                },
            ),
            (
                "initial residual bits",
                ProducerReportRecordV2 {
                    initial_residual_norm: f64::from_bits(
                        baseline.initial_residual_norm.to_bits() + 1,
                    ),
                    ..baseline
                },
            ),
            (
                "reported residual bits",
                ProducerReportRecordV2 {
                    reported_residual_norm: f64::from_bits(
                        baseline.reported_residual_norm.to_bits() + 1,
                    ),
                    ..baseline
                },
            ),
            (
                "true residual bits",
                ProducerReportRecordV2 {
                    true_residual_norm: f64::from_bits(baseline.true_residual_norm.to_bits() + 1),
                    ..baseline
                },
            ),
            (
                "target bits",
                ProducerReportRecordV2 {
                    residual_target: f64::from_bits(baseline.residual_target.to_bits() + 1),
                    ..baseline
                },
            ),
            (
                "algorithm",
                ProducerReportRecordV2 {
                    algorithm: LinearSolver::BiConjugateGradientStabilized,
                    ..baseline
                },
            ),
            (
                "preconditioner",
                ProducerReportRecordV2 {
                    preconditioner: PreconditionerPolicy::Identity,
                    ..baseline
                },
            ),
            (
                "reduction",
                ProducerReportRecordV2 {
                    reduction: ReductionPolicy::Fast,
                    ..baseline
                },
            ),
            (
                "relative tolerance bits",
                ProducerReportRecordV2 {
                    relative_tolerance: f64::from_bits(baseline.relative_tolerance.to_bits() + 1),
                    ..baseline
                },
            ),
            (
                "absolute tolerance bits",
                ProducerReportRecordV2 {
                    absolute_tolerance: f64::from_bits(baseline.absolute_tolerance.to_bits() + 1),
                    ..baseline
                },
            ),
            (
                "maximum iterations",
                ProducerReportRecordV2 {
                    maximum_iterations: NonZeroUsize::new(11).unwrap(),
                    ..baseline
                },
            ),
        ];
        for (axis, changed) in cases {
            assert_ne!(changed.summarize().unwrap(), summary, "missing {axis} axis");
        }

        let minres = ProducerReportRecordV2 {
            algorithm: LinearSolver::MinimumResidual,
            preconditioner: PreconditionerPolicy::Identity,
            ..baseline
        }
        .summarize()
        .expect("stable producer-report tag 2 represents MINRES");
        assert_ne!(minres, summary);
    }

    fn arbitrary_partition() -> Partition {
        Partition::new(
            GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), eqiora_core::ScalarType::F64),
            NonZeroUsize::new(3).unwrap(),
            vec![
                PartitionId::new(2),
                PartitionId::new(0),
                PartitionId::new(1),
                PartitionId::new(2),
                PartitionId::new(0),
            ],
        )
        .unwrap()
    }
}
