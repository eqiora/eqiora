use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_distributed::{
    DistributedAdmissionFingerprintV1, DistributedLayoutAgreementIdentityV1,
    PartitionAgreementIdentityV1,
};
use eqiora_solver::{CanonicalCsrAgreementFingerprintV1, SolverPlan};

use crate::binding::{ProcessGroupSlot, invalid};

/// Transport-neutral communication phase observed during one distributed run.
///
/// MPI, another message transport, or a deterministic protocol oracle may
/// normalize its own operation vocabulary into these phases. Transport
/// handles and library identities never enter this L2 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DistributedExecutionPhaseV1 {
    /// Collective agreement of the complete system/layout/plan admission.
    AdmissionAgreement,
    /// Readiness immediately before exchanging ghost values.
    HaloReadiness,
    /// Local operator or preconditioner work over owned rows.
    OwnedAction,
    /// An owned-vector update between communication boundaries.
    OwnedVectorUpdate,
    /// A scalar collective under the selected reduction policy.
    CollectiveReduction,
    /// Construction or cross-partition agreement of the producer report.
    ProducerReportAgreement,
    /// Readiness before the paired owner-index and owner-value gathers.
    OwnerGatherPreparation,
    /// Validation of the reconstructed complete candidate.
    OwnerGatherValidation,
    /// Native complete-host acceptance performed by every partition.
    NativeHostAcceptance,
    /// Cross-partition agreement of the accepted result.
    AcceptedResultAgreement,
}

/// One actual synchronized boundary in global execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistributedCollectiveStepV1 {
    phase: DistributedExecutionPhaseV1,
    iteration: usize,
    ordinal: usize,
}

impl DistributedCollectiveStepV1 {
    /// Record one transport-normalized synchronized boundary.
    #[must_use]
    pub const fn new(phase: DistributedExecutionPhaseV1, iteration: usize, ordinal: usize) -> Self {
        Self {
            phase,
            iteration,
            ordinal,
        }
    }

    /// Normalized operation phase.
    #[must_use]
    pub const fn phase(self) -> DistributedExecutionPhaseV1 {
        self.phase
    }

    /// Numerical iteration, with zero used by setup and finalization phases.
    #[must_use]
    pub const fn iteration(self) -> usize {
        self.iteration
    }

    /// Dense global ordinal within this admitted distributed run.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
}

/// Conservative checked capacity for the actual synchronized trace.
///
/// The portable graph retains one bounded iterative region rather than
/// expanding `maximum_iterations` nodes. Runtime adapters reserve this many
/// actual boundary records before their first numerical collective.
///
/// # Errors
/// Returns `EQ0807` when the capacity would overflow `usize`.
fn distributed_collective_trace_capacity(plan: SolverPlan) -> Result<usize, Diagnostic> {
    const SETUP_AND_FINALIZATION: usize = 64;
    const PER_ITERATION: usize = 32;
    plan.maximum_iterations()
        .get()
        .checked_mul(PER_ITERATION)
        .and_then(|count| count.checked_add(SETUP_AND_FINALIZATION))
        .ok_or_else(|| invalid("distributed collective trace capacity overflowed"))
}

/// Immutable, bounded trace of one accepted distributed linear execution.
///
/// The fixed macro DAG remains small; this value records only boundaries that
/// actually occurred inside its iterative region. Identity fields bind those
/// observations to the exact complete system, owner map, derived halo layout,
/// admission plan, and selected logical process group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedLinearExecutionTrace {
    system: CanonicalCsrAgreementFingerprintV1,
    partition: PartitionAgreementIdentityV1,
    layout: DistributedLayoutAgreementIdentityV1,
    admission: DistributedAdmissionFingerprintV1,
    process_group: ProcessGroupSlot,
    partitions: NonZeroUsize,
    workers_per_partition: NonZeroUsize,
    owner_gather_dimension: usize,
    trace_capacity: usize,
    steps: Vec<DistributedCollectiveStepV1>,
}

impl DistributedLinearExecutionTrace {
    /// Maximum collective-step inventory admitted by one solver plan.
    ///
    /// # Errors
    /// Returns a structured diagnostic when the plan's iteration bound cannot
    /// be represented by the fixed distributed trace schedule.
    pub fn collective_capacity(plan: SolverPlan) -> Result<usize, Diagnostic> {
        distributed_collective_trace_capacity(plan)
    }

    /// Seal one actual trace after native distributed result agreement.
    ///
    /// # Errors
    /// Returns `EQ0807` for an exceeded reservation, non-dense global
    /// ordinals, an iteration outside the admitted solve bound, or an absent
    /// or misordered required communication/finalization phase.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        system: CanonicalCsrAgreementFingerprintV1,
        partition: PartitionAgreementIdentityV1,
        layout: DistributedLayoutAgreementIdentityV1,
        admission: DistributedAdmissionFingerprintV1,
        process_group: ProcessGroupSlot,
        partitions: NonZeroUsize,
        workers_per_partition: NonZeroUsize,
        owner_gather_dimension: usize,
        trace_capacity: usize,
        steps: Vec<DistributedCollectiveStepV1>,
        plan: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        let maximum_capacity = distributed_collective_trace_capacity(plan)?;
        if trace_capacity > maximum_capacity || steps.len() > trace_capacity {
            return Err(invalid(
                "distributed collective trace exceeds its admitted reservation",
            ));
        }
        if owner_gather_dimension == 0 {
            return Err(invalid(
                "distributed owner gather must reconstruct a nonempty complete vector",
            ));
        }
        for (ordinal, step) in steps.iter().enumerate() {
            if step.ordinal != ordinal {
                return Err(invalid(
                    "distributed collective trace ordinals are not dense and global",
                ));
            }
            if step.iteration > plan.maximum_iterations().get() {
                return Err(invalid(
                    "distributed collective trace exceeds the admitted iteration bound",
                ));
            }
        }
        require_ordered_phases(&steps)?;
        Ok(Self {
            system,
            partition,
            layout,
            admission,
            process_group,
            partitions,
            workers_per_partition,
            owner_gather_dimension,
            trace_capacity,
            steps,
        })
    }

    /// Complete canonical system executed by this trace.
    #[must_use]
    pub const fn system(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.system
    }

    /// Exact unique-owner identity.
    #[must_use]
    pub const fn partition(&self) -> PartitionAgreementIdentityV1 {
        self.partition
    }

    /// Exact local-layout and halo identity.
    #[must_use]
    pub const fn layout(&self) -> DistributedLayoutAgreementIdentityV1 {
        self.layout
    }

    /// Exact system/layout/plan admission agreed by all partitions.
    #[must_use]
    pub const fn admission(&self) -> DistributedAdmissionFingerprintV1 {
        self.admission
    }

    /// Logical process-group slot selected by deployment.
    #[must_use]
    pub const fn process_group(&self) -> ProcessGroupSlot {
        self.process_group
    }

    /// Participating process/rank count.
    #[must_use]
    pub const fn partitions(&self) -> NonZeroUsize {
        self.partitions
    }

    /// Host workers admitted within every partition.
    #[must_use]
    pub const fn workers_per_partition(&self) -> NonZeroUsize {
        self.workers_per_partition
    }

    /// Complete vector extent reconstructed by the paired owner gathers.
    #[must_use]
    pub const fn owner_gather_dimension(&self) -> usize {
        self.owner_gather_dimension
    }

    /// Storage reserved before the first numerical collective.
    #[must_use]
    pub const fn trace_capacity(&self) -> usize {
        self.trace_capacity
    }

    /// Actual synchronized boundaries in dense global order.
    #[must_use]
    pub fn steps(&self) -> &[DistributedCollectiveStepV1] {
        &self.steps
    }
}

fn require_ordered_phases(steps: &[DistributedCollectiveStepV1]) -> Result<(), Diagnostic> {
    use DistributedExecutionPhaseV1::{
        AcceptedResultAgreement, AdmissionAgreement, CollectiveReduction, HaloReadiness,
        NativeHostAcceptance, OwnedAction, OwnedVectorUpdate, OwnerGatherPreparation,
        OwnerGatherValidation, ProducerReportAgreement,
    };

    if steps.first().map(|step| step.phase) != Some(AdmissionAgreement) {
        return Err(invalid(
            "distributed collective trace does not begin with admission agreement",
        ));
    }

    for required in [
        HaloReadiness,
        OwnedAction,
        CollectiveReduction,
        OwnedVectorUpdate,
    ] {
        if !steps.iter().any(|step| step.phase == required) {
            return Err(invalid(format!(
                "distributed collective trace omitted required phase {required:?}"
            )));
        }
    }

    let mut cursor = 0;
    for required in [
        ProducerReportAgreement,
        OwnerGatherPreparation,
        OwnerGatherValidation,
        NativeHostAcceptance,
        AcceptedResultAgreement,
    ] {
        let Some(offset) = steps[cursor..]
            .iter()
            .position(|step| step.phase == required)
        else {
            return Err(invalid(format!(
                "distributed collective trace omitted ordered final phase {required:?}"
            )));
        };
        cursor += offset + 1;
    }
    if steps.last().map(|step| step.phase) != Some(AcceptedResultAgreement) {
        return Err(invalid(
            "distributed collective trace does not terminate in accepted-result agreement",
        ));
    }
    Ok(())
}
