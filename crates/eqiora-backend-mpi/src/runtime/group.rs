//! Collective construction of the application-owned MPI execution group.
#[cfg(feature = "mpi-test-hooks")]
use super::TestFault;
use super::{
    EXECUTION_RECEIPT_SUMMARY_BYTES, MpiExecutionGroup, MpiThreadSupport, RANK_DEVICE_RECORD_BYTES,
    checked_extent, invalid_realization, selection, zeroed,
};
use eqiora_core::Diagnostic;
use eqiora_distributed::PartitionId;
use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::CommunicatorCollectives;
use std::num::NonZeroUsize;

struct PreparedGroup {
    partitions: NonZeroUsize,
    partition: PartitionId,
    thread_support: MpiThreadSupport,
    preparation_statuses: Vec<u8>,
    receipt_summary_bytes: Vec<u8>,
    rank_device_bytes: Vec<u8>,
    planning_agreement_bytes: Vec<u8>,
    #[cfg(feature = "mpi-test-hooks")]
    fault: Option<TestFault>,
}

impl PreparedGroup {
    fn new<C: Communicator>(
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
        #[cfg(feature = "mpi-test-hooks")]
        if injected_preparation_failure(rank) {
            return Err(invalid_realization(
                "injected MPI group preparation failure",
            ));
        }
        Ok(Self {
            partitions: size,
            partition: PartitionId::new(rank),
            thread_support: provided,
            preparation_statuses: zeroed(size.get(), "MPI preparation statuses")?,
            receipt_summary_bytes: zeroed(
                checked_extent(
                    EXECUTION_RECEIPT_SUMMARY_BYTES,
                    size.get(),
                    "MPI execution receipt summaries",
                )?,
                "MPI execution receipt summaries",
            )?,
            rank_device_bytes: zeroed(
                checked_extent(
                    RANK_DEVICE_RECORD_BYTES,
                    size.get(),
                    "MPI rank-device records",
                )?,
                "MPI rank-device records",
            )?,
            planning_agreement_bytes: selection::agreement_buffer(size)?,
            #[cfg(feature = "mpi-test-hooks")]
            fault: TestFault::from_environment(rank),
        })
    }
}

impl MpiExecutionGroup {
    /// Collectively prepare storage, then duplicate an initialized communicator.
    ///
    /// MPI initialization/finalization remains entirely with the application.
    /// `provided` must be the level returned by `initialize_with_threading`.
    /// Every rank must call this constructor in the same collective order.
    /// A stack-scalar readiness reduction uses the supplied communicator before
    /// any rank returns a validation/allocation error or enters `Comm_dup`.
    ///
    /// # Errors
    /// Returns a common `EQ0807` diagnostic if any rank has invalid rank/size,
    /// insufficient thread support, or a fallible storage preparation failure.
    pub fn duplicate<C: Communicator>(
        communicator: &C,
        provided: Threading,
        required: MpiThreadSupport,
    ) -> Result<Self, Diagnostic> {
        let local = PreparedGroup::new(communicator, provided, required);
        let local_failed = i32::from(local.is_err());
        let mut any_failed = 0_i32;
        communicator.all_reduce_into(&local_failed, &mut any_failed, SystemOperation::max());
        if any_failed != 0 {
            return Err(invalid_realization(
                "MPI group preparation failed before communicator duplication",
            ));
        }
        let prepared = local.map_err(|_| {
            invalid_realization("MPI group preparation failed despite all-rank readiness agreement")
        })?;
        Ok(Self {
            communicator: communicator.duplicate(),
            partitions: prepared.partitions,
            partition: prepared.partition,
            thread_support: prepared.thread_support,
            preparation_statuses: prepared.preparation_statuses,
            receipt_summary_bytes: prepared.receipt_summary_bytes,
            rank_device_bytes: prepared.rank_device_bytes,
            planning_agreement_bytes: prepared.planning_agreement_bytes,
            #[cfg(feature = "mpi-test-hooks")]
            fault: prepared.fault,
        })
    }
}

#[cfg(feature = "mpi-test-hooks")]
fn injected_preparation_failure(rank: usize) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static USED: AtomicBool = AtomicBool::new(false);
    std::env::var("EQIORA_MPI_TEST_FAULT").ok().as_deref()
        == Some(format!("group-preparation:{rank}").as_str())
        && !USED.swap(true, Ordering::Relaxed)
}
