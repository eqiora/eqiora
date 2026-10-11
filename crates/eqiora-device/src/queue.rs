use std::fmt::Debug;
use std::num::NonZeroU64;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::DeviceId;

/// Deployment-selected logical queue position on one device.
///
/// A slot is not a materialized command queue and establishes no submission
/// order. Runtime adapters turn it into a process-unique [`QueueId`] only after
/// creating the concrete vendor queue or stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueSlot {
    device: DeviceId,
    ordinal: u32,
}

impl QueueSlot {
    /// Select one adapter-defined logical queue slot on a device.
    #[must_use]
    pub const fn new(device: DeviceId, ordinal: u32) -> Self {
        Self { device, ordinal }
    }

    /// Device that owns the queue.
    #[must_use]
    pub const fn device(self) -> DeviceId {
        self.device
    }

    /// Adapter-scoped queue ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
}

/// Runtime-scoped identity for one materialized ordered command queue.
///
/// `materialization` distinguishes separate vendor queues created for the same
/// deployment slot. Submission order is valid only within this complete
/// identity, never merely within a device and slot ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueId {
    slot: QueueSlot,
    materialization: NonZeroU64,
}

impl QueueId {
    /// Name one concrete materialization of a logical queue slot.
    #[must_use]
    pub const fn new(slot: QueueSlot, materialization: NonZeroU64) -> Self {
        Self {
            slot,
            materialization,
        }
    }

    /// Logical deployment slot materialized by this queue.
    #[must_use]
    pub const fn slot(self) -> QueueSlot {
        self.slot
    }

    /// Device that owns the materialized queue.
    #[must_use]
    pub const fn device(self) -> DeviceId {
        self.slot.device()
    }

    /// Adapter-defined logical slot ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.slot.ordinal()
    }

    /// Process-unique materialization generation assigned by the adapter.
    #[must_use]
    pub const fn materialization(self) -> NonZeroU64 {
        self.materialization
    }
}

/// Identity of one successfully enqueued operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionId {
    queue: QueueId,
    sequence: NonZeroU64,
}

impl SubmissionId {
    /// Queue that establishes the order.
    #[must_use]
    pub const fn queue(self) -> QueueId {
        self.queue
    }

    /// Monotone sequence within the queue.
    #[must_use]
    pub const fn sequence(self) -> NonZeroU64 {
        self.sequence
    }
}

/// Monotone submission identity source owned by an adapter queue.
#[derive(Debug)]
pub struct QueueTimeline {
    queue: QueueId,
    last: u64,
}

impl QueueTimeline {
    /// Begin an empty queue timeline.
    #[must_use]
    pub const fn new(queue: QueueId) -> Self {
        Self { queue, last: 0 }
    }

    /// Allocate the next identity after a successful enqueue.
    ///
    /// # Errors
    /// Returns `EQ0807` if the sequence is exhausted.
    pub fn next_submission(&mut self) -> Result<SubmissionId, Diagnostic> {
        self.last = self
            .last
            .checked_add(1)
            .ok_or_else(|| invalid_queue("queue submission sequence exhausted"))?;
        Ok(SubmissionId {
            queue: self.queue,
            sequence: NonZeroU64::new(self.last).expect("incremented sequence is nonzero"),
        })
    }
}

/// Completion identity for one submitted device operation.
///
/// This is execution evidence, never a state-machine or hybrid-model event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Completion(SubmissionId);

impl Completion {
    /// Associate a backend fence with the operation it completes.
    #[must_use]
    pub const fn new(submission: SubmissionId) -> Self {
        Self(submission)
    }

    /// Completed submission identity.
    #[must_use]
    pub const fn submission(self) -> SubmissionId {
        self.0
    }

    /// Whether this completion strictly precedes another in the same queue.
    ///
    /// # Errors
    /// Returns `EQ0807` rather than inventing an order across queues.
    pub fn happens_before(self, other: Self) -> Result<bool, Diagnostic> {
        if self.0.queue != other.0.queue {
            return Err(invalid_queue(
                "completion order is defined only within one command queue",
            ));
        }
        Ok(self.0.sequence < other.0.sequence)
    }
}

/// Ordered asynchronous submission seam implemented by an adapter queue.
pub trait CommandQueue: Debug + Send + Sync {
    /// Eqiora-owned queue identity.
    fn id(&self) -> QueueId;
}

/// Backend fence whose vendor event remains private to the adapter.
pub trait Fence: Debug + Send + Sync {
    /// Eqiora-owned completion identity.
    fn completion(&self) -> Completion;

    /// Wait until the associated operation has completed.
    ///
    /// # Errors
    /// Returns a stable diagnostic if synchronization fails.
    fn wait(&self) -> Result<(), Diagnostic>;
}

/// Immutable evidence that the host successfully waited for one backend
/// fence.
///
/// A [`Completion`] identifies submitted work; it does not by itself prove
/// host visibility. This value can be created only by invoking [`Fence::wait`]
/// through [`WaitedCompletion::wait`]. It is execution evidence, not hardware
/// attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaitedCompletion(Completion);

impl WaitedCompletion {
    /// Wait for a concrete backend fence and retain the exact completed
    /// submission only after the wait succeeds.
    ///
    /// # Errors
    /// Returns the backend's stable synchronization diagnostic without
    /// constructing visibility evidence.
    pub fn wait(fence: &dyn Fence) -> Result<Self, Diagnostic> {
        let completion = fence.completion();
        fence.wait()?;
        if fence.completion() != completion {
            return Err(invalid_queue(
                "backend fence changed completion identity while being waited",
            ));
        }
        Ok(Self(completion))
    }

    /// Exact submission made visible by the successful wait.
    #[must_use]
    pub const fn completion(self) -> Completion {
        self.0
    }
}

fn invalid_queue(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
