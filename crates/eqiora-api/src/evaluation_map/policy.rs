use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_realization::Target;

use super::invalid;

/// Lifetime of accepted numerical state in an independent map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMapRetention {
    /// Keep each accepted evaluation with its linearization, or each complete time Result.
    Retain,
    /// Release numerical state after ordered delivery; retain its exact acceptance evidence.
    /// Explicit recomputation and mapped products must reaccept the same frozen
    /// input and reproduce that acceptance evidence. Time Runs reuse their original
    /// stochastic path identity; parameter samplers are never called by execution.
    Recompute,
}

/// Bounded host scheduling and storage charge for one independent map.
///
/// The existing host target supplies the outer worker cap. The first parallel
/// profile requires serial host execution and verification for every member;
/// workers do not share mutable solver/preparation state. Chunk and worker
/// positions are scheduling facts, never mathematical or sample identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationMapExecutionPolicy {
    chunk_size: NonZeroUsize,
    workers: NonZeroUsize,
    retention: EvaluationMapRetention,
    storage_bytes_limit: usize,
}

impl EvaluationMapExecutionPolicy {
    /// Admit a nonempty chunk and host worker cap no greater than that chunk.
    ///
    /// The byte limit charges planned points, indexed outcomes, retained
    /// acceptance evidence, numerical member buffers and bounded scheduling metadata.
    /// Temporary deployment metadata is charged by its canonical encoded size,
    /// not asserted to be its heap size. Allocator overhead, solver scratch,
    /// OS thread stacks, diagnostics and callback-owned allocations are not a
    /// process/peak-memory claim. Counts and all charge arithmetic are checked
    /// again against the actual program before map allocation.
    ///
    /// # Errors
    /// Rejects zero chunk size, non-host targets, excess workers, or
    /// unaddressable scheduling inventories before allocation.
    pub fn new(
        chunk_size: usize,
        target: Target,
        retention: EvaluationMapRetention,
        storage_bytes_limit: usize,
    ) -> Result<Self, Diagnostic> {
        let chunk_size = NonZeroUsize::new(chunk_size)
            .ok_or_else(|| invalid("evaluation map chunk size must be nonzero"))?;
        let Target::HostCpu { threads: workers } = target else {
            return Err(invalid("bounded evaluation maps require a host CPU target"));
        };
        if workers.get() > chunk_size.get()
            || chunk_size.get() > isize::MAX as usize / size_of::<usize>()
        {
            return Err(invalid(
                "evaluation map worker or chunk inventory is out of bounds",
            ));
        }
        Ok(Self {
            chunk_size,
            workers,
            retention,
            storage_bytes_limit,
        })
    }

    /// Serial execution with one active member and retained accepted numerical state.
    #[must_use]
    pub const fn retained(storage_bytes_limit: usize) -> Self {
        Self {
            chunk_size: NonZeroUsize::MIN,
            workers: NonZeroUsize::MIN,
            retention: EvaluationMapRetention::Retain,
            storage_bytes_limit,
        }
    }

    /// Maximum occurrences in one ordering buffer.
    #[must_use]
    pub const fn chunk_size(self) -> usize {
        self.chunk_size.get()
    }

    /// Maximum concurrently executing one-thread members.
    #[must_use]
    pub const fn workers(self) -> usize {
        self.workers.get()
    }

    /// Explicit accepted-state lifetime.
    #[must_use]
    pub const fn retention(self) -> EvaluationMapRetention {
        self.retention
    }

    /// Maximum defined storage estimate, not a process-memory ceiling.
    #[must_use]
    pub const fn storage_bytes_limit(self) -> usize {
        self.storage_bytes_limit
    }
}
