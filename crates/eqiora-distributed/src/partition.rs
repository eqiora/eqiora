use std::num::NonZeroUsize;

use eqiora_core::{Diagnostic, ScalarType};

use crate::error::invalid_realization;

/// One execution-group partition index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionId(usize);

impl PartitionId {
    /// Construct a zero-based partition index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based partition index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Global algebraic vector space before distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalVectorSpace {
    dimension: NonZeroUsize,
    scalar_type: ScalarType,
}

impl GlobalVectorSpace {
    /// Construct a nonempty global vector space.
    #[must_use]
    pub const fn new(dimension: NonZeroUsize, scalar_type: ScalarType) -> Self {
        Self {
            dimension,
            scalar_type,
        }
    }

    /// Number of global algebraic entries.
    #[must_use]
    pub const fn dimension(self) -> NonZeroUsize {
        self.dimension
    }

    /// Scalar representation of every entry.
    #[must_use]
    pub const fn scalar_type(self) -> ScalarType {
        self.scalar_type
    }
}

/// Unique owner of every global degree of freedom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    space: GlobalVectorSpace,
    count: NonZeroUsize,
    owners: Vec<PartitionId>,
}

impl Partition {
    /// Validate an explicit unique-owner map.
    ///
    /// Every global entry has exactly one owner by construction. V0 also
    /// requires every declared partition to own at least one entry.
    ///
    /// # Errors
    /// Returns `EQ0807` for a length mismatch, out-of-range owner, empty
    /// partition, or a partition count that cannot be indexed by `usize`.
    pub fn new(
        space: GlobalVectorSpace,
        count: NonZeroUsize,
        owners: Vec<PartitionId>,
    ) -> Result<Self, Diagnostic> {
        if owners.len() != space.dimension.get() {
            return Err(invalid_realization(format!(
                "partition owner map has {} entries for global dimension {}",
                owners.len(),
                space.dimension
            )));
        }
        if count.get() > owners.len() {
            return Err(invalid_realization(format!(
                "{} partitions cannot each own an entry in dimension {}",
                count, space.dimension
            )));
        }
        if let Some(owner) = owners.iter().find(|owner| owner.0 >= count.get()) {
            return Err(invalid_realization(format!(
                "partition owner {} is outside 0..{}",
                owner.0, count
            )));
        }
        let mut owned_counts = Vec::new();
        owned_counts
            .try_reserve_exact(count.get())
            .map_err(|_| invalid_realization("could not reserve partition owner counts"))?;
        owned_counts.resize(count.get(), 0_usize);
        for owner in &owners {
            owned_counts[owner.0] += 1;
        }
        if let Some(empty) = owned_counts.iter().position(|owned| *owned == 0) {
            return Err(invalid_realization(format!(
                "partition {empty} owns no global entries"
            )));
        }
        Ok(Self {
            space,
            count,
            owners,
        })
    }

    /// Deterministic balanced contiguous ownership.
    ///
    /// # Errors
    /// Returns `EQ0807` when more partitions than global entries are
    /// requested.
    pub fn balanced_contiguous(
        space: GlobalVectorSpace,
        count: NonZeroUsize,
    ) -> Result<Self, Diagnostic> {
        if count.get() > space.dimension.get() {
            return Err(invalid_realization(format!(
                "{} partitions cannot each own an entry in dimension {}",
                count, space.dimension
            )));
        }
        let dimension = space.dimension.get();
        let base = dimension / count.get();
        let longer_partitions = dimension % count.get();
        let longer_extent = (base + 1) * longer_partitions;
        let mut owners = Vec::new();
        owners
            .try_reserve_exact(dimension)
            .map_err(|_| invalid_realization("could not reserve balanced partition owner map"))?;
        for global in 0..dimension {
            let owner = if global < longer_extent {
                global / (base + 1)
            } else {
                longer_partitions + (global - longer_extent) / base
            };
            owners.push(PartitionId(owner));
        }
        Self::new(space, count, owners)
    }

    /// Global vector space being partitioned.
    #[must_use]
    pub const fn space(&self) -> GlobalVectorSpace {
        self.space
    }

    /// Execution-group partition count.
    #[must_use]
    pub const fn count(&self) -> NonZeroUsize {
        self.count
    }

    /// Unique owner of one global index.
    #[must_use]
    pub fn owner(&self, global: usize) -> Option<PartitionId> {
        self.owners.get(global).copied()
    }

    /// Complete owner map in ascending global-index order.
    #[must_use]
    pub fn owners(&self) -> &[PartitionId] {
        &self.owners
    }

    /// Ascending global indices uniquely owned by one declared partition.
    ///
    /// An out-of-range partition yields an empty iterator.
    pub fn owned_indices(&self, partition: PartitionId) -> impl Iterator<Item = usize> + '_ {
        self.owners
            .iter()
            .enumerate()
            .filter_map(move |(global, owner)| (*owner == partition).then_some(global))
    }
}
