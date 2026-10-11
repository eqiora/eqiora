mod loopback;

pub use loopback::LoopbackExecutor;

use std::num::NonZeroUsize;

use eqiora_core::{Diagnostic, ScalarType};
use eqiora_solver::{CanonicalCsrSystemView, DiagonalAvailability};

use crate::allocation::{checked_count_add, realization_vector};
use crate::error::{invalid_realization, solve_failed};
use crate::layout::{HaloPlan, LocalLayout};
use crate::partition::{Partition, PartitionId};

#[derive(Debug, PartialEq)]
struct LocalCsrStorage {
    partition: PartitionId,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct OwnedShardMaterialization {
    operator: DistributedCsr,
    local_right_hand_sides: Vec<Vec<f64>>,
}

impl OwnedShardMaterialization {
    pub(crate) fn into_parts(self) -> (DistributedCsr, Vec<Vec<f64>>) {
        (self.operator, self.local_right_hand_sides)
    }
}

/// One partition's canonical owned rows for an already assembled linear system.
///
/// Rows and columns retain global indices. This is the transport-neutral seam
/// for promoting an accepted owner-row assembly without choosing another
/// partition or rebuilding shards from a complete matrix.
#[derive(Debug, PartialEq)]
pub struct OwnedLinearSystemShard {
    partition: PartitionId,
    global_dimension: NonZeroUsize,
    rows: Vec<usize>,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl OwnedLinearSystemShard {
    /// Validate one compact owned-row CSR shard.
    ///
    /// # Errors
    /// Returns `EQ0807` for noncanonical rows/CSR, an out-of-range index,
    /// shape disagreement, or non-finite numerical content.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition: PartitionId,
        global_dimension: NonZeroUsize,
        rows: Vec<usize>,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
        right_hand_side: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let expected_row_offsets = rows.len().checked_add(1).ok_or_else(|| {
            invalid_realization("owned linear-system shard row count exceeds addressable storage")
        })?;
        if rows.windows(2).any(|pair| pair[0] >= pair[1])
            || rows.iter().any(|row| *row >= global_dimension.get())
            || right_hand_side.len() != rows.len()
            || right_hand_side.iter().any(|value| !value.is_finite())
            || row_offsets.len() != expected_row_offsets
            || row_offsets.first() != Some(&0)
            || row_offsets.last() != Some(&column_indices.len())
            || column_indices.len() != values.len()
            || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || column_indices
                .iter()
                .any(|column| *column >= global_dimension.get())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "owned linear-system shard has noncanonical shape, indices, or values",
            ));
        }
        for row in 0..rows.len() {
            if column_indices[row_offsets[row]..row_offsets[row + 1]]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid_realization(
                    "owned linear-system shard columns must be strictly ordered within every row",
                ));
            }
        }
        Ok(Self {
            partition,
            global_dimension,
            rows,
            row_offsets,
            column_indices,
            values,
            right_hand_side,
        })
    }

    /// Owning execution-group partition.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Strictly increasing owned global rows.
    #[must_use]
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }
}

/// Borrowed pairing of one canonical local layout and its owned-row CSR data.
///
/// [`DistributedCsr`] owns each [`LocalLayout`] exactly once. This view keeps
/// shard action convenient without cloning that identity-bearing layout into
/// the numerical storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalCsrShard<'a> {
    layout: &'a LocalLayout,
    storage: &'a LocalCsrStorage,
}

/// Owned remapping from one accepted shard to its rectangular execution view.
///
/// The accepted shard remains the sole source of rows, values, and global
/// layout identity. This capture owns only the backend-independent local
/// column ordinals and the exactly permuted coefficients needed to present a
/// canonical rectangular sparse execution view.
#[derive(Debug, PartialEq)]
pub struct LocalCsrExecutionCapture<'a> {
    shard: LocalCsrShard<'a>,
    local_column_indices: Vec<usize>,
    local_values: Vec<f64>,
}

impl LocalCsrExecutionCapture<'_> {
    /// Borrow the rectangular CSR view represented by this capture.
    ///
    /// Columns address a local input vector ordered as all owned entries in
    /// [`LocalLayout::owned`] order followed by all ghosts in
    /// [`LocalLayout::ghosts`] order.
    #[must_use]
    pub fn view(&self) -> LocalCsrExecutionView<'_> {
        LocalCsrExecutionView {
            partition: self.shard.layout.partition(),
            owned_global_indices: self.shard.layout.owned(),
            ghost_global_indices: self.shard.layout.ghosts(),
            row_offsets: &self.shard.storage.row_offsets,
            column_indices: &self.local_column_indices,
            values: &self.local_values,
        }
    }
}

/// Borrowed rectangular CSR with deterministic partition-local columns.
///
/// Its rows are the shard's owned global rows in ascending order. Its input
/// columns use the canonical `[owned | ghost]` layout described by
/// [`LocalCsrExecutionCapture::view`]. This is an execution projection and
/// deliberately carries no independent algebraic identity. Entries within
/// each row are ordered by the local column ordinal; coefficients are copied
/// only to preserve their exact pairing through that deterministic
/// permutation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalCsrExecutionView<'a> {
    partition: PartitionId,
    owned_global_indices: &'a [usize],
    ghost_global_indices: &'a [usize],
    row_offsets: &'a [usize],
    column_indices: &'a [usize],
    values: &'a [f64],
}

impl LocalCsrExecutionView<'_> {
    /// Partition whose owned rows form this view.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Number of owned rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.owned_global_indices.len()
    }

    /// Number of local input columns: owned entries followed by ghosts.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.owned_global_indices.len() + self.ghost_global_indices.len()
    }

    /// Ascending global rows and leading local columns owned by this partition.
    #[must_use]
    pub const fn owned_global_indices(&self) -> &[usize] {
        self.owned_global_indices
    }

    /// Ascending global indices represented by trailing local columns.
    #[must_use]
    pub const fn ghost_global_indices(&self) -> &[usize] {
        self.ghost_global_indices
    }

    /// CSR row offsets for the owned rows.
    #[must_use]
    pub const fn row_offsets(&self) -> &[usize] {
        self.row_offsets
    }

    /// CSR column ordinals into the canonical local `[owned | ghost]` input.
    #[must_use]
    pub const fn column_indices(&self) -> &[usize] {
        self.column_indices
    }

    /// CSR values borrowed from the capture-owned exact permutation.
    #[must_use]
    pub const fn values(&self) -> &[f64] {
        self.values
    }
}

impl<'a> LocalCsrShard<'a> {
    /// Owned/ghost vector layout used by this shard.
    #[must_use]
    pub const fn layout(&self) -> &LocalLayout {
        self.layout
    }

    /// Whether two views borrow the exact same sealed layout and CSR storage.
    ///
    /// This process-local authority check is intentionally distinct from
    /// structural equality and from durable algebraic identity. A repeated
    /// solver action can prove it received the admitted objects in constant
    /// time without walking resident coefficients; an equal clone is not the
    /// same origin.
    #[must_use]
    pub fn same_origin(self, other: Self) -> bool {
        std::ptr::eq(self.layout, other.layout) && std::ptr::eq(self.storage, other.storage)
    }

    /// Capture a backend-independent rectangular CSR execution projection.
    ///
    /// Every global column is deterministically remapped into a local input
    /// ordered as `[owned | ghost]`. Rows, values, and layout identity remain
    /// derived from this accepted shard; the capture owns only the local
    /// remap and its exact coefficient permutation.
    ///
    /// # Errors
    /// Returns `EQ0807` if the shard and layout disagree, the CSR shape is not
    /// canonical, or a global column is absent from both owned and ghost maps.
    pub fn capture_execution(self) -> Result<LocalCsrExecutionCapture<'a>, Diagnostic> {
        let expected_offsets = self.layout.owned().len().checked_add(1).ok_or_else(|| {
            invalid_realization("local CSR execution row-offset count overflowed")
        })?;
        let local_columns = self
            .layout
            .owned()
            .len()
            .checked_add(self.layout.ghosts().len())
            .ok_or_else(|| invalid_realization("local CSR execution column count overflowed"))?;
        if self.storage.partition != self.layout.partition()
            || self.layout.owned().is_empty()
            || self
                .layout
                .owned()
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .layout
                .ghosts()
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .layout
                .owned()
                .iter()
                .any(|global| self.layout.ghosts().binary_search(global).is_ok())
            || self.storage.row_offsets.len() != expected_offsets
            || self.storage.row_offsets.first() != Some(&0)
            || self.storage.row_offsets.last() != Some(&self.storage.column_indices.len())
            || self
                .storage
                .row_offsets
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || self.storage.column_indices.len() != self.storage.values.len()
            || self.storage.values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "local CSR execution capture has noncanonical layout or shape",
            ));
        }
        for local_row in 0..self.layout.owned().len() {
            let start = self.storage.row_offsets[local_row];
            let end = self.storage.row_offsets[local_row + 1];
            if self.storage.column_indices[start..end]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid_realization(
                    "local CSR execution columns must be globally ordered within every row",
                ));
            }
        }

        let mut local_column_indices = realization_vector(
            self.storage.column_indices.len(),
            "local CSR execution column remap",
        )?;
        let mut local_values = realization_vector(
            self.storage.values.len(),
            "local CSR execution coefficient permutation",
        )?;
        let maximum_row_entries = self
            .storage
            .row_offsets
            .windows(2)
            .map(|range| range[1] - range[0])
            .max()
            .unwrap_or(0);
        let mut row_entries =
            realization_vector(maximum_row_entries, "local CSR execution row permutation")?;
        for local_row in 0..self.layout.owned().len() {
            row_entries.clear();
            let start = self.storage.row_offsets[local_row];
            let end = self.storage.row_offsets[local_row + 1];
            for (&global, &value) in self.storage.column_indices[start..end]
                .iter()
                .zip(&self.storage.values[start..end])
            {
                let local = match self.layout.owned().binary_search(&global) {
                    Ok(owned) => owned,
                    Err(_) => {
                        let ghost = self.layout.ghosts().binary_search(&global).map_err(|_| {
                            invalid_realization(format!(
                                "local CSR execution column {global} is neither owned nor ghosted"
                            ))
                        })?;
                        self.layout
                            .owned()
                            .len()
                            .checked_add(ghost)
                            .ok_or_else(|| {
                                invalid_realization("local CSR execution ghost ordinal overflowed")
                            })?
                    }
                };
                if local >= local_columns {
                    return Err(invalid_realization(
                        "local CSR execution remap exceeds its rectangular column extent",
                    ));
                }
                row_entries.push((local, value));
            }
            row_entries.sort_unstable_by_key(|(column, _)| *column);
            if row_entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                return Err(invalid_realization(
                    "local CSR execution remap produced duplicate local columns",
                ));
            }
            local_column_indices.extend(row_entries.iter().map(|(column, _)| *column));
            local_values.extend(row_entries.iter().map(|(_, value)| *value));
        }
        Ok(LocalCsrExecutionCapture {
            shard: self,
            local_column_indices,
            local_values,
        })
    }

    /// Apply owned rows from owned and already-exchanged ghost values.
    ///
    /// Inputs and output use the exact ascending orders in [`LocalLayout`].
    ///
    /// # Errors
    /// Returns `EQ0802` for shape/non-finite values or an internally missing
    /// local/ghost column.
    pub fn apply(
        &self,
        owned: &[f64],
        ghosts: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if owned.len() != self.layout.owned().len()
            || ghosts.len() != self.layout.ghosts().len()
            || output.len() != self.layout.owned().len()
            || owned.iter().chain(ghosts).any(|value| !value.is_finite())
        {
            return Err(solve_failed(
                "local CSR input/output shapes or values contradict its layout",
            ));
        }
        for (local_row, value) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for entry in
                self.storage.row_offsets[local_row]..self.storage.row_offsets[local_row + 1]
            {
                let global = self.storage.column_indices[entry];
                let input = match self.layout.owned().binary_search(&global) {
                    Ok(local) => owned[local],
                    Err(_) => {
                        let local = self.layout.ghosts().binary_search(&global).map_err(|_| {
                            solve_failed(format!(
                                "local CSR column {global} is neither owned nor ghosted"
                            ))
                        })?;
                        ghosts[local]
                    }
                };
                sum = finite_sum(sum, self.storage.values[entry] * input)?;
            }
            *value = sum;
        }
        Ok(())
    }

    /// Write diagonal entries for the owned rows in local-layout order.
    ///
    /// # Errors
    /// Returns `EQ0802` when `output` does not match the owned-row extent.
    pub fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if output.len() != self.layout.owned().len() {
            return Err(solve_failed(
                "local CSR diagonal output must match its owned-row extent",
            ));
        }
        for (local, (&global, value)) in self.layout.owned().iter().zip(output).enumerate() {
            let start = self.storage.row_offsets[local];
            let end = self.storage.row_offsets[local + 1];
            let Ok(offset) = self.storage.column_indices[start..end].binary_search(&global) else {
                return Ok(DiagonalAvailability::Unavailable);
            };
            *value = self.storage.values[start + offset];
        }
        Ok(DiagonalAvailability::Available)
    }
}

/// One square CSR operator lowered into owned-row shards and a halo plan.
#[derive(Debug, PartialEq)]
pub struct DistributedCsr {
    partition: Partition,
    layouts: Vec<LocalLayout>,
    shards: Vec<LocalCsrStorage>,
    halo: HaloPlan,
}

impl DistributedCsr {
    /// Partition a finite square global CSR artifact.
    ///
    /// This is an explicit realization copy: each shard retains only its
    /// owned rows, with global column indices used to derive ghost storage.
    ///
    /// # Errors
    /// Returns `EQ0807` for malformed CSR structure, non-finite values, or a
    /// non-`f64` global space.
    pub fn from_global_csr(
        partition: Partition,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        let dimension = partition.space().dimension().get();
        if partition.space().scalar_type() != ScalarType::F64 {
            return Err(invalid_realization(
                "distributed CSR v0 executes only f64 vectors",
            ));
        }
        validate_csr(dimension, row_offsets, column_indices, values)?;

        #[derive(Debug, Clone, Copy, Default)]
        struct ShardShape {
            owned_rows: usize,
            nonzeros: usize,
            ghost_candidates: usize,
        }

        let mut shapes = realization_vector(partition.count().get(), "local CSR shape table")?;
        shapes.resize(partition.count().get(), ShardShape::default());
        for row in 0..dimension {
            let owner = partition.owners()[row].index();
            let row_nonzeros = row_offsets[row + 1] - row_offsets[row];
            shapes[owner].owned_rows =
                checked_count_add(shapes[owner].owned_rows, 1, "local owned-row count")?;
            shapes[owner].nonzeros = checked_count_add(
                shapes[owner].nonzeros,
                row_nonzeros,
                "local CSR nonzero count",
            )?;
            for &column in &column_indices[row_offsets[row]..row_offsets[row + 1]] {
                if partition.owners()[column].index() != owner {
                    shapes[owner].ghost_candidates = checked_count_add(
                        shapes[owner].ghost_candidates,
                        1,
                        "local ghost-candidate count",
                    )?;
                }
            }
        }

        let mut layouts = realization_vector(partition.count().get(), "local layout table")?;
        for id in 0..partition.count().get() {
            let id = PartitionId::new(id);
            let shape = shapes[id.index()];
            layouts.push(LocalLayout::reserved(
                &partition,
                id,
                shape.owned_rows,
                shape.ghost_candidates,
            )?);
        }
        for row in 0..dimension {
            let owner = partition.owners()[row];
            let layout = &mut layouts[owner.index()];
            layout.push_owned(row);
            for &column in &column_indices[row_offsets[row]..row_offsets[row + 1]] {
                if partition.owners()[column] != owner {
                    layout.push_ghost_candidate(column);
                }
            }
        }
        for (layout, shape) in layouts.iter_mut().zip(&shapes) {
            layout.canonicalize(&partition, shape.owned_rows, shape.ghost_candidates)?;
        }

        let mut shards = realization_vector(layouts.len(), "local CSR shard table")?;
        for (layout, shape) in layouts.iter().zip(&shapes) {
            let offset_count = shape
                .owned_rows
                .checked_add(1)
                .ok_or_else(|| invalid_realization("local CSR offset count overflowed"))?;
            let mut local_offsets = realization_vector(offset_count, "local CSR row offsets")?;
            let mut local_columns = realization_vector(shape.nonzeros, "local CSR column indices")?;
            let mut local_values = realization_vector(shape.nonzeros, "local CSR values")?;
            local_offsets.push(0);
            for &row in layout.owned() {
                for entry in row_offsets[row]..row_offsets[row + 1] {
                    local_columns.push(column_indices[entry]);
                    local_values.push(values[entry]);
                }
                local_offsets.push(local_columns.len());
            }
            if local_offsets.len() != offset_count
                || local_columns.len() != shape.nonzeros
                || local_values.len() != shape.nonzeros
            {
                return Err(invalid_realization(
                    "pre-counted local CSR shape changed during materialization",
                ));
            }
            shards.push(LocalCsrStorage {
                partition: layout.partition(),
                row_offsets: local_offsets,
                column_indices: local_columns,
                values: local_values,
            });
        }
        let halo = HaloPlan::derive(&partition, &layouts)?;
        Ok(Self {
            partition,
            layouts,
            shards,
            halo,
        })
    }

    pub(crate) fn from_owned_shards(
        complete: &CanonicalCsrSystemView,
        partition: Partition,
        mut shards: Vec<OwnedLinearSystemShard>,
    ) -> Result<OwnedShardMaterialization, Diagnostic> {
        if partition.space().scalar_type() != ScalarType::F64
            || partition.space().dimension().get() != complete.rows()
            || complete.rows() != complete.columns()
        {
            return Err(invalid_realization(format!(
                "complete {}x{} f64 CSR and owned-row partition {:?} dimension {} do not agree",
                complete.rows(),
                complete.columns(),
                partition.space().scalar_type(),
                partition.space().dimension()
            )));
        }
        shards.sort_unstable_by_key(|shard| shard.partition.index());
        if shards.len() != partition.count().get() {
            return Err(invalid_realization(format!(
                "owned-row inventory has {} shards for {} partitions",
                shards.len(),
                partition.count()
            )));
        }

        for (index, shard) in shards.iter().enumerate() {
            let id = PartitionId::new(index);
            if shard.partition != id
                || shard.global_dimension != partition.space().dimension()
                || !shard.rows.iter().copied().eq(partition.owned_indices(id))
            {
                return Err(invalid_realization(format!(
                    "owned-row shard {index} contradicts the accepted partition",
                )));
            }
            for (local, &global) in shard.rows.iter().enumerate() {
                let complete_range =
                    complete.row_offsets()[global]..complete.row_offsets()[global + 1];
                let local_range = shard.row_offsets[local]..shard.row_offsets[local + 1];
                if shard.column_indices[local_range.clone()]
                    != complete.column_indices()[complete_range.clone()]
                    || !same_f64_bits(
                        &shard.values[local_range],
                        &complete.values()[complete_range],
                    )
                    || shard.right_hand_side[local].to_bits()
                        != complete.right_hand_side()[global].to_bits()
                {
                    return Err(invalid_realization(format!(
                        "owned-row shard {index} row {global} differs from the complete canonical system",
                    )));
                }
            }
        }

        let mut layouts = realization_vector(partition.count().get(), "owned-row layout table")?;
        for shard in &shards {
            let ghost_candidates = shard
                .column_indices
                .iter()
                .filter(|column| partition.owner(**column) != Some(shard.partition))
                .count();
            let mut layout = LocalLayout::reserved(
                &partition,
                shard.partition,
                shard.rows.len(),
                ghost_candidates,
            )?;
            layout.extend_owned(&shard.rows);
            layout.extend_ghost_candidates(
                shard
                    .column_indices
                    .iter()
                    .copied()
                    .filter(|column| partition.owner(*column) != Some(shard.partition)),
            );
            layout.canonicalize(&partition, shard.rows.len(), ghost_candidates)?;
            layouts.push(layout);
        }
        let halo = HaloPlan::derive(&partition, &layouts)?;
        let mut local_right_hand_sides =
            realization_vector(shards.len(), "accepted local RHS table")?;
        let mut storage = realization_vector(shards.len(), "accepted local CSR table")?;
        for shard in shards {
            local_right_hand_sides.push(shard.right_hand_side);
            storage.push(LocalCsrStorage {
                partition: shard.partition,
                row_offsets: shard.row_offsets,
                column_indices: shard.column_indices,
                values: shard.values,
            });
        }
        Ok(OwnedShardMaterialization {
            operator: Self {
                partition,
                layouts,
                shards: storage,
                halo,
            },
            local_right_hand_sides,
        })
    }

    /// Unique global ownership map.
    #[must_use]
    pub const fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Local layouts in partition-index order.
    #[must_use]
    pub fn layouts(&self) -> &[LocalLayout] {
        &self.layouts
    }

    /// Derived deterministic halo plan.
    #[must_use]
    pub const fn halo(&self) -> &HaloPlan {
        &self.halo
    }

    /// Owned-row shard for one partition.
    #[must_use]
    pub fn shard(&self, partition: PartitionId) -> Option<LocalCsrShard<'_>> {
        let layout = self.layouts.get(partition.index())?;
        let storage = self.shards.get(partition.index())?;
        if layout.partition() != partition || storage.partition != partition {
            return None;
        }
        Some(LocalCsrShard { layout, storage })
    }
}

fn validate_csr(
    dimension: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[f64],
) -> Result<(), Diagnostic> {
    if dimension
        .checked_add(1)
        .is_none_or(|expected| row_offsets.len() != expected)
        || row_offsets.first() != Some(&0)
        || row_offsets.last() != Some(&column_indices.len())
        || column_indices.len() != values.len()
        || row_offsets
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return Err(invalid_realization(
            "global CSR offsets, columns, and values have inconsistent shape",
        ));
    }
    if column_indices.iter().any(|column| *column >= dimension)
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_realization(
            "global CSR contains an out-of-range column or non-finite value",
        ));
    }
    for row in 0..dimension {
        let columns = &column_indices[row_offsets[row]..row_offsets[row + 1]];
        if columns.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_realization(
                "global CSR columns must be strictly ordered within every row",
            ));
        }
    }
    Ok(())
}

fn finite_sum(sum: f64, value: f64) -> Result<f64, Diagnostic> {
    let next = sum + value;
    next.is_finite()
        .then_some(next)
        .ok_or_else(|| solve_failed("distributed reduction overflowed"))
}

fn same_f64_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_capture_rejects_noncanonical_rectangular_shape() {
        let layout = LocalLayout::from_parts(PartitionId::new(0), vec![0, 1], vec![2]);
        let storage = LocalCsrStorage {
            partition: PartitionId::new(0),
            row_offsets: vec![0, 1],
            column_indices: vec![0],
            values: vec![1.0],
        };

        assert!(
            LocalCsrShard {
                layout: &layout,
                storage: &storage,
            }
            .capture_execution()
            .is_err()
        );
    }

    #[test]
    fn execution_capture_rejects_column_outside_owned_and_ghost_maps() {
        let layout = LocalLayout::from_parts(PartitionId::new(0), vec![1], vec![0]);
        let storage = LocalCsrStorage {
            partition: PartitionId::new(0),
            row_offsets: vec![0, 1],
            column_indices: vec![2],
            values: vec![1.0],
        };

        assert!(
            LocalCsrShard {
                layout: &layout,
                storage: &storage,
            }
            .capture_execution()
            .is_err()
        );
    }
}
