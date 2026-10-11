use eqiora_core::Diagnostic;
use eqiora_solver::ReductionPolicy;

use crate::allocation::solve_vector;
use crate::error::{invalid_realization, solve_failed};
use crate::layout::LocalLayout;
use crate::partition::Partition;

use super::{DistributedCsr, finite_sum};

#[derive(Debug)]
struct LocalValues {
    owned: Vec<f64>,
    ghosts: Vec<Option<f64>>,
}

impl LocalValues {
    fn value(&self, layout: &LocalLayout, global: usize) -> Option<f64> {
        match layout.owned().binary_search(&global) {
            Ok(local) => self.owned.get(local).copied(),
            Err(_) => layout
                .ghosts()
                .binary_search(&global)
                .ok()
                .and_then(|local| self.ghosts.get(local).copied().flatten()),
        }
    }
}

/// Single-process protocol oracle for partition, halo, and owned-row action.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopbackExecutor;

impl LoopbackExecutor {
    /// Compute a distributed dot product from unique-owner contributions.
    ///
    /// Each partition accumulates its owned entries in ascending global-index
    /// order. The loopback collective then combines partials in partition
    /// order. V0 admits only the reproducible policy; `Fast` requires a native
    /// transport collective rather than a misleading loopback implementation.
    ///
    /// # Errors
    /// Returns `EQ0807` for a policy unsupported by this executor, or `EQ0802`
    /// for shape/non-finite inputs or reduction overflow.
    pub fn dot(
        &self,
        partition: &Partition,
        left: &[f64],
        right: &[f64],
        policy: ReductionPolicy,
    ) -> Result<f64, Diagnostic> {
        if policy != ReductionPolicy::Reproducible {
            return Err(invalid_realization(
                "the loopback collective admits only reproducible reductions",
            ));
        }
        let dimension = partition.space().dimension().get();
        if left.len() != dimension
            || right.len() != dimension
            || left.iter().chain(right).any(|value| !value.is_finite())
        {
            return Err(solve_failed(format!(
                "distributed dot operands must each contain {dimension} finite values"
            )));
        }
        let mut partials = solve_vector(partition.count().get(), "loopback dot partials")?;
        partials.resize(partition.count().get(), 0.0);
        for global in 0..dimension {
            let owner = partition.owners()[global].index();
            partials[owner] = finite_sum(partials[owner], left[global] * right[global])?;
        }
        partials
            .into_iter()
            .try_fold(0.0, finite_sum)
            .map_err(|_| solve_failed("distributed dot reduction overflowed"))
    }

    /// Apply a distributed CSR plan by splitting owned input, executing the
    /// exact halo plan, applying owned rows, and gathering owned output.
    ///
    /// # Errors
    /// Returns `EQ0802` for shape/non-finite input, a missing halo value, or a
    /// non-finite local action.
    pub fn apply(&self, operator: &DistributedCsr, input: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        let dimension = operator.partition().space().dimension().get();
        if input.len() != dimension || input.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(format!(
                "distributed input must contain {dimension} finite values"
            )));
        }
        let mut locals = solve_vector(operator.layouts().len(), "loopback local-vector table")?;
        for layout in operator.layouts() {
            let mut owned = solve_vector(layout.owned().len(), "loopback owned values")?;
            for &global in layout.owned() {
                owned.push(input[global]);
            }
            let mut ghosts = solve_vector(layout.ghosts().len(), "loopback ghost values")?;
            ghosts.resize(layout.ghosts().len(), None);
            locals.push(LocalValues { owned, ghosts });
        }
        for exchange in operator.halo().exchanges() {
            for &global in exchange.indices() {
                let owner_layout = operator
                    .layouts
                    .get(exchange.owner().index())
                    .ok_or_else(|| solve_failed("halo plan names an unknown owner layout"))?;
                let value = locals
                    .get(exchange.owner().index())
                    .and_then(|local| local.value(owner_layout, global))
                    .ok_or_else(|| solve_failed("halo plan source is not owner-resident"))?;
                let receiver_layout = operator
                    .layouts
                    .get(exchange.receiver().index())
                    .ok_or_else(|| solve_failed("halo plan names an unknown receiver layout"))?;
                let ghost = receiver_layout
                    .ghosts()
                    .binary_search(&global)
                    .map_err(|_| solve_failed("halo plan target is not a declared ghost"))?;
                let receiver = locals
                    .get_mut(exchange.receiver().index())
                    .ok_or_else(|| solve_failed("halo receiver has no local vector"))?;
                receiver.ghosts[ghost] = Some(value);
            }
        }

        let mut output = solve_vector(dimension, "loopback gathered output")?;
        output.resize(dimension, 0.0);
        for ((layout, shard), local) in operator.layouts().iter().zip(&operator.shards).zip(&locals)
        {
            if shard.partition != layout.partition() {
                return Err(solve_failed(
                    "loopback CSR shard contradicts its canonical local layout",
                ));
            }
            for (local_row, &global_row) in layout.owned().iter().enumerate() {
                let mut value = 0.0;
                for entry in shard.row_offsets[local_row]..shard.row_offsets[local_row + 1] {
                    let global_column = shard.column_indices[entry];
                    let input_value = local.value(layout, global_column).ok_or_else(|| {
                        solve_failed(format!(
                            "partition {} lacks required owned/ghost value {global_column}",
                            layout.partition().index()
                        ))
                    })?;
                    value += shard.values[entry] * input_value;
                }
                if !value.is_finite() {
                    return Err(solve_failed(format!(
                        "distributed row {global_row} produced a non-finite value"
                    )));
                }
                output[global_row] = value;
            }
        }
        Ok(output)
    }
}
