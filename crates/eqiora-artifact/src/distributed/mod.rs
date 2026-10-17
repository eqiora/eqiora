mod content_dag;
mod layout;
mod linear_system;
mod partition;
mod preflight;

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{JsonLimits, invalid_artifact};

pub use content_dag::validate_distributed_content_dag;
pub use layout::DistributedLayoutEnvelopeV1;
pub use linear_system::LinearSystemEnvelopeV1;
pub use partition::PartitionEnvelopeV1;

/// Semantic work budgets shared by distributed algebra artifact generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedDecoderLimits {
    /// Common JSON syntax admission.
    pub json: JsonLimits,
    /// Maximum dimension of one decoded distributed algebra artifact.
    pub max_distributed_dimension: usize,
    /// Maximum partitions in one decoded unique-owner map.
    pub max_distributed_partitions: usize,
    /// Maximum nonzeros in one decoded complete CSR system.
    pub max_distributed_nonzeros: usize,
    /// Maximum owner-map entries in one decoded partition artifact.
    pub max_distributed_owner_entries: usize,
    /// Maximum owned and ghost indices summed across one layout artifact.
    pub max_distributed_local_indices: usize,
    /// Maximum halo records in one decoded layout artifact.
    pub max_distributed_halo_records: usize,
    /// Maximum halo indices summed across one decoded layout artifact.
    pub max_distributed_halo_indices: usize,
    /// Maximum aggregate scalar work admitted before reconstruction.
    pub max_distributed_aggregate_work: usize,
}

impl Default for DistributedDecoderLimits {
    fn default() -> Self {
        Self {
            json: JsonLimits::default(),
            max_distributed_dimension: 4_000_000,
            max_distributed_partitions: 65_536,
            max_distributed_nonzeros: 32_000_000,
            max_distributed_owner_entries: 4_000_000,
            max_distributed_local_indices: 16_000_000,
            max_distributed_halo_records: 4_000_000,
            max_distributed_halo_indices: 16_000_000,
            max_distributed_aggregate_work: 96_000_000,
        }
    }
}

pub(super) const LINEAR_SYSTEM_SCHEMA: &str = "eqiora.linear-system-envelope/v1";
pub(super) const PARTITION_SCHEMA: &str = "eqiora.partition-envelope/v1";
pub(super) const DISTRIBUTED_LAYOUT_SCHEMA: &str = "eqiora.distributed-layout-envelope/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireF64Scalar {
    F64,
}

fn checked_sum(label: &str, counts: impl IntoIterator<Item = usize>) -> Result<usize, Diagnostic> {
    counts.into_iter().try_fold(0_usize, |sum, count| {
        sum.checked_add(count)
            .ok_or_else(|| invalid_artifact(format!("{label} overflows usize")))
    })
}

fn require_limit(label: &str, actual: usize, limit: usize) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_artifact(format!(
            "{label} count {actual} exceeds decoder limit {limit}"
        )))
    } else {
        Ok(())
    }
}

fn portable_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds local usize")))
}

fn portable_u64(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds portable u64")))
}

fn validate_canonical_f64(values: &[f64], label: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_artifact(format!("{label} must be finite")));
    }
    if values
        .iter()
        .any(|value| *value == 0.0 && value.is_sign_negative())
    {
        return Err(invalid_artifact(format!(
            "{label} must use canonical positive zero"
        )));
    }
    Ok(())
}

fn strictly_ascending(values: &[u64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn try_collect<T>(
    len: usize,
    label: &str,
    values: impl IntoIterator<Item = Result<T, Diagnostic>>,
) -> Result<Vec<T>, Diagnostic> {
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(|_| {
        invalid_artifact(format!(
            "cannot reserve storage for {len} decoded {label} entries"
        ))
    })?;
    for value in values {
        result.push(value?);
    }
    Ok(result)
}

fn try_copy_slice<T: Copy>(values: &[T], label: &str) -> Result<Vec<T>, Diagnostic> {
    try_collect(values.len(), label, values.iter().copied().map(Ok))
}
