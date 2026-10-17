use eqiora_core::Diagnostic;
use eqiora_distributed::DistributedLinearSystem;
use serde::{Deserialize, Serialize};

use super::{
    DISTRIBUTED_LAYOUT_SCHEMA, LinearSystemEnvelopeV1, PartitionEnvelopeV1, checked_sum,
    portable_u64, portable_usize, preflight, require_limit, strictly_ascending, try_collect,
};
use crate::{ArtifactDigest, CANONICAL_ENCODING, DistributedDecoderLimits, invalid_artifact};

/// Durable exact projection of sparsity-derived local layouts and halo plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedLayoutEnvelopeV1 {
    wire: WireDistributedLayoutV1,
}

impl DistributedLayoutEnvelopeV1 {
    /// Freshly derive every local and halo record from the linked complete
    /// system and unique-owner partition.
    ///
    /// # Errors
    /// Returns `EQ0901` if either input is invalid, dimensions disagree, a
    /// portable count is unavailable, or default decoder limits are exceeded.
    pub fn derive(
        system: &LinearSystemEnvelopeV1,
        partition: &PartitionEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let complete = system.to_complete()?;
        let partition_value = partition.to_partition()?;
        let distributed = DistributedLinearSystem::from_complete(&complete, partition_value)
            .map_err(|error| {
                invalid_artifact(format!(
                    "cannot derive distributed layout from linked inputs: {}",
                    error.message()
                ))
            })?;
        let layouts = distributed.operator().layouts();
        let local_layouts = try_collect(
            layouts.len(),
            "local-layout record",
            layouts.iter().map(|layout| {
                Ok(WireLocalLayoutV1 {
                    partition: portable_u64(layout.partition().index(), "layout partition")?,
                    owned: try_collect(
                        layout.owned().len(),
                        "owned global-index",
                        layout
                            .owned()
                            .iter()
                            .map(|&index| portable_u64(index, "owned global index")),
                    )?,
                    ghosts: try_collect(
                        layout.ghosts().len(),
                        "ghost global-index",
                        layout
                            .ghosts()
                            .iter()
                            .map(|&index| portable_u64(index, "ghost global index")),
                    )?,
                })
            }),
        )?;
        let exchanges = distributed.operator().halo().exchanges();
        let halo_exchanges = try_collect(
            exchanges.len(),
            "halo-exchange record",
            exchanges.iter().map(|exchange| {
                Ok(WireHaloExchangeV1 {
                    owner: portable_u64(exchange.owner().index(), "halo owner")?,
                    receiver: portable_u64(exchange.receiver().index(), "halo receiver")?,
                    indices: try_collect(
                        exchange.indices().len(),
                        "halo global-index",
                        exchange
                            .indices()
                            .iter()
                            .map(|&index| portable_u64(index, "halo global index")),
                    )?,
                })
            }),
        )?;
        let envelope = Self {
            wire: WireDistributedLayoutV1 {
                schema: DISTRIBUTED_LAYOUT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                linear_system_sha256: system.digest()?.to_string(),
                partition_sha256: partition.digest()?.to_string(),
                local_layouts,
                halo_exchanges,
            },
        };
        envelope.validate_shape(DistributedDecoderLimits::default())?;
        envelope.validate_against(system, partition)?;
        Ok(envelope)
    }

    /// Decode and structurally validate a bounded derived-layout DTO.
    ///
    /// Linked system and partition content is deliberately supplied later to
    /// [`Self::validate_against`]; decoding alone never claims derivation.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, noncanonical, nonportable, or
    /// oversized wire data.
    pub fn from_json(bytes: &[u8], limits: DistributedDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        preflight::distributed_layout(bytes, limits)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid distributed-layout envelope JSON: {error}"))
        })?;
        let envelope = Self { wire };
        envelope.validate_shape(limits)?;
        Ok(envelope)
    }

    /// Deterministic fixed-field-order canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize distributed-layout envelope: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete canonical DTO.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            DISTRIBUTED_LAYOUT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Recompute the linked distributed system and require exact digest,
    /// scalar, dimension, property, local-layout, and halo equality.
    ///
    /// # Errors
    /// Returns `EQ0901` for a cross-wire or any forged/stale derived record.
    pub fn validate_against(
        &self,
        system: &LinearSystemEnvelopeV1,
        partition: &PartitionEnvelopeV1,
    ) -> Result<DistributedLinearSystem, Diagnostic> {
        if self.wire.linear_system_sha256 != system.digest()?.as_str()
            || self.wire.partition_sha256 != partition.digest()?.as_str()
        {
            return Err(invalid_artifact(
                "distributed layout references different system or partition content",
            ));
        }
        let complete = system.to_complete()?;
        let partition_value = partition.to_partition()?;
        let fresh = DistributedLinearSystem::from_complete(&complete, partition_value).map_err(
            |error| {
                invalid_artifact(format!(
                    "linked system and partition cannot derive a distributed layout: {}",
                    error.message()
                ))
            },
        )?;
        if self.wire.local_layouts.len() != fresh.operator().layouts().len()
            || !self
                .wire
                .local_layouts
                .iter()
                .zip(fresh.operator().layouts())
                .all(|(stored, derived)| stored.equals_derived(derived))
            || self.wire.halo_exchanges.len() != fresh.operator().halo().exchanges().len()
            || !self
                .wire
                .halo_exchanges
                .iter()
                .zip(fresh.operator().halo().exchanges())
                .all(|(stored, derived)| stored.equals_derived(derived))
        {
            return Err(invalid_artifact(
                "distributed layout records do not equal their fresh derivation",
            ));
        }
        Ok(fresh)
    }

    /// Linked complete-system artifact digest.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn linear_system_digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        ArtifactDigest::from_hex(self.wire.linear_system_sha256.clone())
    }

    /// Linked unique-owner partition artifact digest.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn partition_digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        ArtifactDigest::from_hex(self.wire.partition_sha256.clone())
    }

    fn validate_shape(&self, limits: DistributedDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != DISTRIBUTED_LAYOUT_SCHEMA || self.wire.encoding != CANONICAL_ENCODING
        {
            return Err(invalid_artifact(
                "unsupported distributed-layout schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.linear_system_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.partition_sha256.clone())?;
        if self.wire.local_layouts.is_empty() {
            return Err(invalid_artifact(
                "distributed layout must contain at least one local record",
            ));
        }
        require_limit(
            "distributed local layout",
            self.wire.local_layouts.len(),
            limits.max_distributed_partitions,
        )?;
        require_limit(
            "distributed halo record",
            self.wire.halo_exchanges.len(),
            limits.max_distributed_halo_records,
        )?;
        let local_indices = self
            .wire
            .local_layouts
            .iter()
            .try_fold(0_usize, |sum, layout| {
                let count = checked_sum(
                    "distributed local-index work",
                    [layout.owned.len(), layout.ghosts.len()],
                )?;
                sum.checked_add(count)
                    .ok_or_else(|| invalid_artifact("distributed local-index work overflows usize"))
            })?;
        let halo_indices = self
            .wire
            .halo_exchanges
            .iter()
            .try_fold(0_usize, |sum, exchange| {
                sum.checked_add(exchange.indices.len())
                    .ok_or_else(|| invalid_artifact("distributed halo-index work overflows usize"))
            })?;
        require_limit(
            "distributed local index",
            local_indices,
            limits.max_distributed_local_indices,
        )?;
        require_limit(
            "distributed halo index",
            halo_indices,
            limits.max_distributed_halo_indices,
        )?;
        let aggregate = checked_sum(
            "distributed artifact aggregate work",
            [
                self.wire.local_layouts.len(),
                local_indices,
                self.wire.halo_exchanges.len(),
                halo_indices,
            ],
        )?;
        require_limit(
            "distributed artifact aggregate work",
            aggregate,
            limits.max_distributed_aggregate_work,
        )?;
        for (expected, layout) in self.wire.local_layouts.iter().enumerate() {
            if portable_usize(layout.partition, "layout partition")? != expected
                || layout.owned.is_empty()
                || !strictly_ascending(&layout.owned)
                || !strictly_ascending(&layout.ghosts)
                || layout
                    .owned
                    .iter()
                    .any(|index| layout.ghosts.binary_search(index).is_ok())
            {
                return Err(invalid_artifact(
                    "local records must be partition-ordered with nonempty sorted disjoint owned/ghost indices",
                ));
            }
            for &index in layout.owned.iter().chain(&layout.ghosts) {
                portable_usize(index, "local global index")?;
            }
        }
        let mut previous = None;
        for exchange in &self.wire.halo_exchanges {
            let owner = portable_usize(exchange.owner, "halo owner")?;
            let receiver = portable_usize(exchange.receiver, "halo receiver")?;
            let pair = (owner, receiver);
            if owner == receiver
                || owner >= self.wire.local_layouts.len()
                || receiver >= self.wire.local_layouts.len()
                || previous.is_some_and(|last| last >= pair)
                || exchange.indices.is_empty()
                || !strictly_ascending(&exchange.indices)
            {
                return Err(invalid_artifact(
                    "halo records must be unique, peer-ordered, nonempty, and strictly indexed",
                ));
            }
            for &index in &exchange.indices {
                portable_usize(index, "halo global index")?;
            }
            previous = Some(pair);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDistributedLayoutV1 {
    schema: String,
    encoding: String,
    linear_system_sha256: String,
    partition_sha256: String,
    local_layouts: Vec<WireLocalLayoutV1>,
    halo_exchanges: Vec<WireHaloExchangeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLocalLayoutV1 {
    partition: u64,
    owned: Vec<u64>,
    ghosts: Vec<u64>,
}

impl WireLocalLayoutV1 {
    fn equals_derived(&self, derived: &eqiora_distributed::LocalLayout) -> bool {
        u64::try_from(derived.partition().index()) == Ok(self.partition)
            && exact_indices(&self.owned, derived.owned())
            && exact_indices(&self.ghosts, derived.ghosts())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireHaloExchangeV1 {
    owner: u64,
    receiver: u64,
    indices: Vec<u64>,
}

impl WireHaloExchangeV1 {
    fn equals_derived(&self, derived: &eqiora_distributed::HaloExchange) -> bool {
        u64::try_from(derived.owner().index()) == Ok(self.owner)
            && u64::try_from(derived.receiver().index()) == Ok(self.receiver)
            && exact_indices(&self.indices, derived.indices())
    }
}

fn exact_indices(stored: &[u64], derived: &[usize]) -> bool {
    stored.len() == derived.len()
        && stored
            .iter()
            .zip(derived)
            .all(|(&left, &right)| u64::try_from(right) == Ok(left))
}
