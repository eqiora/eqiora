use std::num::NonZeroUsize;

use eqiora_core::{Diagnostic, ScalarType};
use eqiora_distributed::{GlobalVectorSpace, Partition, PartitionId};
use serde::{Deserialize, Serialize};

use super::{
    PARTITION_SCHEMA, WireF64Scalar, checked_sum, portable_u64, portable_usize, preflight,
    require_limit, try_collect,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DistributedDecoderLimits, check_json_limits,
    invalid_artifact,
};

/// Durable unique-owner partition over one nonempty global `f64` vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEnvelopeV1 {
    wire: WirePartitionV1,
}

impl PartitionEnvelopeV1 {
    /// Project one validated unique-owner partition into canonical artifact
    /// bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if a portable count is unavailable or the partition
    /// exceeds default decoder limits.
    pub fn from_partition(partition: &Partition) -> Result<Self, Diagnostic> {
        if partition.space().scalar_type() != ScalarType::F64 {
            return Err(invalid_artifact(
                "partition-envelope/v1 admits only f64 global vector spaces",
            ));
        }
        let wire = WirePartitionV1 {
            schema: PARTITION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            scalar: WireF64Scalar::F64,
            dimension: portable_u64(partition.space().dimension().get(), "partition dimension")?,
            partition_count: portable_u64(partition.count().get(), "partition count")?,
            owners: try_collect(
                partition.owners().len(),
                "owner-map",
                partition
                    .owners()
                    .iter()
                    .map(|owner| portable_u64(owner.index(), "owner partition")),
            )?,
        };
        let envelope = Self { wire };
        envelope.validate(DistributedDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Decode and reconstruct a bounded unique-owner partition.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, oversized, nonportable, or
    /// invalid partition data.
    pub fn from_json(bytes: &[u8], limits: DistributedDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        preflight::partition(bytes, limits)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid partition envelope JSON: {error}"))
        })?;
        let envelope = Self { wire };
        envelope.validate(limits)?;
        Ok(envelope)
    }

    /// Deterministic fixed-field-order canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize partition envelope: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of the complete canonical DTO.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            PARTITION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Reconstruct the validated L2 unique-owner partition.
    ///
    /// # Errors
    /// Returns `EQ0901` for portable-conversion, allocation, or partition
    /// invariant failure.
    pub fn to_partition(&self) -> Result<Partition, Diagnostic> {
        self.to_partition_with_limits(DistributedDecoderLimits::default())
    }

    pub(super) fn to_partition_with_limits(
        &self,
        limits: DistributedDecoderLimits,
    ) -> Result<Partition, Diagnostic> {
        self.validate_shape(limits)?;
        let dimension =
            NonZeroUsize::new(portable_usize(self.wire.dimension, "partition dimension")?)
                .ok_or_else(|| invalid_artifact("partition dimension must be nonzero"))?;
        let count = NonZeroUsize::new(portable_usize(
            self.wire.partition_count,
            "partition count",
        )?)
        .ok_or_else(|| invalid_artifact("partition count must be nonzero"))?;
        let owners = try_collect(
            self.wire.owners.len(),
            "owner-map",
            self.wire
                .owners
                .iter()
                .map(|&owner| portable_usize(owner, "owner partition").map(PartitionId::new)),
        )?;
        Partition::new(
            GlobalVectorSpace::new(dimension, ScalarType::F64),
            count,
            owners,
        )
        .map_err(|error| {
            invalid_artifact(format!(
                "partition envelope violates unique-owner validation: {}",
                error.message()
            ))
        })
    }

    fn validate(&self, limits: DistributedDecoderLimits) -> Result<(), Diagnostic> {
        self.to_partition_with_limits(limits).map(|_| ())
    }

    fn validate_shape(&self, limits: DistributedDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != PARTITION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported partition schema or canonical encoding",
            ));
        }
        let dimension = portable_usize(self.wire.dimension, "partition dimension")?;
        let count = portable_usize(self.wire.partition_count, "partition count")?;
        if dimension == 0 || count == 0 {
            return Err(invalid_artifact(
                "partition dimension and count must be nonzero",
            ));
        }
        require_limit(
            "partition dimension",
            dimension,
            limits.max_distributed_dimension,
        )?;
        require_limit("partition", count, limits.max_distributed_partitions)?;
        require_limit(
            "partition owner-map entry",
            self.wire.owners.len(),
            limits.max_distributed_owner_entries,
        )?;
        if self.wire.owners.len() != dimension {
            return Err(invalid_artifact(
                "partition owner map must contain one entry per global index",
            ));
        }
        let aggregate = checked_sum(
            "distributed artifact aggregate work",
            [dimension, count, self.wire.owners.len()],
        )?;
        require_limit(
            "distributed artifact aggregate work",
            aggregate,
            limits.max_distributed_aggregate_work,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePartitionV1 {
    schema: String,
    encoding: String,
    scalar: WireF64Scalar,
    dimension: u64,
    partition_count: u64,
    owners: Vec<u64>,
}
