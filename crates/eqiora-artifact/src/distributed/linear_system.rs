use eqiora_core::Diagnostic;
use eqiora_solver::{CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties};
use serde::{Deserialize, Serialize};

use super::{
    LINEAR_SYSTEM_SCHEMA, WireF64Scalar, checked_sum, portable_u64, portable_usize, preflight,
    require_limit, try_collect, try_copy_slice, validate_canonical_f64,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DistributedDecoderLimits, check_json_limits,
    invalid_artifact,
};

/// Durable complete square `f64` CSR system projected from one canonical view.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSystemEnvelopeV1 {
    wire: WireLinearSystemV1,
}

impl LinearSystemEnvelopeV1 {
    /// Project one validated Eqiora-owned complete CSR action into canonical
    /// artifact bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if a portable count is unavailable or the system
    /// exceeds the default decoder bounds.
    pub fn from_complete(complete: &CanonicalCsrSystemView) -> Result<Self, Diagnostic> {
        let values = try_copy_slice(complete.values(), "CSR value")?;
        let right_hand_side = try_copy_slice(complete.right_hand_side(), "right-hand-side value")?;
        let wire = WireLinearSystemV1 {
            schema: LINEAR_SYSTEM_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            scalar: WireF64Scalar::F64,
            dimension: portable_u64(complete.rows(), "linear-system dimension")?,
            row_offsets: try_collect(
                complete.row_offsets().len(),
                "CSR row-offset",
                complete
                    .row_offsets()
                    .iter()
                    .map(|&value| portable_u64(value, "CSR row offset")),
            )?,
            column_indices: try_collect(
                complete.column_indices().len(),
                "CSR column-index",
                complete
                    .column_indices()
                    .iter()
                    .map(|&value| portable_u64(value, "CSR column index")),
            )?,
            values,
            right_hand_side,
            properties: WireOperatorProperties::try_from(complete.properties())?,
        };
        let envelope = Self { wire };
        envelope.validate(DistributedDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Decode, bound, and reconstruct one complete CSR system through the
    /// same canonical view validation used by execution.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, noncanonical, oversized, or
    /// numerically invalid wire data.
    pub fn from_json(bytes: &[u8], limits: DistributedDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        preflight::linear_system(bytes, limits)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid linear-system envelope JSON: {error}"))
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
            invalid_artifact(format!("cannot serialize linear-system envelope: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of the complete canonical DTO.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            LINEAR_SYSTEM_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Reconstruct the sole Eqiora-owned fixed CSR action from this artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` if portable conversion or canonical CSR validation
    /// fails.
    pub fn to_complete(&self) -> Result<CanonicalCsrSystemView, Diagnostic> {
        self.to_complete_with_limits(DistributedDecoderLimits::default())
    }

    pub(super) fn to_complete_with_limits(
        &self,
        limits: DistributedDecoderLimits,
    ) -> Result<CanonicalCsrSystemView, Diagnostic> {
        self.validate_shape(limits)?;
        let row_offsets = try_collect(
            self.wire.row_offsets.len(),
            "CSR row-offset",
            self.wire
                .row_offsets
                .iter()
                .map(|&value| portable_usize(value, "CSR row offset")),
        )?;
        let column_indices = try_collect(
            self.wire.column_indices.len(),
            "CSR column-index",
            self.wire
                .column_indices
                .iter()
                .map(|&value| portable_usize(value, "CSR column index")),
        )?;
        let storage = DecodedStorage {
            dimension: portable_usize(self.wire.dimension, "linear-system dimension")?,
            row_offsets,
            column_indices,
            values: &self.wire.values,
            right_hand_side: &self.wire.right_hand_side,
        };
        CanonicalCsrSystemView::new(&storage, self.wire.properties.into()).map_err(|error| {
            invalid_artifact(format!(
                "linear-system envelope violates canonical CSR validation: {}",
                error.message()
            ))
        })
    }

    fn validate(&self, limits: DistributedDecoderLimits) -> Result<(), Diagnostic> {
        self.to_complete_with_limits(limits).map(|_| ())
    }

    fn validate_shape(&self, limits: DistributedDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != LINEAR_SYSTEM_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported linear-system schema or canonical encoding",
            ));
        }
        let dimension = portable_usize(self.wire.dimension, "linear-system dimension")?;
        if dimension == 0 {
            return Err(invalid_artifact("linear-system dimension must be nonzero"));
        }
        require_limit(
            "linear-system dimension",
            dimension,
            limits.max_distributed_dimension,
        )?;
        require_limit(
            "linear-system nonzero",
            self.wire.column_indices.len(),
            limits.max_distributed_nonzeros,
        )?;
        if self.wire.values.len() != self.wire.column_indices.len() {
            return Err(invalid_artifact(
                "linear-system columns and values must have equal length",
            ));
        }
        let expected_offsets = dimension
            .checked_add(1)
            .ok_or_else(|| invalid_artifact("linear-system row-offset extent overflows usize"))?;
        if self.wire.row_offsets.len() != expected_offsets
            || self.wire.right_hand_side.len() != dimension
        {
            return Err(invalid_artifact(
                "linear-system row offsets and right-hand side contradict its dimension",
            ));
        }
        let aggregate = checked_sum(
            "distributed artifact aggregate work",
            [
                dimension,
                self.wire.row_offsets.len(),
                self.wire.column_indices.len(),
                self.wire.values.len(),
                self.wire.right_hand_side.len(),
            ],
        )?;
        require_limit(
            "distributed artifact aggregate work",
            aggregate,
            limits.max_distributed_aggregate_work,
        )?;
        validate_canonical_f64(&self.wire.values, "linear-system values")?;
        validate_canonical_f64(&self.wire.right_hand_side, "linear-system right-hand side")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLinearSystemV1 {
    schema: String,
    encoding: String,
    scalar: WireF64Scalar,
    dimension: u64,
    row_offsets: Vec<u64>,
    column_indices: Vec<u64>,
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
    properties: WireOperatorProperties,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOperatorProperties {
    General,
    SymmetricPositiveDefinite,
}

impl TryFrom<LinearOperatorProperties> for WireOperatorProperties {
    type Error = Diagnostic;

    fn try_from(value: LinearOperatorProperties) -> Result<Self, Self::Error> {
        match value {
            LinearOperatorProperties::General => Ok(Self::General),
            LinearOperatorProperties::SymmetricPositiveDefinite => {
                Ok(Self::SymmetricPositiveDefinite)
            }
            LinearOperatorProperties::SymmetricIndefinite
            | LinearOperatorProperties::Symmetric
            | LinearOperatorProperties::ComplexSymmetric
            | LinearOperatorProperties::Hermitian
            | LinearOperatorProperties::HermitianPositiveDefinite => Err(invalid_artifact(
                "distributed linear-system artifact v1 only admits general or real symmetric-positive-definite properties",
            )),
        }
    }
}

impl From<WireOperatorProperties> for LinearOperatorProperties {
    fn from(value: WireOperatorProperties) -> Self {
        match value {
            WireOperatorProperties::General => Self::General,
            WireOperatorProperties::SymmetricPositiveDefinite => Self::SymmetricPositiveDefinite,
        }
    }
}

#[derive(Debug)]
struct DecodedStorage<'a> {
    dimension: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: &'a [f64],
    right_hand_side: &'a [f64],
}

impl CompleteCsrStorage for DecodedStorage<'_> {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    fn values(&self) -> &[f64] {
        self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        self.right_hand_side
    }
}
