//! Canonical array values referenced by external-import provenance.

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{ArtifactDigest, CANONICAL_ENCODING, check_json_limits, invalid_artifact};

const RESOLVED_ARRAY_SCHEMA: &str = "eqiora.resolved-array/v1";

/// Semantic work budgets shared by artifacts that reconstruct resolved arrays.
///
/// This contains no JSON admission policy. A decoder that embeds resolved
/// arrays composes this contract with its own JSON and artifact-family budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedArrayLimits {
    /// Maximum rank of one resolved array.
    pub max_rank: usize,
    /// Maximum scalar values in one resolved array.
    pub max_values: usize,
}

impl Default for ResolvedArrayLimits {
    fn default() -> Self {
        Self {
            max_rank: 8,
            max_values: 16_000_000,
        }
    }
}

/// Admission contract for a standalone canonical resolved-array artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResolvedArrayDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Work admitted while reconstructing the array payload.
    pub array: ResolvedArrayLimits,
}

/// Closed scalar grammar for a canonical resolved-array reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvedArrayScalarV1 {
    /// Unsigned portable integers.
    U64,
    /// Finite IEEE 754 binary64 values with positive-zero normalization.
    F64,
}

/// Exact normalized array presented to Eqiora by an external-format adapter.
///
/// This value identifies shape, scalar grammar, and flat row-major values. It
/// is a provenance reference, not a general array artifact: it carries no
/// mesh, field association, chunking, device residency, or storage identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedArrayV1 {
    wire: AcceptedArray,
}

#[derive(Debug, Clone, PartialEq)]
enum AcceptedArray {
    U64(WireResolvedArrayU64),
    F64(eqiora_schema::resolved_array::ResolvedF64Array),
}

impl ResolvedArrayV1 {
    /// Construct a row-major `u64` array.
    ///
    /// # Errors
    /// Returns `EQ0901` when shape is empty, a dimension is zero, shape
    /// arithmetic overflows local `usize`, or value count differs from the
    /// exact shape product.
    pub fn from_u64(shape: Vec<u64>, values: Vec<u64>) -> Result<Self, Diagnostic> {
        let wire = WireResolvedArrayV1::U64(WireResolvedArrayU64 {
            schema: RESOLVED_ARRAY_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            scalar: WireU64Scalar::U64,
            shape,
            values,
        });
        validate_wire(&wire, None)?;
        Ok(Self {
            wire: match wire {
                WireResolvedArrayV1::U64(wire) => AcceptedArray::U64(wire),
                WireResolvedArrayV1::F64(wire) => AcceptedArray::F64(
                    eqiora_schema::resolved_array::ResolvedF64Array::new(wire.shape, wire.values)?,
                ),
            },
        })
    }

    /// Construct a row-major `f64` array and normalize every zero to `+0.0`.
    ///
    /// # Errors
    /// Returns `EQ0901` when shape is invalid, value count differs from its
    /// exact product, or any value is NaN or infinite.
    pub fn from_f64(shape: Vec<u64>, values: Vec<f64>) -> Result<Self, Diagnostic> {
        Ok(Self {
            wire: AcceptedArray::F64(eqiora_schema::resolved_array::ResolvedF64Array::new(
                shape, values,
            )?),
        })
    }

    /// Immutable verified binary64 semantic value, if selected.
    #[must_use]
    pub const fn resolved_f64(&self) -> Option<&eqiora_schema::resolved_array::ResolvedF64Array> {
        match &self.wire {
            AcceptedArray::F64(array) => Some(array),
            AcceptedArray::U64(_) => None,
        }
    }

    /// Decode the exact closed DTO under byte, nesting, rank, and value limits.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed/unknown data, scalar-tag mismatch,
    /// invalid shape/count, resource excess, non-finite `f64`, or negative zero.
    pub fn from_json(bytes: &[u8], limits: ResolvedArrayDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid resolved array JSON: {error}")))?;
        validate_wire(&wire, Some(limits))?;
        Ok(Self {
            wire: match wire {
                WireResolvedArrayV1::U64(wire) => AcceptedArray::U64(wire),
                WireResolvedArrayV1::F64(wire) => AcceptedArray::F64(
                    eqiora_schema::resolved_array::ResolvedF64Array::new(wire.shape, wire.values)?,
                ),
            },
        })
    }

    /// Deterministic ordered DTO bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        match &self.wire {
            AcceptedArray::F64(array) => array.canonical_json(),
            AcceptedArray::U64(wire) => serde_json::to_vec(wire).map_err(|error| {
                invalid_artifact(format!("cannot serialize resolved array: {error}"))
            }),
        }
    }

    /// Domain-separated SHA-256 identity of the complete normalized DTO.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        if let AcceptedArray::F64(array) = &self.wire {
            return Ok(ArtifactDigest::from_sha256(array.digest()?));
        }
        Ok(ArtifactDigest::compute(
            RESOLVED_ARRAY_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Closed scalar grammar selected by this array.
    #[must_use]
    pub const fn scalar(&self) -> ResolvedArrayScalarV1 {
        match &self.wire {
            AcceptedArray::U64(_) => ResolvedArrayScalarV1::U64,
            AcceptedArray::F64(_) => ResolvedArrayScalarV1::F64,
        }
    }

    /// Ordered positive portable dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        match &self.wire {
            AcceptedArray::U64(wire) => &wire.shape,
            AcceptedArray::F64(wire) => wire.shape(),
        }
    }

    /// Flat row-major unsigned values, when this is a `u64` array.
    #[must_use]
    pub fn u64_values(&self) -> Option<&[u64]> {
        match &self.wire {
            AcceptedArray::U64(wire) => Some(&wire.values),
            AcceptedArray::F64(_) => None,
        }
    }

    /// Flat row-major binary64 values, when this is an `f64` array.
    #[must_use]
    pub fn f64_values(&self) -> Option<&[f64]> {
        match &self.wire {
            AcceptedArray::U64(_) => None,
            AcceptedArray::F64(wire) => Some(wire.values()),
        }
    }
}

fn validate_wire(
    wire: &WireResolvedArrayV1,
    limits: Option<ResolvedArrayDecoderLimits>,
) -> Result<(), Diagnostic> {
    let (schema, encoding, shape, value_count) = match wire {
        WireResolvedArrayV1::U64(wire) => (
            wire.schema.as_str(),
            wire.encoding.as_str(),
            wire.shape.as_slice(),
            wire.values.len(),
        ),
        WireResolvedArrayV1::F64(wire) => {
            if wire.values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_artifact(
                    "resolved f64 array values must all be finite",
                ));
            }
            if wire
                .values
                .iter()
                .any(|value| *value == 0.0 && value.is_sign_negative())
            {
                return Err(invalid_artifact(
                    "resolved f64 array wire values must use canonical positive zero",
                ));
            }
            (
                wire.schema.as_str(),
                wire.encoding.as_str(),
                wire.shape.as_slice(),
                wire.values.len(),
            )
        }
    };
    if schema != RESOLVED_ARRAY_SCHEMA || encoding != CANONICAL_ENCODING {
        return Err(invalid_artifact(
            "unsupported resolved-array schema or canonical encoding",
        ));
    }
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid_artifact(
            "resolved array shape must contain positive dimensions",
        ));
    }
    if let Some(limits) = limits {
        require_count("resolved array rank", shape.len(), limits.array.max_rank)?;
    }
    let required_values = shape.iter().try_fold(1_usize, |product, &dimension| {
        let dimension = usize::try_from(dimension)
            .map_err(|_| invalid_artifact("resolved array dimension exceeds local usize"))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid_artifact("resolved array shape product overflows usize"))
    })?;
    if let Some(limits) = limits {
        require_count(
            "resolved array scalar values",
            required_values,
            limits.array.max_values,
        )?;
    }
    if value_count != required_values {
        return Err(invalid_artifact(format!(
            "resolved array shape requires {required_values} values, received {value_count}",
        )));
    }
    Ok(())
}

fn require_count(label: &str, actual: usize, limit: usize) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_artifact(format!(
            "{label} count {actual} exceeds decoder limit {limit}",
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
enum WireResolvedArrayV1 {
    U64(WireResolvedArrayU64),
    F64(WireResolvedArrayF64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResolvedArrayU64 {
    schema: String,
    encoding: String,
    scalar: WireU64Scalar,
    shape: Vec<u64>,
    values: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResolvedArrayF64 {
    schema: String,
    encoding: String,
    scalar: WireF64Scalar,
    shape: Vec<u64>,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireU64Scalar {
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireF64Scalar {
    F64,
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn exact_ordered_dto_and_domain_match_an_independent_oracle() {
        let array = ResolvedArrayV1::from_u64(vec![2, 2], vec![0, 1, 2, u64::MAX]).unwrap();
        let expected = concat!(
            "{\"schema\":\"eqiora.resolved-array/v1\",",
            "\"encoding\":\"eqiora.canonical-json/v1\",",
            "\"scalar\":\"u64\",\"shape\":[2,2],",
            "\"values\":[0,1,2,18446744073709551615]}"
        )
        .as_bytes();
        assert_eq!(array.canonical_json().unwrap(), expected);

        let mut oracle = Sha256::new();
        oracle.update(RESOLVED_ARRAY_SCHEMA.as_bytes());
        oracle.update([0]);
        oracle.update(expected);
        assert_eq!(
            array.digest().unwrap().sha256_bytes(),
            <[u8; 32]>::from(oracle.finalize()),
        );
    }

    #[test]
    fn f64_schema_owner_preserves_bytes_digest_and_bounded_decode() {
        let semantic =
            eqiora_schema::resolved_array::ResolvedF64Array::new(vec![2], vec![-0.0, 1.5]).unwrap();
        let array = ResolvedArrayV1::from_f64(vec![2], vec![-0.0, 1.5]).unwrap();
        assert_eq!(
            array.canonical_json().unwrap(),
            semantic.canonical_json().unwrap()
        );
        assert_eq!(
            array.digest().unwrap().sha256_bytes(),
            semantic.digest().unwrap()
        );
        let decoded = ResolvedArrayV1::from_json(
            &semantic.canonical_json().unwrap(),
            ResolvedArrayDecoderLimits::default(),
        )
        .unwrap();
        assert_eq!(decoded.resolved_f64(), Some(&semantic));
        let limits = ResolvedArrayDecoderLimits {
            array: ResolvedArrayLimits {
                max_values: 1,
                ..ResolvedArrayLimits::default()
            },
            ..ResolvedArrayDecoderLimits::default()
        };
        assert!(ResolvedArrayV1::from_json(&semantic.canonical_json().unwrap(), limits).is_err());
    }

    #[test]
    fn scalar_grammar_and_shape_order_are_identity() {
        let unsigned = ResolvedArrayV1::from_u64(vec![1], vec![1]).unwrap();
        let float = ResolvedArrayV1::from_f64(vec![1], vec![1.0]).unwrap();
        assert_ne!(unsigned.digest().unwrap(), float.digest().unwrap());

        let rows = ResolvedArrayV1::from_u64(vec![2, 3], vec![0, 1, 2, 3, 4, 5]).unwrap();
        let columns = ResolvedArrayV1::from_u64(vec![3, 2], vec![0, 1, 2, 3, 4, 5]).unwrap();
        assert_ne!(rows.digest().unwrap(), columns.digest().unwrap());
    }

    #[test]
    fn f64_zero_normalizes_and_negative_zero_wire_fails() {
        let array = ResolvedArrayV1::from_f64(vec![2], vec![-0.0, 0.0]).unwrap();
        assert_eq!(array.f64_values().unwrap()[0].to_bits(), 0.0_f64.to_bits());

        let bytes = array.canonical_json().unwrap();
        let index = bytes
            .windows(3)
            .position(|window| window == b"0.0")
            .unwrap();
        let mut negative = bytes;
        negative.splice(index..index + 3, b"-0.0".iter().copied());
        assert!(
            ResolvedArrayV1::from_json(&negative, ResolvedArrayDecoderLimits::default())
                .unwrap_err()
                .message()
                .contains("positive zero")
        );
    }

    #[test]
    fn tag_shape_unknown_data_and_limits_fail_closed() {
        for invalid in [
            br#"{"schema":"eqiora.resolved-array/v1","encoding":"eqiora.canonical-json/v1","scalar":"u64","shape":[],"values":[]}"#.as_slice(),
            br#"{"schema":"eqiora.resolved-array/v1","encoding":"eqiora.canonical-json/v1","scalar":"u64","shape":[1],"values":[1.5]}"#.as_slice(),
            br#"{"schema":"eqiora.resolved-array/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","shape":[1],"values":[1.0],"unit":"m"}"#.as_slice(),
        ] {
            assert!(ResolvedArrayV1::from_json(invalid, ResolvedArrayDecoderLimits::default()).is_err());
        }

        let array = ResolvedArrayV1::from_u64(vec![2, 2], vec![0, 1, 2, 3]).unwrap();
        let limits = ResolvedArrayDecoderLimits {
            array: ResolvedArrayLimits {
                max_values: 3,
                ..ResolvedArrayLimits::default()
            },
            ..ResolvedArrayDecoderLimits::default()
        };
        assert!(ResolvedArrayV1::from_json(&array.canonical_json().unwrap(), limits).is_err());
    }

    #[test]
    fn constructor_rejects_nonfinite_mismatch_zero_and_overflow() {
        assert!(ResolvedArrayV1::from_f64(vec![1], vec![f64::NAN]).is_err());
        assert!(ResolvedArrayV1::from_u64(vec![2], vec![1]).is_err());
        assert!(ResolvedArrayV1::from_u64(vec![0], Vec::new()).is_err());
        assert!(ResolvedArrayV1::from_u64(vec![u64::MAX, 2], Vec::new()).is_err());
    }
}
