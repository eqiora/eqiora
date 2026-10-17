//! Discrete-field-specific storage realization.

use std::fmt;
use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, FieldDecoderLimits,
    invalid_artifact,
};

const STORAGE_SCHEMA: &str = "eqiora.discrete-field-storage-envelope/v1";

/// Raw SHA-256 of one exact storage chunk.
///
/// The distinct type prevents raw storage bytes from being substituted for a
/// domain-separated logical [`ArtifactDigest`] or an external-source digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageChunkSha256V1(String);

impl StorageChunkSha256V1 {
    /// Hash one complete raw storage chunk without a domain prefix.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            use fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        Self(encoded)
    }

    /// Parse canonical lowercase hexadecimal SHA-256.
    ///
    /// # Errors
    /// Returns `EQ0901` for any other spelling.
    pub fn from_hex(value: impl Into<String>) -> Result<Self, Diagnostic> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_artifact(
                "storage chunk digest must be 64 lowercase hexadecimal SHA-256 characters",
            ));
        }
        Ok(Self(value))
    }

    /// Canonical lowercase hexadecimal representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StorageChunkSha256V1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Independently content-addressed raw canonical-byte chunk.
///
/// This is a storage object, not logical Field identity. It intentionally has
/// no path, locator, codec, compression, or container metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageChunkV1 {
    digest: StorageChunkSha256V1,
    bytes: Vec<u8>,
}

impl StorageChunkV1 {
    /// Content-address arbitrary raw storage bytes in the closed v1 domain.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            digest: StorageChunkSha256V1::from_bytes(&bytes),
            bytes,
        }
    }

    /// Raw SHA-256 byte identity in the storage-chunk digest type.
    #[must_use]
    pub const fn digest(&self) -> &StorageChunkSha256V1 {
        &self.digest
    }

    /// Exact raw bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One concrete storage realization of canonical discrete-Field bytes.
///
/// Rechunking changes this envelope identity but cannot change the referenced
/// logical mesh-bound Field identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscreteFieldStorageEnvelopeV1 {
    wire: WireDiscreteFieldStorageV1,
}

impl DiscreteFieldStorageEnvelopeV1 {
    /// Split exact canonical discrete-Field bytes into contiguous raw chunks.
    ///
    /// # Errors
    /// Returns `EQ0901` for a byte-count conversion or offset overflow.
    pub fn pack_raw(
        field: &DiscreteFieldEnvelopeV1,
        chunk_size: NonZeroUsize,
    ) -> Result<(Self, Vec<StorageChunkV1>), Diagnostic> {
        let bytes = field.canonical_json()?;
        let chunks = bytes
            .chunks(chunk_size.get())
            .map(|bytes| StorageChunkV1::from_bytes(bytes.to_vec()))
            .collect::<Vec<_>>();
        let mut offset = 0_u64;
        let references = chunks
            .iter()
            .enumerate()
            .map(|(ordinal, chunk)| {
                let byte_count = u64::try_from(chunk.bytes.len())
                    .map_err(|_| invalid_artifact("storage chunk byte count exceeds u64"))?;
                let reference = WireStorageChunkReference {
                    ordinal: u64::try_from(ordinal)
                        .map_err(|_| invalid_artifact("storage chunk ordinal exceeds u64"))?,
                    offset,
                    byte_count,
                    chunk_sha256: chunk.digest.to_string(),
                };
                offset = offset
                    .checked_add(byte_count)
                    .ok_or_else(|| invalid_artifact("storage chunk offsets overflow u64"))?;
                Ok(reference)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let value = Self {
            wire: WireDiscreteFieldStorageV1 {
                schema: STORAGE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                logical_field_sha256: field.digest()?.to_string(),
                storage_encoding: WireStorageEncoding::CanonicalDiscreteFieldJsonBytes,
                total_bytes: u64::try_from(bytes.len())
                    .map_err(|_| invalid_artifact("discrete Field byte count exceeds u64"))?,
                chunks: references,
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok((value, chunks))
    }

    /// Decode the storage envelope without loading any raw chunks.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid discrete Field storage JSON: {error}"))
        })?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical storage-envelope bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize discrete Field storage: {error}"))
        })
    }

    /// Domain-separated storage realization identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            STORAGE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact logical discrete Field, independent of chunk layout.
    #[must_use]
    pub fn logical_field(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.logical_field_sha256.clone())
    }

    /// Reconstruct and validate the exact logical discrete Field from ordered chunks.
    ///
    /// # Errors
    /// Returns `EQ0901` for a missing, reordered, substituted, truncated, or
    /// oversized chunk, noncanonical reconstructed bytes, or logical digest drift.
    pub fn restore(
        &self,
        chunks: &[StorageChunkV1],
        limits: FieldDecoderLimits,
    ) -> Result<DiscreteFieldEnvelopeV1, Diagnostic> {
        if chunks.len() != self.wire.chunks.len() {
            return Err(invalid_artifact(
                "discrete Field storage realization is missing one or more chunks",
            ));
        }
        let byte_count = usize::try_from(self.wire.total_bytes)
            .map_err(|_| invalid_artifact("discrete Field byte count exceeds local usize"))?;
        if byte_count > limits.json.max_bytes {
            return Err(invalid_artifact(
                "reconstructed discrete Field exceeds the decoder byte limit",
            ));
        }
        let mut bytes = Vec::with_capacity(byte_count);
        for (expected, chunk) in self.wire.chunks.iter().zip(chunks) {
            let actual_offset = u64::try_from(bytes.len())
                .map_err(|_| invalid_artifact("snapshot storage offset exceeds u64"))?;
            let actual_count = u64::try_from(chunk.bytes.len())
                .map_err(|_| invalid_artifact("snapshot storage chunk size exceeds u64"))?;
            let actual_digest = StorageChunkSha256V1::from_bytes(&chunk.bytes);
            if expected.offset != actual_offset
                || expected.byte_count != actual_count
                || expected.chunk_sha256 != actual_digest.to_string()
                || chunk.digest != actual_digest
            {
                return Err(invalid_artifact(
                    "discrete Field storage chunk order, extent, or raw digest differs",
                ));
            }
            bytes.extend_from_slice(&chunk.bytes);
        }
        if bytes.len() != byte_count {
            return Err(invalid_artifact(
                "discrete Field storage chunks do not cover the exact logical byte count",
            ));
        }
        let field = DiscreteFieldEnvelopeV1::from_json(&bytes, limits)?;
        if field.canonical_json()? != bytes || field.digest()? != self.logical_field() {
            return Err(invalid_artifact(
                "restored discrete Field bytes are noncanonical or have logical digest drift",
            ));
        }
        Ok(field)
    }

    fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != STORAGE_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported Field-snapshot-storage schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.logical_field_sha256.clone())?;
        if self.wire.storage_encoding != WireStorageEncoding::CanonicalDiscreteFieldJsonBytes
            || self.wire.total_bytes == 0
            || self.wire.chunks.is_empty()
            || self.wire.chunks.len() > limits.max_field_storage_chunks
        {
            return Err(invalid_artifact(
                "discrete Field storage extent/chunk count is empty or exceeds a decoder limit",
            ));
        }
        let mut offset = 0_u64;
        for (ordinal, chunk) in self.wire.chunks.iter().enumerate() {
            StorageChunkSha256V1::from_hex(chunk.chunk_sha256.clone())?;
            if chunk.ordinal != u64::try_from(ordinal).unwrap_or(u64::MAX)
                || chunk.offset != offset
                || chunk.byte_count == 0
            {
                return Err(invalid_artifact(
                    "discrete Field storage chunks must be nonempty, contiguous, and canonically ordered",
                ));
            }
            offset = offset
                .checked_add(chunk.byte_count)
                .ok_or_else(|| invalid_artifact("discrete Field storage extent overflows u64"))?;
        }
        if offset != self.wire.total_bytes {
            return Err(invalid_artifact(
                "discrete Field storage chunks do not exactly cover logical bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDiscreteFieldStorageV1 {
    schema: String,
    encoding: String,
    logical_field_sha256: String,
    storage_encoding: WireStorageEncoding,
    total_bytes: u64,
    chunks: Vec<WireStorageChunkReference>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireStorageEncoding {
    CanonicalDiscreteFieldJsonBytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStorageChunkReference {
    ordinal: u64,
    offset: u64,
    byte_count: u64,
    chunk_sha256: String,
}
