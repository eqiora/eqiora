//! Stable sample/channel identity shared by parameter and time-path realizations.
use eqiora_core::{Diagnostic, diagnostic::codes};
use sha2::{Digest, Sha256};

/// Full client identity, retained independently of numerical Parameter values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "WireSamplingIdentity", into = "WireSamplingIdentity")]
pub struct SamplingIdentity {
    sample: Vec<u8>,
    channel: Vec<u8>,
    lineage: Vec<u64>,
    coupling: Option<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSamplingIdentity {
    sample: Vec<u8>,
    channel: Vec<u8>,
    lineage: Vec<u64>,
    coupling: Option<Vec<u8>>,
}

impl From<SamplingIdentity> for WireSamplingIdentity {
    fn from(value: SamplingIdentity) -> Self {
        Self {
            sample: value.sample,
            channel: value.channel,
            lineage: value.lineage,
            coupling: value.coupling,
        }
    }
}

impl TryFrom<WireSamplingIdentity> for SamplingIdentity {
    type Error = String;
    fn try_from(value: WireSamplingIdentity) -> Result<Self, Self::Error> {
        Self::new(
            &value.sample,
            &value.channel,
            &value.lineage,
            value.coupling.as_deref(),
        )
        .map_err(|_| "invalid persisted sampling identity".to_owned())
    }
}

impl SamplingIdentity {
    /// Admit nonempty IDs of at most 256 bytes and at most 32 lineage entries.
    /// Lineage records request association only and never changes random draws.
    /// `coupling` explicitly selects a shared stream group; None derives the stream
    /// from the sample ID. Neither request proves statistical independence.
    ///
    /// # Errors
    /// Rejects empty/oversized IDs or oversized lineage before copying them.
    pub fn new(
        sample: &[u8],
        channel: &[u8],
        lineage: &[u64],
        coupling: Option<&[u8]>,
    ) -> Result<Self, Diagnostic> {
        for id in [sample, channel] {
            validate_id(id)?;
        }
        if let Some(id) = coupling {
            validate_id(id)?;
        }
        if lineage.len() > 32 {
            return Err(invalid("sampling occurrence lineage exceeds 32 entries"));
        }
        Ok(Self {
            sample: sample.to_vec(),
            channel: channel.to_vec(),
            lineage: lineage.to_vec(),
            coupling: coupling.map(<[u8]>::to_vec),
        })
    }

    /// Derive one versioned stream from the full sample/channel or explicit coupling.
    ///
    /// The generator name is an algorithm-owned domain separator. Request lineage
    /// records association only and does not change this stream. A stream digest
    /// establishes deterministic addressing, not statistical independence.
    #[must_use]
    pub fn stream_digest(&self, master: &[u8; 32], generator: &str) -> [u8; 32] {
        stream_digest(master, self, generator)
    }

    /// Complete client sample ID, including under explicit coupling.
    #[must_use]
    pub fn sample_id(&self) -> &[u8] {
        &self.sample
    }

    /// Complete stream/noise-channel role.
    #[must_use]
    pub fn channel(&self) -> &[u8] {
        &self.channel
    }

    /// Ordered occurrence association metadata, excluded from stream derivation.
    #[must_use]
    pub fn lineage(&self) -> &[u64] {
        &self.lineage
    }

    /// Explicit common-random-number group, or None for an independent-replicate request.
    #[must_use]
    pub fn coupling_id(&self) -> Option<&[u8]> {
        self.coupling.as_deref()
    }
}

fn validate_id(id: &[u8]) -> Result<(), Diagnostic> {
    if id.is_empty() || id.len() > 256 {
        return Err(invalid("sampling IDs require 1..=256 bytes"));
    }
    Ok(())
}

fn stream_digest(master: &[u8; 32], identity: &SamplingIdentity, generator: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(generator.as_bytes());
    hash.update([0]);
    hash.update(master);
    append_id(&mut hash, &identity.channel);
    match identity.coupling_id() {
        None => {
            hash.update([0]);
            append_id(&mut hash, &identity.sample);
        }
        Some(id) => {
            hash.update([1]);
            append_id(&mut hash, id);
        }
    }
    hash.finalize().into()
}

fn append_id(hash: &mut Sha256, id: &[u8]) {
    hash.update((id.len() as u16).to_be_bytes());
    hash.update(id);
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXECUTION_CONFIG, message)
}

#[cfg(test)]
mod tests;
