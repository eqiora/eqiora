//! Stream the unchanged canonical CSR identity in bounded SHA-256 batches.
use super::{CanonicalCsrAgreementFingerprintV1, portable_count};
use crate::LinearOperatorProperties;
use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;
use sha2::{Digest, Sha256};

const AGREEMENT_DOMAIN_V1: &[u8] = b"eqiora.canonical-csr-agreement/v1\0";

#[allow(clippy::too_many_arguments)]
pub(super) fn agreement_fingerprint<S: Scalar + ComplexFloat<Real = f64>>(
    rows: usize,
    columns: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[S],
    right_hand_side: &[S],
    properties: LinearOperatorProperties,
) -> Result<CanonicalCsrAgreementFingerprintV1, Diagnostic> {
    let mut hash = AgreementHash::new(if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
        2
    } else {
        1
    }); // binary64 domain tag
    hash.count(rows, "row count")?;
    hash.count(columns, "column count")?;
    hash.count(row_offsets.len(), "row-offset count")?;
    for &offset in row_offsets {
        hash.count(offset, "row offset")?;
    }
    hash.count(column_indices.len(), "column-index count")?;
    for &column in column_indices {
        hash.count(column, "column index")?;
    }
    hash.count(values.len(), "value count")?;
    for value in values {
        hash.word(value.re().to_bits());
        if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
            hash.word(value.im().to_bits());
        }
    }
    hash.count(right_hand_side.len(), "right-hand-side count")?;
    for value in right_hand_side {
        hash.word(value.re().to_bits());
        if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
            hash.word(value.im().to_bits());
        }
    }
    Ok(hash.finish(agreement_property_tag_v1(properties)))
}

/// Frozen tags in the `eqiora.canonical-csr-agreement/v1` binary domain.
///
/// `SymmetricIndefinite` is an additive L2 agreement tag. This does not widen
/// any durable artifact-v1 schema, whose encoder rejects that property until
/// a separately versioned wire extension exists.
const fn agreement_property_tag_v1(properties: LinearOperatorProperties) -> u8 {
    match properties {
        LinearOperatorProperties::General => 0,
        LinearOperatorProperties::SymmetricPositiveDefinite => 1,
        LinearOperatorProperties::SymmetricIndefinite => 2,
        LinearOperatorProperties::Symmetric => 3,
        LinearOperatorProperties::ComplexSymmetric => 4,
        LinearOperatorProperties::Hermitian => 5,
        LinearOperatorProperties::HermitianPositiveDefinite => 6,
    }
}

// Counts and coefficient bits are all big-endian u64 words in the v1 stream.
// Buffer only a fixed stack-sized batch, never a complete encoded matrix.
struct AgreementHash {
    hash: Sha256,
    words: [u8; 1024],
    used: usize,
}

impl AgreementHash {
    fn new(scalar_tag: u8) -> Self {
        let mut hash = Sha256::new();
        hash.update(AGREEMENT_DOMAIN_V1);
        hash.update([scalar_tag]);
        Self {
            hash,
            words: [0; 1024],
            used: 0,
        }
    }

    fn word(&mut self, value: u64) {
        if self.used == self.words.len() {
            self.hash.update(self.words.as_slice());
            self.used = 0;
        }
        self.words[self.used..self.used + 8].copy_from_slice(&value.to_be_bytes());
        self.used += 8;
    }

    fn count(&mut self, value: usize, name: &'static str) -> Result<(), Diagnostic> {
        self.word(portable_count(value, name)?);
        Ok(())
    }

    fn finish(mut self, property_tag: u8) -> CanonicalCsrAgreementFingerprintV1 {
        self.hash.update(&self.words[..self.used]);
        self.hash.update([property_tag]);
        CanonicalCsrAgreementFingerprintV1(self.hash.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn real_and_complex_v1_streams_cross_batch_boundaries_without_changing_identity() {
        // Independently encoded with Python's struct.pack('>Q' / '>d') and
        // hashlib.sha256, before changing the implementation. The v1 stream is
        // the literal domain, scalar tag, counted CSR arrays, counted RHS and
        // General property byte. Diagonal entries alternate signed eighths;
        // complex entries add imag = -2 * real - 1/4. RHS real = (i - 3)/2,
        // imag = (i mod 7 - 3)/4. Sizes straddle one and several 1 KiB batches.
        let cases = [
            (
                20,
                "2e88b73453ce6a958efded128b7b9cd7e117216bb5090016551f2a2e7484da1e",
                "32190a1aaed81040e55602ef0ec11b8fa91b66368986b2d82ec219cbe4bad969",
            ),
            (
                21,
                "02f0aa9d51beb0e8689e50ba34709427d232bf282d6fba3175ac7814427ec167",
                "f35a5b7949706d63dae6c7cd8b4c11260e0930bbf3293db2f0fc15d90421fa50",
            ),
            (
                30,
                "4d2ee953a3cc944af80abe98590a9efa038a214105ff2bac3470457a78d37fbb",
                "e90b690a8f7fb9877ef81bc5031c229e336a3bde2c8588d45330af682cd392c6",
            ),
            (
                31,
                "842c250401cc8a6123be83b5a3789a9a23333c4c4c119b71e31fb5994703de2a",
                "0dcf076a717f794c1a1c7c57b9d94e8bc576895d4f0662921d68e8120044b08a",
            ),
            (
                127,
                "2e7047c78d09d419d7e47b28b049113a7d7277dc434fd41c8cd80cf4923c8adb",
                "a0d2e3c8d851deb51945ebace9d37f5dbba983fdae47b7c7a94199b8ebdde88a",
            ),
            (
                257,
                "6ebecc3ad9743afd921b3745aae9652e54a3c29aaab153ff8f0e58047e502c97",
                "43d36e68ef77c73cae0cc0d0e15bef37e8de101a0b4cba0e66160fb47b1089b7",
            ),
        ];
        let hex = |identity: CanonicalCsrAgreementFingerprintV1| {
            identity
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        for (size, expected_real, expected_complex) in cases {
            let offsets = (0..=size).collect::<Vec<_>>();
            let columns = (0..size).collect::<Vec<_>>();
            let values = (0..size)
                .map(|i| (i + 1) as f64 * 0.125 * if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect::<Vec<_>>();
            let rhs = (0..size)
                .map(|i| (i as f64 - 3.0) * 0.5)
                .collect::<Vec<_>>();
            let complex_values = values
                .iter()
                .map(|&value| Complex64::new(value, -2.0 * value - 0.25))
                .collect::<Vec<_>>();
            let complex_rhs = rhs
                .iter()
                .enumerate()
                .map(|(i, &value)| Complex64::new(value, ((i % 7) as f64 - 3.0) * 0.25))
                .collect::<Vec<_>>();
            let real = agreement_fingerprint(
                size,
                size,
                &offsets,
                &columns,
                &values,
                &rhs,
                LinearOperatorProperties::General,
            )
            .unwrap();
            let complex = agreement_fingerprint(
                size,
                size,
                &offsets,
                &columns,
                &complex_values,
                &complex_rhs,
                LinearOperatorProperties::General,
            )
            .unwrap();
            assert_eq!(hex(real), expected_real, "real size {size}");
            assert_eq!(hex(complex), expected_complex, "complex size {size}");
        }
    }
}
