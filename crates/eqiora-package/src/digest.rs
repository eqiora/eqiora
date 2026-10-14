use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

use crate::ContractError;

macro_rules! digest_type {
    ($name:ident, $description:expr, $error:expr) => {
        #[doc = $description]
        #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name([u8; 32]);

        impl $name {
            pub fn parse(value: &str) -> Result<Self, ContractError> {
                if value.len() != 64
                    || !value
                        .as_bytes()
                        .iter()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
                {
                    return Err(ContractError::new($error));
                }
                let mut bytes = [0_u8; 32];
                for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
                    bytes[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
                }
                Ok(Self(bytes))
            }

            #[must_use]
            pub fn to_hex(self) -> String {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let mut output = String::with_capacity(64);
                for byte in self.0 {
                    output.push(char::from(HEX[usize::from(byte >> 4)]));
                    output.push(char::from(HEX[usize::from(byte & 0x0f)]));
                }
                output
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.to_hex())
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.to_hex())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.to_hex())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::parse(&value).map_err(serde::de::Error::custom)
            }
        }
    };
    ($name:ident, $domain:literal) => {
        digest_type!(
            $name,
            concat!("A lowercase SHA-256 digest in the `", $domain, "` domain."),
            concat!(
                stringify!($name),
                " must be 64 lowercase hexadecimal characters"
            )
        );
        impl $name {
            pub(crate) fn compute(canonical_bytes: &[u8]) -> Self {
                let mut hasher = Sha256::new();
                hasher.update($domain.as_bytes());
                hasher.update([0]);
                hasher.update(canonical_bytes);
                Self(hasher.finalize().into())
            }
        }
    };
}

fn nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("validated hexadecimal digit"),
    }
}

digest_type!(PackageSemanticDigest, "eqiora.package-semantic.sha256.v1");
digest_type!(SourceBundleDigest, "eqiora.source-bundle.sha256.v1");
digest_type!(ResolutionDigest, "eqiora.package-resolution.sha256.v1");
digest_type!(
    PackageCompilationDigest,
    "eqiora.package-compilation.sha256.v1"
);
digest_type!(
    PackageRunBindingDigest,
    "eqiora.package-run-binding.sha256.v1"
);
digest_type!(
    PackageExecutionBindingDigest,
    "eqiora.package-execution-binding.sha256.v1"
);

digest_type!(
    CanonicalModelDigest,
    "SHA-256 emitted by the canonical model-artifact boundary. This wrapper validates and carries that identity without redefining its preimage.",
    "canonical model digest must be 64 lowercase hexadecimal characters"
);

digest_type!(
    CanonicalRunDigest,
    "SHA-256 emitted by a canonical run-manifest boundary. This wrapper validates and carries that identity without redefining its preimage.",
    "canonical run digest must be 64 lowercase hexadecimal characters"
);

digest_type!(
    CanonicalRealizationDigest,
    "SHA-256 emitted by a canonical Realization-artifact boundary. This wrapper validates and carries that identity without redefining its preimage.",
    "canonical Realization digest must be 64 lowercase hexadecimal characters"
);

#[cfg(test)]
mod tests;
