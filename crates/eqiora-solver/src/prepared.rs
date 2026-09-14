use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use sha2::{Digest, Sha256};

use crate::{LinearProblem, LinearSolution};

const IDENTITY_DOMAIN: &[u8] = b"eqiora.prepared-linear-structure/v1\0";

/// Exact assembly-owned identity for one run-local linear structure.
///
/// The canonical encoding covers the complete structural authority, including
/// ordering, constraints, packet maps, fixed structural values and sparse
/// topology. Its digest is a compact observation only: equality compares the
/// complete encoding so a provider never authorizes reuse from a hash alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedLinearStructureIdentity {
    encoding: Arc<[u8]>,
    digest: [u8; 32],
}

impl PreparedLinearStructureIdentity {
    /// Capture one nonempty canonical structural encoding.
    ///
    /// # Errors
    /// Returns `EQ0807` when the supplied identity is empty or too large for
    /// its portable length prefix.
    pub fn new(encoding: impl Into<Arc<[u8]>>) -> Result<Self, Diagnostic> {
        let encoding = encoding.into();
        if encoding.is_empty() {
            return Err(invalid(
                "prepared linear structure identity must be nonempty",
            ));
        }
        let length = u64::try_from(encoding.len())
            .map_err(|_| invalid("prepared linear structure identity exceeds portable u64"))?;
        let mut hash = Sha256::new();
        hash.update(IDENTITY_DOMAIN);
        hash.update(length.to_be_bytes());
        hash.update(&encoding);
        Ok(Self {
            encoding,
            digest: hash.finalize().into(),
        })
    }

    /// Compact observation of this exact identity.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Complete canonical authority used by exact equality.
    #[must_use]
    pub fn encoding(&self) -> &[u8] {
        &self.encoding
    }
}

/// Provider-private state for repeated solves inside one prepared run.
pub trait PreparedLinearSolver: std::fmt::Debug {
    /// Solve one matrix candidate under its exact assembly structure identity.
    ///
    /// A provider must compare both the complete identity and actual canonical
    /// CSR topology before reusing symbolic state.
    ///
    /// # Errors
    /// Returns a structured identity, capability, or numerical diagnostic.
    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_>,
    ) -> Result<LinearSolution, Diagnostic>;
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_uses_the_complete_canonical_encoding() {
        let first = PreparedLinearStructureIdentity::new(&b"ordering-a"[..]).unwrap();
        let equal = PreparedLinearStructureIdentity::new(&b"ordering-a"[..]).unwrap();
        let changed = PreparedLinearStructureIdentity::new(&b"ordering-b"[..]).unwrap();
        assert_eq!(first, equal);
        assert_ne!(first, changed);
        assert_eq!(first.encoding(), b"ordering-a");
        assert_eq!(first.digest(), equal.digest());
    }

    #[test]
    fn empty_identity_is_rejected() {
        assert!(PreparedLinearStructureIdentity::new(&b""[..]).is_err());
    }
}
