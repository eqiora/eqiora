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
#[derive(Debug, Clone, Eq)]
pub struct PreparedLinearStructureIdentity {
    encoding: Arc<[u8]>,
    digest: [u8; 32],
}

impl PartialEq for PreparedLinearStructureIdentity {
    fn eq(&self, other: &Self) -> bool {
        // Clones share immutable authority. Make this constant-time path explicit
        // rather than relying on the toolchain's Arc<[u8]> equality optimization.
        // Independently owned encodings still require complete byte equality.
        (Arc::ptr_eq(&self.encoding, &other.encoding) || self.encoding == other.encoding)
            && self.digest == other.digest
    }
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
pub trait PreparedLinearSolver<S = f64>: std::fmt::Debug {
    /// Begin one caller-owned action, retaining the last committed resources.
    ///
    /// # Errors
    /// Rejects a nested action, including an action awaiting discard after failure.
    fn begin_action(&mut self) -> Result<(), Diagnostic>;

    /// Commit tentative resources after caller-owned numerical and lineage acceptance.
    /// An action with no solves may also commit.
    ///
    /// # Errors
    /// Rejects idle or failed actions without changing committed resources.
    fn commit_action(&mut self) -> Result<(), Diagnostic>;

    /// Drop tentative resources and retain the last committed resources.
    ///
    /// # Errors
    /// Rejects calls without an active or failed action.
    fn discard_action(&mut self) -> Result<(), Diagnostic>;

    /// Solve one matrix candidate under its exact assembly structure identity.
    /// Successful solves update tentative resources only. During an active action,
    /// a solve error poisons the action until discard; committed resources remain
    /// unchanged. Calls outside an active action reject without mutation.
    ///
    /// A provider must compare both the complete identity and actual canonical
    /// CSR topology before reusing symbolic state.
    ///
    /// # Errors
    /// Returns a structured identity, capability, or numerical diagnostic.
    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_, S>,
    ) -> Result<LinearSolution<S>, Diagnostic>;
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
        assert_eq!(first, first.clone());
        assert_eq!(first, equal);
        assert_ne!(first, changed);
        assert_eq!(first.encoding(), b"ordering-a");
        assert_eq!(first.digest(), equal.digest());

        let mut same_digest = changed;
        same_digest.digest = first.digest;
        assert_ne!(first, same_digest);

        let mut changed_digest = first.clone();
        changed_digest.digest[0] ^= 1;
        assert_ne!(first, changed_digest);
    }

    #[test]
    fn empty_identity_is_rejected() {
        assert!(PreparedLinearStructureIdentity::new(&b""[..]).is_err());
    }
}
