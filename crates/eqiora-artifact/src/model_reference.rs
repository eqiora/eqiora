//! Typed identity and replay of the single current Model artifact.

use eqiora_core::{Diagnostic, OntologyId};
use eqiora_graph::Transaction;
use eqiora_realization::SemanticRevision;
use eqiora_schema::Model;
use eqiora_sem::KernelProgram;

use crate::{ArtifactDigest, ModelDecoderLimits, ModelEnvelope, invalid_artifact};

mod sealed {
    pub trait Sealed {}
}

/// One canonical current Model artifact that can yield a closed identity
/// reference.
///
/// The trait is sealed: artifact adapters own the digest domain, Model
/// identity, and revision extraction. Callers cannot manufacture a reference
/// by implementing a permissive metadata interface.
pub trait CanonicalModelArtifact: sealed::Sealed {
    /// Construct the exact typed reference for this artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` only when validated artifact state cannot be decoded.
    fn artifact_reference(&self) -> Result<ModelArtifactReference, Diagnostic>;
}

/// One canonical Model artifact replayed through the current decoder and the
/// ordinary whole-model validator.
///
/// The exact artifact reference and immutable program are produced together;
/// a consumer cannot accidentally combine metadata from one artifact with
/// semantic content replayed from another.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayedCanonicalModel {
    reference: ModelArtifactReference,
    program: KernelProgram,
}

impl ReplayedCanonicalModel {
    /// Exact wire-domain identity of the replayed artifact.
    #[must_use]
    pub const fn artifact_reference(&self) -> &ModelArtifactReference {
        &self.reference
    }

    /// Immutable, completely validated Semantic Kernel projection.
    #[must_use]
    pub const fn program(&self) -> &KernelProgram {
        &self.program
    }
}

/// A canonical Model artifact that can replay its exact bytes into a validated
/// immutable Semantic Kernel program.
///
/// The [`CanonicalModelArtifact`] supertrait seals this extension. An
/// identity-only reference deliberately does not implement replay: having a
/// digest, Model ID, and revision does not imply possession of canonical
/// content. The current Model owner implements this boundary beside its decoder
/// before semantic consumers can admit it.
pub trait ReplayableCanonicalModelArtifact: CanonicalModelArtifact {
    /// Replay the canonical bytes and return their exact identity and validated
    /// semantic content as one indivisible value.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical reconstruction unexpectedly fails or if
    /// the replayed Model identity/revision contradicts the artifact reference.
    fn replay_model(&self) -> Result<ReplayedCanonicalModel, Diagnostic>;
}

/// Closed identity of one canonical Model artifact.
///
/// The content digest remains domain-separated by the current Model schema.
/// This type gives downstream artifacts one stable way to retain the exact
/// digest, ontology identity, and semantic revision without owning the
/// envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactReference {
    artifact: ArtifactDigest,
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
}

impl ModelArtifactReference {
    fn new(artifact: ArtifactDigest, model: OntologyId<Model>, source_revision: u64) -> Self {
        Self {
            artifact,
            model,
            semantic_revision: SemanticRevision::new(source_revision),
        }
    }

    /// Content digest in the current Model artifact domain.
    #[must_use]
    pub const fn artifact(&self) -> &ArtifactDigest {
        &self.artifact
    }

    /// Typed Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Semantic graph revision serialized by the Model artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Prove that another Model artifact is this exact artifact, including its
    /// content digest.
    ///
    /// # Errors
    /// Returns `EQ0901` for digest, Model identity, or revision drift.
    pub fn validate_artifact(
        &self,
        artifact: &impl CanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        let candidate = artifact.artifact_reference()?;
        if self != &candidate {
            return Err(invalid_artifact(
                "Model artifact digest, ontology identity, or semantic revision differs from the typed reference",
            ));
        }
        Ok(())
    }
}

impl sealed::Sealed for ModelArtifactReference {}

impl CanonicalModelArtifact for ModelArtifactReference {
    fn artifact_reference(&self) -> Result<ModelArtifactReference, Diagnostic> {
        Ok(self.clone())
    }
}

/// One admitted artifact under the single current Model contract.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedModelArtifact {
    envelope: ModelEnvelope,
}

impl AcceptedModelArtifact {
    /// Encode one validated Kernel Program through the current Model contract.
    ///
    /// # Errors
    /// Returns `EQ0901` when the program cannot be represented by the bounded
    /// current artifact.
    pub fn from_program(program: &KernelProgram) -> Result<Self, Diagnostic> {
        ModelEnvelope::from_program(program).map(|envelope| Self { envelope })
    }

    /// Decode bytes through the current Model contract.
    ///
    /// No schema sniffing, retry, migration, or compatibility fallback occurs.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, noncanonical, unsupported,
    /// or wrong-schema data.
    pub fn from_json(bytes: &[u8], limits: ModelDecoderLimits) -> Result<Self, Diagnostic> {
        ModelEnvelope::from_json(bytes, limits).map(|envelope| Self { envelope })
    }

    /// Deterministic compact canonical JSON.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        self.envelope.canonical_json()
    }

    /// Domain-separated current Model identity.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        self.envelope.digest()
    }

    /// Reconstruct the exact transaction and typed Model identity.
    pub fn to_transaction(&self) -> Result<(Transaction, OntologyId<Model>), Vec<Diagnostic>> {
        self.envelope.to_transaction()
    }

    /// Source graph revision retained by the current artifact.
    #[must_use]
    pub const fn source_revision(&self) -> u64 {
        self.envelope.source_revision()
    }
}

impl sealed::Sealed for ModelEnvelope {}

impl CanonicalModelArtifact for ModelEnvelope {
    fn artifact_reference(&self) -> Result<ModelArtifactReference, Diagnostic> {
        Ok(ModelArtifactReference::new(
            self.digest()?,
            self.model()?,
            self.source_revision(),
        ))
    }
}

impl ReplayableCanonicalModelArtifact for ModelEnvelope {
    fn replay_model(&self) -> Result<ReplayedCanonicalModel, Diagnostic> {
        let reference = self.artifact_reference()?;
        let program = self.to_program().map_err(|diagnostics| {
            invalid_artifact(format!(
                "cannot replay canonical Model artifact: {}",
                diagnostics
                    .iter()
                    .map(Diagnostic::message)
                    .collect::<Vec<_>>()
                    .join("; ")
            ))
        })?;
        if program.model() != reference.model()
            || SemanticRevision::new(program.revision().0) != reference.semantic_revision()
        {
            return Err(invalid_artifact(
                "replayed Model identity or semantic revision differs from its exact artifact reference",
            ));
        }
        Ok(ReplayedCanonicalModel { reference, program })
    }
}

impl sealed::Sealed for AcceptedModelArtifact {}

impl CanonicalModelArtifact for AcceptedModelArtifact {
    fn artifact_reference(&self) -> Result<ModelArtifactReference, Diagnostic> {
        self.envelope.artifact_reference()
    }
}

impl ReplayableCanonicalModelArtifact for AcceptedModelArtifact {
    fn replay_model(&self) -> Result<ReplayedCanonicalModel, Diagnostic> {
        self.envelope.replay_model()
    }
}
