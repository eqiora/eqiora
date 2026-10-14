//! Version-neutral identity and execution projection of one Realization artifact.

use crate::{ArtifactDigest, LayoutArtifacts, RealizationEnvelope, invalid_artifact};
use eqiora_core::Diagnostic;
use eqiora_realization::{PlacementRequirementNode, SemanticRevision, VectorLayoutKind};
use eqiora_solver::ReductionPolicy;

mod sealed {
    pub trait Sealed {}
}

/// One explicitly versioned canonical Realization artifact that can yield a
/// closed identity and execution-policy reference.
///
/// The trait is sealed. Artifact adapters handle digest domains and guarantee
/// that the placement, layout, and reduction policy come from one fully validated
/// payload.
pub trait CanonicalRealizationArtifact: sealed::Sealed {
    /// Construct the version-neutral reference for this artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` only when validated artifact state cannot be decoded.
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic>;
}

/// Closed identity and run-relevant policy of one Realization artifact.
///
/// The content digest stays domain-separated by the selected Realization schema.
/// Run manifests use this reference to validate the current graph envelope while
/// still checking the digest, Model, revision, layout, placement, and reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationArtifactReference {
    artifact: ArtifactDigest,
    model_artifact: ArtifactDigest,
    semantic_revision: SemanticRevision,
    placement: PlacementRequirementNode,
    vector_layout: VectorLayoutKind,
    layout_artifacts: LayoutArtifacts,
    reduction: ReductionPolicy,
}

impl RealizationArtifactReference {
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        artifact: ArtifactDigest,
        model_artifact: ArtifactDigest,
        semantic_revision: SemanticRevision,
        placement: PlacementRequirementNode,
        vector_layout: VectorLayoutKind,
        layout_artifacts: LayoutArtifacts,
        reduction: ReductionPolicy,
    ) -> Self {
        Self {
            artifact,
            model_artifact,
            semantic_revision,
            placement,
            vector_layout,
            layout_artifacts,
            reduction,
        }
    }

    /// Content digest in the selected Realization wire's domain.
    #[must_use]
    pub const fn artifact(&self) -> &ArtifactDigest {
        &self.artifact
    }

    /// Canonical Model artifact selected by the Realization.
    #[must_use]
    pub const fn model_artifact(&self) -> &ArtifactDigest {
        &self.model_artifact
    }

    /// Semantic graph revision selected by the Realization.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Portable compute requirement, independent of the observed deployment.
    #[must_use]
    pub const fn placement(&self) -> PlacementRequirementNode {
        self.placement
    }

    /// Resolved algebraic vector layout.
    #[must_use]
    pub const fn vector_layout(&self) -> VectorLayoutKind {
        self.vector_layout
    }

    /// Content-addressed replicated or distributed layout inputs.
    #[must_use]
    pub fn layout_artifacts(&self) -> LayoutArtifacts {
        self.layout_artifacts.clone()
    }

    /// Reduction policy selected by the solver plan.
    #[must_use]
    pub const fn reduction(&self) -> ReductionPolicy {
        self.reduction
    }

    /// Check that another Realization artifact has this identity and execution
    /// policy.
    ///
    /// # Errors
    /// Returns `EQ0901` for identity or policy drift.
    pub fn validate_artifact(
        &self,
        artifact: &(impl CanonicalRealizationArtifact + ?Sized),
    ) -> Result<(), Diagnostic> {
        let candidate = artifact.artifact_reference()?;
        if self != &candidate {
            return Err(invalid_artifact(
                "Realization artifact identity or execution policy differs from the typed reference",
            ));
        }
        Ok(())
    }
}

impl sealed::Sealed for RealizationArtifactReference {}

impl CanonicalRealizationArtifact for RealizationArtifactReference {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        Ok(self.clone())
    }
}

impl sealed::Sealed for RealizationEnvelope {}

impl CanonicalRealizationArtifact for RealizationEnvelope {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        Ok(RealizationArtifactReference::new(
            self.digest()?,
            self.model_artifact(),
            self.semantic_revision(),
            self.placement(),
            self.vector_layout(),
            self.layout_artifacts(),
            self.solver_plan().reduction(),
        ))
    }
}
