//! Version-neutral identity and execution projection of one Realization artifact.

use crate::{
    ArtifactDigest, CanonicalModelArtifact, LayoutArtifacts, RealizationEnvelopeV1,
    RealizationEnvelopeV7, RealizationEnvelopeV9, RealizationEnvelopeV10, SimplicialMeshEnvelopeV1,
    invalid_artifact,
};
use eqiora_core::Diagnostic;
use eqiora_realization::{
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequirements,
    SemanticRevision, Target, VectorLayoutKind,
};
use eqiora_solver::ReductionPolicy;

mod sealed {
    pub trait Sealed {}
}

/// One explicitly versioned canonical Realization artifact that can yield a
/// closed identity and execution-policy reference.
///
/// The trait is sealed. Artifact adapters remain responsible for digest
/// domains and for proving that the projected target, layout, and reduction
/// policy came from one completely validated wire payload.
pub trait CanonicalRealizationArtifact: sealed::Sealed {
    /// Construct the exact version-neutral reference for this artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` only when validated artifact state cannot be decoded.
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic>;
}

/// One canonical fixed-topology ALE Realization that can be replayed against
/// its exact Model and immutable reference mesh.
///
/// Moving spatial artifacts replay the current ALE wire through this sealed
/// boundary, including its Model, mesh, and complete plan validation.
pub trait ReplayableFixedTopologyAleRealizationArtifact: CanonicalRealizationArtifact {
    /// Replay the exact fixed-topology ALE lowerer requirements.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    fn ale_requirements(
        &self,
    ) -> Result<FixedTopologyAleCoupledRealizationRequirements, Diagnostic>;

    /// Replay the complete typed fixed-topology ALE plan.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    fn ale_plan(&self) -> Result<FixedTopologyAleCoupledRealizationPlan, Diagnostic>;

    /// Validate the exact canonical Model selected by this ALE artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model identity or revision drift.
    fn validate_ale_model_artifact(
        &self,
        model: &impl CanonicalModelArtifact,
    ) -> Result<(), Diagnostic>;

    /// Validate the exact immutable reference mesh selected by this artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` for mesh identity or admitted-dimension drift.
    fn validate_ale_mesh_artifact(&self, mesh: &SimplicialMeshEnvelopeV1)
    -> Result<(), Diagnostic>;
}

/// Closed identity and run-relevant policy of one Realization artifact.
///
/// The content digest remains domain-separated by the selected Realization
/// schema. This projection permits Run manifests to validate either wire
/// generation without weakening the exact digest, Model, revision, layout,
/// placement, or reduction linkage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationArtifactReference {
    artifact: ArtifactDigest,
    model_artifact: ArtifactDigest,
    semantic_revision: SemanticRevision,
    target: Target,
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
        target: Target,
        vector_layout: VectorLayoutKind,
        layout_artifacts: LayoutArtifacts,
        reduction: ReductionPolicy,
    ) -> Self {
        Self {
            artifact,
            model_artifact,
            semantic_revision,
            target,
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

    /// Exact canonical Model artifact selected by the Realization.
    #[must_use]
    pub const fn model_artifact(&self) -> &ArtifactDigest {
        &self.model_artifact
    }

    /// Semantic graph revision selected by the Realization.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Resolved deployment target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
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

    /// Reduction policy selected by the exact solver plan.
    #[must_use]
    pub const fn reduction(&self) -> ReductionPolicy {
        self.reduction
    }

    /// Prove that another explicitly selected Realization artifact is this
    /// exact artifact and execution policy.
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

impl sealed::Sealed for RealizationEnvelopeV1 {}

impl CanonicalRealizationArtifact for RealizationEnvelopeV1 {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        let plan = self.plan()?;
        Ok(RealizationArtifactReference::new(
            self.digest()?,
            self.model_artifact(),
            self.semantic_revision(),
            plan.target(),
            self.requirements()?.vector_layout(),
            self.layout_artifacts(),
            plan.solver().reduction(),
        ))
    }
}

impl sealed::Sealed for RealizationEnvelopeV7 {}

impl CanonicalRealizationArtifact for RealizationEnvelopeV7 {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        let plan = self.plan()?;
        Ok(RealizationArtifactReference::new(
            self.digest()?,
            self.model_artifact(),
            self.semantic_revision(),
            plan.target(),
            self.requirements()?.execution().vector_layout(),
            self.layout_artifacts(),
            plan.solver().reduction(),
        ))
    }
}

impl sealed::Sealed for RealizationEnvelopeV9 {}

impl CanonicalRealizationArtifact for RealizationEnvelopeV9 {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        let plan = self.plan()?;
        Ok(RealizationArtifactReference::new(
            self.digest()?,
            self.model_artifact(),
            self.semantic_revision(),
            plan.target(),
            self.requirements()?.execution().vector_layout(),
            self.layout_artifacts(),
            plan.solver().reduction(),
        ))
    }
}

impl sealed::Sealed for RealizationEnvelopeV10 {}

impl CanonicalRealizationArtifact for RealizationEnvelopeV10 {
    fn artifact_reference(&self) -> Result<RealizationArtifactReference, Diagnostic> {
        let plan = self.plan()?;
        Ok(RealizationArtifactReference::new(
            self.digest()?,
            self.model_artifact(),
            self.semantic_revision(),
            plan.coupled().target(),
            self.requirements()?.coupled().execution().vector_layout(),
            self.layout_artifacts(),
            plan.coupled().solver().reduction(),
        ))
    }
}

impl ReplayableFixedTopologyAleRealizationArtifact for RealizationEnvelopeV10 {
    fn ale_requirements(
        &self,
    ) -> Result<FixedTopologyAleCoupledRealizationRequirements, Diagnostic> {
        self.requirements()
    }

    fn ale_plan(&self) -> Result<FixedTopologyAleCoupledRealizationPlan, Diagnostic> {
        self.plan()
    }

    fn validate_ale_model_artifact(
        &self,
        model: &impl CanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        self.validate_model_artifact(model)
    }

    fn validate_ale_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        self.validate_mesh_artifact(mesh)
    }
}
