//! Field-wise Realization artifact with exact mixed-algebra identity.

use eqiora_core::{Diagnostic, OntologyId};
use eqiora_realization::{
    FieldwiseRealizationPlan, FieldwiseRealizationRequirements, RealizationRevision,
    ResolvedFieldwiseRealization, SemanticRevision,
};
use eqiora_schema::Model;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalModelArtifact, LayoutArtifacts,
    RealizationDecoderLimits, SimplicialMeshEnvelopeV1, check_json_limits, invalid_artifact,
};

pub(crate) mod wire;

use wire::{WireFieldwisePlan, WireFieldwiseRequirements, WireLayoutArtifacts};

const REALIZATION_SCHEMA: &str = "eqiora.realization-envelope/v7";

/// Versioned serialization of one resolved field-wise Realization.
///
/// V7 retains exact Semantic Domain and Field identities, mixed spaces,
/// algebraic constraints, operator properties, and dimensional congruence
/// scales with exact rational physical dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationEnvelopeV7 {
    wire: WireRealizationEnvelopeV7,
}

impl RealizationEnvelopeV7 {
    /// Encode a resolved field-wise Realization and its exact Model/layout inputs.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model lineage drift, contradictory layout
    /// artifacts, or a value outside the portable wire contract.
    pub fn from_resolved(
        model: &impl CanonicalModelArtifact,
        resolved: &ResolvedFieldwiseRealization,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let model = model.artifact_reference()?;
        if model.model() != resolved.model()
            || model.semantic_revision() != resolved.semantic_revision()
        {
            return Err(invalid_artifact(
                "resolved field-wise realization does not identify the supplied model artifact and source revision",
            ));
        }
        require_layout_artifacts(
            resolved.requirements().execution().vector_layout(),
            &layout_artifacts,
        )?;
        let wire = WireRealizationEnvelopeV7 {
            schema: REALIZATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.artifact().to_string(),
            model_ulid: resolved.model().ulid().to_string(),
            semantic_revision: resolved.semantic_revision().get(),
            source: WireFieldwiseSource::Explicit {
                realization_revision: resolved.realization_revision().get(),
            },
            requirements: WireFieldwiseRequirements::encode(resolved.requirements())?,
            plan: WireFieldwisePlan::encode(resolved.plan())?,
            layout_artifacts: WireLayoutArtifacts::encode(layout_artifacts),
        };
        let envelope = Self { wire };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Decode and locally validate a field-wise realization envelope.
    ///
    /// This validates canonical wire form and the complete internal typed
    /// contract. It does not prove that referenced Domain/Field identities
    /// belong to an externally loaded Model or rerun backend capability
    /// resolution; the canonical finalized adapter must perform those checks
    /// before numerical lowering.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// noncanonical, or internally inconsistent data.
    pub fn from_json(bytes: &[u8], limits: RealizationDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireRealizationEnvelopeV7 = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid field-wise realization envelope JSON: {error}"
            ))
        })?;
        wire.requirements.validate_limits(limits)?;
        wire.plan.validate_limits(limits)?;
        let envelope = Self { wire };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize field-wise realization envelope: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete V7 bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            REALIZATION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical Semantic Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Typed Semantic Model identity.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn model(&self) -> Result<OntologyId<Model>, Diagnostic> {
        parse_ulid(&self.wire.model_ulid).map(OntologyId::from_ulid)
    }

    /// Validate the exact Model artifact selected by this Realization.
    ///
    /// # Errors
    /// Returns `EQ0901` for digest, Model identity, or revision drift.
    pub fn validate_model_artifact(
        &self,
        model: &impl CanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        let reference = model.artifact_reference()?;
        if self.model_artifact() != *reference.artifact()
            || self.model()? != reference.model()
            || self.semantic_revision() != reference.semantic_revision()
        {
            return Err(invalid_artifact(
                "Model artifact digest, ontology identity, or semantic revision differs from the field-wise Realization",
            ));
        }
        Ok(())
    }

    /// Semantic graph revision realized by this artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        SemanticRevision::new(self.wire.semantic_revision)
    }

    /// Explicit field-wise Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        match self.wire.source {
            WireFieldwiseSource::Explicit {
                realization_revision,
            } => RealizationRevision::new(realization_revision),
        }
    }

    /// Exact lowerer and execution requirements used during resolution.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn requirements(&self) -> Result<FieldwiseRealizationRequirements, Diagnostic> {
        self.wire.requirements.clone().decode()
    }

    /// Complete typed field-wise Realization plan.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn plan(&self) -> Result<FieldwiseRealizationPlan, Diagnostic> {
        self.wire.plan.clone().decode()
    }

    /// Referenced replicated or distributed layout artifacts.
    #[must_use]
    pub fn layout_artifacts(&self) -> LayoutArtifacts {
        self.wire.layout_artifacts.decode()
    }

    /// Referenced imported-mesh artifact, absent for generated meshes.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn mesh_artifact(&self) -> Result<Option<ArtifactDigest>, Diagnostic> {
        Ok(match self.plan()?.spatial().discretization().mesh() {
            eqiora_realization::MeshPolicy::GeneratedUniform { .. } => None,
            eqiora_realization::MeshPolicy::SuppliedCartesian { artifact, .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian1d { artifact, .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian3d { artifact, .. } => {
                Some(ArtifactDigest::from_sha256(artifact.sha256()))
            }
            eqiora_realization::MeshPolicy::ImportedSimplicial { artifact } => {
                Some(ArtifactDigest::from_sha256(artifact.sha256()))
            }
        })
    }

    /// Validate an imported mesh against exact content and admitted dimension.
    ///
    /// # Errors
    /// Returns `EQ0901` if this plan generates its mesh, the digest differs,
    /// or the mesh dimension contradicts the resolved requirements.
    pub fn validate_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let Some(expected) = self.mesh_artifact()? else {
            return Err(invalid_artifact(
                "generated-mesh field-wise realization does not consume an imported mesh artifact",
            ));
        };
        if mesh.digest()? != expected {
            return Err(invalid_artifact(
                "imported mesh artifact digest differs from the field-wise realization plan",
            ));
        }
        let required_dimension = self.requirements()?.execution().spatial_dimension().get();
        if mesh.dimension() != required_dimension {
            return Err(invalid_artifact(format!(
                "imported mesh dimension {} differs from admitted field-wise realization dimension {required_dimension}",
                mesh.dimension(),
            )));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != REALIZATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported realization-envelope/v7 schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        let requirements = self.wire.requirements.clone().decode()?;
        let plan = self.wire.plan.clone().decode()?;
        if WireFieldwiseRequirements::encode(&requirements)? != self.wire.requirements
            || WireFieldwisePlan::encode(&plan)? != self.wire.plan
        {
            return Err(invalid_artifact(
                "field-wise realization arrays or values are not in canonical closed form",
            ));
        }
        if requirements.domain() != plan.spatial().domain()
            || requirements.unknown_fields()
                != plan
                    .spatial()
                    .field_spaces()
                    .iter()
                    .map(|binding| binding.field())
                    .collect::<Vec<_>>()
        {
            return Err(invalid_artifact(
                "field-wise plan does not bind the exact required Domain and unknown-Field inventory",
            ));
        }
        let artifacts = self.wire.layout_artifacts.decode_validated()?;
        require_layout_artifacts(requirements.execution().vector_layout(), &artifacts)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRealizationEnvelopeV7 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    source: WireFieldwiseSource,
    requirements: WireFieldwiseRequirements,
    plan: WireFieldwisePlan,
    layout_artifacts: WireLayoutArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireFieldwiseSource {
    Explicit { realization_revision: u64 },
}

fn require_layout_artifacts(
    layout: eqiora_realization::VectorLayoutKind,
    artifacts: &LayoutArtifacts,
) -> Result<(), Diagnostic> {
    if matches!(
        (layout, artifacts),
        (
            eqiora_realization::VectorLayoutKind::Replicated,
            LayoutArtifacts::Replicated
        ) | (
            eqiora_realization::VectorLayoutKind::Distributed,
            LayoutArtifacts::Distributed { .. }
        )
    ) {
        Ok(())
    } else {
        Err(invalid_artifact(
            "field-wise realization layout artifacts contradict the admitted vector-layout requirement",
        ))
    }
}

fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid_artifact("model ULID is malformed"))
}
