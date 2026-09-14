//! Multi-Domain Field-wise Realization artifact with exact trace quotients.

use eqiora_core::{Diagnostic, OntologyId};
use eqiora_realization::{
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequirements, RealizationRevision,
    ResolvedCoupledFieldwiseRealization, SemanticRevision,
};
use eqiora_schema::Model;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalModelArtifact, LayoutArtifacts,
    RealizationDecoderLimits, SimplicialMeshEnvelopeV1, check_json_limits, invalid_artifact,
};

pub(crate) mod wire;

use self::wire::{WireCoupledPlan, WireCoupledRequirements};
use super::realization_v7::wire::WireLayoutArtifacts;

const REALIZATION_SCHEMA: &str = "eqiora.realization-envelope/v8";

/// Versioned serialization of one resolved multi-Domain Field-wise Realization.
///
/// Contains canonical exact Domain/Field and conforming trace-quotient inventories,
/// one content-addressed shared imported mesh, and one fixed
/// Backward Euler step, with exact rational physical dimensions.
/// Current coupled payloads require plural trace inventories, including when nested
/// in an ALE envelope; the displaced singular `trace_quotient` field is rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationEnvelopeV8 {
    wire: WireRealizationEnvelopeV8,
}

impl RealizationEnvelopeV8 {
    /// Encode a resolved coupled Realization and its exact Model/layout inputs.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model lineage drift, contradictory layout
    /// artifacts, or a value outside the closed portable contract.
    pub fn from_resolved(
        model: &impl CanonicalModelArtifact,
        resolved: &ResolvedCoupledFieldwiseRealization,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let model = model.artifact_reference()?;
        if model.model() != resolved.model()
            || model.semantic_revision() != resolved.semantic_revision()
        {
            return Err(invalid_artifact(
                "resolved coupled realization does not identify the supplied Model artifact and source revision",
            ));
        }
        require_layout_artifacts(
            resolved.requirements().execution().vector_layout(),
            &layout_artifacts,
        )?;
        let wire = WireRealizationEnvelopeV8 {
            schema: REALIZATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.artifact().to_string(),
            model_ulid: resolved.model().ulid().to_string(),
            semantic_revision: resolved.semantic_revision().get(),
            source: WireCoupledSource::Explicit {
                realization_revision: resolved.realization_revision().get(),
            },
            requirements: WireCoupledRequirements::encode(resolved.requirements())?,
            plan: WireCoupledPlan::encode(resolved.plan())?,
            layout_artifacts: WireLayoutArtifacts::encode(layout_artifacts),
        };
        let envelope = Self { wire };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Decode and locally validate a V8 realization envelope.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// noncanonical, or internally inconsistent data.
    pub fn from_json(bytes: &[u8], limits: RealizationDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireRealizationEnvelopeV8 = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid coupled realization envelope JSON: {error}"
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
                "cannot serialize coupled realization envelope: {error}"
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete V8 bytes.
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
                "Model artifact digest, ontology identity, or semantic revision differs from the coupled Realization",
            ));
        }
        Ok(())
    }

    /// Semantic graph revision realized by this artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        SemanticRevision::new(self.wire.semantic_revision)
    }

    /// Explicit Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        match self.wire.source {
            WireCoupledSource::Explicit {
                realization_revision,
            } => RealizationRevision::new(realization_revision),
        }
    }

    /// Exact lowerer requirements used during resolution.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn requirements(&self) -> Result<CoupledFieldwiseRealizationRequirements, Diagnostic> {
        self.wire.requirements.clone().decode()
    }

    /// Complete typed multi-Domain Field-wise plan.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn plan(&self) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
        self.wire.plan.clone().decode()
    }

    /// Referenced replicated or distributed layout artifacts.
    #[must_use]
    pub fn layout_artifacts(&self) -> LayoutArtifacts {
        self.wire.layout_artifacts.decode()
    }

    /// Sole referenced imported-mesh artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn mesh_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        match self.plan()?.spatial().discretization().mesh() {
            eqiora_realization::MeshPolicy::ImportedSimplicial { artifact } => {
                Ok(ArtifactDigest::from_sha256(artifact.sha256()))
            }
            eqiora_realization::MeshPolicy::GeneratedUniform { .. } => Err(invalid_artifact(
                "coupled realization must reference one imported mesh artifact",
            )),
            eqiora_realization::MeshPolicy::SuppliedCartesian { .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian1d { .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian3d { .. } => Err(invalid_artifact(
                "coupled realization does not admit a supplied Cartesian mesh",
            )),
        }
    }

    /// Validate the sole imported mesh against exact content and dimension.
    ///
    /// # Errors
    /// Returns `EQ0901` for digest or admitted-dimension drift.
    pub fn validate_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        if mesh.digest()? != self.mesh_artifact()? {
            return Err(invalid_artifact(
                "imported mesh digest differs from the coupled realization plan",
            ));
        }
        let dimension = self.requirements()?.execution().spatial_dimension().get();
        if mesh.dimension() != dimension {
            return Err(invalid_artifact(format!(
                "imported mesh dimension {} differs from admitted coupled realization dimension {dimension}",
                mesh.dimension(),
            )));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != REALIZATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported realization-envelope/v8 schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        let requirements = self.wire.requirements.clone().decode()?;
        let plan = self.wire.plan.clone().decode()?;
        if WireCoupledRequirements::encode(&requirements)? != self.wire.requirements
            || WireCoupledPlan::encode(&plan)? != self.wire.plan
        {
            return Err(invalid_artifact(
                "coupled realization arrays or values are not in canonical closed form",
            ));
        }
        let eliminated = plan
            .time_step()
            .eliminated_states()
            .iter()
            .map(|state| state.pair())
            .collect::<Vec<_>>();
        let represented = plan
            .represented_physical_fields()
            .map_err(|error| invalid_artifact(error.to_string()))?;
        let selected_domains = plan
            .spatial()
            .domains()
            .iter()
            .map(|domain| {
                eqiora_realization::DomainFieldInventory::new(
                    domain.domain(),
                    represented
                        .iter()
                        .filter(|field| field.domain() == domain.domain())
                        .map(|field| field.field()),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| invalid_artifact(error.to_string()))?;
        if requirements.domains() != selected_domains
            || requirements.trace_quotients() != plan.spatial().trace_quotients()
            || requirements.eliminated_states() != eliminated
        {
            return Err(invalid_artifact(
                "coupled plan does not bind the exact required Domain, Field, Connection, and trace-pair inventory",
            ));
        }
        let artifacts = self.wire.layout_artifacts.decode_validated()?;
        require_layout_artifacts(requirements.execution().vector_layout(), &artifacts)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRealizationEnvelopeV8 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    source: WireCoupledSource,
    requirements: WireCoupledRequirements,
    plan: WireCoupledPlan,
    layout_artifacts: WireLayoutArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireCoupledSource {
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
            "coupled realization layout artifacts contradict the admitted vector-layout requirement",
        ))
    }
}

fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid_artifact("model ULID is malformed"))
}
