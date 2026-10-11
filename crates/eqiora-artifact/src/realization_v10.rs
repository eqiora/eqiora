//! Dimension-explicit simplex ALE Realization artifact.

use eqiora_core::{Diagnostic, OntologyId};
use eqiora_realization::{
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequirements,
    QuadraturePolicy, RealizationRevision, ResolvedFixedTopologyAleCoupledRealization,
    SemanticRevision,
};
use eqiora_schema::Model;
use serde::{Deserialize, Serialize};

use crate::ale_wire::{WireAleEnvelope, WireAleSource, parse_ulid, validate_ale_mesh_quality_gate};
use crate::realization_v7::wire::{WireQuadratureCodec, decode_nonzero_usize, encode_usize};
use crate::{
    ArtifactDigest, CanonicalModelArtifact, LayoutArtifacts, RealizationDecoderLimits,
    SimplicialMeshEnvelopeV1, invalid_artifact,
};

const REALIZATION_SCHEMA: &str = "eqiora.realization-envelope/v10";

type WireRealizationEnvelopeV10 = WireAleEnvelope<WireQuadratureV10>;

/// Versioned fixed-topology ALE Realization with dimension-explicit simplex
/// quadrature.
///
/// The payload carries exact rational physical dimensions and explicit
/// simplex quadrature dimensions for the complete ALE graph.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationEnvelopeV10 {
    wire: WireRealizationEnvelopeV10,
}

impl RealizationEnvelopeV10 {
    /// Encode one completely resolved fixed-topology ALE Realization.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model lineage drift, contradictory layout
    /// artifacts, an invalid portable graph projection, quadrature-dimension
    /// drift, or a value outside the closed V10 contract.
    pub fn from_resolved(
        model: &impl CanonicalModelArtifact,
        resolved: &ResolvedFixedTopologyAleCoupledRealization,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let wire = WireRealizationEnvelopeV10::from_resolved(
            REALIZATION_SCHEMA,
            model,
            resolved,
            layout_artifacts,
        )?;
        let envelope = Self { wire };
        envelope.validate_policy()?;
        Ok(envelope)
    }

    /// Decode and locally validate a V10 ALE realization envelope.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version,
    /// noncanonical, resource-excess, graph-inconsistent, or
    /// dimension-inconsistent data.
    pub fn from_json(bytes: &[u8], limits: RealizationDecoderLimits) -> Result<Self, Diagnostic> {
        let wire = WireRealizationEnvelopeV10::from_json(REALIZATION_SCHEMA, bytes, limits)?;
        let envelope = Self { wire };
        envelope.validate_policy()?;
        Ok(envelope)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize dimension-explicit ALE realization envelope: {error}",
            ))
        })
    }

    /// Domain-separated SHA-256 identity of the complete V10 bytes.
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
                "Model artifact identity or semantic revision differs from the dimension-explicit ALE Realization",
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
            WireAleSource::Explicit {
                realization_revision,
            } => RealizationRevision::new(realization_revision),
        }
    }

    /// Exact lowerer requirements used during resolution.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn requirements(
        &self,
    ) -> Result<FixedTopologyAleCoupledRealizationRequirements, Diagnostic> {
        self.wire.requirements.clone().decode()
    }

    /// Complete typed fixed-topology ALE plan reconstructed from V10.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn plan(&self) -> Result<FixedTopologyAleCoupledRealizationPlan, Diagnostic> {
        let requirements = self.requirements()?;
        self.wire.plan.clone().decode(&requirements)
    }

    /// Referenced replicated or distributed layout artifacts.
    #[must_use]
    pub fn layout_artifacts(&self) -> LayoutArtifacts {
        self.wire.layout_artifacts.decode()
    }

    /// Sole referenced immutable simplex mesh artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` only if validated internal state was corrupted.
    pub fn mesh_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        match self.plan()?.coupled().spatial().discretization().mesh() {
            eqiora_realization::MeshPolicy::ImportedSimplicial { artifact } => {
                Ok(ArtifactDigest::from_sha256(artifact.sha256()))
            }
            eqiora_realization::MeshPolicy::GeneratedUniform { .. } => Err(invalid_artifact(
                "fixed-topology ALE realization must reference one imported simplex mesh",
            )),
            eqiora_realization::MeshPolicy::SuppliedCartesian { .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian1d { .. }
            | eqiora_realization::MeshPolicy::SuppliedCartesian3d { .. } => Err(invalid_artifact(
                "fixed-topology ALE realization does not admit a supplied Cartesian mesh",
            )),
        }
    }

    /// Validate the immutable reference mesh against exact content, dimension,
    /// and ALE quality policy.
    ///
    /// # Errors
    /// Returns `EQ0901` for digest, admitted-dimension, or quality-gate drift.
    pub fn validate_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        if mesh.digest()? != self.mesh_artifact()? {
            return Err(invalid_artifact(
                "reference mesh digest differs from the ALE realization plan",
            ));
        }
        let dimension = self
            .requirements()?
            .coupled()
            .execution()
            .spatial_dimension()
            .get();
        if mesh.dimension() != dimension {
            return Err(invalid_artifact(format!(
                "reference mesh dimension {} differs from admitted ALE dimension {dimension}",
                mesh.dimension(),
            )));
        }
        validate_ale_mesh_quality_gate(&self.plan()?, mesh)?;
        Ok(())
    }

    fn validate_policy(&self) -> Result<(), Diagnostic> {
        let required_dimension = self
            .requirements()?
            .coupled()
            .execution()
            .spatial_dimension();
        let policy = self
            .plan()?
            .coupled()
            .spatial()
            .discretization()
            .quadrature();
        let policy_dimension = match policy {
            QuadraturePolicy::TriangleDuffyGaussLegendre { .. } => 2,
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension, ..
            } => spatial_dimension.get(),
            _ => {
                return Err(invalid_artifact(
                    "dimension-explicit ALE realization requires simplex Duffy quadrature",
                ));
            }
        };
        if policy_dimension != required_dimension.get() {
            return Err(invalid_artifact(format!(
                "quadrature policy dimension {policy_dimension} differs from required spatial dimension {required_dimension}",
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireQuadratureV10 {
    GaussLegendre {
        points_per_axis: u64,
    },
    CellCentroid,
    SimplexCentroid,
    TriangleDuffyGaussLegendre {
        spatial_dimension: u64,
        points_per_axis: u64,
    },
    SimplexDuffyGaussLegendre {
        spatial_dimension: u64,
        points_per_axis: u64,
    },
}

impl WireQuadratureCodec for WireQuadratureV10 {
    fn encode(value: QuadraturePolicy) -> Result<Self, Diagnostic> {
        Ok(match value {
            QuadraturePolicy::GaussLegendre { points_per_axis } => Self::GaussLegendre {
                points_per_axis: encode_usize(points_per_axis.get(), "quadrature points per axis")?,
            },
            QuadraturePolicy::CellCentroid => Self::CellCentroid,
            QuadraturePolicy::SimplexCentroid => Self::SimplexCentroid,
            QuadraturePolicy::TriangleDuffyGaussLegendre { points_per_axis } => {
                Self::TriangleDuffyGaussLegendre {
                    spatial_dimension: 2,
                    points_per_axis: encode_usize(
                        points_per_axis.get(),
                        "Duffy quadrature points per axis",
                    )?,
                }
            }
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Self::SimplexDuffyGaussLegendre {
                spatial_dimension: encode_usize(
                    spatial_dimension.get(),
                    "simplex quadrature spatial dimension",
                )?,
                points_per_axis: encode_usize(
                    points_per_axis.get(),
                    "Duffy quadrature points per axis",
                )?,
            },
        })
    }

    fn decode(self) -> Result<QuadraturePolicy, Diagnostic> {
        match self {
            Self::GaussLegendre { points_per_axis } => Ok(QuadraturePolicy::GaussLegendre {
                points_per_axis: decode_nonzero_usize(
                    points_per_axis,
                    "quadrature points per axis",
                )?,
            }),
            Self::CellCentroid => Ok(QuadraturePolicy::CellCentroid),
            Self::SimplexCentroid => Ok(QuadraturePolicy::SimplexCentroid),
            Self::TriangleDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => {
                if spatial_dimension != 2 {
                    return Err(invalid_artifact(
                        "legacy triangle Duffy quadrature must explicitly declare spatial dimension two",
                    ));
                }
                Ok(QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: decode_nonzero_usize(
                        points_per_axis,
                        "Duffy quadrature points per axis",
                    )?,
                })
            }
            Self::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Ok(QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: decode_nonzero_usize(
                    spatial_dimension,
                    "simplex quadrature spatial dimension",
                )?,
                points_per_axis: decode_nonzero_usize(
                    points_per_axis,
                    "Duffy quadrature points per axis",
                )?,
            }),
        }
    }
}
