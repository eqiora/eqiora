//! The current Realization artifact: exact Model identity and one portable graph.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, OntologyId};
use eqiora_realization::{
    GeometryActionNode, MeshPolicy, PlacementRequirementNode, PortableRealizationGraph,
    ResolutionSource, SemanticRevision, SystemBlock, VectorLayoutKind,
};
use eqiora_schema::Model;
use eqiora_solver::SolverPlan;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalModelArtifact, ReplayableCanonicalModelArtifact,
    SimplicialMeshEnvelopeV1, invalid_artifact,
};

const REALIZATION_SCHEMA: &str = "eqiora.realization-envelope/v11";

/// Semantic work budgets for the current Realization artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealizationDecoderLimits {
    /// Shared JSON size and nesting limits.
    pub json: crate::JsonLimits,
    /// Maximum Semantic Fields and Field-space bindings.
    pub max_realization_fields: usize,
    /// Maximum entries in an algebraic-constraint or transformation inventory.
    pub max_realization_constraints: usize,
    /// Maximum scaled algebraic blocks.
    pub max_realization_blocks: usize,
}

impl Default for RealizationDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonLimits::default(),
            max_realization_fields: 100_000,
            max_realization_constraints: 100_000,
            max_realization_blocks: 200_000,
        }
    }
}

/// Content-addressed layout inputs required by a realization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutArtifacts {
    /// A complete vector is resident in one process.
    Replicated,
    /// A distributed layout and its unique-owner partition artifact.
    Distributed {
        /// Digest of the ordered owned/ghost layout artifact.
        layout: ArtifactDigest,
        /// Digest of the global unique-owner partition artifact.
        partition: ArtifactDigest,
    },
}

/// Content-addressed numerical intent over the sole accepted portable graph.
///
/// Model/revision/source and every numerical policy belong to the graph. The
/// envelope adds only exact external artifact references. Runtime device ordinals
/// and provider observations belong to execution binding and Run provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationEnvelope {
    model_artifact: ArtifactDigest,
    graph: PortableRealizationGraph,
    layout_artifacts: LayoutArtifacts,
}

impl RealizationEnvelope {
    /// Bind one accepted graph to its exact Model and layout inputs.
    ///
    /// # Errors
    /// Rejects Model identity/revision drift or contradictory layout references.
    pub fn new(
        model: &impl CanonicalModelArtifact,
        graph: PortableRealizationGraph,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let reference = model.artifact_reference()?;
        let value = Self {
            model_artifact: reference.artifact().clone(),
            graph,
            layout_artifacts,
        };
        value.validate_model_artifact(model)?;
        value.validate_layout()?;
        Ok(value)
    }

    /// Decode only the current closed wire, revalidating its canonical graph.
    ///
    /// # Errors
    /// Rejects retired schemas, malformed/unknown fields, noncanonical graph
    /// bytes, invalid policies, contradictory layout, or exceeded caller budgets.
    pub fn from_json(bytes: &[u8], limits: RealizationDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire: WireRealizationEnvelope = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid realization envelope JSON: {error}"))
        })?;
        if wire.schema != REALIZATION_SCHEMA || wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported realization-envelope schema or canonical encoding",
            ));
        }
        let value = Self {
            model_artifact: ArtifactDigest::from_hex(wire.model_sha256)?,
            graph: PortableRealizationGraph::from_bytes(wire.graph.get().as_bytes())
                .map_err(|error| invalid_artifact(error.message()))?,
            layout_artifacts: wire.layout_artifacts.decode_validated()?,
        };
        value.validate_limits(limits)?;
        value.validate_layout()?;
        Ok(value)
    }

    /// Deterministic canonical JSON, embedding the graph's own exact bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if graph encoding or serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes = self
            .graph
            .to_bytes()
            .map_err(|error| invalid_artifact(error.message()))?;
        let graph = serde_json::from_slice::<Box<RawValue>>(&bytes)
            .map_err(|error| invalid_artifact(error.to_string()))?;
        let wire = WireRealizationEnvelope {
            schema: REALIZATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: self.model_artifact.to_string(),
            graph,
            layout_artifacts: WireLayoutArtifacts::encode(self.layout_artifacts.clone()),
        };
        serde_json::to_vec(&wire).map_err(|error| invalid_artifact(error.to_string()))
    }

    /// Schema-domain-separated identity of the complete current envelope.
    ///
    /// # Errors
    /// Preserves canonical encoding diagnostics.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            REALIZATION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Sole numerical graph, including lineage and every selected policy.
    #[must_use]
    pub const fn graph(&self) -> &PortableRealizationGraph {
        &self.graph
    }

    /// Exact referenced Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        self.model_artifact.clone()
    }

    /// Model identity retained by graph lineage.
    #[must_use]
    pub fn model(&self) -> OntologyId<Model> {
        self.graph.lineage().model()
    }

    /// Semantic revision retained by graph lineage.
    #[must_use]
    pub fn semantic_revision(&self) -> SemanticRevision {
        self.graph.lineage().semantic_revision()
    }

    /// Default-policy or explicit-revision origin retained by graph lineage.
    #[must_use]
    pub fn source(&self) -> ResolutionSource {
        self.graph.lineage().source()
    }

    /// Exact referenced layout inputs.
    #[must_use]
    pub fn layout_artifacts(&self) -> LayoutArtifacts {
        self.layout_artifacts.clone()
    }

    /// Sole algebraic partition requirement.
    #[must_use]
    pub fn vector_layout(&self) -> VectorLayoutKind {
        self.graph.systems()[0].partition()
    }

    /// Portable compute requirement; contains no deployment-local device ordinal.
    #[must_use]
    pub fn placement(&self) -> PlacementRequirementNode {
        self.graph.placements()[0]
    }

    /// Linear controls, including the inner linear solve of a nonlinear root.
    #[must_use]
    pub fn solver_plan(&self) -> SolverPlan {
        self.graph.linear_solves()[0].plan()
    }

    /// Verify the exact Model digest, ontology identity and semantic revision.
    ///
    /// # Errors
    /// Rejects drift in any of the three identity components.
    pub fn validate_model_artifact(
        &self,
        model: &impl CanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        let reference = model.artifact_reference()?;
        if self.model_artifact != *reference.artifact()
            || self.model() != reference.model()
            || self.semantic_revision() != reference.semantic_revision()
        {
            return Err(invalid_artifact(
                "Model artifact digest, ontology identity, or semantic revision differs from the Realization",
            ));
        }
        Ok(())
    }

    /// Imported or supplied mesh selected by one exact graph Domain.
    ///
    /// # Errors
    /// Rejects a Domain absent from this Realization.
    pub fn mesh_artifact(
        &self,
        domain: Id<kinds::Domain>,
    ) -> Result<Option<ArtifactDigest>, Diagnostic> {
        let node = self
            .graph
            .domains()
            .iter()
            .find(|node| node.domain() == domain)
            .ok_or_else(|| {
                invalid_artifact("mesh lookup selects a Domain absent from the Realization")
            })?;
        Ok(mesh_digest(node.discretization().mesh()))
    }

    /// Authenticate one supplied simplex mesh against its graph Domains and Model.
    ///
    /// Dimension comes from accepted semantic spatial support, never a repeated
    /// envelope setting. ALE quality gates retain their exact mesh agreement.
    ///
    /// # Errors
    /// Rejects Model drift, absent mesh references, dimension drift, or quality drift.
    pub fn validate_mesh_artifact(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        self.validate_model_artifact(model)?;
        let replayed = model.replay_model()?;
        let digest = mesh.digest()?;
        let domains = self
            .graph
            .domains()
            .iter()
            .filter(|node| mesh_digest(node.discretization().mesh()).as_ref() == Some(&digest))
            .collect::<Vec<_>>();
        if domains.is_empty() {
            return Err(invalid_artifact(
                "mesh artifact is not selected by any Realization Domain",
            ));
        }
        for node in domains {
            let dimension = replayed
                .program()
                .spatial_support(node.domain())
                .map(|support| {
                    support
                        .ambient_dimensions()
                        .unwrap_or_else(|| support.intrinsic_dimensions())
                })
                .ok_or_else(|| {
                    invalid_artifact("simplex mesh Domain lacks admitted spatial support")
                })?;
            if let eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension,
                ..
            } = node.discretization().quadrature()
                && spatial_dimension.get() != dimension
            {
                return Err(invalid_artifact(
                    "simplex quadrature dimension differs from the selected Semantic Domain",
                ));
            }
            if dimension != mesh.dimension() {
                return Err(invalid_artifact(
                    "mesh dimension differs from the selected Semantic Domain",
                ));
            }
        }
        for action in self.graph.geometry_actions() {
            let GeometryActionNode::P1HarmonicExtension {
                harmonic_domains,
                drivers,
                quality_gate,
                ..
            } = action;
            let uses_mesh = harmonic_domains
                .iter()
                .copied()
                .chain(
                    drivers
                        .iter()
                        .filter_map(|id| self.graph.field(*id).map(|field| field.domain())),
                )
                .filter_map(|id| self.graph.domain(id))
                .any(|domain| {
                    mesh_digest(domain.discretization().mesh()).as_ref() == Some(&digest)
                });
            if uses_mesh
                && quality_gate.minimum_mean_ratio().to_bits()
                    != mesh.mesh().quality_gate().minimum_mean_ratio().to_bits()
            {
                return Err(invalid_artifact(
                    "reference mesh quality gate differs from the ALE mesh-motion quality gate",
                ));
            }
        }
        Ok(())
    }

    fn validate_layout(&self) -> Result<(), Diagnostic> {
        if matches!(
            (self.vector_layout(), &self.layout_artifacts),
            (VectorLayoutKind::Replicated, LayoutArtifacts::Replicated)
                | (
                    VectorLayoutKind::Distributed,
                    LayoutArtifacts::Distributed { .. }
                )
        ) {
            Ok(())
        } else {
            Err(invalid_artifact(
                "realization layout artifacts contradict the admitted vector-layout requirement",
            ))
        }
    }

    fn validate_limits(&self, limits: RealizationDecoderLimits) -> Result<(), Diagnostic> {
        let constraints = self
            .graph
            .systems()
            .iter()
            .flat_map(|system| system.blocks())
            .filter(|block| matches!(block, SystemBlock::ConstraintMultiplier(_)))
            .count();
        let blocks = self
            .graph
            .systems()
            .iter()
            .map(|system| system.blocks().len())
            .sum::<usize>();
        if self.graph.fields().len() > limits.max_realization_fields
            || constraints > limits.max_realization_constraints
            || self.graph.transformations().len() > limits.max_realization_constraints
            || blocks > limits.max_realization_blocks
        {
            return Err(invalid_artifact(
                "Realization graph exceeds the selected Field, constraint, transformation, or block budget",
            ));
        }
        Ok(())
    }
}

fn mesh_digest(policy: MeshPolicy) -> Option<ArtifactDigest> {
    match policy {
        MeshPolicy::GeneratedUniform { .. } => None,
        MeshPolicy::ImportedSimplicial { artifact }
        | MeshPolicy::SuppliedCartesian { artifact, .. }
        | MeshPolicy::SuppliedCartesian1d { artifact, .. }
        | MeshPolicy::SuppliedCartesian3d { artifact, .. } => {
            Some(ArtifactDigest::from_sha256(artifact.sha256()))
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRealizationEnvelope {
    schema: String,
    encoding: String,
    model_sha256: String,
    graph: Box<RawValue>,
    layout_artifacts: WireLayoutArtifacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireLayoutArtifacts {
    Replicated,
    Distributed {
        layout_sha256: String,
        partition_sha256: String,
    },
}

impl WireLayoutArtifacts {
    fn encode(value: LayoutArtifacts) -> Self {
        match value {
            LayoutArtifacts::Replicated => Self::Replicated,
            LayoutArtifacts::Distributed { layout, partition } => Self::Distributed {
                layout_sha256: layout.0,
                partition_sha256: partition.0,
            },
        }
    }

    fn decode_validated(&self) -> Result<LayoutArtifacts, Diagnostic> {
        match self {
            Self::Replicated => Ok(LayoutArtifacts::Replicated),
            Self::Distributed {
                layout_sha256,
                partition_sha256,
            } => Ok(LayoutArtifacts::Distributed {
                layout: ArtifactDigest::from_hex(layout_sha256.clone())?,
                partition: ArtifactDigest::from_hex(partition_sha256.clone())?,
            }),
        }
    }
}
