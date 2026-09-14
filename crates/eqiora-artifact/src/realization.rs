use std::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use std::str::FromStr;

use eqiora_core::{Diagnostic, OntologyId, ScalarType};
use eqiora_realization::{
    DefaultPolicyVersion, Discretization, DiscretizationMethod, ExecutionSchedule,
    MeshArtifactReference, MeshPolicy, QuadraturePolicy, RealizationPlan, RealizationRequirements,
    RealizationRevision, ResolutionSource, ResolvedRealization, SemanticRevision, Space,
    SpaceFamily, Target, VectorLayoutKind,
};
use eqiora_schema::Model;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalModelArtifact, SimplicialMeshEnvelopeV1,
    check_json_limits, invalid_artifact,
};

const REALIZATION_SCHEMA: &str = "eqiora.realization-envelope/v1";

/// Semantic work budgets shared by Realization artifact generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealizationDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum exact Semantic Fields and Field-space bindings.
    pub max_realization_fields: usize,
    /// Maximum entries in an algebraic-constraint or trace-quotient inventory.
    pub max_realization_constraints: usize,
    /// Maximum scaled algebraic blocks.
    pub max_realization_blocks: usize,
}

impl Default for RealizationDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
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

/// Versioned serialization of one resolved, validated Realization selection.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationEnvelopeV1 {
    wire: WireRealizationEnvelopeV1,
}

impl RealizationEnvelopeV1 {
    /// Encode a resolved realization and its content-addressed model/layout inputs.
    ///
    /// # Errors
    /// Returns `EQ0901` if the supplied layout artifact form contradicts the
    /// requirements admitted during resolution or a platform-sized value
    /// cannot be represented by the portable wire contract.
    pub fn from_resolved(
        model: &impl CanonicalModelArtifact,
        resolved: &ResolvedRealization,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let model = model.artifact_reference()?;
        if model.model() != resolved.model()
            || model.semantic_revision() != resolved.semantic_revision()
        {
            return Err(invalid_artifact(
                "resolved realization does not identify the supplied model artifact and source revision",
            ));
        }
        require_layout_artifacts(resolved.requirements().vector_layout(), &layout_artifacts)?;
        let wire = WireRealizationEnvelopeV1 {
            schema: REALIZATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.artifact().to_string(),
            model_ulid: resolved.model().ulid().to_string(),
            semantic_revision: resolved.semantic_revision().get(),
            source: WireResolutionSource::encode(resolved.source()),
            requirements: WireRequirements::encode(resolved.requirements())?,
            plan: WirePlan::encode(resolved.plan())?,
            layout_artifacts: WireLayoutArtifacts::encode(layout_artifacts),
        };
        let envelope = Self { wire };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Decode and validate a realization envelope without resolving a backend.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, or locally
    /// inconsistent data. Typed constructors revalidate the complete plan.
    pub fn from_json(bytes: &[u8], limits: RealizationDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid realization envelope JSON: {error}"))
        })?;
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
            invalid_artifact(format!("cannot serialize realization envelope: {error}"))
        })
    }

    /// Domain-separated SHA-256 identity of the complete realization bytes.
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
    /// Returns `EQ0901` only if internal validated state was corrupted.
    pub fn model(&self) -> Result<OntologyId<Model>, Diagnostic> {
        parse_ulid(&self.wire.model_ulid).map(OntologyId::from_ulid)
    }

    /// Validate the exact explicitly selected Model artifact referenced by
    /// this Realization.
    ///
    /// # Errors
    /// Returns `EQ0901` for wire-domain digest, Model identity, or semantic
    /// revision drift.
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
                "Model artifact digest, ontology identity, or semantic revision differs from the Realization",
            ));
        }
        Ok(())
    }

    /// Semantic graph revision realized by this artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        SemanticRevision::new(self.wire.semantic_revision)
    }

    /// Default-policy or explicit-revision origin.
    #[must_use]
    pub const fn source(&self) -> ResolutionSource {
        self.wire.source.decode()
    }

    /// Model/lowering requirements against which the plan was admitted.
    ///
    /// # Errors
    /// Returns `EQ0901` only if internal validated state was corrupted.
    pub fn requirements(&self) -> Result<RealizationRequirements, Diagnostic> {
        self.wire.requirements.decode()
    }

    /// Complete typed realization plan.
    ///
    /// # Errors
    /// Returns `EQ0901` only if internal validated state was corrupted.
    pub fn plan(&self) -> Result<RealizationPlan, Diagnostic> {
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
    /// Returns `EQ0901` only if internal validated state was corrupted.
    pub fn mesh_artifact(&self) -> Result<Option<ArtifactDigest>, Diagnostic> {
        Ok(match self.plan()?.discretization().mesh() {
            MeshPolicy::GeneratedUniform { .. } => None,
            MeshPolicy::SuppliedCartesian { artifact, .. }
            | MeshPolicy::SuppliedCartesian1d { artifact, .. }
            | MeshPolicy::SuppliedCartesian3d { artifact, .. } => {
                Some(ArtifactDigest::from_sha256(artifact.sha256()))
            }
            MeshPolicy::ImportedSimplicial { artifact } => {
                Some(ArtifactDigest::from_sha256(artifact.sha256()))
            }
        })
    }

    /// Validate a resolved imported mesh against this realization's exact
    /// content identity and admitted spatial dimension.
    ///
    /// # Errors
    /// Returns `EQ0901` if this plan generates its mesh, the digest differs,
    /// or the mesh dimension contradicts the admitted lowering requirements.
    pub fn validate_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let Some(expected) = self.mesh_artifact()? else {
            return Err(invalid_artifact(
                "generated-mesh realization does not consume an imported mesh artifact",
            ));
        };
        if mesh.digest()? != expected {
            return Err(invalid_artifact(
                "imported mesh artifact digest differs from the realization plan",
            ));
        }
        let required_dimension = self.requirements()?.spatial_dimension().get();
        if mesh.dimension() != required_dimension {
            return Err(invalid_artifact(format!(
                "imported mesh dimension {} differs from admitted realization dimension {required_dimension}",
                mesh.dimension(),
            )));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != REALIZATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported realization-envelope schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid)?;
        if matches!(
            self.wire.source,
            WireResolutionSource::Default { policy_version }
                if policy_version != DefaultPolicyVersion::V0.get()
        ) {
            return Err(invalid_artifact(
                "realization-envelope/v1 supports only default-policy/v0",
            ));
        }
        let requirements = self.wire.requirements.decode()?;
        self.wire.plan.clone().decode()?;
        let artifacts = self.wire.layout_artifacts.decode_validated()?;
        require_layout_artifacts(requirements.vector_layout(), &artifacts)
    }
}

fn require_layout_artifacts(
    layout: VectorLayoutKind,
    artifacts: &LayoutArtifacts,
) -> Result<(), Diagnostic> {
    if matches!(
        (layout, artifacts),
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

fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact("model ULID is malformed"))
}

fn encode_usize(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds wire u64")))
}

fn decode_nonzero_usize(value: u64, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    let value = usize::try_from(value)
        .map_err(|_| invalid_artifact(format!("{label} exceeds local usize")))?;
    NonZeroUsize::new(value).ok_or_else(|| invalid_artifact(format!("{label} must be non-zero")))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRealizationEnvelopeV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    source: WireResolutionSource,
    requirements: WireRequirements,
    plan: WirePlan,
    layout_artifacts: WireLayoutArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireResolutionSource {
    Default { policy_version: u16 },
    Explicit { realization_revision: u64 },
}

impl WireResolutionSource {
    const fn encode(value: ResolutionSource) -> Self {
        match value {
            ResolutionSource::Default(version) => Self::Default {
                policy_version: version.get(),
            },
            ResolutionSource::Explicit(revision) => Self::Explicit {
                realization_revision: revision.get(),
            },
        }
    }

    const fn decode(self) -> ResolutionSource {
        match self {
            Self::Default { policy_version } => {
                ResolutionSource::Default(DefaultPolicyVersion::new(policy_version))
            }
            Self::Explicit {
                realization_revision,
            } => ResolutionSource::Explicit(RealizationRevision::new(realization_revision)),
        }
    }
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

    fn decode(&self) -> LayoutArtifacts {
        self.decode_validated()
            .expect("validated realization envelope retains valid artifact digests")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRequirements {
    spatial_dimension: u64,
    scalar_type: WireScalarType,
    vector_layout: WireVectorLayout,
}

impl WireRequirements {
    fn encode(value: RealizationRequirements) -> Result<Self, Diagnostic> {
        Ok(Self {
            spatial_dimension: encode_usize(value.spatial_dimension().get(), "spatial dimension")?,
            scalar_type: WireScalarType::encode(value.scalar_type()),
            vector_layout: WireVectorLayout::encode(value.vector_layout()),
        })
    }

    fn decode(self) -> Result<RealizationRequirements, Diagnostic> {
        Ok(RealizationRequirements::new(
            decode_nonzero_usize(self.spatial_dimension, "spatial dimension")?,
            self.scalar_type.decode(),
            self.vector_layout.decode(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePlan {
    space: WireSpace,
    discretization: WireDiscretization,
    solver: WireSolverPlan,
    target: WireTarget,
    schedule: WireSchedule,
}

impl WirePlan {
    fn encode(value: &RealizationPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            space: WireSpace::encode(value.space())?,
            discretization: WireDiscretization::encode(value.discretization())?,
            solver: WireSolverPlan::encode(value.solver())?,
            target: WireTarget::encode(value.target())?,
            schedule: WireSchedule::encode(value.schedule()),
        })
    }

    fn decode(self) -> Result<RealizationPlan, Diagnostic> {
        RealizationPlan::new(
            self.space.decode()?,
            self.discretization.decode()?,
            self.solver.decode()?,
            self.target.decode()?,
            self.schedule.decode()?,
        )
        .map_err(|error| invalid_artifact(error.message()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSpace {
    ContinuousLagrange { order: u16 },
    CellConstant,
}

impl WireSpace {
    fn encode(value: Space) -> Result<Self, Diagnostic> {
        match value.family() {
            SpaceFamily::ContinuousLagrange { order } => {
                Ok(Self::ContinuousLagrange { order: order.get() })
            }
            SpaceFamily::CellConstant => Ok(Self::CellConstant),
            SpaceFamily::SimplexP1Bubble => Err(invalid_artifact(
                "realization artifact v1 cannot encode a simplex P1-bubble space; a versioned wire extension is required",
            )),
        }
    }

    fn decode(self) -> Result<Space, Diagnostic> {
        match self {
            Self::ContinuousLagrange { order } => NonZeroU16::new(order)
                .map(Space::continuous_lagrange)
                .ok_or_else(|| invalid_artifact("Lagrange order must be non-zero")),
            Self::CellConstant => Ok(Space::cell_constant()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDiscretization {
    method: WireDiscretizationMethod,
    mesh: WireMesh,
    quadrature: WireQuadrature,
}

impl WireDiscretization {
    fn encode(value: Discretization) -> Result<Self, Diagnostic> {
        Ok(Self {
            method: WireDiscretizationMethod::encode(value.method()),
            mesh: WireMesh::encode(value.mesh())?,
            quadrature: WireQuadrature::encode(value.quadrature())?,
        })
    }

    fn decode(self) -> Result<Discretization, Diagnostic> {
        Ok(Discretization::new(
            self.method.decode(),
            self.mesh.decode()?,
            self.quadrature.decode()?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireDiscretizationMethod {
    ContinuousGalerkin,
    CellCenteredFiniteVolume,
}

impl WireDiscretizationMethod {
    const fn encode(value: DiscretizationMethod) -> Self {
        match value {
            DiscretizationMethod::ContinuousGalerkin => Self::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume => Self::CellCenteredFiniteVolume,
        }
    }

    const fn decode(self) -> DiscretizationMethod {
        match self {
            Self::ContinuousGalerkin => DiscretizationMethod::ContinuousGalerkin,
            Self::CellCenteredFiniteVolume => DiscretizationMethod::CellCenteredFiniteVolume,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireMesh {
    GeneratedUniform { cells_per_axis: u64 },
    ImportedSimplicial { artifact_sha256: String },
}

impl WireMesh {
    fn encode(value: MeshPolicy) -> Result<Self, Diagnostic> {
        match value {
            MeshPolicy::GeneratedUniform { cells_per_axis } => Ok(Self::GeneratedUniform {
                cells_per_axis: encode_usize(cells_per_axis.get(), "cells per axis")?,
            }),
            MeshPolicy::ImportedSimplicial { artifact } => Ok(Self::ImportedSimplicial {
                artifact_sha256: ArtifactDigest::from_sha256(artifact.sha256()).to_string(),
            }),
            MeshPolicy::SuppliedCartesian { .. }
            | MeshPolicy::SuppliedCartesian1d { .. }
            | MeshPolicy::SuppliedCartesian3d { .. } => Err(invalid_artifact(
                "realization-envelope/v1 cannot encode supplied Cartesian mesh policy",
            )),
        }
    }

    fn decode(self) -> Result<MeshPolicy, Diagnostic> {
        match self {
            Self::GeneratedUniform { cells_per_axis } => Ok(MeshPolicy::GeneratedUniform {
                cells_per_axis: decode_nonzero_usize(cells_per_axis, "cells per axis")?,
            }),
            Self::ImportedSimplicial { artifact_sha256 } => {
                let digest = ArtifactDigest::from_hex(artifact_sha256)?;
                Ok(MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256(digest.sha256_bytes()),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireQuadrature {
    GaussLegendre { points_per_axis: u64 },
    CellCentroid,
    SimplexCentroid,
}

impl WireQuadrature {
    fn encode(value: QuadraturePolicy) -> Result<Self, Diagnostic> {
        match value {
            QuadraturePolicy::GaussLegendre { points_per_axis } => Ok(Self::GaussLegendre {
                points_per_axis: encode_usize(points_per_axis.get(), "quadrature points per axis")?,
            }),
            QuadraturePolicy::CellCentroid => Ok(Self::CellCentroid),
            QuadraturePolicy::SimplexCentroid => Ok(Self::SimplexCentroid),
            QuadraturePolicy::TriangleDuffyGaussLegendre { .. } => Err(invalid_artifact(
                "realization artifact v1 cannot encode triangle Duffy quadrature; a versioned wire extension is required",
            )),
            QuadraturePolicy::SimplexDuffyGaussLegendre { .. } => Err(invalid_artifact(
                "realization artifact v1 cannot encode dimension-explicit simplex Duffy quadrature; a versioned wire extension is required",
            )),
        }
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSolverPlan {
    algorithm: WireLinearSolver,
    preconditioner: WirePreconditioner,
    reduction: WireReduction,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
}

impl WireSolverPlan {
    fn encode(value: SolverPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            algorithm: WireLinearSolver::encode(value.algorithm())?,
            preconditioner: WirePreconditioner::encode(value.preconditioner()),
            reduction: WireReduction::encode(value.reduction()),
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerance: value.absolute_tolerance(),
            maximum_iterations: encode_usize(
                value.maximum_iterations().get(),
                "maximum iterations",
            )?,
        })
    }

    fn decode(self) -> Result<SolverPlan, Diagnostic> {
        SolverPlan::new(
            self.algorithm.decode(),
            self.relative_tolerance,
            self.absolute_tolerance,
            decode_nonzero_usize(self.maximum_iterations, "maximum iterations")?,
        )
        .map(|plan| {
            plan.with_preconditioner(self.preconditioner.decode())
                .with_reduction(self.reduction.decode())
        })
        .map_err(|error| invalid_artifact(error.message()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLinearSolver {
    ConjugateGradient,
    BiConjugateGradientStabilized,
}

impl WireLinearSolver {
    fn encode(value: LinearSolver) -> Result<Self, Diagnostic> {
        match value {
            LinearSolver::ConjugateGradient => Ok(Self::ConjugateGradient),
            LinearSolver::BiConjugateGradientStabilized => Ok(Self::BiConjugateGradientStabilized),
            LinearSolver::MinimumResidual => Err(invalid_artifact(
                "realization artifact v1 cannot encode MINRES; a versioned wire extension is required",
            )),
            LinearSolver::SparseLu => Err(invalid_artifact(
                "realization artifact v1 cannot encode sparse LU; a versioned wire extension is required",
            )),
        }
    }

    const fn decode(self) -> LinearSolver {
        match self {
            Self::ConjugateGradient => LinearSolver::ConjugateGradient,
            Self::BiConjugateGradientStabilized => LinearSolver::BiConjugateGradientStabilized,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePreconditioner {
    Identity,
    Jacobi,
}

impl WirePreconditioner {
    const fn encode(value: PreconditionerPolicy) -> Self {
        match value {
            PreconditionerPolicy::Identity => Self::Identity,
            PreconditionerPolicy::Jacobi => Self::Jacobi,
        }
    }

    const fn decode(self) -> PreconditionerPolicy {
        match self {
            Self::Identity => PreconditionerPolicy::Identity,
            Self::Jacobi => PreconditionerPolicy::Jacobi,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireReduction {
    Reproducible,
    Fast,
}

impl WireReduction {
    const fn encode(value: ReductionPolicy) -> Self {
        match value {
            ReductionPolicy::Reproducible => Self::Reproducible,
            ReductionPolicy::Fast => Self::Fast,
        }
    }

    const fn decode(self) -> ReductionPolicy {
        match self {
            Self::Reproducible => ReductionPolicy::Reproducible,
            Self::Fast => ReductionPolicy::Fast,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarType {
    F32,
    F64,
}

impl WireScalarType {
    const fn encode(value: ScalarType) -> Self {
        match value {
            ScalarType::F32 => Self::F32,
            ScalarType::F64 => Self::F64,
        }
    }

    const fn decode(self) -> ScalarType {
        match self {
            Self::F32 => ScalarType::F32,
            Self::F64 => ScalarType::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireVectorLayout {
    Replicated,
    Distributed,
}

impl WireVectorLayout {
    const fn encode(value: VectorLayoutKind) -> Self {
        match value {
            VectorLayoutKind::Replicated => Self::Replicated,
            VectorLayoutKind::Distributed => Self::Distributed,
        }
    }

    const fn decode(self) -> VectorLayoutKind {
        match self {
            Self::Replicated => VectorLayoutKind::Replicated,
            Self::Distributed => VectorLayoutKind::Distributed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTarget {
    HostCpu { threads: u64 },
    CudaGpu { device: u16 },
}

impl WireTarget {
    fn encode(value: Target) -> Result<Self, Diagnostic> {
        match value {
            Target::HostCpu { threads } => Ok(Self::HostCpu {
                threads: encode_usize(threads.get(), "host threads")?,
            }),
            Target::CudaGpu { device } => Ok(Self::CudaGpu { device }),
        }
    }

    fn decode(self) -> Result<Target, Diagnostic> {
        match self {
            Self::HostCpu { threads } => Ok(Target::HostCpu {
                threads: decode_nonzero_usize(threads, "host threads")?,
            }),
            Self::CudaGpu { device } => Ok(Target::CudaGpu { device }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSchedule {
    Offline,
    RealTime { priority: u16, deadline_ns: u64 },
}

impl WireSchedule {
    const fn encode(value: ExecutionSchedule) -> Self {
        match value {
            ExecutionSchedule::Offline => Self::Offline,
            ExecutionSchedule::RealTime {
                priority,
                deadline_ns,
            } => Self::RealTime {
                priority,
                deadline_ns: deadline_ns.get(),
            },
        }
    }

    fn decode(self) -> Result<ExecutionSchedule, Diagnostic> {
        match self {
            Self::Offline => Ok(ExecutionSchedule::Offline),
            Self::RealTime {
                priority,
                deadline_ns,
            } => Ok(ExecutionSchedule::RealTime {
                priority,
                deadline_ns: NonZeroU64::new(deadline_ns)
                    .ok_or_else(|| invalid_artifact("real-time deadline must be non-zero"))?,
            }),
        }
    }
}
