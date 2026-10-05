//! Model-bound finite Hermitian spectral admission.
use super::invalid;
use eqiora_artifact::{CanonicalModelArtifact, ModelEnvelope};
use eqiora_core::{Diagnostic, Id, ValueLiteral, entity::kinds};
use eqiora_sem::KernelProgram;
use eqiora_solver::{HermitianEigenproblem, LinearSolverBackend, SolverProvider};
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod exclusion;
mod request;
mod source;
pub use request::CommonEigenRequest;

/// Reconstructed source Fields and the maximum relative coordinate residual.
type LiftedMode = (Vec<(Id<kinds::Field>, ValueLiteral)>, f64);

/// One exact source Model and its finite Hermitian eigensolve selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonEigenPlan {
    model: Arc<ModelEnvelope>,
    kernel: KernelProgram,
    source: source::SourcePencil,
    exclusion: Option<exclusion::Exclusion>,
    request: CommonEigenRequest,
    provider: SolverProvider,
    identity: String,
    model_id: String,
    model_digest: String,
    model_revision: u64,
}

impl CommonEigenPlan {
    /// Resolve a finite source pencil and its explicit coordinate embedding
    /// chain without an initial guess. Other source constraints and unsupported
    /// providers reject before execution.
    pub fn resolve(
        model: &ModelEnvelope,
        request: CommonEigenRequest,
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        let kernel = model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("spectral Model replay failed"))
        })?;
        let source = source::SourcePencil::lower(&kernel)?;
        let (operator, metric) = source
            .projected
            .as_ref()
            .map_or((&source.operator, &source.metric), |value| {
                (&value.operator, &value.metric)
            });
        let problem = HermitianEigenproblem::new(operator, metric)?;
        request.validate(&problem)?;
        backend.provider().validate()?;
        backend.require_hermitian_eigenproblem(&problem)?;
        let exclusion = exclusion::Exclusion::derive(&source, request)?;
        let reference = model.artifact_reference()?;
        let model_digest = reference.artifact().to_string();
        let provider = backend.provider();
        let bytes = serde_json::to_vec(&(
            &model_digest,
            source.relation.ulid().to_string(),
            source.mode.ulid().to_string(),
            source.eigenvalue.ulid().to_string(),
            request.identity_bytes()?,
            provider.id().as_str(),
            provider.implementation_version(),
            provider
                .libraries()
                .iter()
                .map(|p| (p.name(), p.version()))
                .collect::<Vec<_>>(),
        ))
        .map_err(|error| invalid(format!("cannot identify spectral Plan: {error}")))?;
        let mut hash = Sha256::new();
        hash.update(b"eqiora.common-eigen-plan/v1\0");
        hash.update(bytes);
        let identity = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(Self {
            model: Arc::new(model.clone()),
            kernel,
            source,
            exclusion,
            request,
            provider,
            identity,
            model_id: reference.model().ulid().to_string(),
            model_digest,
            model_revision: reference.semantic_revision().get(),
        })
    }
    /// Excluded source directions as `(projector, dimension, operator_defect, metric_defect)`.
    /// The dimensionless metric projector retains the physical Field basis.
    /// Defects are `||A E||_F/(||A||_F ||E||_F)` and the analogous B action,
    /// with zero defect for a zero operator. Compare each separately with the
    /// request's residual tolerance; numerical nullity does not establish gauge
    /// freedom. None means that the source embedding excludes no directions.
    pub fn excluded_space(&self) -> Option<(&ValueLiteral, usize, f64, f64)> {
        self.exclusion
            .as_ref()
            .map(exclusion::Exclusion::description)
    }

    /// Execute with the exact admitted provider and verify the original pencil.
    pub fn run_result(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        if backend.provider() != self.provider {
            return Err(invalid(
                "spectral execution provider differs from the exact Plan",
            ));
        }
        let problem = self.admitted_problem()?;
        backend.require_hermitian_eigenproblem(&problem)?;
        let start = std::time::Instant::now();
        let candidates = backend.hermitian_eigenpairs(&problem)?;
        crate::CommonResult::from_eigen(self, candidates)?
            .with_elapsed_seconds(start.elapsed().as_secs_f64())
    }

    /// Exact Plan identity, including Model, source roles, controls and provider release.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Original Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    /// Exact Model artifact digest.
    pub fn model_digest(&self) -> &str {
        &self.model_digest
    }
    /// Original semantic revision.
    pub const fn model_revision(&self) -> u64 {
        self.model_revision
    }
    /// Retained source Model artifact.
    pub fn model_artifact(&self) -> &ModelEnvelope {
        &self.model
    }
    /// Accepted source semantics.
    pub fn kernel(&self) -> &KernelProgram {
        &self.kernel
    }
    /// Original equality Relation, never a manufactured RHS system.
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.source.relation
    }
    /// Source mode Field in the exact nominal finite basis.
    pub const fn mode_field(&self) -> Id<kinds::Field> {
        self.source.mode
    }
    /// Source real eigenvalue Field.
    pub const fn eigenvalue_field(&self) -> Id<kinds::Field> {
        self.source.eigenvalue
    }
    /// Exact source coordinate equalities, ordered from the physical mode to
    /// the admitted coordinates: (Relation, target Field, coordinate Field, map).
    /// Each map means `target = map * coordinate`; no excluded directions are
    /// silently identified as null modes or discarded from the source Model.
    pub fn coordinate_embeddings(
        &self,
    ) -> impl Iterator<
        Item = (
            Id<kinds::Relation>,
            Id<kinds::Field>,
            Id<kinds::Field>,
            &ValueLiteral,
        ),
    > {
        self.source
            .projected
            .iter()
            .flat_map(|value| value.embedding.steps.iter())
            .map(|step| (step.relation, step.target, step.coordinate, &step.map))
    }
    /// Typed operator derived from the original Relation.
    pub fn operator(&self) -> &ValueLiteral {
        &self.source.operator
    }
    /// Typed metric derived from the original Relation. It is positive definite
    /// on the admitted coordinate space; excluded original directions may be singular.
    pub fn metric(&self) -> &ValueLiteral {
        &self.source.metric
    }
    /// Requested dense spectral selection and acceptance controls.
    pub const fn request(&self) -> CommonEigenRequest {
        self.request
    }
    /// Exact admitted numerical provider release.
    pub const fn solver_provider(&self) -> SolverProvider {
        self.provider
    }

    pub(crate) fn admitted_problem(&self) -> Result<HermitianEigenproblem<'_>, Diagnostic> {
        let (operator, metric) = self
            .source
            .projected
            .as_ref()
            .map_or((self.operator(), self.metric()), |value| {
                (&value.operator, &value.metric)
            });
        HermitianEigenproblem::new(operator, metric)
    }

    pub(crate) fn lift_mode(&self, coordinate: &ValueLiteral) -> Result<LiftedMode, Diagnostic> {
        self.source.projected.as_ref().map_or_else(
            || Ok((vec![(self.mode_field(), coordinate.clone())], 0.)),
            |value| value.embedding.lift(coordinate),
        )
    }
}
