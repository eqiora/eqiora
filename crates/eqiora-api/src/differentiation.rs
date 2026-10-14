//! Exact common-Plan-bound differentiable application programs.

use eqiora_artifact::{CanonicalModelArtifact, ModelArtifactReference};
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::{Entity, kinds};
use eqiora_core::{Diagnostic, EntityKind, Id, RawId, ScalarType};
use eqiora_differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora_execution::{AcceptedOutputFingerprintV1, ExecutionReceipt};
use eqiora_ir::{LinearizedOutput, LinearizedRelation};
use eqiora_numerics::{CommonAlgebraicState, CommonDifferentiationPoint, ResolvedCommonPlan};
use eqiora_solver::{LinearSolveRequest, LinearSolverBackend, SolverPlan};

use crate::ModelDocument;

mod evidence;
mod map_admission;
mod partials;
mod program;
pub(crate) mod signature;
mod values;

pub use evidence::{
    DerivativeImplementation, DifferentiationEvidence, DifferentiationMode, LinearizationState,
};
pub use signature::DifferentiableValueSignature;
use std::sync::Arc;
pub use values::{DifferentiableValue, DifferentiableValues};

/// One entity selected from an exact immutable Model artifact.
///
/// The identifier is a nominal `Id<E>` or a `RawId` that retains its entity kind.
/// Construction and kind erasure preserve the same exact Model owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntityRef<I = RawId> {
    model: ModelArtifactReference,
    id: I,
}

impl<I: Copy> ModelEntityRef<I> {
    /// Exact Model artifact owning this entity.
    #[must_use]
    pub const fn model(&self) -> &ModelArtifactReference {
        &self.model
    }

    /// Canonical identity retaining its nominal entity kind.
    #[must_use]
    pub const fn id(&self) -> I {
        self.id
    }
}

impl<E: Entity> ModelEntityRef<Id<E>> {
    /// Retain exact Model ownership while erasing only the compile-time entity kind.
    #[must_use]
    pub fn erase(&self) -> ModelEntityRef {
        ModelEntityRef {
            model: self.model.clone(),
            id: self.id.erase(),
        }
    }
}

impl ModelDocument {
    /// Resolve a source alias or exact ULID once into a Model-bound Parameter.
    ///
    /// # Errors
    /// Returns a structured lookup/kind diagnostic if the selection is absent
    /// or does not identify a Parameter in this exact Model.
    pub fn parameter_ref(
        &self,
        selection: &str,
    ) -> Result<ModelEntityRef<Id<kinds::Parameter>>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Parameter)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Parameter"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }

    /// Resolve a source alias or exact ULID once into a Model-bound Field.
    ///
    /// # Errors
    /// Returns a structured lookup/kind diagnostic if the selection is absent
    /// or does not identify a Field in this exact Model.
    pub fn field_ref(
        &self,
        selection: &str,
    ) -> Result<ModelEntityRef<Id<kinds::Field>>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Field)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Field"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }

    /// Resolve one exact instantaneous or spatial Observable in this Model.
    pub fn observable_ref(
        &self,
        selection: &str,
    ) -> Result<ModelEntityRef<Id<kinds::Observable>>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Observable)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Observable"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }

    /// Resolve a source alias or exact ULID once into a Model-bound Domain.
    pub fn domain_ref(
        &self,
        selection: &str,
    ) -> Result<ModelEntityRef<Id<kinds::Domain>>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Domain)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Domain"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }
}

/// Complete typed identity of one differentiable application program.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableProgramIdentity {
    model: ModelArtifactReference,
    plan_identity: String,
    inputs: Arc<[DifferentiableValueSignature]>,
    outputs: Arc<[DifferentiableValueSignature]>,
    initial_state_identity: Option<String>,
    scalar_type: ScalarType,
    solver: SolverPlan,
}

impl DifferentiableProgramIdentity {
    /// Exact canonical Model artifact.
    #[must_use]
    pub const fn model(&self) -> &ModelArtifactReference {
        &self.model
    }

    /// Exact common Plan identity.
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    /// Ordered canonical Parameter inputs.
    #[must_use]
    pub fn inputs(&self) -> &[DifferentiableValueSignature] {
        &self.inputs
    }

    /// Ordered selected Fields and Observables with their exact types and layouts.
    #[must_use]
    pub fn outputs(&self) -> &[DifferentiableValueSignature] {
        &self.outputs
    }

    /// Exact finite seed bound by this Program; absent for spatial linear execution.
    #[must_use]
    pub fn initial_state_identity(&self) -> Option<&str> {
        self.initial_state_identity.as_deref()
    }

    /// Flat tangent/gradient input dimension.
    #[must_use]
    pub fn input_dimension(&self) -> usize {
        signature::dimension(&self.inputs).expect("admitted input signature")
    }

    /// Total real-coordinate dimension of the ordered selected outputs.
    #[must_use]
    pub fn output_dimension(&self) -> usize {
        signature::dimension(&self.outputs).expect("admitted output signature")
    }

    /// Shared storage representation; current program admission requires [`ScalarType::F64`].
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }

    /// Solver policy shared by normal and transposed actions.
    #[must_use]
    pub const fn solver(&self) -> SolverPlan {
        self.solver
    }
}

/// Accepted selected outputs and their producer evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiablePrimal {
    output: Vec<f64>,
    evidence: DifferentiationEvidence,
}

impl DifferentiablePrimal {
    /// Complete selected output coordinates in Program order.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Typed occurrence provenance.
    #[must_use]
    pub const fn evidence(&self) -> &DifferentiationEvidence {
        &self.evidence
    }

    /// Consume the occurrence into its owned Field values and evidence.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, DifferentiationEvidence) {
        (self.output, self.evidence)
    }
}

/// Accepted primal and forward output action.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableJvp {
    output: Vec<f64>,
    tangent: Vec<f64>,
    evidence: DifferentiationEvidence,
}

impl DifferentiableJvp {
    /// Complete selected output coordinates at the accepted point.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Complete selected output tangent coordinates.
    #[must_use]
    pub fn tangent(&self) -> &[f64] {
        &self.tangent
    }

    /// Typed occurrence provenance.
    #[must_use]
    pub const fn evidence(&self) -> &DifferentiationEvidence {
        &self.evidence
    }

    /// Consume the occurrence into primal values, tangent, and evidence.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>, DifferentiationEvidence) {
        (self.output, self.tangent, self.evidence)
    }
}

/// Accepted primal and reverse input action.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableVjp {
    output: Vec<f64>,
    input_cotangent: Vec<f64>,
    evidence: DifferentiationEvidence,
}

impl DifferentiableVjp {
    /// Complete selected output coordinates at the accepted point.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Total cotangent in exact selected-input order.
    #[must_use]
    pub fn input_cotangent(&self) -> &[f64] {
        &self.input_cotangent
    }

    /// Typed occurrence provenance.
    #[must_use]
    pub const fn evidence(&self) -> &DifferentiationEvidence {
        &self.evidence
    }

    /// Consume the occurrence into primal values, input cotangent, and evidence.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>, DifferentiationEvidence) {
        (self.output, self.input_cotangent, self.evidence)
    }
}

/// One immutable accepted Parameter point and its paired primal linearization.
#[derive(Debug, Clone)]
pub struct DifferentiableEvaluation {
    identity: DifferentiableProgramIdentity,
    point: DifferentiableValues,
    native: CommonDifferentiationPoint,
    accepted_state: AcceptedOutputFingerprintV1,
    accepted_output: AcceptedOutputFingerprintV1,
    primal_residual_norm: f64,
    residual_tolerance: f64,
    backend: &'static dyn LinearSolverBackend,
}

/// Opaque immutable differentiable program over one fixed input coordinate set.
#[derive(Debug, Clone)]
pub struct DifferentiableProgram {
    identity: DifferentiableProgramIdentity,
    plan: ResolvedCommonPlan,
    initial: Option<CommonAlgebraicState>,
    backend: &'static dyn LinearSolverBackend,
    default: DifferentiableEvaluation,
}

impl DifferentiableEvaluation {
    /// Static program identity shared by every accepted point.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Exact immutable Parameter point accepted by this evaluation.
    #[must_use]
    pub const fn point(&self) -> &DifferentiableValues {
        &self.point
    }

    /// Return this point's accepted selected outputs.
    #[must_use]
    pub fn primal(&self) -> DifferentiablePrimal {
        DifferentiablePrimal {
            output: self.native.output_values().to_vec(),
            evidence: self.primal_evidence(),
        }
    }

    /// Apply this point's total output JVP in exact selected-input order.
    ///
    /// # Errors
    /// Preserves shape, non-finite input, relation, and solver diagnostics.
    pub fn jvp(&self, tangent: &[f64]) -> Result<DifferentiableJvp, Diagnostic> {
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            self.native.relation(),
            &self.native,
            self.native.relation().state_jacobian(),
            self.residual_tolerance,
        )?;
        let sensitivity = forward_output_sensitivity(
            &accepted,
            tangent,
            self.native.relation().state_jacobian().properties(),
            LinearSolveRequest::new(self.backend, self.identity.solver),
        )?;
        let (state, tangent) = sensitivity.into_parts();
        let (_, solve) = state.into_parts();
        Ok(DifferentiableJvp {
            output: self.native.output_values().to_vec(),
            tangent,
            evidence: self.evidence(
                DifferentiationMode::Jvp,
                Some(solve),
                LinearizationState::Reused,
            ),
        })
    }

    /// Apply this point's total output VJP in exact selected-input order.
    ///
    /// # Errors
    /// Preserves shape, non-finite input, output, relation, and transposed
    /// solver diagnostics.
    pub fn vjp(&self, cotangent: &[f64]) -> Result<DifferentiableVjp, Diagnostic> {
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            self.native.relation(),
            &self.native,
            self.native.relation().state_jacobian(),
            self.residual_tolerance,
        )?;
        let gradient = adjoint_output_gradient(
            &accepted,
            cotangent,
            self.native.relation().state_jacobian().properties(),
            LinearSolveRequest::new(self.backend, self.identity.solver),
        )?;
        let (adjoint, input_cotangent) = gradient.into_parts();
        let (_, solve) = adjoint.into_parts();
        Ok(DifferentiableVjp {
            output: self.native.output_values().to_vec(),
            input_cotangent,
            evidence: self.evidence(
                DifferentiationMode::Vjp,
                Some(solve),
                LinearizationState::Reused,
            ),
        })
    }
}

fn resolve_entity(
    model: &ModelDocument,
    selection: &str,
    expected: EntityKind,
) -> Result<RawId, Diagnostic> {
    let id = model.aliases().get(selection).copied().or_else(|| {
        model
            .program()
            .nodes()
            .map(|node| node.id())
            .find(|id| id.ulid().to_string() == selection)
    });
    let Some(id) = id else {
        return Err(Diagnostic::error(
            codes::NODE_NOT_FOUND,
            format!("{expected:?} selection {selection:?} is not present in this Model"),
        ));
    };
    if id.kind() != expected {
        return Err(wrong_kind(selection, &format!("{expected:?}")));
    }
    Ok(id)
}

fn wrong_kind(selection: &str, expected: &str) -> Diagnostic {
    Diagnostic::error(
        codes::ID_KIND_MISMATCH,
        format!("selection {selection:?} does not identify a canonical {expected}"),
    )
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

fn single(diagnostic: Diagnostic) -> Vec<Diagnostic> {
    vec![diagnostic]
}
