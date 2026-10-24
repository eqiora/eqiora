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
use eqiora_numerics::{
    CommonDifferentiationLinearization, CommonDifferentiationPoint, CommonInitialState,
    ResolvedCommonPlan,
};
use eqiora_solver::{LinearSolveRequest, LinearSolverBackend, SolverPlan};

use crate::ModelDocument;

pub(crate) mod context;
use context::Context;
mod evidence;
mod record;
pub use record::RecordedDifferentiableEvaluation;
mod identity;
pub use identity::DifferentiableProgramIdentity;
mod map_admission;
mod partials;
mod products;
mod program;
mod result;
pub(crate) mod signature;
mod values;

pub use evidence::{
    DerivativeImplementation, DifferentiationEvidence, DifferentiationMode, LinearizationState,
};
pub use products::{DifferentiableCotangent, DifferentiableTangent};
pub use result::DifferentiableResult;
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
    /// Select a nominal entity from a canonical Model artifact, without source aliases.
    ///
    /// # Errors
    /// Rejects an identifier outside this exact Model. This selects identity only;
    /// the consuming Plan owns semantic admission with its external resources.
    pub fn from_model_artifact(
        model: &eqiora_artifact::ModelEnvelope,
        id: Id<E>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let (transaction, _) = model.to_transaction()?;
        if !transaction.ops().iter().any(|operation| {
            matches!(operation, eqiora_graph::Op::DefineKernelNode { node } if node.id() == id.erase())
        }) {
            return Err(single(Diagnostic::error(
                codes::NODE_NOT_FOUND,
                "selected entity is outside the exact Model artifact",
            )));
        }
        Ok(Self {
            model: model.artifact_reference().map_err(single)?,
            id,
        })
    }

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

/// One immutable accepted Parameter point and its paired primal linearization.
#[derive(Debug, Clone)]
pub struct DifferentiableEvaluation {
    identity: DifferentiableProgramIdentity,
    point: DifferentiableValues,
    native: CommonDifferentiationPoint,
    acceptance: evidence::Acceptance,
    backend: &'static dyn LinearSolverBackend,
}

/// Opaque immutable differentiable program over one fixed input coordinate set.
#[derive(Debug, Clone)]
pub struct DifferentiableProgram {
    identity: DifferentiableProgramIdentity,
    context: Context<'static>,
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

    fn output_values(&self) -> DifferentiableValues {
        DifferentiableValues::from_coordinates(
            self.identity.outputs.clone(),
            self.native.linearization().output_values().to_vec(),
        )
        .expect("accepted output has exact finite value signatures")
    }

    /// Return this point's accepted selected outputs.
    #[must_use]
    pub fn primal(&self) -> DifferentiableResult {
        DifferentiableResult {
            derivative: (),
            output: self.output_values(),
            evidence: self.primal_evidence(),
        }
    }

    /// Apply this point's total output JVP in exact selected-input order.
    ///
    /// # Errors
    /// Rejects unequal ordered input signatures before the action. Preserves
    /// relation and solver diagnostics and rejects nonfinite result components.
    pub fn jvp(
        &self,
        tangent: &DifferentiableTangent,
    ) -> Result<DifferentiableResult<DifferentiableTangent>, Diagnostic> {
        tangent.values.validate_signatures(self.identity.inputs())?;
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            self.native.linearization().relation(),
            self.native.linearization(),
            self.native.linearization().relation().state_jacobian(),
            self.native.linearization().state_scalar_domain(),
            self.acceptance.residual_tolerance,
        )?;
        let sensitivity = forward_output_sensitivity(
            &accepted,
            tangent.values.coordinates(),
            self.native
                .linearization()
                .relation()
                .state_jacobian()
                .properties(),
            LinearSolveRequest::new(self.backend, self.identity.solver),
        )?;
        let (state, tangent) = sensitivity.into_parts();
        let (_, solve) = state.into_parts();
        Ok(DifferentiableResult {
            output: self.output_values(),
            derivative: DifferentiableTangent {
                values: DifferentiableValues::from_coordinates(
                    self.identity.outputs.clone(),
                    tangent,
                )?,
            },
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
    /// Rejects unequal ordered output base spaces before the action. Preserves
    /// relation and transposed-solver diagnostics and rejects nonfinite results.
    pub fn vjp(
        &self,
        cotangent: &DifferentiableCotangent,
    ) -> Result<DifferentiableResult<DifferentiableCotangent>, Diagnostic> {
        cotangent
            .values
            .validate_signatures(self.identity.outputs())?;
        let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
            self.native.linearization().relation(),
            self.native.linearization(),
            self.native.linearization().relation().state_jacobian(),
            self.native.linearization().state_scalar_domain(),
            self.acceptance.residual_tolerance,
        )?;
        let gradient = adjoint_output_gradient(
            &accepted,
            cotangent.values.coordinates(),
            self.native
                .linearization()
                .relation()
                .state_jacobian()
                .properties(),
            LinearSolveRequest::new(self.backend, self.identity.solver),
        )?;
        let (adjoint, input_cotangent) = gradient.into_parts();
        let (_, solve) = adjoint.into_parts();
        Ok(DifferentiableResult {
            output: self.output_values(),
            derivative: DifferentiableCotangent {
                values: DifferentiableValues::from_coordinates(
                    self.identity.inputs.clone(),
                    input_cotangent,
                )?,
            },
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
