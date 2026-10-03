//! Exact common-Plan-bound differentiable application programs.

use eqiora_artifact::{CanonicalModelArtifact, ModelArtifactReference};
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::{Entity, kinds};
use eqiora_core::{Diagnostic, EntityKind, Id, RawId};
use eqiora_differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora_execution::ExecutionReceipt;
use eqiora_ir::{LinearizedOutput, LinearizedRelation};
use eqiora_numerics::{CommonAlgebraicState, CommonScalarDifferentiationPoint, ResolvedCommonPlan};
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, LinearSolveRequest, LinearSolverBackend, SolveReport,
    SolverPlan,
};

use crate::ModelDocument;

mod map_admission;
mod partials;
mod program;

/// One nominally typed entity selected from an exact immutable Model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntityRef<E: Entity> {
    model: ModelArtifactReference,
    id: Id<E>,
}

impl<E: Entity> ModelEntityRef<E> {
    /// Exact Model artifact owning this entity.
    #[must_use]
    pub const fn model(&self) -> &ModelArtifactReference {
        &self.model
    }

    /// Canonical identity retaining its nominal entity kind.
    #[must_use]
    pub const fn id(&self) -> Id<E> {
        self.id
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
    ) -> Result<ModelEntityRef<kinds::Parameter>, Diagnostic> {
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
    pub fn field_ref(&self, selection: &str) -> Result<ModelEntityRef<kinds::Field>, Diagnostic> {
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
    ) -> Result<ModelEntityRef<kinds::Observable>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Observable)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Observable"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }

    /// Resolve a source alias or exact ULID once into a Model-bound Domain.
    pub fn domain_ref(&self, selection: &str) -> Result<ModelEntityRef<kinds::Domain>, Diagnostic> {
        let id = resolve_entity(self, selection, EntityKind::Domain)?
            .downcast()
            .ok_or_else(|| wrong_kind(selection, "Domain"))?;
        Ok(ModelEntityRef {
            model: self.artifact_reference()?,
            id,
        })
    }
}

/// Scalar representation admitted by a differentiable program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiableScalarType {
    /// Native IEEE-754 binary64 values.
    F64,
}

/// Device boundary admitted by a differentiable program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiableDevice {
    /// Host CPU device zero.
    HostCpu,
}

/// Derivative meaning admitted by a differentiable program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeContract {
    /// First derivative of a converged implicit relation and selected output.
    ImplicitFirstOrder,
}

/// Complete typed identity of one differentiable application program.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableProgramIdentity {
    model: ModelArtifactReference,
    plan_identity: String,
    inputs: Vec<Id<kinds::Parameter>>,
    output: RawId,
    initial_state_identity: Option<String>,
    input_dimension: usize,
    output_dimension: usize,
    scalar_type: DifferentiableScalarType,
    device: DifferentiableDevice,
    derivative: DerivativeContract,
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
    pub fn inputs(&self) -> &[Id<kinds::Parameter>] {
        &self.inputs
    }

    /// Selected canonical primary Field.
    #[must_use]
    pub const fn output(&self) -> RawId {
        self.output
    }

    /// Exact finite seed bound by this Program; absent for spatial linear execution.
    #[must_use]
    pub fn initial_state_identity(&self) -> Option<&str> {
        self.initial_state_identity.as_deref()
    }

    /// Flat tangent/gradient input dimension.
    #[must_use]
    pub const fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    /// Flat primary-Field output dimension.
    #[must_use]
    pub const fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    /// Exact scalar representation.
    #[must_use]
    pub const fn scalar_type(&self) -> DifferentiableScalarType {
        self.scalar_type
    }

    /// Exact execution device boundary.
    #[must_use]
    pub const fn device(&self) -> DifferentiableDevice {
        self.device
    }

    /// Mathematical derivative contract.
    #[must_use]
    pub const fn derivative(&self) -> DerivativeContract {
        self.derivative
    }

    /// Solver policy shared by normal and transposed actions.
    #[must_use]
    pub const fn solver(&self) -> SolverPlan {
        self.solver
    }
}

/// One immutable numerical point in a program's ordered Parameter coordinates.
///
/// The canonical Model stores the program's default values. A point binds
/// only the Parameters promoted to program inputs; unselected Parameters stay
/// frozen at their canonical values. Point values are coherent-SI `f64`
/// scalars and never mutate the Model or Plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableParameterPoint {
    inputs: Vec<Id<kinds::Parameter>>,
    values: Vec<f64>,
}

impl DifferentiableParameterPoint {
    /// Ordered canonical Parameter identities.
    #[must_use]
    pub fn inputs(&self) -> &[Id<kinds::Parameter>] {
        &self.inputs
    }

    /// Complete finite values in exact input order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Action represented by one differentiation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationMode {
    /// Accepted primary Field value.
    Primal,
    /// Forward output Jacobian action.
    Jvp,
    /// Reverse output Jacobian action.
    Vjp,
}

/// Lowered derivative implementation used by this bounded slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeImplementation {
    /// Analytically assembled `R_w`, `R_p`, `O_w`, and `O_p` actions.
    AnalyticAssembled,
    /// Exact original-expression Operator IR partial actions.
    OperatorIr,
}

/// Whether an occurrence reused the evaluation's accepted linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationState {
    /// The action publishes the primal that established this evaluation.
    Established,
    /// The derivative action reused the immutable state owned by the evaluation.
    Reused,
}

#[derive(Debug, Clone, PartialEq)]
enum PrimalEvidence {
    Linear(Box<ExecutionReceipt>),
    Nonlinear {
        initial_state_identity: String,
        iterations: usize,
        initial_residual_norm: f64,
        accepted_unknowns: Vec<f64>,
    },
}

/// Typed in-memory provenance for one primal or derivative occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiationEvidence {
    identity: DifferentiableProgramIdentity,
    point: DifferentiableParameterPoint,
    mode: DifferentiationMode,
    implementation: DerivativeImplementation,
    linearization_state: LinearizationState,
    primal_residual_norm: f64,
    residual_tolerance: f64,
    state_system: CanonicalCsrAgreementFingerprintV1,
    primal: PrimalEvidence,
    derivative_solve: Option<SolveReport>,
}

impl DifferentiationEvidence {
    /// Exact program identity.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Exact accepted numerical point used by this occurrence.
    #[must_use]
    pub const fn point(&self) -> &DifferentiableParameterPoint {
        &self.point
    }

    /// Primal, JVP, or VJP action.
    #[must_use]
    pub const fn mode(&self) -> DifferentiationMode {
        self.mode
    }

    /// Source of derivative actions.
    #[must_use]
    pub const fn implementation(&self) -> DerivativeImplementation {
        self.implementation
    }

    /// Primal establishment or derivative reuse of the accepted state.
    #[must_use]
    pub const fn linearization_state(&self) -> LinearizationState {
        self.linearization_state
    }

    /// Exact algebraic state-system identity at the accepted point.
    #[must_use]
    pub const fn state_system(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.state_system
    }

    /// Independently evaluated primal residual norm.
    #[must_use]
    pub const fn primal_residual_norm(&self) -> f64 {
        self.primal_residual_norm
    }

    /// Threshold used to admit the linearization.
    #[must_use]
    pub const fn residual_tolerance(&self) -> f64 {
        self.residual_tolerance
    }

    /// Solve that established the accepted primal point.
    #[must_use]
    pub fn primal_solve(&self) -> Option<&SolveReport> {
        self.receipt().map(ExecutionReceipt::report)
    }

    /// Exact deployment, operator, plan, output, and accepted-solve linkage.
    #[must_use]
    pub fn receipt(&self) -> Option<&ExecutionReceipt> {
        match &self.primal {
            PrimalEvidence::Linear(receipt) => Some(receipt),
            PrimalEvidence::Nonlinear { .. } => None,
        }
    }

    /// Exact nonlinear initial State, absent for a linear primal.
    #[must_use]
    pub fn nonlinear_initial_state_identity(&self) -> Option<&str> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                initial_state_identity,
                ..
            } => Some(initial_state_identity),
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Accepted nonlinear update count, including zero-update acceptance.
    #[must_use]
    pub const fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            PrimalEvidence::Nonlinear { iterations, .. } => Some(*iterations),
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Original nonlinear residual norm at the bound seed and Parameter point.
    #[must_use]
    pub const fn nonlinear_initial_residual_norm(&self) -> Option<f64> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                initial_residual_norm,
                ..
            } => Some(*initial_residual_norm),
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Accepted finite unknowns paired with the nonlinear Jacobian.
    #[must_use]
    pub fn nonlinear_accepted_unknowns(&self) -> Option<&[f64]> {
        match &self.primal {
            PrimalEvidence::Nonlinear {
                accepted_unknowns, ..
            } => Some(accepted_unknowns),
            PrimalEvidence::Linear(_) => None,
        }
    }

    /// Normal or transposed derivative solve, absent for primal publication.
    #[must_use]
    pub const fn derivative_solve(&self) -> Option<&SolveReport> {
        self.derivative_solve.as_ref()
    }
}

/// Accepted primary output and its producer evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiablePrimal {
    output: Vec<f64>,
    evidence: DifferentiationEvidence,
}

impl DifferentiablePrimal {
    /// Complete primary Field values.
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
    /// Complete primary Field values at the accepted point.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Complete primary Field tangent.
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
    /// Complete primary Field values at the accepted point.
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
    point: DifferentiableParameterPoint,
    native: CommonScalarDifferentiationPoint,
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
    pub const fn point(&self) -> &DifferentiableParameterPoint {
        &self.point
    }

    /// Return this point's accepted complete primary Field.
    #[must_use]
    pub fn primal(&self) -> DifferentiablePrimal {
        DifferentiablePrimal {
            output: self.native.output_values(),
            evidence: self.evidence(
                DifferentiationMode::Primal,
                None,
                LinearizationState::Established,
            ),
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
            output: self.native.output_values(),
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
            output: self.native.output_values(),
            input_cotangent,
            evidence: self.evidence(
                DifferentiationMode::Vjp,
                Some(solve),
                LinearizationState::Reused,
            ),
        })
    }

    fn evidence(
        &self,
        mode: DifferentiationMode,
        derivative_solve: Option<SolveReport>,
        linearization_state: LinearizationState,
    ) -> DifferentiationEvidence {
        DifferentiationEvidence {
            identity: self.identity.clone(),
            point: self.point.clone(),
            mode,
            implementation: if self.native.receipt().is_some() {
                DerivativeImplementation::AnalyticAssembled
            } else {
                DerivativeImplementation::OperatorIr
            },
            linearization_state,
            primal_residual_norm: self.primal_residual_norm,
            residual_tolerance: self.residual_tolerance,
            state_system: self
                .native
                .relation()
                .state_jacobian()
                .agreement_fingerprint(),
            primal: match self.native.receipt() {
                Some(receipt) => PrimalEvidence::Linear(Box::new(receipt.clone())),
                None => PrimalEvidence::Nonlinear {
                    initial_state_identity: self
                        .native
                        .nonlinear_initial_state()
                        .expect("accepted nonlinear seed")
                        .identity()
                        .to_owned(),
                    iterations: self
                        .native
                        .nonlinear_iterations()
                        .expect("accepted nonlinear count"),
                    initial_residual_norm: self
                        .native
                        .nonlinear_initial_residual_norm()
                        .expect("accepted nonlinear initial residual"),
                    accepted_unknowns: self.native.relation().accepted_unknowns().to_vec(),
                },
            },
            derivative_solve,
        }
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
