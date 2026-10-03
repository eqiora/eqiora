//! Exact common-Plan-bound differentiable application programs.

use eqiora_artifact::ModelArtifactReference;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::{Entity, kinds};
use eqiora_core::{Diagnostic, EntityKind, Id, RawId};
use eqiora_differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora_execution::ExecutionReceipt;
use eqiora_ir::{LinearizedOutput, LinearizedRelation};
use eqiora_numerics::{CommonScalarDifferentiationPoint, CommonScalarPlan};
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, LinearSolveRequest, REFERENCE_LINEAR_SOLVER, SolveReport,
    SolverPlan,
};

use crate::ModelDocument;

mod map_admission;

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
    output: Id<kinds::Field>,
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
    pub const fn output(&self) -> Id<kinds::Field> {
        self.output
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
}

/// Whether an occurrence reused the evaluation's accepted linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationState {
    /// The action publishes the primal that established this evaluation.
    Established,
    /// The derivative action reused the immutable state owned by the evaluation.
    Reused,
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
    receipt: ExecutionReceipt,
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
        self.receipt.operator()
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
    pub const fn primal_solve(&self) -> &SolveReport {
        self.receipt.report()
    }

    /// Exact deployment, operator, plan, output, and accepted-solve linkage.
    #[must_use]
    pub const fn receipt(&self) -> &ExecutionReceipt {
        &self.receipt
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
}

/// Opaque immutable differentiable program over one fixed input coordinate set.
#[derive(Debug, Clone)]
pub struct DifferentiableProgram {
    identity: DifferentiableProgramIdentity,
    plan: CommonScalarPlan,
    default: DifferentiableEvaluation,
}

impl DifferentiableProgram {
    /// Compile one exact common-Plan program and accept its default point.
    ///
    /// The current bounded slice admits one supplied-Cartesian scalar
    /// elliptic primary Field on host CPU `f64`. Selected Parameters become
    /// the program's ordered numerical inputs; all unselected Parameters stay
    /// frozen at their canonical Model values. Evaluating another point does
    /// not mutate or replace the Model or Plan.
    ///
    /// # Errors
    /// Returns structured identity, role, capability, primal-solve, output, or
    /// linearization diagnostics. No program is published before its primal
    /// relation has been independently accepted.
    pub fn compile(
        plan: CommonScalarPlan,
        inputs: &[ModelEntityRef<kinds::Parameter>],
        output: &ModelEntityRef<kinds::Field>,
    ) -> Result<Self, Vec<Diagnostic>> {
        if inputs.is_empty() {
            return Err(single(invalid(
                "differentiable program requires at least one selected Parameter",
            )));
        }
        let model_reference = plan.model_reference().map_err(single)?;
        if output.model != model_reference
            || inputs.iter().any(|input| input.model != model_reference)
        {
            return Err(single(invalid(
                "differentiable inputs and output must belong to the exact compiled Model artifact",
            )));
        }
        if inputs
            .iter()
            .enumerate()
            .any(|(index, input)| inputs[..index].iter().any(|seen| seen.id == input.id))
        {
            return Err(single(invalid(
                "differentiable program input selection contains a duplicate Parameter",
            )));
        }

        if !plan.fields().any(|(field, _)| field == output.id) {
            return Err(single(invalid(
                "selected output is not a scalar Field of this Plan",
            )));
        }
        let selected = inputs.iter().map(|input| input.id).collect::<Vec<_>>();
        let native = plan.differentiate(&selected, None).map_err(single)?;
        let relation = native.relation();
        let field_output = &native;
        let receipt = native
            .receipt()
            .expect("scalar Plan supplies a linear receipt");

        let residual_tolerance = receipt.report().residual_target();
        let accepted_linearization =
            AcceptedOutputLinearization::new(relation, field_output, residual_tolerance)
                .map_err(single)?;
        if relation.state_jacobian().agreement_fingerprint() != receipt.operator() {
            return Err(single(invalid(
                "accepted execution receipt differs from the paired linearized state system",
            )));
        }
        let primal_residual_norm = accepted_linearization.relation().primal_residual_norm();
        let identity = DifferentiableProgramIdentity {
            model: model_reference,
            plan_identity: plan.identity().to_owned(),
            inputs: selected,
            output: output.id,
            input_dimension: relation.parameter_dimension(),
            output_dimension: field_output.output_dimension(),
            scalar_type: DifferentiableScalarType::F64,
            device: DifferentiableDevice::HostCpu,
            derivative: DerivativeContract::ImplicitFirstOrder,
            solver: plan.linear(),
        };
        let point = DifferentiableParameterPoint {
            inputs: identity.inputs.clone(),
            values: relation.design_values().to_vec(),
        };
        let default = DifferentiableEvaluation {
            identity: identity.clone(),
            point,
            native,
            primal_residual_norm,
            residual_tolerance,
        };
        Ok(Self {
            identity,
            plan,
            default,
        })
    }

    /// Complete exact program identity.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Canonical Model values promoted into this program's input order.
    #[must_use]
    pub const fn default_point(&self) -> &DifferentiableParameterPoint {
        &self.default.point
    }

    /// Evaluate one complete finite Parameter point without changing program identity.
    ///
    /// The returned value owns the accepted primal, linearized relation,
    /// output projection, point values, and solve evidence. It is therefore
    /// safe to retain for a paired reverse action while the same program is
    /// evaluated concurrently at other points.
    ///
    /// # Errors
    /// Returns structured shape, value, capability, solve, or linearization
    /// diagnostics. An inadmissible point is rejected before publication.
    pub fn evaluate(
        &self,
        parameters: &[f64],
    ) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        if parameters.len() != self.identity.input_dimension
            || parameters.iter().any(|value| !value.is_finite())
        {
            return Err(single(invalid(format!(
                "differentiable Parameter point must contain {} finite values",
                self.identity.input_dimension
            ))));
        }
        if parameters
            .iter()
            .zip(self.default.point.values())
            .all(|(candidate, default)| candidate.to_bits() == default.to_bits())
        {
            return Ok(self.default.clone());
        }

        let native = self
            .plan
            .differentiate(&self.identity.inputs, Some(parameters))
            .map_err(single)?;
        let relation = native.relation();
        let output = &native;
        let receipt = native
            .receipt()
            .expect("scalar Plan supplies a linear receipt");
        if relation.design_values().len() != parameters.len()
            || parameters
                .iter()
                .zip(relation.design_values())
                .any(|(requested, accepted)| requested.to_bits() != accepted.to_bits())
        {
            return Err(single(invalid(
                "accepted linearized relation differs from the requested Parameter point",
            )));
        }
        if output.output_dimension() != self.identity.output_dimension {
            return Err(single(invalid(
                "Parameter-point output shape differs from the static differentiable program",
            )));
        }
        let residual_tolerance = receipt.report().residual_target();
        let accepted = AcceptedOutputLinearization::new(relation, output, residual_tolerance)
            .map_err(single)?;
        if relation.state_jacobian().agreement_fingerprint() != receipt.operator() {
            return Err(single(invalid(
                "Parameter-point execution receipt differs from its linearized state system",
            )));
        }
        Ok(DifferentiableEvaluation {
            identity: self.identity.clone(),
            point: DifferentiableParameterPoint {
                inputs: self.identity.inputs.clone(),
                values: relation.design_values().to_vec(),
            },
            primal_residual_norm: accepted.relation().primal_residual_norm(),
            residual_tolerance,
            native,
        })
    }

    /// Return the already accepted complete primary Field at the default point.
    #[must_use]
    pub fn primal(&self) -> DifferentiablePrimal {
        self.default.primal()
    }

    /// Apply the default-point total output JVP in exact selected-input order.
    ///
    /// # Errors
    /// Preserves shape, non-finite input, relation, and solver diagnostics.
    pub fn jvp(&self, tangent: &[f64]) -> Result<DifferentiableJvp, Diagnostic> {
        self.default.jvp(tangent)
    }

    /// Apply the default-point total output VJP in exact selected-input order.
    ///
    /// # Errors
    /// Preserves shape, non-finite input, output, relation, and transposed
    /// solver diagnostics.
    pub fn vjp(&self, cotangent: &[f64]) -> Result<DifferentiableVjp, Diagnostic> {
        self.default.vjp(cotangent)
    }
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
        let accepted = AcceptedOutputLinearization::new(
            self.native.relation(),
            &self.native,
            self.residual_tolerance,
        )?;
        let sensitivity = forward_output_sensitivity(
            &accepted,
            tangent,
            self.native.relation().state_jacobian().properties(),
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, self.identity.solver),
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
        let accepted = AcceptedOutputLinearization::new(
            self.native.relation(),
            &self.native,
            self.residual_tolerance,
        )?;
        let gradient = adjoint_output_gradient(
            &accepted,
            cotangent,
            self.native.relation().state_jacobian().properties(),
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, self.identity.solver),
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
            implementation: DerivativeImplementation::AnalyticAssembled,
            linearization_state,
            primal_residual_norm: self.primal_residual_norm,
            residual_tolerance: self.residual_tolerance,
            receipt: self
                .native
                .receipt()
                .expect("scalar Program owns a linear receipt")
                .clone(),
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
