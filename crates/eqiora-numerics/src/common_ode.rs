//! Model-first admission for non-spatial canonical explicit ODEs.

use std::sync::Arc;

use crate::common_time::{
    CommonTimeRunRequest, CommonTimeState, TimeStateSpace, digest, dimension_bytes, push,
};
use eqiora_artifact::{CanonicalModelArtifact, ModelEnvelope, TimeLoweringEnvelopeV3};
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id};
use eqiora_runtime::{CpuProgram, FirstOrderProgram};
use eqiora_schema::kernel::{ActivationKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_time::{
    InitialConditionPolicy, TimeBackendIdentity, TimeEquationClass, TimeMethod, TimePlan,
    TimeProblem,
};

mod checkpoint;
mod controls;
use crate::CommonTimeCheckpoint;
pub(crate) use checkpoint::AdaptiveCheckpoint;
mod hermitian;
mod history;
mod midpoint;
mod norm;
pub(crate) use norm::ConservedNorm;
mod events;
mod forward_policy;
pub(crate) use forward_policy::{CommonForwardSensitivity, CommonSensitivityTolerance};
pub(crate) mod parameter_system;
mod run;
pub(crate) use events::{CommonEventPolicy, CommonGuardTolerance};

mod sensitivity;

/// ODE method and numerical controls before Model admission.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonOdePolicy {
    method: TimeMethod,
    initial_step_s: f64,
    relative_tolerance: f64,
    absolute_tolerances: Vec<(eqiora_core::TimeStateCoordinate, f64)>,
    events: Option<CommonEventPolicy>,
    forward_sensitivities: Option<CommonForwardSensitivity>,
    pub(crate) conserved_norms: Vec<ConservedNorm>,
    pub(crate) hermitian_parameters: Vec<Id<kinds::Parameter>>,
}

impl CommonOdePolicy {
    /// Construct finite positive controls with `(State coordinate, SI tolerance)`
    /// pairs. This owner validates values and duplicates; Plan resolution checks
    /// complete coverage of the Model's admitted coordinates.
    pub fn new(
        method: TimeMethod,
        initial_step_s: f64,
        relative_tolerance: f64,
        mut absolute_tolerances: Vec<(eqiora_core::TimeStateCoordinate, f64)>,
    ) -> Result<Self, Diagnostic> {
        if !matches!(
            method,
            TimeMethod::Tsitouras45 | TimeMethod::ImplicitMidpoint
        ) {
            return Err(invalid(
                "common ODE policy requires Tsitouras45 or ImplicitMidpoint",
            ));
        }
        require_positive(initial_step_s, "ODE step_s")?;
        require_positive(relative_tolerance, "ODE relative_tolerance")?;
        if absolute_tolerances.is_empty() {
            return Err(invalid(
                "Tsitouras45 requires one exact Field-bound absolute tolerance per state",
            ));
        }
        for (_, value) in &absolute_tolerances {
            require_positive(*value, "ODE absolute tolerances")?;
        }
        absolute_tolerances.sort_by_key(|entry| {
            (
                entry.0.field().ulid(),
                entry.0.derivative_order(),
                entry.0.component(),
                entry.0.is_imaginary(),
            )
        });
        if absolute_tolerances
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(invalid(
                "ODE absolute tolerances contain a duplicate exact state coordinate",
            ));
        }
        Ok(Self {
            method,
            initial_step_s,
            relative_tolerance,
            absolute_tolerances,
            events: None,
            forward_sensitivities: None,
            conserved_norms: Vec::new(),
            hermitian_parameters: Vec::new(),
        })
    }

    pub(crate) fn validate_backend(
        &self,
        capabilities: eqiora_time::TimeBackendCapabilities,
    ) -> Result<(), Diagnostic> {
        if self.method() == TimeMethod::ImplicitMidpoint
            && capabilities.identity() != eqiora_time::ImplicitMidpointTimeBackend::IDENTITY
        {
            return Err(invalid(
                "implicit midpoint requires its available host backend identity",
            ));
        }
        Ok(())
    }

    /// Exact numerical integration method selected by this policy.
    #[must_use]
    pub const fn method(&self) -> TimeMethod {
        self.method
    }

    /// Initial adaptive step-size guess in coherent SI seconds.
    #[must_use]
    pub const fn initial_step_s(&self) -> f64 {
        self.initial_step_s
    }

    /// Relative local-error tolerance (Tsitouras) or Newton correction tolerance (midpoint).
    #[must_use]
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Canonically coordinate-ordered absolute tolerances.
    #[must_use]
    pub fn absolute_tolerances(&self) -> &[(eqiora_core::TimeStateCoordinate, f64)] {
        &self.absolute_tolerances
    }
}

/// Opaque no-Mesh Plan for one structurally proven ODE.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonOdePlan {
    model: Arc<ModelEnvelope>,
    program: FirstOrderProgram,
    temporal: CommonOdePolicy,
    ordered_absolute_tolerances: Vec<f64>,
    ordered_guard_tolerances: Vec<eqiora_core::DynQuantity>,
    forward_sensitivity_plan: Option<eqiora_time::ForwardSensitivityPlan>,
    forward_parameter_ids: Option<Vec<Id<kinds::Parameter>>>,
    state_space: Arc<TimeStateSpace>,
    identity: String,
    lowering_digest: String,
    model_id: String,
    model_revision: u64,
    backend: TimeBackendIdentity,
    pub(crate) nonlinear: Option<eqiora_realization::NonlinearSolvePlan>,
    pub(crate) linear: Option<crate::numerical_admission::NativeLinearPolicy>,
}

impl CommonOdePlan {
    pub(crate) fn state_space(&self) -> &TimeStateSpace {
        &self.state_space
    }

    pub(crate) fn system(&self) -> &FirstOrderProgram {
        &self.program
    }

    pub(crate) fn model_artifact(&self) -> &ModelEnvelope {
        &self.model
    }

    /// Resolve one exact Model through canonical first-order structural lowering.
    pub fn resolve(
        model: &ModelEnvelope,
        kernel: &KernelProgram,
        temporal: CommonOdePolicy,
        capabilities: eqiora_time::TimeBackendCapabilities,
        solve: Option<crate::CommonSolvePolicy>,
        linear_backend: &dyn eqiora_solver::LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        let linear = match (temporal.method(), solve) {
            (TimeMethod::ImplicitMidpoint, Some(solve)) => {
                let request = match solve {
                    crate::CommonSolvePolicy::Linear(linear)
                    | crate::CommonSolvePolicy::Newton { linear, .. } => linear,
                };
                Some(crate::numerical_admission::solver_planning::resolve_linear(
                    request,
                    eqiora_solver::LinearOperatorProperties::General,
                    None,
                    None,
                    None,
                    linear_backend,
                )?)
            }
            (TimeMethod::Tsitouras45, None) => None,
            (TimeMethod::ImplicitMidpoint, None) => {
                return Err(invalid(
                    "implicit midpoint requires an explicit Linear or Newton solve policy",
                ));
            }
            _ => {
                return Err(invalid(
                    "explicit time integration does not consume a solve policy",
                ));
            }
        };
        let backend = capabilities.identity();
        temporal.validate_backend(capabilities)?;
        if temporal.method() == TimeMethod::ImplicitMidpoint
            && (temporal.events().is_some() || temporal.forward_sensitivities().is_some())
        {
            return Err(invalid(
                "implicit midpoint event and forward-sensitivity execution is not implemented",
            ));
        }
        let reference = model.artifact_reference()?;
        let flow = events::flow(kernel, temporal.events())?;
        let program = FirstOrderProgram::lower(kernel, flow)?;
        for coordinate in program.state_coordinates() {
            let Some(KernelNode::Field(field)) = kernel.node(coordinate.field().erase()) else {
                return Err(invalid("time coordinate has no exact Field type"));
            };
            capabilities.admit_real_coordinates(
                field.value_type().scalar_domain(),
                eqiora_core::ScalarType::F64,
            )?;
        }
        let ordered_guard_tolerances = if let Some(policy) = temporal.events() {
            let cpu = CpuProgram::lower(kernel).map_err(|diagnostics| {
                diagnostics.into_iter().next().unwrap_or_else(|| {
                    invalid("canonical explicit-ODE lowering failed without a diagnostic")
                })
            })?;
            let roots = events::roots(model, kernel, &cpu, &program, policy)?;
            roots
                .events()
                .iter()
                .map(|event| {
                    policy
                        .guard_tolerances()
                        .iter()
                        .find(|entry| entry.activation() == event.activations()[0])
                        .map(CommonGuardTolerance::quantity)
                        .ok_or_else(|| invalid("root group omits its representative tolerance"))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let admitted = match temporal.method() {
            TimeMethod::Tsitouras45 => program.equation_class() == TimeEquationClass::ExplicitOde,
            TimeMethod::ImplicitMidpoint => matches!(
                program.equation_class(),
                TimeEquationClass::ExplicitOde
                    | TimeEquationClass::MassMatrix {
                        rank: eqiora_time::MassMatrixRank::Full
                    }
            ),
            _ => false,
        };
        if !admitted {
            return Err(invalid(
                "selected ODE method does not admit this equation class",
            ));
        }
        if program.initial_condition_policy() != InitialConditionPolicy::Provided {
            return Err(invalid(
                "explicit-ODE resolution requires complete Model-owned initial values",
            ));
        }
        let state_coordinates = program.state_coordinates();
        let ordered_absolute_tolerances =
            temporal.ordered_absolute_tolerances(state_coordinates)?;
        let lowering = TimeLoweringEnvelopeV3::from_proof(model, kernel, program.lowering_proof())?;
        let lowering_digest = lowering.digest()?.to_string();
        let model_id = reference.model().ulid().to_string();
        let model_revision = reference.semantic_revision().get();

        let mut state_space = Vec::new();
        push(&mut state_space, lowering_digest.as_bytes());
        push(&mut state_space, b"declared-conserved-norms/v1");
        state_space.extend_from_slice(&(temporal.conserved_norms.len() as u64).to_be_bytes());
        for norm in &temporal.conserved_norms {
            push(&mut state_space, &norm.identity_bytes());
        }
        push(&mut state_space, b"declared-hermitian-parameters/v1");
        state_space.extend_from_slice(&(temporal.hermitian_parameters.len() as u64).to_be_bytes());
        for parameter in &temporal.hermitian_parameters {
            push(&mut state_space, parameter.ulid().to_string().as_bytes());
        }
        let state_space = Arc::new(TimeStateSpace::new(
            model,
            state_coordinates.to_vec(),
            &state_space,
        )?);

        let mut identity = state_space.identity().as_bytes().to_vec();
        identity.extend_from_slice(&temporal.initial_step_s().to_bits().to_be_bytes());
        identity.extend_from_slice(&temporal.relative_tolerance().to_bits().to_be_bytes());
        for tolerance in &ordered_absolute_tolerances {
            identity.extend_from_slice(&tolerance.to_bits().to_be_bytes());
        }
        if let Some(policy) = temporal.events() {
            push(&mut identity, &policy.identity_bytes());
        }
        if let Some(policy) = temporal.forward_sensitivities() {
            push(&mut identity, &policy.identity_bytes());
        }
        push(
            &mut identity,
            match temporal.method() {
                TimeMethod::Tsitouras45 => b"tsitouras45",
                TimeMethod::ImplicitMidpoint => b"implicit-midpoint",
                _ => unreachable!("validated common ODE method"),
            },
        );
        push(&mut identity, backend.id().as_bytes());
        push(&mut identity, backend.version().as_bytes());
        push(&mut identity, b"host-serial");
        if let Some(solve) = solve {
            push(
                &mut identity,
                &crate::numerical_admission::plan_artifact::solve_intent_bytes(solve)?,
            );
        }
        if let Some(linear) = &linear {
            push(
                &mut identity,
                &crate::numerical_admission::plan_artifact::solve_intent_bytes(
                    crate::CommonSolvePolicy::Linear(crate::CommonLinearRequest::exact(
                        linear.solver,
                        linear.provider,
                    )?),
                )?,
            );
        }
        let identity = digest(b"eqiora.common-ode-plan/v3\0", &identity);

        let mut plan = Self {
            model: Arc::new(model.clone()),
            program,
            temporal,
            ordered_absolute_tolerances,
            ordered_guard_tolerances,
            forward_sensitivity_plan: None,
            forward_parameter_ids: None,
            state_space,
            identity,
            lowering_digest,
            model_id,
            model_revision,
            backend,
            nonlinear: match solve {
                Some(crate::CommonSolvePolicy::Newton { nonlinear, .. }) => Some(nonlinear),
                _ => None,
            },
            linear,
        };
        plan.admit_forward_policy(kernel)?;
        plan.admit_hermitian_parameters(kernel)?;
        plan.admit_conserved_norms(kernel)?;
        Ok(plan)
    }

    /// Exact Plan identity excluding Run horizon and output schedule.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn model_digest(&self) -> &str {
        self.state_space.model_digest()
    }

    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        self.model_revision
    }

    #[must_use]
    pub fn lowering_digest(&self) -> &str {
        &self.lowering_digest
    }

    #[must_use]
    pub const fn temporal(&self) -> &CommonOdePolicy {
        &self.temporal
    }

    pub fn state_coordinates(
        &self,
    ) -> impl ExactSizeIterator<Item = eqiora_core::TimeStateCoordinate> + '_ {
        self.state_space.coordinates().iter().copied()
    }

    #[must_use]
    pub fn state_dimensions(&self) -> &[DimExponents] {
        self.state_space.dimensions()
    }

    #[must_use]
    pub fn state_space_identity(&self) -> &str {
        self.state_space.identity()
    }

    /// Equation class structurally admitted by this Plan's time method.
    #[must_use]
    pub const fn equation_class(&self) -> TimeEquationClass {
        self.program.equation_class()
    }

    /// Exact time backend selected by resolution.
    #[must_use]
    pub const fn backend(&self) -> TimeBackendIdentity {
        self.backend
    }

    /// Construct the exact Model-owned initial state at the explicit timeline coordinate.
    pub fn initial_state(&self, time_s: f64) -> Result<CommonTimeState, Diagnostic> {
        self.model.artifact_reference()?;
        self.accept_state(
            time_s,
            self.program
                .initialize(time_s, eqiora_sem::ReferenceConfig::new(0.0, 1.0)?)?
                .state()
                .to_vec(),
            "initial",
        )
    }

    /// Reauthenticate one complete physical time State against this exact Plan.
    /// # Errors
    /// Rejects foreign layouts, malformed artifacts and violated conserved norms.
    pub fn state_from_bytes(&self, bytes: &[u8]) -> Result<CommonTimeState, Diagnostic> {
        let state = CommonTimeState::from_bytes(bytes, self.state_space.clone())?;
        self.check_conserved_norms(state.values())?;
        Ok(state)
    }

    pub(crate) fn accept_state(
        &self,
        time_s: f64,
        values: Vec<f64>,
        source_kind: &'static str,
    ) -> Result<CommonTimeState, Diagnostic> {
        let state = CommonTimeState::new(self.state_space.clone(), time_s, values, source_kind)?;
        self.check_conserved_norms(state.values())?;
        Ok(state)
    }

    pub(crate) fn run_time_plan(
        &self,
        state: &CommonTimeState,
        execution_times: Vec<f64>,
    ) -> Result<TimePlan, Diagnostic> {
        let plan = TimePlan::new(
            self.temporal.method(),
            state.time_s(),
            self.temporal.initial_step_s(),
            self.temporal.relative_tolerance(),
            self.ordered_absolute_tolerances.clone(),
            execution_times,
        )?;
        plan.validate_for(&self.problem(state)?)?;
        Ok(plan)
    }

    pub(crate) fn problem<'a>(
        &'a self,
        state: &CommonTimeState,
    ) -> Result<TimeProblem<'a>, Diagnostic> {
        if state.state_space_identity() != self.state_space_identity() {
            return Err(invalid(
                "State belongs to a different exact no-Mesh ODE state space",
            ));
        }
        self.check_conserved_norms(state.values())?;
        TimeProblem::new(
            &self.program,
            self.program.equation_class(),
            InitialConditionPolicy::Provided,
            state.values().to_vec(),
        )
    }
}

fn require_positive(value: f64, label: &str) -> Result<(), Diagnostic> {
    if !value.is_finite() || value <= 0.0 || value.to_bits() == (-0.0_f64).to_bits() {
        return Err(invalid(format!(
            "{label} must be finite and strictly positive"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod higher_order_tests;
#[cfg(test)]
mod tests;
