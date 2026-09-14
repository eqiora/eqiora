use std::collections::{BTreeSet, HashMap, HashSet};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId};
use eqiora_graph::EdgeKind;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::{ActivationKind, EventDirection, KernelNode, SymbolRef};
use eqiora_time::{
    EventFlowLinearization, EventGuardLinearization, EventResetLinearization,
    RegisteredRootProblem, RootFunctions, RootProposal, RootRegistrationId, RootRegistrationProof,
    TimeEquationClass, TimeSystem, TransversalEventLinearization,
};

mod validation;
use validation::*;

use crate::{CpuProgram, FirstOrderProgram};

/// Canonical explicit-ODE event group lowered to one differentiable reset.
///
/// An event group is selected by one Activation. Every Event Activation with
/// the same structural guard and direction is then included automatically, so
/// split reset Relations commit and differentiate as one transition. This
/// first seam admits one flow Relation and a constant full-monomial implicit
/// `Next` Jacobian. Coupled reset solves, DAE events, distinct coincident
/// guards, and mode-dependent post-event flow remain explicit future classes.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalEventProgram {
    flow: FirstOrderProgram,
    activations: Vec<Id<kinds::Activation>>,
    direction: EventDirection,
    guard: BoundOperator,
    guard_dimension: DimExponents,
    resets: Vec<BoundOperator>,
    parameters: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    next_projection: MonomialNextProjection,
}

impl CanonicalEventProgram {
    /// Lower one continuous explicit ODE and the complete structural event
    /// group containing `event`.
    ///
    /// # Errors
    /// Returns `EQ0705` unless the flow is an explicit ODE, the selected node
    /// is an Event Activation, every structurally identical activation is
    /// included, reset symbols are limited to `Pre`/`Next`/Parameter/time, and
    /// the combined implicit reset has one constant monomial `Next` equation
    /// per flow state.
    pub fn lower(
        program: &CpuProgram,
        flow: Id<kinds::Relation>,
        event: Id<kinds::Activation>,
    ) -> Result<Self, Diagnostic> {
        let flow = FirstOrderProgram::lower(program, flow)?;
        if flow.equation_class() != TimeEquationClass::ExplicitOde {
            return Err(invalid_event(
                event.erase(),
                "hybrid differentiation currently requires an explicit ODE flow",
            ));
        }
        let (guard_expression, direction) = match program.kernel().node(event.erase()) {
            Some(KernelNode::Activation(activation)) => match activation.kind() {
                ActivationKind::Event { guard, direction } => (guard, *direction),
                _ => {
                    return Err(invalid_event(
                        event.erase(),
                        "selected Activation is not an event",
                    ));
                }
            },
            _ => {
                return Err(invalid_event(
                    event.erase(),
                    "selected event Activation is absent from the kernel",
                ));
            }
        };

        let mut activations = program
            .kernel()
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Activation(activation) => match activation.kind() {
                    ActivationKind::Event {
                        guard,
                        direction: candidate,
                    } if guard == guard_expression && *candidate == direction => {
                        Some(activation.id())
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        activations.sort_by_key(|activation| activation.erase());

        let activation_ids = activations
            .iter()
            .map(|activation| activation.erase())
            .collect::<BTreeSet<_>>();
        let relation_ids = program
            .kernel()
            .edges()
            .iter()
            .filter(|edge| {
                edge.kind() == EdgeKind::Activates && activation_ids.contains(&edge.from())
            })
            .map(|edge| edge.to())
            .collect::<BTreeSet<_>>();
        if relation_ids.is_empty() {
            return Err(invalid_event(
                event.erase(),
                "event group activates no reset Relation",
            ));
        }

        let typed_guard = program
            .kernel()
            .typed_event_guard(event)
            .map_err(|diagnostics| {
                invalid_event(
                    event.erase(),
                    diagnostics.first().map_or(
                        "event guard has no accepted scalar type",
                        Diagnostic::message,
                    ),
                )
            })?;
        let guard_dimension = typed_guard
            .node_type(guard_expression.roots()[0])
            .expect("semantic guard typing includes its root")
            .dimension();
        let guard_operator = ScalarOperatorIr::lower(guard_expression)?;
        validate_guard_symbols(event.erase(), &guard_operator, flow.state_fields())?;
        let reset_operators = relation_ids
            .iter()
            .map(|relation| {
                let operator = program.operator(*relation).ok_or_else(|| {
                    invalid_event(*relation, "reset Relation has no scalar Operator IR")
                })?;
                validate_reset_symbols(*relation, operator, flow.state_fields())?;
                Ok((*relation, operator.clone()))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;

        let mut parameters = flow.parameter_fields().to_vec();
        append_operator_parameters(&mut parameters, &guard_operator);
        for (_, operator) in &reset_operators {
            append_operator_parameters(&mut parameters, operator);
        }
        let parameter_values = parameters
            .iter()
            .map(|parameter| {
                let value = program
                    .kernel()
                    .value(parameter.erase())
                    .ok_or_else(|| {
                        invalid_event(parameter.erase(), "event Parameter has no bound value")
                    })?
                    .value();
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(invalid_event(
                        parameter.erase(),
                        "event Parameter value must be finite",
                    ))
                }
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let state_coordinates = flow
            .state_fields()
            .iter()
            .enumerate()
            .map(|(coordinate, field)| (*field, coordinate))
            .collect::<HashMap<_, _>>();
        let parameter_coordinates = parameters
            .iter()
            .enumerate()
            .map(|(coordinate, parameter)| (*parameter, coordinate))
            .collect::<HashMap<_, _>>();

        let guard = BoundOperator::guard(
            event.erase(),
            guard_operator,
            &state_coordinates,
            &parameter_coordinates,
        )?;
        let resets = reset_operators
            .into_iter()
            .map(|(relation, operator)| {
                BoundOperator::reset(
                    relation,
                    operator,
                    &state_coordinates,
                    &parameter_coordinates,
                )
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let next_projection =
            MonomialNextProjection::prove(event.erase(), &resets, flow.state_fields())?;

        Ok(Self {
            flow,
            activations,
            direction,
            guard,
            guard_dimension,
            resets,
            parameters,
            parameter_values,
            next_projection,
        })
    }

    /// Continuous flow projection used on both sides of this first event seam.
    #[must_use]
    pub const fn flow(&self) -> &FirstOrderProgram {
        &self.flow
    }

    /// Structurally identical Event Activations included in the atomic group.
    #[must_use]
    pub fn activations(&self) -> &[Id<kinds::Activation>] {
        &self.activations
    }

    /// Canonical zero-crossing direction.
    #[must_use]
    pub const fn direction(&self) -> EventDirection {
        self.direction
    }

    /// Physical dimension of the normalized real scalar zero-crossing guard.
    ///
    /// Inferred by the immutable semantic Model's existing activation type contract.
    #[must_use]
    pub const fn guard_dimension(&self) -> DimExponents {
        self.guard_dimension
    }

    /// Selected Parameter order shared by guard, reset, and sensitivity data.
    #[must_use]
    pub fn parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        &self.parameters
    }

    /// Revision-captured values in [`Self::parameter_fields`] order.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.parameter_values
    }

    fn guard_value(&self, time: f64, state: &[f64]) -> Result<f64, Diagnostic> {
        if !time.is_finite()
            || state.len() != self.flow.state_fields().len()
            || state.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_event(
                self.activations[0].erase(),
                "root evaluation requires finite time/state and exact state shape",
            ));
        }
        let inputs = self.guard.inputs(time, state, &[], &self.parameter_values);
        let residual = self.guard.operator.evaluate(&inputs)?;
        residual.first().copied().ok_or_else(|| {
            invalid_event(
                self.activations[0].erase(),
                "canonical Event guard produced no scalar residual",
            )
        })
    }

    /// Linearize the canonical guard, implicit reset, and pre/post flow at one
    /// localized event point.
    ///
    /// `guard_tolerance` is explicit localization evidence, not a hidden
    /// derivative tolerance. The transversality test itself rejects only an
    /// exactly grazing guard and enforces the canonical crossing direction.
    ///
    /// # Errors
    /// Returns `EQ0704`/`EQ0705` for invalid state shape, a point outside the
    /// guard band, a direction mismatch, grazing, or non-finite actions.
    pub fn linearize_at(
        &self,
        time: f64,
        pre_state: &[f64],
        guard_tolerance: f64,
    ) -> Result<CanonicalEventLinearization, Diagnostic> {
        let state_dimension = self.flow.state_fields().len();
        if !time.is_finite()
            || !guard_tolerance.is_finite()
            || guard_tolerance < 0.0
            || pre_state.len() != state_dimension
            || pre_state.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_event(
                self.activations[0].erase(),
                "event point requires finite time/state, exact state shape, and a non-negative finite guard tolerance",
            ));
        }

        let zero_post = vec![0.0; state_dimension];
        let guard = self.guard.derivatives(
            time,
            pre_state,
            &zero_post,
            &self.parameter_values,
            state_dimension,
            self.parameters.len(),
        )?;
        if guard.residual.len() != 1 || guard.residual[0].abs() > guard_tolerance {
            return Err(invalid_event(
                self.activations[0].erase(),
                "event linearization point lies outside the declared guard-localization tolerance",
            ));
        }
        let guard_residual = guard.residual[0];

        let reset_at_zero = self.reset_residuals(time, pre_state, &zero_post)?;
        let post_state = self.next_projection.solve(&reset_at_zero)?;
        let reset_derivatives = self.reset_derivatives(time, pre_state, &post_state)?;
        let reset_state = self
            .next_projection
            .project_jacobian(&reset_derivatives.state, state_dimension)?;
        let reset_parameter = self
            .next_projection
            .project_jacobian(&reset_derivatives.parameter, self.parameters.len())?;
        let reset_time = self
            .next_projection
            .project_vector(&reset_derivatives.time)?;

        let mut before_flow = vec![0.0; state_dimension];
        let mut after_flow = vec![0.0; state_dimension];
        self.flow.rhs(time, pre_state, &mut before_flow)?;
        self.flow.rhs(time, &post_state, &mut after_flow)?;
        let transversality = guard.time[0]
            + guard.state[..state_dimension]
                .iter()
                .zip(&before_flow)
                .map(|(gradient, flow)| gradient * flow)
                .sum::<f64>();
        match self.direction {
            EventDirection::Rising if transversality <= 0.0 => {
                return Err(invalid_event(
                    self.activations[0].erase(),
                    "localized event does not cross in the canonical rising direction",
                ));
            }
            EventDirection::Falling if transversality >= 0.0 => {
                return Err(invalid_event(
                    self.activations[0].erase(),
                    "localized event does not cross in the canonical falling direction",
                ));
            }
            EventDirection::Any | EventDirection::Rising | EventDirection::Falling => {}
        }

        let flow = EventFlowLinearization::new(before_flow, after_flow)?;
        let guard = EventGuardLinearization::new(guard.state, guard.parameter, guard.time[0])?;
        let reset = EventResetLinearization::new(
            state_dimension,
            self.parameters.len(),
            reset_state,
            reset_parameter,
            reset_time,
        )?;
        let derivatives = TransversalEventLinearization::new(flow, guard, reset)?;
        Ok(CanonicalEventLinearization {
            time,
            guard_residual,
            pre_state: pre_state.to_vec(),
            post_state,
            derivatives,
        })
    }

    fn reset_residuals(
        &self,
        time: f64,
        pre_state: &[f64],
        post_state: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        let mut residuals = Vec::new();
        for reset in &self.resets {
            let inputs = reset.inputs(time, pre_state, post_state, &self.parameter_values);
            residuals.extend(reset.operator.evaluate(&inputs)?);
        }
        Ok(residuals)
    }

    fn reset_derivatives(
        &self,
        time: f64,
        pre_state: &[f64],
        post_state: &[f64],
    ) -> Result<OperatorDerivatives, Diagnostic> {
        let state_dimension = self.flow.state_fields().len();
        let parameter_dimension = self.parameters.len();
        let mut combined = OperatorDerivatives::empty();
        for reset in &self.resets {
            combined.append(reset.derivatives(
                time,
                pre_state,
                post_state,
                &self.parameter_values,
                state_dimension,
                parameter_dimension,
            )?);
        }
        Ok(combined)
    }
}

/// Canonical root callbacks reconstructed in one registration's proof order.
///
/// This is the only runtime bridge from a backend-local root index to Event
/// Activations. Construction independently proves that the registration is a
/// complete partition of the program's Event Activations and that every group
/// lowers to the exact structural guard represented by its callback slot.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalRootSet {
    registration: RootRegistrationId,
    proof: RootRegistrationProof,
    events: Vec<CanonicalEventProgram>,
}

impl CanonicalRootSet {
    /// Rebuild all root callbacks from canonical semantics and registration
    /// proof, in proof order.
    ///
    /// # Errors
    /// Returns `EQ0705` if the proof omits, repeats, splits, or combines Event
    /// Activations, or any group cannot lower against the selected ODE flow.
    pub fn lower(
        program: &CpuProgram,
        flow: Id<kinds::Relation>,
        registration: RootRegistrationId,
        proof: RootRegistrationProof,
    ) -> Result<Self, Diagnostic> {
        let canonical_activations = program
            .kernel()
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Activation(activation)
                    if matches!(activation.kind(), ActivationKind::Event { .. }) =>
                {
                    Some(activation.id())
                }
                _ => None,
            })
            .collect::<HashSet<_>>();
        let registered_activations = proof
            .groups()
            .iter()
            .flat_map(|group| group.activations().iter().copied())
            .collect::<HashSet<_>>();
        if registered_activations != canonical_activations {
            return Err(invalid_event(
                flow.erase(),
                "root registration is not a complete partition of canonical Event Activations",
            ));
        }

        let events = proof
            .groups()
            .iter()
            .map(|group| {
                let event = CanonicalEventProgram::lower(program, flow, group.representative())?;
                if event.activations() != group.activations() {
                    return Err(invalid_event(
                        group.representative().erase(),
                        "root registration group differs from its canonical structural guard group",
                    ));
                }
                Ok(event)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(Self {
            registration,
            proof,
            events,
        })
    }

    /// Content-addressed identity carried by every backend proposal.
    #[must_use]
    pub const fn registration(&self) -> RootRegistrationId {
        self.registration
    }

    /// Canonical root callback order and atomic Activation grouping.
    #[must_use]
    pub const fn proof(&self) -> &RootRegistrationProof {
        &self.proof
    }

    /// Canonical event programs in the exact registration proof/root-index order.
    ///
    /// Each program exposes its real flow, guard, and reset Parameter columns.
    #[must_use]
    pub fn events(&self) -> &[CanonicalEventProgram] {
        &self.events
    }

    /// Borrow this callback set through the backend-neutral registered seam.
    ///
    /// # Errors
    /// Returns `EQ0705` only if internal callback/proof shape was corrupted.
    pub fn root_problem(&self) -> Result<RegisteredRootProblem<'_>, Diagnostic> {
        RegisteredRootProblem::new(self.registration, self.proof.clone(), self)
    }

    /// Validate one registered numerical proposal and lower its canonical
    /// reset, event-time derivative, and saltation operator.
    ///
    /// # Errors
    /// Returns `EQ0705` for registration/index/equation drift or the selected
    /// event's existing direction, localization, reset, or transversality
    /// diagnostics.
    pub fn linearize_proposal(
        &self,
        proposal: &RootProposal,
        guard_tolerance: f64,
    ) -> Result<CanonicalEventLinearization, Diagnostic> {
        if proposal.registration() != self.registration
            || proposal.report().equation_class() != TimeEquationClass::ExplicitOde
        {
            return Err(invalid_event(
                self.proof.groups()[0].representative().erase(),
                "root proposal registration or equation class differs from the canonical root set",
            ));
        }
        let event = self.events.get(proposal.root_index()).ok_or_else(|| {
            invalid_event(
                self.proof.groups()[0].representative().erase(),
                "root proposal index is absent from the canonical registration",
            )
        })?;
        event.linearize_at(proposal.time(), proposal.state(), guard_tolerance)
    }
}

impl RootFunctions for CanonicalRootSet {
    fn count(&self) -> usize {
        self.events.len()
    }

    fn evaluate(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != self.events.len() {
            return Err(invalid_event(
                self.proof.groups()[0].representative().erase(),
                "root callback output shape differs from the canonical registration",
            ));
        }
        for (value, event) in output.iter_mut().zip(&self.events) {
            *value = event.guard_value(time, state)?;
        }
        Ok(())
    }
}

/// One localized canonical event point and its lowered derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalEventLinearization {
    time: f64,
    guard_residual: f64,
    pre_state: Vec<f64>,
    post_state: Vec<f64>,
    derivatives: TransversalEventLinearization,
}

impl CanonicalEventLinearization {
    /// Localized model time.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Guard residual at the admitted localization point.
    #[must_use]
    pub const fn guard_residual(&self) -> f64 {
        self.guard_residual
    }

    /// State immediately before the atomic reset.
    #[must_use]
    pub fn pre_state(&self) -> &[f64] {
        &self.pre_state
    }

    /// State solved from the grouped implicit reset Relations.
    #[must_use]
    pub fn post_state(&self) -> &[f64] {
        &self.post_state
    }

    /// Event-time, reset, and saltation linearization.
    #[must_use]
    pub const fn derivatives(&self) -> &TransversalEventLinearization {
        &self.derivatives
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BoundOperator {
    owner: RawId,
    operator: ScalarOperatorIr,
    bindings: Vec<InputBinding>,
    roles: Vec<DifferentiationRole>,
    unknown_sources: Vec<UnknownSource>,
    parameter_sources: Vec<usize>,
}

impl BoundOperator {
    fn guard(
        owner: RawId,
        operator: ScalarOperatorIr,
        states: &HashMap<Id<kinds::Field>, usize>,
        parameters: &HashMap<Id<kinds::Parameter>, usize>,
    ) -> Result<Self, Diagnostic> {
        Self::bind(owner, operator, states, parameters, false)
    }

    fn reset(
        owner: RawId,
        operator: ScalarOperatorIr,
        states: &HashMap<Id<kinds::Field>, usize>,
        parameters: &HashMap<Id<kinds::Parameter>, usize>,
    ) -> Result<Self, Diagnostic> {
        Self::bind(owner, operator, states, parameters, true)
    }

    fn bind(
        owner: RawId,
        operator: ScalarOperatorIr,
        states: &HashMap<Id<kinds::Field>, usize>,
        parameters: &HashMap<Id<kinds::Parameter>, usize>,
        reset: bool,
    ) -> Result<Self, Diagnostic> {
        let mut bindings = Vec::with_capacity(operator.symbols().len());
        let mut roles = Vec::with_capacity(operator.symbols().len());
        let mut unknown_sources = Vec::new();
        let mut parameter_sources = Vec::new();
        for symbol in operator.symbols().iter().copied() {
            let (binding, role) = match symbol {
                SymbolRef::Field(field) if !reset => {
                    let coordinate = state_coordinate(owner, states, field)?;
                    unknown_sources.push(UnknownSource::State(coordinate));
                    (InputBinding::Pre(coordinate), DifferentiationRole::Unknown)
                }
                SymbolRef::Pre(field) if reset => {
                    let coordinate = state_coordinate(owner, states, field)?;
                    unknown_sources.push(UnknownSource::State(coordinate));
                    (InputBinding::Pre(coordinate), DifferentiationRole::Unknown)
                }
                SymbolRef::Next(field) if reset => (
                    InputBinding::Post(state_coordinate(owner, states, field)?),
                    DifferentiationRole::Frozen,
                ),
                SymbolRef::Parameter(parameter) => {
                    let coordinate = parameters.get(&parameter).copied().ok_or_else(|| {
                        invalid_event(owner, "Parameter is absent from event coordinate order")
                    })?;
                    parameter_sources.push(coordinate);
                    (
                        InputBinding::Parameter(coordinate),
                        DifferentiationRole::Parameter,
                    )
                }
                SymbolRef::Time => {
                    unknown_sources.push(UnknownSource::Time);
                    (InputBinding::Time, DifferentiationRole::Unknown)
                }
                _ => {
                    return Err(invalid_event(
                        owner,
                        "event operator contains a symbol outside its admitted guard/reset form",
                    ));
                }
            };
            bindings.push(binding);
            roles.push(role);
        }
        Ok(Self {
            owner,
            operator,
            bindings,
            roles,
            unknown_sources,
            parameter_sources,
        })
    }

    fn inputs(
        &self,
        time: f64,
        pre_state: &[f64],
        post_state: &[f64],
        parameters: &[f64],
    ) -> Vec<f64> {
        self.bindings
            .iter()
            .map(|binding| match *binding {
                InputBinding::Pre(coordinate) => pre_state[coordinate],
                InputBinding::Post(coordinate) => post_state[coordinate],
                InputBinding::Parameter(coordinate) => parameters[coordinate],
                InputBinding::Time => time,
            })
            .collect()
    }

    fn derivatives(
        &self,
        time: f64,
        pre_state: &[f64],
        post_state: &[f64],
        parameters: &[f64],
        state_dimension: usize,
        parameter_dimension: usize,
    ) -> Result<OperatorDerivatives, Diagnostic> {
        let inputs = self.inputs(time, pre_state, post_state, parameters);
        let linearization = self.operator.linearize(&inputs, &self.roles)?;
        let residual_dimension = self.operator.residual_count();
        let mut residual = vec![0.0; residual_dimension];
        linearization.primal(&mut residual)?;
        let mut state = vec![0.0; residual_dimension * state_dimension];
        let mut time_derivative = vec![0.0; residual_dimension];
        let mut parameter = vec![0.0; residual_dimension * parameter_dimension];

        for coordinate in 0..state_dimension {
            let tangent = self
                .unknown_sources
                .iter()
                .map(|source| usize::from(*source == UnknownSource::State(coordinate)) as f64)
                .collect::<Vec<_>>();
            let mut action = vec![0.0; residual_dimension];
            linearization.jvp(RelationTangent::Unknown(&tangent), &mut action)?;
            for (row, value) in action.into_iter().enumerate() {
                state[row * state_dimension + coordinate] = value;
            }
        }
        let tangent = self
            .unknown_sources
            .iter()
            .map(|source| usize::from(*source == UnknownSource::Time) as f64)
            .collect::<Vec<_>>();
        linearization.jvp(RelationTangent::Unknown(&tangent), &mut time_derivative)?;

        for coordinate in 0..parameter_dimension {
            let tangent = self
                .parameter_sources
                .iter()
                .map(|source| usize::from(*source == coordinate) as f64)
                .collect::<Vec<_>>();
            let mut action = vec![0.0; residual_dimension];
            linearization.jvp(RelationTangent::Parameter(&tangent), &mut action)?;
            for (row, value) in action.into_iter().enumerate() {
                parameter[row * parameter_dimension + coordinate] = value;
            }
        }
        Ok(OperatorDerivatives {
            residual,
            state,
            parameter,
            time: time_derivative,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputBinding {
    Pre(usize),
    Post(usize),
    Parameter(usize),
    Time,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnknownSource {
    State(usize),
    Time,
}

struct OperatorDerivatives {
    residual: Vec<f64>,
    state: Vec<f64>,
    parameter: Vec<f64>,
    time: Vec<f64>,
}

impl OperatorDerivatives {
    fn empty() -> Self {
        Self {
            residual: Vec::new(),
            state: Vec::new(),
            parameter: Vec::new(),
            time: Vec::new(),
        }
    }

    fn append(&mut self, other: Self) {
        self.residual.extend(other.residual);
        self.state.extend(other.state);
        self.parameter.extend(other.parameter);
        self.time.extend(other.time);
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MonomialNextProjection {
    owner: RawId,
    row_for_state: Vec<usize>,
    coefficients: Vec<f64>,
}

impl MonomialNextProjection {
    fn prove(
        owner: RawId,
        resets: &[BoundOperator],
        states: &[Id<kinds::Field>],
    ) -> Result<Self, Diagnostic> {
        let next_symbols = states
            .iter()
            .copied()
            .map(SymbolRef::Next)
            .collect::<Vec<_>>();
        let mut matrix = Vec::new();
        let mut rows = 0usize;
        for reset in resets {
            let jacobian = reset
                .operator
                .constant_symbol_jacobian(&next_symbols)
                .map_err(|failure| next_structure_error(reset.owner, failure))?;
            rows += jacobian.row_count();
            matrix.extend_from_slice(jacobian.coefficients());
        }
        if rows != states.len() {
            return Err(invalid_event(
                owner,
                "event reset group requires one residual equation per flow state",
            ));
        }
        let mut row_for_state = vec![usize::MAX; states.len()];
        let mut coefficients = vec![0.0; states.len()];
        for (row, values) in matrix.chunks_exact(states.len()).enumerate() {
            let nonzero = values
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
                .collect::<Vec<_>>();
            if nonzero.len() != 1 {
                return Err(invalid_event(
                    owner,
                    "event reset Next Jacobian must be full monomial in the first hybrid seam",
                ));
            }
            let (state, coefficient) = nonzero[0];
            if row_for_state[state] != usize::MAX {
                return Err(invalid_event(
                    owner,
                    "event reset Next Jacobian assigns one state more than once",
                ));
            }
            row_for_state[state] = row;
            coefficients[state] = *coefficient;
        }
        if row_for_state.contains(&usize::MAX) {
            return Err(invalid_event(
                owner,
                "event reset Next Jacobian leaves a flow state undefined",
            ));
        }
        Ok(Self {
            owner,
            row_for_state,
            coefficients,
        })
    }

    fn solve(&self, residual_at_zero: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        if residual_at_zero.len() != self.row_for_state.len() {
            return Err(invalid_event(
                self.owner,
                "event reset residual shape changed after lowering",
            ));
        }
        let values = self
            .row_for_state
            .iter()
            .enumerate()
            .map(|(state, row)| -residual_at_zero[*row] / self.coefficients[state])
            .collect::<Vec<_>>();
        if values.iter().all(|value| value.is_finite()) {
            Ok(values)
        } else {
            Err(invalid_event(
                self.owner,
                "implicit event reset produced a non-finite state",
            ))
        }
    }

    fn project_vector(&self, residual_derivative: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        if residual_derivative.len() != self.row_for_state.len() {
            return Err(invalid_event(
                self.owner,
                "event reset derivative vector has an invalid shape",
            ));
        }
        Ok(self
            .row_for_state
            .iter()
            .enumerate()
            .map(|(state, row)| -residual_derivative[*row] / self.coefficients[state])
            .collect())
    }

    fn project_jacobian(
        &self,
        residual_jacobian: &[f64],
        columns: usize,
    ) -> Result<Vec<f64>, Diagnostic> {
        if self.row_for_state.len().checked_mul(columns) != Some(residual_jacobian.len()) {
            return Err(invalid_event(
                self.owner,
                "event reset derivative matrix has an invalid shape",
            ));
        }
        let mut result = vec![0.0; residual_jacobian.len()];
        for (state, row) in self.row_for_state.iter().copied().enumerate() {
            for column in 0..columns {
                result[state * columns + column] =
                    -residual_jacobian[row * columns + column] / self.coefficients[state];
            }
        }
        if result.iter().all(|value| value.is_finite()) {
            Ok(result)
        } else {
            Err(invalid_event(
                self.owner,
                "event reset derivative projection produced a non-finite value",
            ))
        }
    }
}
