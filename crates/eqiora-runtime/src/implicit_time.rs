use std::collections::HashMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id};
use eqiora_ir::{
    DifferentiationRole, DiscreteStepLinearization, LinearizedRelation, RelationCotangent,
    RelationTangent, ScalarLinearization, ScalarOperatorIr, SymbolicLinearityFailure,
};
use eqiora_schema::kernel::SymbolRef;
use eqiora_time::{
    ConstantDerivativeMatrixProof, DaeVariableKind, GeneralImplicitLoweringProof,
    GeneralImplicitReason, ImplicitDaeInitialization, ImplicitDaeProblem, ImplicitTimeSystem,
    InitialConditionPolicy, TimeLoweringProof,
};

use crate::CpuProgram;
use crate::time::{
    invalid_time, require_continuous_activation, require_finite, require_finite_slice, state_order,
};
use eqiora_sem::{KernelProgram, ReferenceConfig};

/// Canonical continuous Relation proven to require residual-native execution.
///
/// Unlike [`crate::FirstOrderProgram`], this projection retains both `y` and
/// `y_dot` as runtime inputs and exposes the paired residual JVP. It accepts
/// only a structural nonconstant/nonlinear derivative-Jacobian obstruction;
/// a valid constant first-order system must use the narrower projection.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralImplicitProgram {
    relation: Id<kinds::Relation>,
    operator: ScalarOperatorIr,
    state_fields: Vec<Id<kinds::Field>>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    kernel: KernelProgram,
    bindings: Vec<ImplicitBinding>,
    roles: Vec<DifferentiationRole>,
    proof: GeneralImplicitLoweringProof,
}

impl GeneralImplicitProgram {
    /// Lower one continuously activated Relation after proving that it cannot
    /// enter the constant first-order seam.
    ///
    /// # Errors
    /// Returns `EQ0705` for invalid activation/symbol/shape data, a pure
    /// algebraic Relation, a malformed derivative analysis, or a Relation that
    /// is already representable by [`crate::FirstOrderProgram`].
    pub fn lower(program: &CpuProgram, relation: Id<kinds::Relation>) -> Result<Self, Diagnostic> {
        require_continuous_activation(program, relation)?;
        let typed = program
            .kernel()
            .typed_relation_residual(relation)
            .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
        let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;
        let state_order = state_order(relation, &operator)?;
        if operator.residual_count() != state_order.fields.len() {
            return Err(invalid_time(
                relation,
                "general implicit system requires one residual equation per state Field",
            ));
        }

        let derivatives = state_order
            .fields
            .iter()
            .copied()
            .map(SymbolRef::Derivative)
            .collect::<Vec<_>>();
        let reason = match operator.constant_symbol_jacobian(&derivatives) {
            Err(SymbolicLinearityFailure::VariableCoefficient { .. }) => {
                GeneralImplicitReason::NonconstantDerivativeJacobian
            }
            Err(SymbolicLinearityFailure::Nonlinear { .. }) => {
                GeneralImplicitReason::NonlinearDerivativeDependence
            }
            Err(failure) => {
                return Err(invalid_time(
                    relation,
                    format!("general implicit derivative analysis failed: {failure:?}"),
                ));
            }
            Ok(jacobian) => {
                let matrix = ConstantDerivativeMatrixProof::new(
                    state_order.fields.len(),
                    jacobian.coefficients().to_vec(),
                )?;
                if matrix.exact_rank() == 0 {
                    return Err(invalid_time(
                        relation,
                        "general implicit time Relation has no structurally effective derivative",
                    ));
                }
                TimeLoweringProof::new(relation, state_order.fields.clone(), matrix)?;
                return Err(invalid_time(
                    relation,
                    "Relation has a valid constant first-order projection; use FirstOrderProgram",
                ));
            }
        };

        let variable_kinds = state_order
            .fields
            .iter()
            .map(|field| derivative_kind(relation, &operator, *field))
            .collect::<Result<Vec<_>, _>>()?;
        let proof = GeneralImplicitLoweringProof::new(
            relation,
            state_order.fields.clone(),
            variable_kinds,
            reason,
        )?;
        let bindings = bind_symbols(program, relation, &operator, &state_order.coordinates)?;
        Ok(Self {
            relation,
            operator,
            state_fields: state_order.fields,
            parameter_fields: bindings.parameter_fields,
            parameter_values: bindings.parameter_values,
            kernel: program.kernel().clone(),
            bindings: bindings.values,
            roles: bindings.roles,
            proof,
        })
    }

    /// Canonical Relation represented by this projection.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic state coordinate order.
    #[must_use]
    pub fn state_fields(&self) -> &[Id<kinds::Field>] {
        &self.state_fields
    }

    /// Solve fresh simultaneous initial equations with explicit numerical controls.
    /// Restart consumes an accepted pair without invoking this operation.
    ///
    /// # Errors
    /// Returns initialization diagnostics for missing, inconsistent, or unsupported conditions.
    pub fn initialize(
        &self,
        config: ReferenceConfig,
    ) -> Result<ImplicitDaeInitialization, Diagnostic> {
        super::initialization::initialize(&self.kernel, &self.state_fields, self.relation, config)
    }

    /// Deterministic first-occurrence Parameter order.
    #[must_use]
    pub fn parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        &self.parameter_fields
    }

    /// Revision-captured Parameter point.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.parameter_values
    }

    /// Structural witness behind residual-native admission.
    #[must_use]
    pub const fn lowering_proof(&self) -> &GeneralImplicitLoweringProof {
        &self.proof
    }

    /// Construct the residual-native problem and its consistency policy.
    ///
    /// # Errors
    /// Retains [`ImplicitDaeProblem`] validation diagnostics if its invariants
    /// change.
    pub fn implicit_problem(&self) -> Result<ImplicitDaeProblem<'_>, Diagnostic> {
        let initial = self.initialize(ReferenceConfig::new(0.0, 1.0)?)?;
        ImplicitDaeProblem::new(
            self,
            self.proof.variable_kinds().to_vec(),
            InitialConditionPolicy::Provided,
            initial.state().to_vec(),
            initial.derivative().to_vec(),
        )
    }

    /// Linearize one accepted implicit-Euler step as a discrete residual.
    ///
    /// The returned relation has `next_state` as its unknown vector. Its
    /// parameter vector is the explicit direct sum
    ///
    /// ```text
    /// [previous_state, canonical model Parameters]
    /// ```
    ///
    /// while model time and step size remain frozen realization data. The
    /// step residual is
    ///
    /// ```text
    /// G(y_next; y_previous, p)
    ///   = F(next_time, y_next, (y_next - y_previous) / step, p).
    /// ```
    ///
    /// # Errors
    /// Returns `EQ0705` for invalid time/step/state data and retains Operator
    /// IR linearization diagnostics for an invalid point.
    pub fn linearize_implicit_euler_step(
        &self,
        previous_time: f64,
        next_time: f64,
        previous_state: &[f64],
        next_state: &[f64],
    ) -> Result<ImplicitEulerStepLinearization<'_>, Diagnostic> {
        let dimension = self.state_fields.len();
        let step = next_time - previous_time;
        let inverse_step = 1.0 / step;
        if !previous_time.is_finite()
            || !next_time.is_finite()
            || !step.is_finite()
            || step <= 0.0
            || !previous_time.is_finite()
            || previous_time >= next_time
            || !inverse_step.is_finite()
            || previous_state.len() != dimension
            || next_state.len() != dimension
        {
            return Err(invalid_time(
                self.relation,
                "implicit-Euler step linearization requires advancing finite time, positive representable step, and exact state shapes",
            ));
        }
        require_finite_slice(
            self.relation,
            previous_state,
            "implicit-Euler previous state",
        )?;
        require_finite_slice(self.relation, next_state, "implicit-Euler next state")?;
        let derivative = next_state
            .iter()
            .zip(previous_state)
            .map(|(next, previous)| (next - previous) / step)
            .collect::<Vec<_>>();
        require_finite_slice(
            self.relation,
            &derivative,
            "implicit-Euler discrete derivative",
        )?;
        let inputs = self.inputs(next_time, next_state, &derivative);
        let roles = self
            .bindings
            .iter()
            .map(|binding| match binding {
                ImplicitBinding::State(_) | ImplicitBinding::Derivative(_) => {
                    DifferentiationRole::Unknown
                }
                ImplicitBinding::Parameter(_) => DifferentiationRole::Parameter,
                ImplicitBinding::Time => DifferentiationRole::Frozen,
            })
            .collect::<Vec<_>>();
        let inner = self.operator.linearize(&inputs, &roles)?;
        Ok(ImplicitEulerStepLinearization {
            relation: self.relation,
            inner,
            bindings: &self.bindings,
            state_fields: &self.state_fields,
            state_dimension: dimension,
            previous_time,
            next_time,
            step,
            inverse_step,
            model_parameter_fields: &self.parameter_fields,
            model_parameter_values: &self.parameter_values,
            previous_state: previous_state.to_vec(),
            next_state: next_state.to_vec(),
        })
    }

    fn inputs(&self, time: f64, state: &[f64], derivative: &[f64]) -> Vec<f64> {
        self.bindings
            .iter()
            .map(|binding| match *binding {
                ImplicitBinding::State(coordinate) => state[coordinate],
                ImplicitBinding::Derivative(coordinate) => derivative[coordinate],
                ImplicitBinding::Parameter(coordinate) => self.parameter_values[coordinate],
                ImplicitBinding::Time => time,
            })
            .collect()
    }

    fn require_shape(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        state_direction: Option<&[f64]>,
        derivative_direction: Option<&[f64]>,
        output: &[f64],
    ) -> Result<(), Diagnostic> {
        let dimension = self.state_fields.len();
        if !time.is_finite()
            || state.len() != dimension
            || derivative.len() != dimension
            || output.len() != dimension
            || state_direction.is_some_and(|direction| direction.len() != dimension)
            || derivative_direction.is_some_and(|direction| direction.len() != dimension)
        {
            return Err(invalid_time(
                self.relation,
                "general implicit action requires finite time and exact state/derivative shapes",
            ));
        }
        require_finite_slice(self.relation, state, "general implicit state")?;
        require_finite_slice(self.relation, derivative, "general implicit derivative")?;
        if let Some(direction) = state_direction {
            require_finite_slice(self.relation, direction, "general implicit state direction")?;
        }
        if let Some(direction) = derivative_direction {
            require_finite_slice(
                self.relation,
                direction,
                "general implicit derivative direction",
            )?;
        }
        Ok(())
    }
}

/// Linearized discrete residual for one accepted implicit-Euler step.
///
/// This is a projection of the canonical residual linearization, not a
/// derivative of the Newton iterations used to find the step. Unknown
/// cotangents are projected to `y_next`; parameter cotangents are ordered as
/// `[y_previous, canonical model Parameters]`.
#[derive(Debug)]
pub struct ImplicitEulerStepLinearization<'a> {
    relation: Id<kinds::Relation>,
    inner: ScalarLinearization<'a>,
    bindings: &'a [ImplicitBinding],
    state_fields: &'a [Id<kinds::Field>],
    state_dimension: usize,
    previous_time: f64,
    next_time: f64,
    step: f64,
    inverse_step: f64,
    model_parameter_fields: &'a [Id<kinds::Parameter>],
    model_parameter_values: &'a [f64],
    previous_state: Vec<f64>,
    next_state: Vec<f64>,
}

impl ImplicitEulerStepLinearization<'_> {
    /// Canonical state order used by both step state blocks.
    #[must_use]
    pub const fn state_fields(&self) -> &[Id<kinds::Field>] {
        self.state_fields
    }

    /// Number of leading parameter coordinates occupied by `y_previous`.
    #[must_use]
    pub const fn previous_state_parameter_dimension(&self) -> usize {
        self.state_dimension
    }

    /// Canonical model Parameters following `y_previous` in first-occurrence
    /// order.
    #[must_use]
    pub const fn model_parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        self.model_parameter_fields
    }

    /// Frozen model time at the accepted start of the step.
    #[must_use]
    pub const fn previous_time(&self) -> f64 {
        self.previous_time
    }

    /// Frozen model time at the accepted end of the step.
    #[must_use]
    pub const fn next_time(&self) -> f64 {
        self.next_time
    }

    /// Frozen implicit-Euler step size.
    #[must_use]
    pub const fn step(&self) -> f64 {
        self.step
    }

    fn validate_tangent(
        &self,
        unknown: Option<&[f64]>,
        parameter: Option<&[f64]>,
    ) -> Result<(), Diagnostic> {
        if unknown.is_some_and(|values| values.len() != self.state_dimension)
            || parameter.is_some_and(|values| values.len() != self.parameter_dimension())
        {
            return Err(invalid_step_linearization(
                self.relation,
                "implicit-Euler step tangent does not match next-state and [previous-state, model-Parameter] dimensions",
            ));
        }
        if unknown.is_some_and(|values| values.iter().any(|value| !value.is_finite()))
            || parameter.is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(invalid_step_linearization(
                self.relation,
                "implicit-Euler step tangent must contain only finite values",
            ));
        }
        Ok(())
    }

    fn projected_jvp_inputs(
        &self,
        unknown: Option<&[f64]>,
        parameter: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut inner_unknown = Vec::with_capacity(self.inner.unknown_dimension());
        let mut inner_parameter = Vec::with_capacity(self.inner.parameter_dimension());
        for binding in self.bindings {
            match *binding {
                ImplicitBinding::State(coordinate) => {
                    inner_unknown.push(unknown.map_or(0.0, |values| values[coordinate]));
                }
                ImplicitBinding::Derivative(coordinate) => {
                    let next = unknown.map_or(0.0, |values| values[coordinate]);
                    let previous = parameter.map_or(0.0, |values| values[coordinate]);
                    inner_unknown.push((next - previous) * self.inverse_step);
                }
                ImplicitBinding::Parameter(coordinate) => {
                    inner_parameter.push(
                        parameter.map_or(0.0, |values| values[self.state_dimension + coordinate]),
                    );
                }
                ImplicitBinding::Time => {}
            }
        }
        debug_assert_eq!(inner_unknown.len(), self.inner.unknown_dimension());
        debug_assert_eq!(inner_parameter.len(), self.inner.parameter_dimension());
        (inner_unknown, inner_parameter)
    }
}

impl LinearizedRelation<f64> for ImplicitEulerStepLinearization<'_> {
    fn unknown_dimension(&self) -> usize {
        self.state_dimension
    }

    fn parameter_dimension(&self) -> usize {
        self.state_dimension + self.model_parameter_fields.len()
    }

    fn residual_dimension(&self) -> usize {
        self.inner.residual_dimension()
    }

    fn primal(&self, residual: &mut [f64]) -> Result<(), Diagnostic> {
        self.inner.primal(residual)
    }

    fn jvp(
        &self,
        tangent: RelationTangent<'_, f64>,
        residual_tangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let (unknown, parameter) = match tangent {
            RelationTangent::Unknown(unknown) => (Some(unknown), None),
            RelationTangent::Parameter(parameter) => (None, Some(parameter)),
            RelationTangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        self.validate_tangent(unknown, parameter)?;
        let (inner_unknown, inner_parameter) = self.projected_jvp_inputs(unknown, parameter);
        self.inner.jvp(
            RelationTangent::Both {
                unknown: &inner_unknown,
                parameter: &inner_parameter,
            },
            residual_tangent,
        )
    }

    fn vjp(
        &self,
        residual_cotangent: &[f64],
        cotangent: RelationCotangent<'_, f64>,
    ) -> Result<(), Diagnostic> {
        let (mut unknown, mut parameter) = match cotangent {
            RelationCotangent::Unknown(unknown) => (Some(unknown), None),
            RelationCotangent::Parameter(parameter) => (None, Some(parameter)),
            RelationCotangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        if unknown
            .as_deref()
            .is_some_and(|values| values.len() != self.state_dimension)
            || parameter
                .as_deref()
                .is_some_and(|values| values.len() != self.parameter_dimension())
        {
            return Err(invalid_step_linearization(
                self.relation,
                "implicit-Euler step cotangent does not match next-state and [previous-state, model-Parameter] dimensions",
            ));
        }

        let mut inner_unknown = vec![0.0; self.inner.unknown_dimension()];
        let mut inner_parameter = vec![0.0; self.inner.parameter_dimension()];
        self.inner.vjp(
            residual_cotangent,
            RelationCotangent::Both {
                unknown: &mut inner_unknown,
                parameter: &mut inner_parameter,
            },
        )?;
        if let Some(values) = unknown.as_deref_mut() {
            values.fill(0.0);
        }
        if let Some(values) = parameter.as_deref_mut() {
            values.fill(0.0);
        }

        let mut inner_unknown_coordinate = 0usize;
        let mut inner_parameter_coordinate = 0usize;
        for binding in self.bindings {
            match *binding {
                ImplicitBinding::State(coordinate) => {
                    if let Some(values) = unknown.as_deref_mut() {
                        values[coordinate] += inner_unknown[inner_unknown_coordinate];
                    }
                    inner_unknown_coordinate += 1;
                }
                ImplicitBinding::Derivative(coordinate) => {
                    let value = inner_unknown[inner_unknown_coordinate] * self.inverse_step;
                    if let Some(values) = unknown.as_deref_mut() {
                        values[coordinate] += value;
                    }
                    if let Some(values) = parameter.as_deref_mut() {
                        values[coordinate] -= value;
                    }
                    inner_unknown_coordinate += 1;
                }
                ImplicitBinding::Parameter(coordinate) => {
                    if let Some(values) = parameter.as_deref_mut() {
                        values[self.state_dimension + coordinate] +=
                            inner_parameter[inner_parameter_coordinate];
                    }
                    inner_parameter_coordinate += 1;
                }
                ImplicitBinding::Time => {}
            }
        }
        debug_assert_eq!(inner_unknown_coordinate, inner_unknown.len());
        debug_assert_eq!(inner_parameter_coordinate, inner_parameter.len());
        if unknown
            .as_deref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
            || parameter
                .as_deref()
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(invalid_step_linearization(
                self.relation,
                "implicit-Euler step VJP produced a non-finite value",
            ));
        }
        Ok(())
    }
}

impl DiscreteStepLinearization for ImplicitEulerStepLinearization<'_> {
    fn state_fields(&self) -> &[Id<kinds::Field>] {
        self.state_fields
    }

    fn model_parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        self.model_parameter_fields
    }

    fn model_parameter_values(&self) -> &[f64] {
        self.model_parameter_values
    }

    fn previous_state(&self) -> &[f64] {
        &self.previous_state
    }

    fn next_state(&self) -> &[f64] {
        &self.next_state
    }

    fn start_time(&self) -> f64 {
        self.previous_time
    }

    fn end_time(&self) -> f64 {
        self.next_time
    }

    fn previous_state_parameter_dimension(&self) -> usize {
        self.state_dimension
    }
}

fn invalid_step_linearization(
    relation: Id<kinds::Relation>,
    message: impl Into<String>,
) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message).with_graph_path(GraphPath::new([
        "implicit-euler-step".to_owned(),
        "relation".to_owned(),
        relation.to_string(),
    ]))
}

fn derivative_kind(
    relation: Id<kinds::Relation>,
    operator: &ScalarOperatorIr,
    field: Id<kinds::Field>,
) -> Result<DaeVariableKind, Diagnostic> {
    match operator.constant_symbol_jacobian(&[SymbolRef::Derivative(field)]) {
        Ok(jacobian) => Ok(
            if jacobian
                .coefficients()
                .iter()
                .any(|coefficient| *coefficient != 0.0)
            {
                DaeVariableKind::Differential
            } else {
                DaeVariableKind::Algebraic
            },
        ),
        Err(
            SymbolicLinearityFailure::VariableCoefficient { .. }
            | SymbolicLinearityFailure::Nonlinear { .. },
        ) => Ok(DaeVariableKind::Differential),
        Err(failure) => Err(invalid_time(
            relation,
            format!("general implicit derivative partition failed: {failure:?}"),
        )),
    }
}

impl ImplicitTimeSystem for GeneralImplicitProgram {
    fn dimension(&self) -> usize {
        self.state_fields.len()
    }

    fn residual(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_shape(time, state, derivative, None, None, output)?;
        let residual = self
            .operator
            .evaluate(&self.inputs(time, state, derivative))?;
        output.copy_from_slice(&residual);
        require_finite_slice(self.relation, output, "general implicit residual")
    }

    fn residual_jvp(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        state_direction: &[f64],
        derivative_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_shape(
            time,
            state,
            derivative,
            Some(state_direction),
            Some(derivative_direction),
            output,
        )?;
        let inputs = self.inputs(time, state, derivative);
        let linearization = self.operator.linearize(&inputs, &self.roles)?;
        let tangent = self
            .bindings
            .iter()
            .filter_map(|binding| match *binding {
                ImplicitBinding::State(coordinate) => Some(state_direction[coordinate]),
                ImplicitBinding::Derivative(coordinate) => Some(derivative_direction[coordinate]),
                ImplicitBinding::Parameter(_) | ImplicitBinding::Time => None,
            })
            .collect::<Vec<_>>();
        linearization.jvp(RelationTangent::Unknown(&tangent), output)?;
        require_finite_slice(self.relation, output, "general implicit residual JVP")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ImplicitBinding {
    State(usize),
    Derivative(usize),
    Parameter(usize),
    Time,
}

struct ImplicitBindings {
    values: Vec<ImplicitBinding>,
    roles: Vec<DifferentiationRole>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
}

fn bind_symbols(
    program: &CpuProgram,
    relation: Id<kinds::Relation>,
    operator: &ScalarOperatorIr,
    state_coordinates: &HashMap<Id<kinds::Field>, usize>,
) -> Result<ImplicitBindings, Diagnostic> {
    let mut values = Vec::with_capacity(operator.symbols().len());
    let mut roles = Vec::with_capacity(operator.symbols().len());
    let mut parameter_fields = Vec::new();
    let mut parameter_values = Vec::new();
    let mut parameter_coordinates = HashMap::new();
    for symbol in operator.symbols().iter().copied() {
        let (binding, role) = match symbol {
            SymbolRef::Field(field) => (
                ImplicitBinding::State(state_coordinate(relation, state_coordinates, field)?),
                DifferentiationRole::Unknown,
            ),
            SymbolRef::Derivative(field) => (
                ImplicitBinding::Derivative(state_coordinate(relation, state_coordinates, field)?),
                DifferentiationRole::Unknown,
            ),
            SymbolRef::Parameter(parameter) => {
                let coordinate = if let Some(coordinate) = parameter_coordinates.get(&parameter) {
                    *coordinate
                } else {
                    let value = program
                        .kernel()
                        .value(parameter.erase())
                        .ok_or_else(|| invalid_time(relation, "Parameter has no bound value"))?
                        .value();
                    require_finite(relation, value, "Parameter value")?;
                    let coordinate = parameter_values.len();
                    parameter_coordinates.insert(parameter, coordinate);
                    parameter_fields.push(parameter);
                    parameter_values.push(value);
                    coordinate
                };
                (
                    ImplicitBinding::Parameter(coordinate),
                    DifferentiationRole::Frozen,
                )
            }
            SymbolRef::Time => (ImplicitBinding::Time, DifferentiationRole::Frozen),
            SymbolRef::Pre(_) | SymbolRef::Next(_) | SymbolRef::Port(_) => {
                return Err(invalid_time(
                    relation,
                    "general implicit lowering admits only state, derivative, Parameter, and time symbols",
                ));
            }
            _ => {
                return Err(invalid_time(
                    relation,
                    "Relation symbol is newer than general implicit lowering",
                ));
            }
        };
        values.push(binding);
        roles.push(role);
    }
    Ok(ImplicitBindings {
        values,
        roles,
        parameter_fields,
        parameter_values,
    })
}

fn state_coordinate(
    relation: Id<kinds::Relation>,
    coordinates: &HashMap<Id<kinds::Field>, usize>,
    field: Id<kinds::Field>,
) -> Result<usize, Diagnostic> {
    coordinates.get(&field).copied().ok_or_else(|| {
        invalid_time(
            relation,
            "Field is absent from general implicit state order",
        )
    })
}

#[cfg(test)]
mod tests {
    use eqiora_schema::kernel::ExprDagBuilder;

    use super::*;

    #[test]
    fn variable_partition_uses_effective_derivative_dependence() {
        let relation = Id::<kinds::Relation>::new();
        let field = Id::<kinds::Field>::new();

        let mut expression = ExprDagBuilder::new();
        let derivative = expression.symbol(SymbolRef::Derivative(field)).unwrap();
        let canceled = expression.sub(derivative, derivative).unwrap();
        let state = expression.symbol(SymbolRef::Field(field)).unwrap();
        let residual = expression.add(canceled, state).unwrap();
        let operator = ScalarOperatorIr::lower(&expression.finish([residual]).unwrap()).unwrap();
        assert_eq!(
            derivative_kind(relation, &operator, field).unwrap(),
            DaeVariableKind::Algebraic
        );

        let mut expression = ExprDagBuilder::new();
        let derivative = expression.symbol(SymbolRef::Derivative(field)).unwrap();
        let state = expression.symbol(SymbolRef::Field(field)).unwrap();
        let residual = expression.mul(derivative, state).unwrap();
        let operator = ScalarOperatorIr::lower(&expression.finish([residual]).unwrap()).unwrap();
        assert_eq!(
            derivative_kind(relation, &operator, field).unwrap(),
            DaeVariableKind::Differential
        );
    }
}
