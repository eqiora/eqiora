mod operator;
use operator::TimeOperator;
mod coordinates;
mod generator;
pub(crate) use coordinates::state_order;

use std::collections::HashMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id};
use eqiora_graph::EdgeKind;
use eqiora_ir::{
    ConstantSymbolJacobian, DifferentiationRole, LinearizedRelation, RelationTangent,
    ScalarOperatorIr, SymbolicLinearityFailure,
};
use eqiora_schema::kernel::{ActivationKind, KernelNode, SymbolRef};
use eqiora_time::{
    ConstantDerivativeMatrixProof, ForwardSensitivityProblem, ImplicitDaeInitialization,
    InitialConditionPolicy, MassParameterDependence, ParametricTimeSystem, TimeEquationClass,
    TimeLoweringProof, TimeProblem, TimeSystem,
};

use eqiora_sem::{KernelProgram, ReferenceConfig};

/// Canonical continuous Relation proven to have first-order form
/// `M y_dot = f(t,y)`.
///
/// State order follows first occurrence of current-value or derivative Field
/// coordinates in Operator IR. A full constant monomial derivative Jacobian
/// is normalized to an explicit ODE. Every other non-zero-rank constant matrix
/// remains a full or rank-deficient mass matrix. State-dependent and
/// derivative-nonlinear systems fail closed; equation class is never inferred
/// from sample evaluations or a floating-point rank threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstOrderProgram {
    relation: Id<kinds::Relation>,
    operator: TimeOperator,
    state_coordinates: Vec<eqiora_core::TimeStateCoordinate>,
    companions: Vec<(usize, usize)>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    parameter_coordinates: Vec<eqiora_ir::ScalarSymbolCoordinate>,
    kernel: KernelProgram,
    bindings: Vec<TimeBinding>,
    roles: Vec<DifferentiationRole>,
    state_symbol_coordinates: Vec<usize>,
    proof: TimeLoweringProof,
    projection: FirstOrderProjection,
}

impl FirstOrderProgram {
    /// Lower one continuously activated Relation from its validated kernel
    /// Model and prove an admitted first-order equation class.
    ///
    /// # Errors
    /// Returns `EQ0705` if activation, symbols, shapes, or
    /// derivative structure cannot enter the first-order seam. Existing
    /// Operator IR diagnostics are retained when scalar evaluation fails.
    pub fn lower(
        program: &KernelProgram,
        relation: Id<kinds::Relation>,
    ) -> Result<Self, Diagnostic> {
        require_continuous_activation(program, relation)?;
        let typed = program
            .typed_relation_residual(relation)
            .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
        let operator = TimeOperator::lower(&typed)?;

        let state_order = coordinates::component_state_order(program, relation, &operator)?;
        let derivatives = state_order.rate_symbols();
        let coefficients = operator.derivative_coefficients(relation, &derivatives)?;
        let matrix = state_order.derivative_matrix(relation, &coefficients)?;
        let classified = classify_first_order(relation, matrix, &state_order.state_coordinates)?;

        let time_bindings = bind_symbols(program, relation, &operator, &state_order.coordinates)?;
        Ok(Self {
            relation,
            operator,
            state_coordinates: state_order.state_coordinates,
            companions: state_order.companions,
            parameter_fields: time_bindings.parameter_fields,
            parameter_values: time_bindings.parameter_values,
            parameter_coordinates: time_bindings.parameter_coordinates,
            kernel: program.clone(),
            bindings: time_bindings.values,
            roles: time_bindings.roles,
            state_symbol_coordinates: time_bindings.state_coordinates,
            proof: classified.proof,
            projection: classified.projection,
        })
    }

    /// Canonical Relation represented by this projection.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic state coordinate order.
    #[must_use]
    pub fn state_coordinates(&self) -> &[eqiora_core::TimeStateCoordinate] {
        &self.state_coordinates
    }

    /// Solve fresh simultaneous initial equations before the first activation.
    /// Numerical tolerances and guesses belong to the supplied execution configuration.
    /// Restart callers use their accepted State directly instead of this operation.
    ///
    /// # Errors
    /// Returns initialization diagnostics for missing, inconsistent, or unsupported conditions,
    /// including a locally singular or high-index constant-mass constraint block.
    pub fn initialize(
        &self,
        initial_time: f64,
        config: ReferenceConfig,
    ) -> Result<ImplicitDaeInitialization, Diagnostic> {
        let initial = super::initialization::initialize(
            &self.kernel,
            &self.state_coordinates,
            self.relation,
            initial_time,
            config,
        )?;
        super::initialization::require_constant_mass_regularity(
            &self.kernel,
            &self.state_coordinates,
            self.relation,
            initial_time,
            &initial,
            self.proof.derivative_matrix(),
        )?;
        Ok(initial)
    }

    /// Deterministic first-occurrence order of bound Parameter symbols.
    #[must_use]
    pub fn parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        &self.parameter_fields
    }

    /// Complete real coordinates of bound Parameters in derivative-vector order.
    #[must_use]
    pub fn parameter_coordinates(&self) -> &[eqiora_ir::ScalarSymbolCoordinate] {
        &self.parameter_coordinates
    }

    /// Revision-captured Parameter values in [`Self::parameter_coordinates`] order.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.parameter_values
    }

    /// Exact Operator-IR witness behind equation-class admission.
    #[must_use]
    pub const fn lowering_proof(&self) -> &TimeLoweringProof {
        &self.proof
    }

    /// Structurally proven equation class.
    #[must_use]
    pub const fn equation_class(&self) -> TimeEquationClass {
        self.proof.equation_class()
    }

    /// Initial-condition meaning required by the proven equation class.
    #[must_use]
    pub const fn initial_condition_policy(&self) -> InitialConditionPolicy {
        self.proof.initial_condition_policy()
    }

    /// Construct the sole backend-neutral time problem from this projection.
    ///
    /// # Errors
    /// Retains `TimeProblem` validation diagnostics if its invariants change.
    pub fn time_problem(&self) -> Result<TimeProblem<'_>, Diagnostic> {
        TimeProblem::new(
            self,
            self.equation_class(),
            self.initial_condition_policy(),
            self.initialize(0.0, ReferenceConfig::new(0.0, 1.0)?)?
                .state()
                .to_vec(),
        )
    }

    /// Construct the parameter-JVP problem at an explicit initial time from the same proven projection.
    ///
    /// # Errors
    /// Retains `ForwardSensitivityProblem` validation diagnostics when the
    /// Relation has no Parameter symbols or its invariants change.
    pub fn forward_sensitivity_problem(
        &self,
        initial_time: f64,
    ) -> Result<ForwardSensitivityProblem<'_>, Diagnostic> {
        self.require_parameter_independent_initial_conditions(initial_time)?;
        ForwardSensitivityProblem::new(
            self,
            self.equation_class(),
            self.initial_condition_policy(),
            self.initialize(initial_time, ReferenceConfig::new(0.0, 1.0)?)?
                .state()
                .to_vec(),
        )
    }

    fn require_parameter_independent_initial_conditions(
        &self,
        initial_time: f64,
    ) -> Result<(), Diagnostic> {
        super::initialization::require_zero_parameter_tangent(
            &self.kernel,
            &self.state_coordinates,
            &self.parameter_fields,
            self.relation,
            initial_time,
        )
    }

    fn inputs(&self, time: f64, state: &[f64]) -> Vec<f64> {
        self.bindings
            .iter()
            .map(|binding| match *binding {
                TimeBinding::State(coordinate) => state[coordinate],
                TimeBinding::DerivativeZero => 0.0,
                TimeBinding::Parameter(coordinate) => self.parameter_values[coordinate],
                TimeBinding::Time => time,
            })
            .collect()
    }

    fn write_rhs(&self, residual: &[f64], output: &mut [f64]) {
        match &self.projection {
            FirstOrderProjection::Explicit {
                residual_rows,
                derivative_scales,
            } => {
                for state in 0..self.state_coordinates.len() {
                    output[state] = -residual[residual_rows[state]] / derivative_scales[state];
                }
            }
            FirstOrderProjection::MassMatrix { .. } => {
                for (output, residual) in output.iter_mut().zip(residual) {
                    *output = -*residual;
                }
            }
        }
    }

    fn require_action_shape(
        &self,
        time: f64,
        state: &[f64],
        direction: Option<&[f64]>,
        output: &[f64],
    ) -> Result<(), Diagnostic> {
        let dimension = self.dimension();
        if !time.is_finite()
            || state.len() != dimension
            || output.len() != dimension
            || direction.is_some_and(|direction| direction.len() != dimension)
        {
            return Err(invalid_time(
                self.relation,
                "first-order action requires finite time and exact state-vector shapes",
            ));
        }
        require_finite_slice(self.relation, state, "first-order state")?;
        if let Some(direction) = direction {
            require_finite_slice(self.relation, direction, "first-order state direction")?;
        }
        Ok(())
    }

    fn require_parameter_action_shape(
        &self,
        time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &[f64],
    ) -> Result<(), Diagnostic> {
        if !time.is_finite()
            || state.len() != self.dimension()
            || output.len() != self.dimension()
            || parameter_direction.len() != self.parameter_values.len()
        {
            return Err(invalid_time(
                self.relation,
                "first-order Parameter action requires finite time and exact state/Parameter shapes",
            ));
        }
        require_finite_slice(self.relation, state, "first-order state")?;
        require_finite_slice(
            self.relation,
            parameter_direction,
            "first-order Parameter direction",
        )
    }
}

impl ParametricTimeSystem for FirstOrderProgram {
    fn parameter_dimension(&self) -> usize {
        self.parameter_values.len()
    }

    fn parameters(&self) -> &[f64] {
        &self.parameter_values
    }

    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        // Constant-symbol Jacobian proof rejects a derivative coefficient that
        // contains any Parameter symbol, so this is a lowering fact rather
        // than a backend assumption.
        MassParameterDependence::Independent
    }

    fn rhs_parameter_jvp(
        &self,
        time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_parameter_action_shape(time, state, parameter_direction, output)?;
        let inputs = self.inputs(time, state);
        let linearization = self.operator.linearize(&inputs, &self.roles)?;
        let mut residual_tangent = vec![0.0; self.operator.residual_count()];
        linearization.jvp(
            RelationTangent::Parameter(parameter_direction),
            &mut residual_tangent,
        )?;
        residual_tangent.resize(self.state_coordinates.len(), 0.);
        self.write_rhs(&residual_tangent, output);
        require_finite_slice(self.relation, output, "first-order Parameter JVP")
    }

    fn initial_parameter_jvp(
        &self,
        time: f64,
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_parameter_independent_initial_conditions(time)?;
        if !time.is_finite()
            || output.len() != self.state_coordinates.len()
            || parameter_direction.len() != self.parameter_values.len()
        {
            return Err(invalid_time(
                self.relation,
                "initial Parameter action requires finite time and exact vector shapes",
            ));
        }
        require_finite_slice(
            self.relation,
            parameter_direction,
            "initial Parameter direction",
        )?;
        output.fill(0.0);
        Ok(())
    }
}

impl TimeSystem for FirstOrderProgram {
    fn dimension(&self) -> usize {
        self.state_coordinates.len()
    }

    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.require_action_shape(time, state, None, output)?;
        let mut residual = self.operator.evaluate(&self.inputs(time, state))?;
        residual.extend(self.companions.iter().map(|&(_, next)| -state[next]));
        self.write_rhs(&residual, output);
        require_finite_slice(self.relation, output, "first-order right-hand side")
    }

    fn rhs_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_action_shape(time, state, Some(direction), output)?;
        let inputs = self.inputs(time, state);
        let linearization = self.operator.linearize(&inputs, &self.roles)?;
        let tangent = self
            .state_symbol_coordinates
            .iter()
            .map(|coordinate| direction[*coordinate])
            .collect::<Vec<_>>();
        let mut residual_tangent = vec![0.0; self.operator.residual_count()];
        linearization.jvp(RelationTangent::Unknown(&tangent), &mut residual_tangent)?;
        residual_tangent.extend(self.companions.iter().map(|&(_, next)| -direction[next]));
        self.write_rhs(&residual_tangent, output);
        require_finite_slice(self.relation, output, "first-order state JVP")
    }

    fn mass_action(
        &self,
        time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_action_shape(time, direction, Some(direction), output)?;
        let FirstOrderProjection::MassMatrix { coefficients } = &self.projection else {
            return Err(invalid_time(
                self.relation,
                "explicit ODE projection has no mass-matrix action",
            ));
        };
        let dimension = self.dimension();
        for (row, output) in coefficients.chunks_exact(dimension).zip(output.iter_mut()) {
            *output = row
                .iter()
                .zip(direction)
                .map(|(coefficient, direction)| coefficient * direction)
                .sum();
        }
        require_finite_slice(self.relation, output, "mass-matrix action")
    }
}

#[derive(Debug, Clone, PartialEq)]
enum FirstOrderProjection {
    Explicit {
        residual_rows: Vec<usize>,
        derivative_scales: Vec<f64>,
    },
    MassMatrix {
        coefficients: Vec<f64>,
    },
}

struct ClassifiedProjection {
    projection: FirstOrderProjection,
    proof: TimeLoweringProof,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TimeBinding {
    State(usize),
    DerivativeZero,
    Parameter(usize),
    Time,
}

struct TimeBindings {
    values: Vec<TimeBinding>,
    roles: Vec<DifferentiationRole>,
    state_coordinates: Vec<usize>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    parameter_coordinates: Vec<eqiora_ir::ScalarSymbolCoordinate>,
}

fn bind_symbols(
    program: &KernelProgram,
    relation: Id<kinds::Relation>,
    operator: &TimeOperator,
    state_coordinates: &HashMap<eqiora_ir::ScalarSymbolCoordinate, usize>,
) -> Result<TimeBindings, Diagnostic> {
    let mut bindings = Vec::with_capacity(operator.symbols().len());
    let mut roles = Vec::with_capacity(operator.symbols().len());
    let mut state_symbol_coordinates = Vec::new();
    let mut parameter_fields = Vec::new();
    let mut parameter_values = Vec::new();
    let mut parameter_indices = HashMap::new();
    let mut parameter_coordinates = Vec::new();
    for source in operator.symbols() {
        let (binding, role) = match source.symbol() {
            SymbolRef::Field(_) => {
                let coordinate = state_coordinates.get(source).copied().ok_or_else(|| {
                    invalid_time(relation, "Field is absent from first-order state order")
                })?;
                state_symbol_coordinates.push(coordinate);
                (TimeBinding::State(coordinate), DifferentiationRole::Unknown)
            }
            SymbolRef::Derivative(field, order) => {
                if let Some(&coordinate) = state_coordinates.get(source) {
                    state_symbol_coordinates.push(coordinate);
                    (TimeBinding::State(coordinate), DifferentiationRole::Unknown)
                } else if state_coordinates.contains_key(&source.with_symbol(if order.get() == 1 {
                    SymbolRef::Field(field)
                } else {
                    SymbolRef::Derivative(
                        field,
                        std::num::NonZeroU32::new(order.get() - 1).unwrap(),
                    )
                })) {
                    (TimeBinding::DerivativeZero, DifferentiationRole::Frozen)
                } else {
                    return Err(invalid_time(
                        relation,
                        "derivative symbol is absent from first-order state order",
                    ));
                }
            }
            SymbolRef::Parameter(parameter) => {
                let coordinate = if let Some(coordinate) = parameter_indices.get(source) {
                    *coordinate
                } else {
                    let value = program.typed_value(parameter.erase()).ok_or_else(|| {
                        invalid_time(relation, "Parameter has no bound typed value")
                    })?;
                    let coordinates = eqiora_ir::ScalarSymbolCoordinate::for_value(
                        SymbolRef::Parameter(parameter),
                        value.value_type(),
                    )?;
                    let index = coordinates
                        .iter()
                        .position(|coordinate| coordinate == source)
                        .ok_or_else(|| {
                            invalid_time(
                                relation,
                                "Parameter coordinate differs from its exact type",
                            )
                        })?;
                    let complex =
                        value.value_type().scalar_domain() == eqiora_core::ScalarDomain::Complex;
                    let component = value
                        .component(index / if complex { 2 } else { 1 })
                        .ok_or_else(|| {
                            invalid_time(relation, "Parameter component is unavailable")
                        })?;
                    let value = if source.is_imaginary() {
                        component.1
                    } else {
                        component.0
                    };
                    require_finite(relation, value, "Parameter value")?;
                    let coordinate = parameter_values.len();
                    parameter_indices.insert(source.clone(), coordinate);
                    parameter_coordinates.push(source.clone());
                    if !parameter_fields.contains(&parameter) {
                        parameter_fields.push(parameter);
                    }
                    parameter_values.push(value);
                    coordinate
                };
                (
                    TimeBinding::Parameter(coordinate),
                    DifferentiationRole::Parameter,
                )
            }
            SymbolRef::Time => (TimeBinding::Time, DifferentiationRole::Frozen),
            SymbolRef::Pre(_) | SymbolRef::Next(_) | SymbolRef::Port(_) => {
                return Err(invalid_time(
                    relation,
                    "first-order lowering admits only state, derivative, Parameter, and time symbols",
                ));
            }
            _ => {
                return Err(invalid_time(
                    relation,
                    "Relation symbol is newer than first-order lowering",
                ));
            }
        };
        bindings.push(binding);
        roles.push(role);
    }
    Ok(TimeBindings {
        values: bindings,
        roles,
        state_coordinates: state_symbol_coordinates,
        parameter_fields,
        parameter_values,
        parameter_coordinates,
    })
}

pub(crate) fn require_continuous_activation(
    program: &KernelProgram,
    relation: Id<kinds::Relation>,
) -> Result<(), Diagnostic> {
    let activation = program
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation.erase())
        .map(|edge| edge.from())
        .ok_or_else(|| invalid_time(relation, "Relation has no Activation"))?;
    match program.node(activation) {
        Some(KernelNode::Activation(activation))
            if matches!(activation.kind(), ActivationKind::Continuous) =>
        {
            Ok(())
        }
        _ => Err(invalid_time(
            relation,
            "only a continuously activated Relation can enter time lowering",
        )),
    }
}

fn classify_first_order(
    relation: Id<kinds::Relation>,
    derivative_matrix: ConstantDerivativeMatrixProof,
    state_coordinates: &[eqiora_core::TimeStateCoordinate],
) -> Result<ClassifiedProjection, Diagnostic> {
    let dimension = state_coordinates.len();
    let proof = TimeLoweringProof::new(relation, state_coordinates.to_vec(), derivative_matrix)?;
    let projection = if proof.equation_class() == TimeEquationClass::ExplicitOde {
        let rows = proof
            .derivative_matrix()
            .monomial_rows()
            .expect("explicit ODE classification requires a full monomial matrix");
        let mut residual_rows = vec![usize::MAX; dimension];
        let mut derivative_scales = vec![0.0; dimension];
        for (row, witness) in rows.into_iter().enumerate() {
            let state_coordinate = witness.state_coordinate();
            let coefficient = witness.coefficient();
            residual_rows[state_coordinate] = row;
            derivative_scales[state_coordinate] = coefficient;
        }
        FirstOrderProjection::Explicit {
            residual_rows,
            derivative_scales,
        }
    } else {
        FirstOrderProjection::MassMatrix {
            coefficients: proof.derivative_matrix().coefficients().to_vec(),
        }
    };
    Ok(ClassifiedProjection { projection, proof })
}

fn derivative_structure_error(
    relation: Id<kinds::Relation>,
    failure: SymbolicLinearityFailure,
) -> Diagnostic {
    invalid_time(
        relation,
        format!("cannot prove constant derivative Jacobian: {failure:?}"),
    )
}

pub(crate) fn require_finite(
    relation: Id<kinds::Relation>,
    value: f64,
    name: &str,
) -> Result<(), Diagnostic> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_time(relation, format!("{name} must be finite")))
    }
}

pub(crate) fn require_finite_slice(
    relation: Id<kinds::Relation>,
    values: &[f64],
    name: &str,
) -> Result<(), Diagnostic> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_time(
            relation,
            format!("{name} must contain only finite values"),
        ))
    }
}

pub(crate) fn invalid_time(
    relation: Id<kinds::Relation>,
    message: impl Into<String>,
) -> Diagnostic {
    Diagnostic::error(codes::INVALID_TIME_LOWERING, message).with_graph_path(GraphPath::new([
        "time-lowering".to_owned(),
        "relation".to_owned(),
        relation.to_string(),
    ]))
}
