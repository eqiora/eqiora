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

use crate::CpuProgram;
use eqiora_sem::{KernelProgram, ReferenceConfig};

/// Canonical continuous Relation proven to have first-order form
/// `M y_dot = f(t,y)`.
///
/// State order follows first occurrence of current-value or derivative Field
/// symbols in scalar Operator IR. A full constant monomial derivative Jacobian
/// is normalized to an explicit ODE. Every other non-zero-rank constant matrix
/// remains a full or rank-deficient mass matrix. State-dependent and
/// derivative-nonlinear systems fail closed; equation class is never inferred
/// from sample evaluations or a floating-point rank threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstOrderProgram {
    relation: Id<kinds::Relation>,
    operator: ScalarOperatorIr,
    state_fields: Vec<Id<kinds::Field>>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    kernel: KernelProgram,
    bindings: Vec<TimeBinding>,
    roles: Vec<DifferentiationRole>,
    state_symbol_coordinates: Vec<usize>,
    proof: TimeLoweringProof,
    projection: FirstOrderProjection,
}

impl FirstOrderProgram {
    /// Lower one continuously activated Relation from an already lowered CPU
    /// program and prove an admitted first-order equation class.
    ///
    /// # Errors
    /// Returns `EQ0705` if activation, symbols, shapes, or
    /// derivative structure cannot enter the first-order seam. Existing
    /// Operator IR diagnostics are retained when scalar evaluation fails.
    pub fn lower(program: &CpuProgram, relation: Id<kinds::Relation>) -> Result<Self, Diagnostic> {
        require_continuous_activation(program, relation)?;
        let typed = program
            .kernel()
            .typed_relation_residual(relation)
            .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
        let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;

        let state_order = state_order(relation, &operator)?;
        let state_fields = state_order.fields;
        let derivatives = state_fields
            .iter()
            .copied()
            .map(SymbolRef::Derivative)
            .collect::<Vec<_>>();
        let jacobian = operator
            .constant_symbol_jacobian(&derivatives)
            .map_err(|failure| derivative_structure_error(relation, failure))?;
        let classified = classify_first_order(relation, &jacobian, &state_fields)?;

        let time_bindings = bind_symbols(program, relation, &operator, &state_order.coordinates)?;
        Ok(Self {
            relation,
            operator,
            state_fields,
            parameter_fields: time_bindings.parameter_fields,
            parameter_values: time_bindings.parameter_values,
            kernel: program.kernel().clone(),
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
    pub fn state_fields(&self) -> &[Id<kinds::Field>] {
        &self.state_fields
    }

    /// Solve fresh simultaneous initial equations before the first activation.
    /// Numerical tolerances and guesses belong to the supplied execution configuration.
    /// Restart callers use their accepted State directly instead of this operation.
    ///
    /// # Errors
    /// Returns initialization diagnostics for missing, inconsistent, or unsupported conditions.
    pub fn initialize(
        &self,
        config: ReferenceConfig,
    ) -> Result<ImplicitDaeInitialization, Diagnostic> {
        super::initialization::initialize(&self.kernel, &self.state_fields, self.relation, config)
    }

    /// Deterministic first-occurrence order of bound Parameter symbols.
    #[must_use]
    pub fn parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        &self.parameter_fields
    }

    /// Revision-captured Parameter values in [`Self::parameter_fields`] order.
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
            self.initialize(ReferenceConfig::new(0.0, 1.0)?)?
                .state()
                .to_vec(),
        )
    }

    /// Construct the parameter-JVP problem from the same proven projection.
    ///
    /// # Errors
    /// Retains `ForwardSensitivityProblem` validation diagnostics when the
    /// Relation has no Parameter symbols or its invariants change.
    pub fn forward_sensitivity_problem(&self) -> Result<ForwardSensitivityProblem<'_>, Diagnostic> {
        self.require_parameter_independent_initial_conditions()?;
        ForwardSensitivityProblem::new(
            self,
            self.equation_class(),
            self.initial_condition_policy(),
            self.initialize(ReferenceConfig::new(0.0, 1.0)?)?
                .state()
                .to_vec(),
        )
    }

    fn require_parameter_independent_initial_conditions(&self) -> Result<(), Diagnostic> {
        super::initialization::require_zero_parameter_tangent(
            &self.kernel,
            &self.state_fields,
            &self.parameter_fields,
            self.relation,
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
                for state in 0..self.state_fields.len() {
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
        self.write_rhs(&residual_tangent, output);
        require_finite_slice(self.relation, output, "first-order Parameter JVP")
    }

    fn initial_parameter_jvp(
        &self,
        time: f64,
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_parameter_independent_initial_conditions()?;
        if !time.is_finite()
            || output.len() != self.state_fields.len()
            || parameter_direction.len() != self.parameter_fields.len()
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
        self.state_fields.len()
    }

    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.require_action_shape(time, state, None, output)?;
        let residual = self.operator.evaluate(&self.inputs(time, state))?;
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

pub(crate) struct StateOrder {
    pub(crate) fields: Vec<Id<kinds::Field>>,
    pub(crate) coordinates: HashMap<Id<kinds::Field>, usize>,
}

struct TimeBindings {
    values: Vec<TimeBinding>,
    roles: Vec<DifferentiationRole>,
    state_coordinates: Vec<usize>,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
}

pub(crate) fn state_order(
    relation: Id<kinds::Relation>,
    operator: &ScalarOperatorIr,
) -> Result<StateOrder, Diagnostic> {
    let mut fields = Vec::new();
    let mut coordinates = HashMap::new();
    for symbol in operator.symbols() {
        let field = match *symbol {
            SymbolRef::Field(field) | SymbolRef::Derivative(field) => Some(field),
            _ => None,
        };
        if let Some(field) = field {
            let next = fields.len();
            coordinates.entry(field).or_insert_with(|| {
                fields.push(field);
                next
            });
        }
    }
    if fields.is_empty() {
        Err(invalid_time(
            relation,
            "first-order Relation has no state Field symbols",
        ))
    } else {
        Ok(StateOrder {
            fields,
            coordinates,
        })
    }
}

fn bind_symbols(
    program: &CpuProgram,
    relation: Id<kinds::Relation>,
    operator: &ScalarOperatorIr,
    state_coordinates: &HashMap<Id<kinds::Field>, usize>,
) -> Result<TimeBindings, Diagnostic> {
    let mut bindings = Vec::with_capacity(operator.symbols().len());
    let mut roles = Vec::with_capacity(operator.symbols().len());
    let mut state_symbol_coordinates = Vec::new();
    let mut parameter_fields = Vec::new();
    let mut parameter_values = Vec::new();
    let mut parameter_coordinates = HashMap::new();
    for symbol in operator.symbols().iter().copied() {
        let (binding, role) = match symbol {
            SymbolRef::Field(field) => {
                let coordinate = state_coordinates.get(&field).copied().ok_or_else(|| {
                    invalid_time(relation, "Field is absent from first-order state order")
                })?;
                state_symbol_coordinates.push(coordinate);
                (TimeBinding::State(coordinate), DifferentiationRole::Unknown)
            }
            SymbolRef::Derivative(field) => {
                if !state_coordinates.contains_key(&field) {
                    return Err(invalid_time(
                        relation,
                        "derivative symbol is absent from first-order state order",
                    ));
                }
                (TimeBinding::DerivativeZero, DifferentiationRole::Frozen)
            }
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
    })
}

pub(crate) fn require_continuous_activation(
    program: &CpuProgram,
    relation: Id<kinds::Relation>,
) -> Result<(), Diagnostic> {
    let activation = program
        .kernel()
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation.erase())
        .map(|edge| edge.from())
        .ok_or_else(|| invalid_time(relation, "Relation has no Activation"))?;
    match program.kernel().node(activation) {
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
    jacobian: &ConstantSymbolJacobian,
    state_fields: &[Id<kinds::Field>],
) -> Result<ClassifiedProjection, Diagnostic> {
    let dimension = state_fields.len();
    if jacobian.row_count() != dimension || jacobian.column_count() != dimension {
        return Err(invalid_time(
            relation,
            "first-order system requires one residual equation per state Field",
        ));
    }

    let derivative_matrix =
        ConstantDerivativeMatrixProof::new(dimension, jacobian.coefficients().to_vec())?;
    let proof = TimeLoweringProof::new(relation, state_fields.to_vec(), derivative_matrix)?;
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
