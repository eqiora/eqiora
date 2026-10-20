//! Pointwise tensor definitions lower through the existing real scalar IR.
use super::*;
mod evaluation;
mod partials;
use eqiora_schema::kernel::pure_operator::{CalculusNode, CalculusNodeId};

struct PureComponent<'a> {
    definition: &'a PureOperatorDefinition,
    arguments: &'a [ExprId],
    component: &'a [u32],
    component_types: &'a [eqiora_core::ValueType],
    remapped: Vec<[Option<ScalarInputValueId>; 2]>,
    formal_values: Option<Vec<ScalarInputValueId>>,
    bound_values: Vec<(CalculusNodeId, ScalarInputValueId)>,
}

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_pure_operator(
        &mut self,
        operator: StandardPureOperator,
        operand: ExprId,
        component: &[u32],
        expected_result: &ExpressionType<I>,
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let definition = match operator {
            StandardPureOperator::SymmetricPart => PureOperatorDefinition::symmetric_part(),
            StandardPureOperator::IsotropicLift => PureOperatorDefinition::isotropic_lift(),
        }
        .map_err(|error| invalid_component_ir(format!("invalid pure operator: {error}")))?;
        self.lower_pure_definition(&definition, &[operand], component, expected_result, part)
    }

    pub(super) fn lower_pure_definition(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        component: &[u32],
        expected_result: &ExpressionType<I>,
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.lower_pure_definition_with_inputs(
            definition,
            arguments,
            component,
            expected_result,
            part,
            None,
        )
    }

    fn lower_pure_definition_with_inputs(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        component: &[u32],
        expected_result: &ExpressionType<I>,
        part: ScalarPart,
        formal_values: Option<Vec<ScalarInputValueId>>,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let argument_types = arguments
            .iter()
            .map(|argument| {
                let index = node_index(*argument, self.expression.nodes().len())?;
                self.node_types.get(index).cloned().ok_or_else(|| {
                    invalid_component_ir("pure operator argument has no inferred type")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expansion = definition.instantiate(&argument_types).map_err(|error| {
            invalid_component_ir(format!("pure operator typing failed: {error}"))
        })?;
        if expansion.result_type() != expected_result {
            return Err(invalid_component_ir(
                "pure operator expansion differs from the inferred Kernel type",
            ));
        }
        validate_component(expected_result.shape(), component)?;
        // Lower the checked definition directly. Exact polynomial expansion is
        // a proof view and must not restrict ordinary numerical execution.
        let work = definition
            .nodes()
            .len()
            .checked_mul(8)
            .ok_or_else(|| invalid_component_ir("pure component expansion work overflows"))?;
        super::finite::charge_products(self.finite_products, work)?;
        let mut state = PureComponent {
            definition,
            arguments,
            component,
            component_types: expansion.component_types(),
            remapped: vec![[None; 2]; definition.nodes().len()],
            formal_values,
            bound_values: Vec::new(),
        };
        self.lower_calculus_component(&mut state, definition.root(), part)
    }

    fn lower_calculus_component(
        &mut self,
        state: &mut PureComponent<'_>,
        value: CalculusNodeId,
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        use ScalarPart::{Imaginary, Real};
        let index = value.index() as usize;
        let part_index = usize::from(part == Imaginary);
        if let Some(mapped) = state.remapped[index][part_index] {
            return Ok(mapped);
        }
        if part == Imaginary
            && state.component_types[index].scalar_domain() == eqiora_core::ScalarDomain::Real
        {
            // A real intermediate stays real inside a complex expression. Keep its
            // demanded value and domains when constructing the typed imaginary zero.
            let real = self.lower_calculus_component(state, value, Real)?;
            let zero = self.builder.sub(real, real)?;
            state.remapped[index][part_index] = Some(zero);
            return Ok(zero);
        }
        let node = state.definition.nodes()[index].clone();
        let mapped = match node {
            CalculusNode::Rational { value, dimension } => {
                self.builder.constant(DynQuantity::new(
                    if part == Real {
                        value
                            .to_f64()
                            .map_err(|error| invalid_component_ir(error.to_string()))?
                    } else {
                        0.0
                    },
                    dimension,
                ))?
            }
            CalculusNode::KroneckerDelta(left, right) => {
                let left = left
                    .resolve(state.component)
                    .map_err(|error| invalid_component_ir(error.to_string()))?;
                let right = right
                    .resolve(state.component)
                    .map_err(|error| invalid_component_ir(error.to_string()))?;
                self.builder.constant(DynQuantity::new(
                    f64::from(part == Real && left == right),
                    eqiora_core::DimExponents::DIMENSIONLESS,
                ))?
            }
            CalculusNode::FormalComponent { formal, axes } => {
                let operand = state.arguments[usize::from(formal)];
                let coordinates = axes
                    .iter()
                    .map(|axis| {
                        axis.resolve(state.component)
                            .map_err(|error| invalid_component_ir(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(values) = &state.formal_values {
                    let real = values[usize::from(formal)];
                    if part == Real {
                        real
                    } else {
                        self.builder.sub(real, real)?
                    }
                } else {
                    self.lower_part(operand, &coordinates, part)?
                }
            }
            CalculusNode::BoundInput(inner) => {
                if let Some((_, mapped)) = state.bound_values.iter().find(|(id, _)| *id == value) {
                    if part == Real {
                        *mapped
                    } else {
                        self.builder.sub(*mapped, *mapped)?
                    }
                } else {
                    self.lower_calculus_component(state, inner, part)?
                }
            }
            CalculusNode::Differentiated { source, wrt, .. } => {
                self.lower_formal_partial(state, source, wrt, part)?
            }
            CalculusNode::Boolean(value) => self.builder.boolean(value)?,
            CalculusNode::Compare(op, left, right) => {
                let left = self.lower_calculus_component(state, left, Real)?;
                let right = self.lower_calculus_component(state, right, Real)?;
                self.builder.compare(op, left, right)?
            }
            CalculusNode::Not(value) => {
                let value = self.lower_calculus_component(state, value, Real)?;
                self.builder.not(value)?
            }
            CalculusNode::And(left, right) | CalculusNode::Or(left, right) => {
                let left = self.lower_calculus_component(state, left, Real)?;
                let right = self.lower_calculus_component(state, right, Real)?;
                if matches!(node, CalculusNode::And(..)) {
                    self.builder.and(left, right)?
                } else {
                    self.builder.or(left, right)?
                }
            }
            CalculusNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.lower_calculus_component(state, condition, Real)?;
                let then_value = self.lower_calculus_component(state, then_value, part)?;
                let else_value = self.lower_calculus_component(state, else_value, part)?;
                self.builder.select(condition, then_value, else_value)?
            }
            CalculusNode::Require { condition, value } => {
                let condition = self.lower_calculus_component(state, condition, Real)?;
                let value = self.lower_calculus_component(state, value, part)?;
                self.builder.require(condition, value)?
            }
            CalculusNode::Complex { real, imag } => {
                self.lower_calculus_component(state, if part == Real { real } else { imag }, Real)?
            }
            CalculusNode::UnaryMath(function, value) => {
                let complex = state.component_types[value.index() as usize].scalar_domain()
                    == eqiora_core::ScalarDomain::Complex;
                self.lower_math_parts(function, complex, part, |this, part| {
                    this.lower_calculus_component(state, value, part)
                })?
            }
            CalculusNode::Neg(value) => {
                let value = self.lower_calculus_component(state, value, part)?;
                self.builder.neg(value)?
            }
            CalculusNode::Add(left, right) => {
                let left = self.lower_calculus_component(state, left, part)?;
                let right = self.lower_calculus_component(state, right, part)?;
                self.builder.add(left, right)?
            }
            CalculusNode::Mul(left, right) => {
                let ar = self.lower_calculus_component(state, left, Real)?;
                let br = self.lower_calculus_component(state, right, Real)?;
                if state.component_types[index].scalar_domain()
                    != eqiora_core::ScalarDomain::Complex
                {
                    self.builder.mul(ar, br)?
                } else {
                    let ai = self.lower_calculus_component(state, left, Imaginary)?;
                    let bi = self.lower_calculus_component(state, right, Imaginary)?;
                    self.multiply_parts([ar, ai], [br, bi], part)?
                }
            }
            CalculusNode::Div(left, right) => {
                let ar = self.lower_calculus_component(state, left, Real)?;
                let br = self.lower_calculus_component(state, right, Real)?;
                if state.component_types[index].scalar_domain()
                    != eqiora_core::ScalarDomain::Complex
                {
                    self.builder.div(ar, br)?
                } else {
                    let ai = self.lower_calculus_component(state, left, Imaginary)?;
                    let bi = self.lower_calculus_component(state, right, Imaginary)?;
                    self.builder
                        .complex_div([ar, ai, br, bi], part == Imaginary)?
                }
            }
        };
        state.remapped[index][part_index] = Some(mapped);
        Ok(mapped)
    }
}
