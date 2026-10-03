//! Pointwise tensor definitions lower through the existing real scalar IR.
use super::*;

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
        let calculus = expansion.component(component).map_err(|error| {
            invalid_component_ir(format!("pure operator component expansion failed: {error}"))
        })?;
        let mut remapped = vec![[None; 2]; calculus.nodes().len()];
        self.lower_calculus_component(&calculus, calculus.root(), arguments, part, &mut remapped)
    }

    fn lower_calculus_component(
        &mut self,
        calculus: &crate::ScalarCalculus<I>,
        value: crate::CalculusNodeId,
        arguments: &[ExprId],
        part: ScalarPart,
        remapped: &mut [[Option<ScalarInputValueId>; 2]],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let index = usize::try_from(value.index())
            .ok()
            .filter(|index| *index < calculus.nodes().len())
            .ok_or_else(|| {
                invalid_component_ir("pure calculus contains an invalid value reference")
            })?;
        let part_index = usize::from(part == ScalarPart::Imaginary);
        if let Some(mapped) = remapped[index][part_index] {
            return Ok(mapped);
        }
        let node = calculus.nodes()[index].clone();
        let mapped = match node {
            ScalarCalculusNode::Rational { value, dimension } => {
                self.builder.constant(DynQuantity::new(
                    if part == ScalarPart::Real {
                        value.as_f64()
                    } else {
                        0.0
                    },
                    dimension,
                ))?
            }
            ScalarCalculusNode::FormalComponent(atom) => {
                let operand = arguments
                    .get(usize::from(atom.formal()))
                    .copied()
                    .ok_or_else(|| {
                        invalid_component_ir("pure calculus referenced an unexpected formal")
                    })?;
                self.lower_part(operand, atom.component(), part)?
            }
            ScalarCalculusNode::BoundInput(value)
            | ScalarCalculusNode::Differentiated { value, .. } => {
                self.lower_calculus_component(calculus, value, arguments, part, remapped)?
            }
            ScalarCalculusNode::Neg(value) => {
                let value =
                    self.lower_calculus_component(calculus, value, arguments, part, remapped)?;
                self.builder.neg(value)?
            }
            ScalarCalculusNode::Add(left, right) => {
                let left =
                    self.lower_calculus_component(calculus, left, arguments, part, remapped)?;
                let right =
                    self.lower_calculus_component(calculus, right, arguments, part, remapped)?;
                self.builder.add(left, right)?
            }
            ScalarCalculusNode::Mul(left, right) => {
                use ScalarPart::{Imaginary, Real};
                let ar =
                    self.lower_calculus_component(calculus, left, arguments, Real, remapped)?;
                let br =
                    self.lower_calculus_component(calculus, right, arguments, Real, remapped)?;
                if calculus.result_type().value_type.scalar_domain()
                    == eqiora_core::ScalarDomain::Real
                {
                    self.builder.mul(ar, br)?
                } else {
                    let ai = self
                        .lower_calculus_component(calculus, left, arguments, Imaginary, remapped)?;
                    let bi = self.lower_calculus_component(
                        calculus, right, arguments, Imaginary, remapped,
                    )?;
                    if part == Real {
                        let real = self.builder.mul(ar, br)?;
                        let imaginary = self.builder.mul(ai, bi)?;
                        self.builder.sub(real, imaginary)?
                    } else {
                        let first = self.builder.mul(ar, bi)?;
                        let second = self.builder.mul(ai, br)?;
                        self.builder.add(first, second)?
                    }
                }
            }
        };
        remapped[index][part_index] = Some(mapped);
        Ok(mapped)
    }
}
