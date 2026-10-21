//! Typed component composition that keeps the authored order of exact arithmetic.
use super::*;
mod argument;
pub use argument::CalculusArgument;
pub(super) use argument::coordinates;

#[cfg(test)]
mod typed_tests;

impl CalculusBuilder {
    /// Select a static channel while retaining the complete input value's domain.
    /// The selected value keeps its remaining axes; every original component is
    /// retained as a value prerequisite, including through later derivatives.
    ///
    /// # Errors
    /// Rejects invalid channel indices, incompatible typed layouts and calculus
    /// resource limits. A failure leaves the caller unchanged.
    pub fn index(
        &mut self,
        argument: &CalculusArgument,
        index: u32,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        let input = argument.value_type(self)?;
        let output = ExpressionType::<()>::new(input.clone(), None)
            .index(index)
            .map_err(|_| PureOperatorError::FormalTypeMismatch)?
            .value_type;
        let mut projection = Self::new(
            [PureValueClass::from_value_type(input.clone())?],
            PureValueClass::from_value_type(output.clone())?,
        )?;
        let mut complete = None;
        for axes in coordinates(&input)? {
            let value = projection.push(CalculusNode::FormalComponent { formal: 0, axes })?;
            complete = Some(match complete {
                Some(prerequisite) => projection.push(CalculusNode::Demand {
                    prerequisite,
                    value,
                })?,
                None => value,
            });
        }
        let value = projection.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: std::iter::once(ComponentIndex::Fixed(index))
                .chain((0..output.shape().rank()).map(|axis| ComponentIndex::Result(axis as u16)))
                .collect(),
        })?;
        let root = projection.push(CalculusNode::Demand {
            prerequisite: complete.ok_or(PureOperatorError::FormalTypeMismatch)?,
            value,
        })?;
        self.apply_inputs(&projection.finish(root)?, std::slice::from_ref(argument))
    }

    pub(super) fn reachable_nodes(
        &self,
        root: CalculusNodeId,
    ) -> Result<Vec<bool>, PureOperatorError> {
        let end = definition_index(root, self.nodes.len())?;
        let mut reachable = vec![false; end + 1];
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            let index = definition_index(id, reachable.len())?;
            if !std::mem::replace(&mut reachable[index], true) {
                pending.extend(self.nodes[index].operands());
            }
        }
        Ok(reachable)
    }

    /// Substitute complete typed inputs into a closed component definition.
    /// Whole-value templates retain their own result axes until projected by the
    /// callee. Scalar components already belong to the caller's result-axis context.
    /// A shaped result retains the caller's complete component layout.
    /// The caller resolves lexical names and rejects recursive definition graphs.
    /// No polynomial normalization or reassociation is performed.
    ///
    /// # Errors
    /// Rejects incompatible argument contracts, foreign formal environments, result
    /// layouts and resource limits. A failure leaves the builder unchanged.
    pub fn apply_inputs(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[CalculusArgument],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        if arguments.len() != definition.formals.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        if !definition.result.is_invariant_scalar()
            && !compatible_component_layout(&definition.result, &self.result)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        for (argument, formal) in arguments.iter().zip(&definition.formals) {
            argument.validate(self, formal)?;
        }
        let retain_bindings = definition
            .nodes
            .iter()
            .any(|node| matches!(node, CalculusNode::Differentiated { .. }));
        let mut staged = Self {
            formals: self.formals.clone(),
            result: self.result.clone(),
            nodes: self.nodes.clone(),
            depths: self.depths.clone(),
        };
        let mut components = std::collections::BTreeMap::new();
        // Keep scalar-only composition's authored order and distinct occurrences.
        for (formal, argument) in arguments.iter().enumerate() {
            if let CalculusArgument::Scalar(value) = argument {
                let value = if retain_bindings {
                    staged.push(CalculusNode::BoundInput(*value))?
                } else {
                    *value
                };
                components.insert((formal as u16, Box::<[ComponentIndex]>::default()), value);
            }
        }
        let mut ids = Vec::with_capacity(definition.nodes.len());
        for node in &definition.nodes {
            let id = match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    let key = (*formal, axes.clone());
                    if let Some(value) = components.get(&key) {
                        *value
                    } else {
                        let value = arguments[usize::from(*formal)].project(&mut staged, axes)?;
                        let value = if retain_bindings {
                            staged.push(CalculusNode::BoundInput(value))?
                        } else {
                            value
                        };
                        components.insert(key, value);
                        value
                    }
                }
                node => staged.push(node.map_operands(|id| ids[id.index() as usize]))?,
            };
            ids.push(id);
        }
        let root =
            staged.embed_result(ids[definition.root.index() as usize], &definition.result)?;
        derive_symbolic_dimension(&staged.formals, &staged.nodes, root)?;
        *self = staged;
        Ok(root)
    }

    fn embed_result(
        &mut self,
        mut root: CalculusNodeId,
        result: &PureValueClass,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        let actual = domains::expression_domain(&self.formals, &self.nodes, root)?;
        let declared = domains::result_domain(result.scalar_domain(), actual)?;
        if declared == Some(eqiora_core::ScalarDomain::Complex)
            && actual == Some(eqiora_core::ScalarDomain::Real)
        {
            // Preserve declared embedding, including positive imaginary zero and units.
            let imag = self.push(CalculusNode::UnaryMath(
                super::super::UnaryMathFunction::Imag,
                root,
            ))?;
            root = self.push(CalculusNode::Complex { real: root, imag })?;
        }
        Ok(root)
    }
}

// Physical dimensions and scalar domain belong to the component expression.
// All other type information must survive an implicit componentwise call.
pub(super) fn compatible_component_layout(left: &PureValueClass, right: &PureValueClass) -> bool {
    let layout = |value: &PureValueClass| {
        value
            .clone()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(eqiora_core::ScalarDomain::Real)
    };
    let (Ok(formal), Ok(actual)) = (layout(left), layout(right)) else {
        return false;
    };
    formal == actual
        || actual.exact_type().is_some_and(|value_type| {
            instantiation::validate_argument_class(
                &formal,
                &ExpressionType::<()>::new(value_type.clone(), None),
            )
            .is_ok()
        })
}

impl CalculusNode {
    pub(super) fn map_operands(
        &self,
        mut map: impl FnMut(CalculusNodeId) -> CalculusNodeId,
    ) -> Self {
        let mut node = self.clone();
        match &mut node {
            Self::Require {
                condition: a,
                value: b,
            }
            | Self::Differentiated { source: a, wrt: b }
            | Self::Demand {
                prerequisite: a,
                value: b,
            }
            | Self::Complex { real: a, imag: b }
            | Self::Add(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Compare(_, a, b)
            | Self::And(a, b)
            | Self::Or(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::BoundInput(a) | Self::Neg(a) | Self::Not(a) | Self::UnaryMath(_, a) => {
                *a = map(*a)
            }
            Self::Select {
                condition,
                then_value,
                else_value,
            } => {
                *condition = map(*condition);
                *then_value = map(*then_value);
                *else_value = map(*else_value);
            }
            Self::Rational { .. }
            | Self::Boolean(_)
            | Self::FormalComponent { .. }
            | Self::KroneckerDelta(..) => {}
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{ScalarDomain, ValueType};

    #[test]
    fn composition_preserves_complex_return_embedding_with_symbolic_units() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        for dimension in [None, Some(length)] {
            let mut real = PureValueClass::invariant_scalar()
                .with_scalar_domain(ScalarDomain::Real)
                .unwrap();
            if let Some(dimension) = dimension {
                real = real.with_dimension(dimension);
            }
            let complex = real
                .clone()
                .with_scalar_domain(ScalarDomain::Complex)
                .unwrap();
            let mut callee = CalculusBuilder::new([real.clone()], complex.clone()).unwrap();
            let root = callee
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            let definition = callee.finish(root).unwrap();
            for (result, admits) in [
                (complex, true),
                (PureValueClass::invariant_scalar(), true),
                (real.clone(), false),
            ] {
                let mut caller = CalculusBuilder::new([real.clone()], result).unwrap();
                let input = caller
                    .push(CalculusNode::FormalComponent {
                        formal: 0,
                        axes: Box::new([]),
                    })
                    .unwrap();
                let root = caller.apply_inputs(&definition, &[input.into()]).unwrap();
                let result = caller.finish(root);
                assert_eq!(result.is_ok(), admits);
                if let Ok(result) = result {
                    let instance = result
                        .instantiate(&[ExpressionType::<()>::new(
                            ValueType::scalar(ScalarDomain::Real, length).unwrap(),
                            None,
                        )])
                        .unwrap();
                    assert_eq!(
                        instance.result_type().value_type,
                        ValueType::scalar(ScalarDomain::Complex, length).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn differentiated_composition_retains_distinct_local_input_occurrences() {
        let class = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        let mut original =
            CalculusBuilder::new([class.clone(), class.clone()], class.clone()).unwrap();
        let x = original
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let y = original
            .push(CalculusNode::FormalComponent {
                formal: 1,
                axes: Box::new([]),
            })
            .unwrap();
        let product = original.push(CalculusNode::Mul(x, y)).unwrap();
        let dx = original.partial(product, 0).unwrap();
        let definition = original.finish(dx).unwrap();
        let mut caller = CalculusBuilder::new([class.clone()], class.clone()).unwrap();
        let p = caller
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let applied = caller
            .apply_inputs(&definition, &[p.into(), p.into()])
            .unwrap();
        let CalculusNode::Differentiated { source, wrt, .. } =
            caller.nodes[applied.index() as usize]
        else {
            panic!("retained derivative")
        };
        let CalculusNode::Mul(left, right) = caller.nodes[source.index() as usize] else {
            panic!("retained source")
        };
        assert_ne!(left, right);
        assert_eq!(wrt, left);
        assert_eq!(
            caller.nodes[left.index() as usize],
            CalculusNode::BoundInput(p)
        );
        assert_eq!(
            caller.nodes[right.index() as usize],
            CalculusNode::BoundInput(p)
        );
        caller.finish(applied).unwrap();
    }

    #[test]
    fn shaped_composition_preserves_axes_frame_channels_and_nominal_bases() {
        use eqiora_core::{FiniteBasis, Id, ValueShape, ValueType};
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let scalar_class = PureValueClass::from_value_type(scalar.clone()).unwrap();
        let spatial = |extents: [u32; 2]| {
            ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new(extents).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap()
        };
        let first = FiniteBasis::new(Id::new(), 2).unwrap();
        let second = FiniteBasis::new(Id::new(), 2).unwrap();
        let map = |a, b| {
            ValueType::linear_map(a, b, ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap()
        };
        let channels = scalar.clone().array(2).unwrap().array(2).unwrap();
        let channel_vectors = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap()
        .array(2)
        .unwrap();
        let layouts = [
            spatial([2, 2]),
            spatial([3, 3]),
            channels,
            channel_vectors,
            map(first, second),
            map(second, first),
        ];
        for (callee_index, value_type) in layouts.iter().enumerate() {
            let result = PureValueClass::from_value_type(value_type.clone()).unwrap();
            let mut callee = CalculusBuilder::new([scalar_class.clone()], result).unwrap();
            let input = callee
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            let delta = callee
                .push(CalculusNode::KroneckerDelta(
                    ComponentIndex::Result(0),
                    ComponentIndex::Result(1),
                ))
                .unwrap();
            let root = callee.push(CalculusNode::Mul(input, delta)).unwrap();
            let definition = callee.finish(root).unwrap();
            for (caller_index, layout) in layouts.iter().enumerate() {
                let result = PureValueClass::from_value_type(
                    layout
                        .clone()
                        .with_scalar_domain(ScalarDomain::Complex)
                        .unwrap(),
                )
                .unwrap();
                let mut caller = CalculusBuilder::new([scalar_class.clone()], result).unwrap();
                let input = caller
                    .push(CalculusNode::FormalComponent {
                        formal: 0,
                        axes: Box::new([]),
                    })
                    .unwrap();
                let before = (caller.nodes.clone(), caller.depths.clone());
                let application = caller.apply_inputs(&definition, &[input.into()]);
                if caller_index == callee_index {
                    caller.finish(application.unwrap()).unwrap();
                } else {
                    assert_eq!(application, Err(PureOperatorError::FormalTypeMismatch));
                    assert_eq!(caller.nodes, before.0);
                    assert_eq!(caller.depths, before.1);
                }
            }
        }
    }
}
