//! Scalar composition that keeps the authored order of exact arithmetic.
use super::*;

impl CalculusBuilder {
    /// Substitute checked scalar argument expressions into a closed scalar definition.
    /// The caller resolves lexical names and rejects recursive definition graphs.
    /// No polynomial normalization or reassociation is performed.
    pub fn apply_scalar(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[CalculusNodeId],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        if arguments.len() != definition.formals.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        if !definition.result.is_invariant_scalar()
            || definition
                .formals
                .iter()
                .any(|formal| !formal.is_invariant_scalar())
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let additional = definition
            .nodes
            .iter()
            .filter(|node| !matches!(node, CalculusNode::FormalComponent { .. }))
            .count();
        let retain_bindings = definition
            .nodes
            .iter()
            .any(|node| matches!(node, CalculusNode::Differentiated { .. }));
        let additional = additional + if retain_bindings { arguments.len() } else { 0 };
        self.nodes
            .len()
            .checked_add(additional)
            .filter(|count| *count <= MAX_NODES)
            .ok_or(PureOperatorError::NodeLimit)?;
        for (argument, formal) in arguments.iter().zip(&definition.formals) {
            definition_index(*argument, self.nodes.len())?;
            let dimension = derive_symbolic_dimension(&self.formals, &self.nodes, *argument)?;
            validate_result_dimension(&self.formals, *formal, &dimension)?;
            if let Some(expected) = formal.scalar_domain()
                && domains::expression_domain(&self.formals, &self.nodes, *argument)?
                    != Some(expected)
            {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
        }
        // Stage the bounded append so any failed depth/type check leaves the caller intact.
        let mut staged = Self {
            formals: self.formals.clone(),
            result: self.result,
            nodes: self.nodes.clone(),
            depths: self.depths.clone(),
        };
        let arguments = arguments
            .iter()
            .map(|argument| {
                if retain_bindings {
                    staged.push(CalculusNode::BoundInput(*argument))
                } else {
                    Ok(*argument)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut ids = Vec::with_capacity(definition.nodes.len());
        for node in &definition.nodes {
            let id = match node {
                CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                    arguments[usize::from(*formal)]
                }
                CalculusNode::FormalComponent { .. } => {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                // The closed definition already owns the topological operand invariant.
                node => staged.push(node.map_operands(|id| ids[id.index() as usize]))?,
            };
            ids.push(id);
        }
        let root = ids[definition_index(definition.root, ids.len())?];
        dimensions::validate_profile(&staged.formals, staged.result, &staged.nodes)?;
        derive_symbolic_dimension(&staged.formals, &staged.nodes, root)?;
        *self = staged;
        Ok(root)
    }
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
    use eqiora_core::ScalarDomain;

    #[test]
    fn differentiated_composition_retains_distinct_local_input_occurrences() {
        let class = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        let mut original = CalculusBuilder::new([class, class], class).unwrap();
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
        let mut caller = CalculusBuilder::new([class], class).unwrap();
        let p = caller
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let applied = caller.apply_scalar(&definition, &[p, p]).unwrap();
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
}
