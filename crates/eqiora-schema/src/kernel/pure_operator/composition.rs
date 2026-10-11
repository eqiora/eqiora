//! Bounded scalar composition preserves authored exact arithmetic order.
use super::*;

impl CalculusBuilder {
    /// Substitute checked scalar argument expressions into one closed scalar definition.
    /// The caller resolves lexical names and rejects recursive definition graphs.
    /// No polynomial normalization or executable reassociation is performed.
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
            let mapped = |id| {
                ids.get(definition_index(id, ids.len())?)
                    .copied()
                    .ok_or(PureOperatorError::InvalidNode)
            };
            let id = match node {
                CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
                    arguments[usize::from(*formal)]
                }
                CalculusNode::Rational { value, dimension } => {
                    staged.push(CalculusNode::Rational {
                        value: *value,
                        dimension: *dimension,
                    })?
                }
                CalculusNode::Boolean(value) => staged.push(CalculusNode::Boolean(*value))?,
                CalculusNode::Compare(op, left, right) => {
                    staged.push(CalculusNode::Compare(*op, mapped(*left)?, mapped(*right)?))?
                }
                CalculusNode::Not(value) => staged.push(CalculusNode::Not(mapped(*value)?))?,
                CalculusNode::And(left, right) => {
                    staged.push(CalculusNode::And(mapped(*left)?, mapped(*right)?))?
                }
                CalculusNode::Or(left, right) => {
                    staged.push(CalculusNode::Or(mapped(*left)?, mapped(*right)?))?
                }
                CalculusNode::UnaryMath(function, value) => {
                    staged.push(CalculusNode::UnaryMath(*function, mapped(*value)?))?
                }
                CalculusNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => staged.push(CalculusNode::Select {
                    condition: mapped(*condition)?,
                    then_value: mapped(*then_value)?,
                    else_value: mapped(*else_value)?,
                })?,
                CalculusNode::Require { condition, value } => {
                    staged.push(CalculusNode::Require {
                        condition: mapped(*condition)?,
                        value: mapped(*value)?,
                    })?
                }
                CalculusNode::Differentiated { value, source, wrt } => {
                    staged.push(CalculusNode::Differentiated {
                        value: mapped(*value)?,
                        source: mapped(*source)?,
                        wrt: mapped(*wrt)?,
                    })?
                }
                CalculusNode::BoundInput(value) => {
                    staged.push(CalculusNode::BoundInput(mapped(*value)?))?
                }
                CalculusNode::Neg(value) => staged.push(CalculusNode::Neg(mapped(*value)?))?,
                CalculusNode::Add(left, right) => {
                    staged.push(CalculusNode::Add(mapped(*left)?, mapped(*right)?))?
                }
                CalculusNode::Mul(left, right) => {
                    staged.push(CalculusNode::Mul(mapped(*left)?, mapped(*right)?))?
                }
                _ => return Err(PureOperatorError::FormalTypeMismatch),
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
