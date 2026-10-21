//! Arguments retain either an ambient scalar component or a complete value template.
use super::*;

/// One argument to component-calculus composition.
#[derive(Debug, Clone)]
pub enum CalculusArgument {
    /// A scalar component already in the caller's arena and result-axis context.
    Scalar(CalculusNodeId),
    /// A complete value defined over exactly the caller's ordered formal contracts.
    /// Its result axes are local to this value, and are rebound during composition.
    Value(Box<PureOperatorDefinition>),
}

impl From<CalculusNodeId> for CalculusArgument {
    fn from(value: CalculusNodeId) -> Self {
        Self::Scalar(value)
    }
}

impl From<PureOperatorDefinition> for CalculusArgument {
    fn from(value: PureOperatorDefinition) -> Self {
        Self::Value(Box::new(value))
    }
}

impl CalculusArgument {
    pub(super) fn validate(
        &self,
        caller: &CalculusBuilder,
        formal: &PureValueClass,
    ) -> Result<(), PureOperatorError> {
        let (nodes, root, declared) = match self {
            Self::Scalar(root) => {
                if !formal.is_invariant_scalar() {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                (&caller.nodes, *root, None)
            }
            Self::Value(value) => {
                if value.formals != caller.formals {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                let actual = caller
                    .formals
                    .iter()
                    .map(|class| {
                        class
                            .exact_type()
                            .cloned()
                            .map(|ty| ExpressionType::<()>::new(ty, None))
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|inputs| {
                        value.instantiate(&inputs).and_then(|instance| {
                            PureValueClass::from_value_type(
                                instance.result_type().value_type.clone(),
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or_else(|| value.result.clone());
                if !compatible_component_layout(formal, &actual) {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                (&value.nodes, value.root, value.result.scalar_domain())
            }
        };
        definition_index(root, nodes.len())?;
        let dimension = derive_symbolic_dimension(&caller.formals, nodes, root)?;
        validate_result_dimension(&caller.formals, formal, &dimension)?;
        let actual = domains::result_domain(
            declared,
            domains::expression_domain(&caller.formals, nodes, root)?,
        )?;
        if formal
            .scalar_domain()
            .is_some_and(|expected| actual != Some(expected))
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        Ok(())
    }

    pub(super) fn project(
        &self,
        caller: &mut CalculusBuilder,
        axes: &[ComponentIndex],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        let Self::Value(value) = self else {
            let Self::Scalar(id) = self else {
                unreachable!()
            };
            return if axes.is_empty() {
                Ok(*id)
            } else {
                Err(PureOperatorError::FormalTypeMismatch)
            };
        };
        let axis = |index: ComponentIndex| match index {
            ComponentIndex::Fixed(_) => Ok(index),
            ComponentIndex::Result(index) => axes
                .get(usize::from(index))
                .copied()
                .ok_or(PureOperatorError::ResultAxisOutOfRange),
        };
        let mut ids = Vec::with_capacity(value.nodes.len());
        for node in &value.nodes {
            let node = match node {
                CalculusNode::FormalComponent { formal, axes } => CalculusNode::FormalComponent {
                    formal: *formal,
                    axes: axes.iter().copied().map(axis).collect::<Result<_, _>>()?,
                },
                CalculusNode::KroneckerDelta(left, right) => match (axis(*left)?, axis(*right)?) {
                    (ComponentIndex::Fixed(a), ComponentIndex::Fixed(b)) => {
                        CalculusNode::Rational {
                            value: ExactRational::integer(i64::from(a == b)),
                            dimension: DimExponents::DIMENSIONLESS,
                        }
                    }
                    (left, right) => CalculusNode::KroneckerDelta(left, right),
                },
                node => node.map_operands(|id| ids[id.index() as usize]),
            };
            ids.push(caller.push(node)?);
        }
        caller.embed_result(ids[value.root.index() as usize], &value.result)
    }
}
