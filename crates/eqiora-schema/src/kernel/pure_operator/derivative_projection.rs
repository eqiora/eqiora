//! On-demand ordered polynomial derivatives for proof consumers only.
use super::*;
use std::borrow::Cow;

impl PureOperatorDefinition {
    /// Expand formal derivatives with the exact polynomial rules for classification.
    /// This proof view never selects executable arithmetic or replaces definition identity.
    /// Definitions without formal derivatives are borrowed unchanged.
    ///
    /// # Errors
    /// Returns an error when a derivative is outside the polynomial profile or its
    /// bounded expansion exceeds the node, depth, dimension, or arithmetic limits.
    /// Such a failure means no polynomial proof, not an invalid numerical definition.
    pub fn polynomial_derivative_view(&self) -> Result<Cow<'_, Self>, PureOperatorError> {
        if !self
            .nodes
            .iter()
            .any(|n| matches!(n, CalculusNode::Differentiated { .. }))
        {
            return Ok(Cow::Borrowed(self));
        }
        let mut builder = CalculusBuilder::new(self.formals.iter().cloned(), self.result.clone())?;
        let mut ids = Vec::<CalculusNodeId>::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let node = node.map_operands(|id| ids[id.index() as usize]);
            let id = if let CalculusNode::Differentiated { source, wrt } = node {
                let selection = match builder.nodes[wrt.index() as usize] {
                    CalculusNode::FormalComponent { formal, ref axes } if axes.is_empty() => {
                        Selection::Formal(formal)
                    }
                    CalculusNode::BoundInput(_) => Selection::Occurrence(wrt),
                    _ => return Err(PureOperatorError::FormalTypeMismatch),
                };
                let input = builder.value_type(wrt)?.dimension();
                let output = builder
                    .value_type(source)?
                    .dimension()
                    .div(input)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                let (expanded, id) = builder.raw_partial(source, selection, input, output)?;
                builder = expanded;
                id
            } else {
                builder.push(node)?
            };
            ids.push(id);
        }
        Ok(Cow::Owned(builder.finish(ids[self.root.index() as usize])?))
    }
}

#[derive(Clone, Copy)]
pub(super) enum Selection {
    Formal(u16),
    Occurrence(CalculusNodeId),
}
impl Selection {
    fn matches(self, id: CalculusNodeId, node: &CalculusNode) -> bool {
        match self {
            Self::Formal(selected) => {
                matches!(node, CalculusNode::FormalComponent { formal, axes } if *formal == selected && axes.is_empty())
            }
            Self::Occurrence(selected) => id == selected,
        }
    }
}

impl CalculusBuilder {
    pub(super) fn raw_partial(
        &self,
        root: CalculusNodeId,
        selection: Selection,
        input_dimension: DimExponents,
        dimension: DimExponents,
    ) -> Result<(Self, CalculusNodeId), PureOperatorError> {
        let root_index = definition_index(root, self.nodes.len())?;
        let mut reachable = vec![false; root_index + 1];
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            let index = definition_index(id, reachable.len())?;
            if std::mem::replace(&mut reachable[index], true) {
                continue;
            }
            if selection.matches(id, &self.nodes[index]) {
                continue;
            }
            // A validity predicate is retained verbatim, not differentiated.
            match &self.nodes[index] {
                CalculusNode::Require { value, .. } | CalculusNode::Demand { value, .. } => {
                    pending.push(*value)
                }
                node => pending.extend(node.operands()),
            }
        }
        let mut staged = Self {
            formals: self.formals.clone(),
            result: self.result.clone(),
            nodes: self.nodes.clone(),
            depths: self.depths.clone(),
        };
        let mut derivatives = vec![None; root_index + 1];
        for (index, node) in self.nodes.iter().take(root_index + 1).enumerate() {
            if !reachable[index] {
                continue;
            }
            let derivative = |id: CalculusNodeId| derivatives[id.index() as usize];
            if selection.matches(CalculusNodeId(index as u32), node) {
                derivatives[index] = Some(staged.push(CalculusNode::Rational {
                    value: ExactRational::integer(1),
                    dimension: DimExponents::DIMENSIONLESS,
                })?);
                continue;
            }
            derivatives[index] = match node {
                CalculusNode::Rational { .. } => None,
                CalculusNode::BoundInput(value) => derivative(*value),
                CalculusNode::Boolean(_)
                | CalculusNode::Compare(..)
                | CalculusNode::Not(_)
                | CalculusNode::And(..)
                | CalculusNode::Or(..) => None,
                CalculusNode::Require { value, .. } | CalculusNode::Demand { value, .. } => {
                    let value = match derivative(*value) {
                        Some(value) => value,
                        None => {
                            let dimension = self
                                .value_type(*value)?
                                .dimension()
                                .div(input_dimension)
                                .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                            staged.push(CalculusNode::Rational {
                                value: ExactRational::integer(0),
                                dimension,
                            })?
                        }
                    };
                    Some(staged.push(match node {
                        CalculusNode::Require { condition, .. } => CalculusNode::Require {
                            condition: *condition,
                            value,
                        },
                        CalculusNode::Demand { prerequisite, .. } => CalculusNode::Demand {
                            prerequisite: *prerequisite,
                            value,
                        },
                        _ => unreachable!("matched value prerequisite"),
                    })?)
                }
                CalculusNode::FormalComponent { axes, .. } if axes.is_empty() => None,
                CalculusNode::Neg(value) => derivative(*value)
                    .map(|value| staged.push(CalculusNode::Neg(value)))
                    .transpose()?,
                CalculusNode::Add(left, right) => {
                    sum(&mut staged, derivative(*left), derivative(*right))?
                }
                CalculusNode::Mul(left, right) => {
                    let first = derivative(*left)
                        .map(|value| staged.push(CalculusNode::Mul(value, *right)))
                        .transpose()?;
                    let second = derivative(*right)
                        .map(|value| staged.push(CalculusNode::Mul(*left, value)))
                        .transpose()?;
                    sum(&mut staged, first, second)?
                }
                _ => return Err(PureOperatorError::FormalTypeMismatch),
            };
        }
        let result = match derivatives[root_index] {
            Some(value) => value,
            None => staged.push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension,
            })?,
        };
        Ok((staged, result))
    }
}

fn sum(
    builder: &mut CalculusBuilder,
    left: Option<CalculusNodeId>,
    right: Option<CalculusNodeId>,
) -> Result<Option<CalculusNodeId>, PureOperatorError> {
    match (left, right) {
        (Some(left), Some(right)) => builder.push(CalculusNode::Add(left, right)).map(Some),
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}
