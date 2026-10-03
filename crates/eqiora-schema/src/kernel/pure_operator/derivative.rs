//! Exact ordered formal differentiation, shared by source compilation and IR.
use super::*;
use eqiora_core::ScalarDomain;

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
    /// Append the first partial of an explicit real scalar polynomial with
    /// respect to one declared formal. All other formals are held fixed.
    ///
    /// The original input occurrences and product-rule order are retained.
    /// A constant result produces zero with the exact quotient dimension.
    /// Explicit validity requirements retain their original predicate, even
    /// when the derivative value is zero; predicates are not differentiated.
    /// The append is atomic and uses the existing calculus node/depth bounds.
    ///
    /// # Errors
    /// Rejects invalid nodes/formals, unconstrained or non-real scalar input
    /// types, unsupported derivative rules, and unrepresentable dimensions.
    pub fn partial(
        &mut self,
        root: CalculusNodeId,
        formal: u16,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        if self.derivative_order(root)? >= 2 {
            return Err(PureOperatorError::DerivativeOrder);
        }
        let selected = self
            .formals
            .get(usize::from(formal))
            .ok_or(PureOperatorError::InvalidFormal(formal))?;
        if self.formals.iter().any(|value| {
            !value.is_invariant_scalar()
                || value.scalar_domain() != Some(ScalarDomain::Real)
                || value.dimension().is_none()
        }) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let output = self.value_type(root)?;
        if output.scalar_domain() != ScalarDomain::Real || !output.shape().is_scalar() {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let dimension = output
            .dimension()
            .div(selected.dimension().unwrap())
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        let (mut staged, result) = self.raw_partial(
            root,
            Selection::Formal(formal),
            selected.dimension().unwrap(),
            dimension,
        )?;
        let wrt = staged.push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })?;
        let result = staged.push(CalculusNode::Differentiated {
            value: result,
            source: root,
            wrt,
        })?;
        *self = staged;
        Ok(result)
    }
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
                CalculusNode::Require { value, .. }
                | CalculusNode::Differentiated { value, .. } => pending.push(*value),
                node => pending.extend(node.operands()),
            }
        }
        let mut staged = Self {
            formals: self.formals.clone(),
            result: self.result,
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
                CalculusNode::Differentiated { value, .. } | CalculusNode::BoundInput(value) => {
                    derivative(*value)
                }
                CalculusNode::Boolean(_)
                | CalculusNode::Compare(..)
                | CalculusNode::Not(_)
                | CalculusNode::And(..)
                | CalculusNode::Or(..) => None,
                CalculusNode::Require { condition, value } => {
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
                    Some(staged.push(CalculusNode::Require {
                        condition: *condition,
                        value,
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

impl CalculusBuilder {
    pub(super) fn derivative_order(&self, root: CalculusNodeId) -> Result<u8, PureOperatorError> {
        let end = definition_index(root, self.nodes.len())?;
        let mut orders: Vec<u8> = Vec::with_capacity(end + 1);
        for node in self.nodes.iter().take(end + 1) {
            let order = match node {
                CalculusNode::Differentiated { source, .. } => orders[source.index() as usize] + 1,
                _ => node
                    .operands()
                    .map(|id| orders[id.index() as usize])
                    .max()
                    .unwrap_or(0),
            };
            if order > 2 {
                return Err(PureOperatorError::DerivativeOrder);
            }
            orders.push(order);
        }
        Ok(orders[end])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_retains_validity_for_both_variable_and_typed_zero_derivatives() {
        let t = DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap();
        let class = |d| {
            PureValueClass::invariant_scalar()
                .with_dimension(d)
                .with_scalar_domain(ScalarDomain::Real)
                .unwrap()
        };
        let mut builder = CalculusBuilder::new([class(t), class(t)], class(t)).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let zero = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension: t,
            })
            .unwrap();
        let predicate = builder
            .push(CalculusNode::Compare(
                crate::kernel::ComparisonOp::Greater,
                x,
                zero,
            ))
            .unwrap();
        let root = builder
            .push(CalculusNode::Require {
                condition: predicate,
                value: x,
            })
            .unwrap();
        for formal in [0, 1] {
            let derived = builder.partial(root, formal).unwrap();
            let CalculusNode::Differentiated {
                value: derived,
                source,
                ..
            } = builder.nodes[derived.index() as usize]
            else {
                panic!("derivative history retained")
            };
            assert_eq!(source, root);
            let CalculusNode::Require { condition, value } =
                builder.nodes[derived.index() as usize]
            else {
                panic!("guard retained")
            };
            assert_eq!(condition, predicate);
            assert!(
                matches!(builder.nodes[value.index() as usize], CalculusNode::Rational { value, dimension }
                if value == ExactRational::integer(if formal == 0 { 1 } else { 0 }) && dimension == DimExponents::DIMENSIONLESS)
            );
        }
    }
    #[test]
    fn ordered_history_survives_a_zero_result_and_rejects_third_order_atomically() {
        let class = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap();
        let mut builder = CalculusBuilder::new([class, class], class).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let first = builder.partial(x, 1).unwrap();
        let second = builder.partial(first, 0).unwrap();
        let CalculusNode::Differentiated { source, wrt, .. } =
            builder.nodes[second.index() as usize]
        else {
            panic!("second history")
        };
        assert_eq!(source, first);
        assert!(matches!(
            builder.nodes[wrt.index() as usize],
            CalculusNode::FormalComponent { formal: 0, .. }
        ));
        let CalculusNode::Differentiated { source, wrt, .. } =
            builder.nodes[first.index() as usize]
        else {
            panic!("first history")
        };
        assert_eq!(source, x);
        assert!(matches!(
            builder.nodes[wrt.index() as usize],
            CalculusNode::FormalComponent { formal: 1, .. }
        ));
        let before = builder.nodes.clone();
        assert_eq!(
            builder.partial(second, 0),
            Err(PureOperatorError::DerivativeOrder)
        );
        assert_eq!(builder.nodes, before);
        builder.finish(second).unwrap();
    }
}
