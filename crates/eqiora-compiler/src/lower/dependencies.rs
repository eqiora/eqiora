//! Original symbol dependencies of shared expression DAGs used by structural elaboration.
use super::{LoweringExpression, LoweringExpressionNode};
use std::collections::BTreeSet;

impl LoweringExpression {
    pub(crate) fn referenced_names(&self) -> BTreeSet<String> {
        self.dependencies(true)
    }

    pub(crate) fn structural_parameters(&self) -> BTreeSet<String> {
        self.dependencies(false)
    }

    pub(crate) fn with_structural_parameters(
        mut self,
        names: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut parameters = self
            .structural_parameters
            .as_deref()
            .cloned()
            .unwrap_or_default();
        parameters.extend(names);
        if !parameters.is_empty() {
            self.structural_parameters = Some(std::sync::Arc::new(parameters));
        }
        self
    }

    fn dependencies(&self, include_symbols: bool) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![self];
        while let Some(value) = pending.pop() {
            if let Some(parameters) = &value.structural_parameters {
                names.extend(parameters.iter().cloned());
            }
            if !seen.insert(std::sync::Arc::as_ptr(&value.node) as usize) {
                continue;
            }
            match value.node.as_ref() {
                LoweringExpressionNode::CoordinateMapFactor { source, at, .. } => {
                    pending.extend(source);
                    pending.extend(
                        at.iter()
                            .flat_map(|(coordinate, mapped)| [coordinate, mapped]),
                    );
                }
                LoweringExpressionNode::Pullback { value, source, at } => {
                    pending.push(value);
                    pending.extend(source);
                    pending.extend(
                        at.iter()
                            .flat_map(|(coordinate, mapped)| [coordinate, mapped]),
                    );
                }
                LoweringExpressionNode::Evaluate { value, at, .. } => {
                    pending.push(value);
                    pending.extend(
                        at.iter()
                            .flat_map(|(coordinate, point)| [coordinate, point]),
                    );
                }
                LoweringExpressionNode::Partial { value, wrt } => {
                    pending.extend([value, wrt]);
                }
                LoweringExpressionNode::Coordinate {
                    support, factor, ..
                } if include_symbols => {
                    names.extend([support.clone(), factor.clone()]);
                }
                LoweringExpressionNode::Name(name) if include_symbols => {
                    names.insert(name.clone());
                }
                LoweringExpressionNode::Boundary {
                    argument, on, from, ..
                } => {
                    pending.push(argument);
                    if include_symbols {
                        names.extend(on.iter().chain(from.iter()).cloned());
                    }
                }
                LoweringExpressionNode::Not(value)
                | LoweringExpressionNode::Neg(value)
                | LoweringExpressionNode::Index { value, .. }
                | LoweringExpressionNode::Call {
                    argument: value, ..
                }
                | LoweringExpressionNode::Sample { value, .. } => pending.push(value),
                LoweringExpressionNode::Array(values)
                | LoweringExpressionNode::IntegerCall {
                    arguments: values, ..
                }
                | LoweringExpressionNode::Finite {
                    arguments: values, ..
                }
                | LoweringExpressionNode::Piecewise {
                    arguments: values, ..
                }
                | LoweringExpressionNode::Property {
                    arguments: values, ..
                }
                | LoweringExpressionNode::PureOperator {
                    arguments: values, ..
                } => pending.extend(values),
                LoweringExpressionNode::Case { value, arms } => {
                    pending.push(value);
                    pending.extend(arms.iter().map(|(_, value)| value));
                }
                LoweringExpressionNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => pending.extend([condition, then_value, else_value]),
                LoweringExpressionNode::Require { condition, value } => {
                    pending.extend([condition, value])
                }
                LoweringExpressionNode::Binary { left, right, .. }
                | LoweringExpressionNode::Extremum { left, right, .. } => {
                    pending.push(left);
                    pending.push(right);
                }
                LoweringExpressionNode::Complex { real, imag } => {
                    pending.push(real);
                    pending.push(imag);
                }
                _ => {}
            }
        }
        names
    }
}
