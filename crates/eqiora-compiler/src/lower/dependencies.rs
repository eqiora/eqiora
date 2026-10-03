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
                LoweringExpressionNode::Partial { value, wrt } => {
                    if include_symbols {
                        names.insert(wrt.clone());
                    }
                    pending.push(value);
                }
                LoweringExpressionNode::Name(name) if include_symbols => {
                    names.insert(name.clone());
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
