//! Original symbol dependencies of shared expression DAGs used by structural elaboration.
use super::{LoweringExpression, LoweringExpressionNode};
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;

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
                LoweringExpressionNode::Coordinate {
                    support, factor, ..
                } if include_symbols => {
                    names.extend([support.clone(), factor.clone()]);
                }
                LoweringExpressionNode::Name(name) if include_symbols => {
                    names.insert(name.clone());
                }
                LoweringExpressionNode::Boundary { on, from, .. } if include_symbols => {
                    names.extend(on.iter().chain(from.iter()).cloned());
                }
                _ => {}
            }
            value.node.push_children(&mut pending);
        }
        names
    }
}
