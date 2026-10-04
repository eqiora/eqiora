use crate::lower::LoweringExpression;

use super::Scope;
use crate::hierarchy::body_check::DependencyActivation;
use crate::hierarchy::parameters::{
    ParameterLineage, ResolvedParameter, SymbolicParameterMap, SymbolicParameterValue,
};

#[derive(Debug, Clone)]
pub(super) enum ScopedValue {
    Coordinate(LoweringExpression),
    Static(ResolvedParameter),
    Runtime {
        expression: LoweringExpression,
        activation: DependencyActivation,
    },
}

impl Scope {
    pub(in crate::hierarchy) fn coordinate(&self, name: &str) -> Option<&LoweringExpression> {
        match self.values.get(name)? {
            ScopedValue::Coordinate(value) => Some(value),
            _ => None,
        }
    }
    pub(in crate::hierarchy) fn insert_coordinate(
        &mut self,
        name: String,
        expression: LoweringExpression,
    ) -> Result<(), &'static str> {
        if self
            .values
            .insert(name, ScopedValue::Coordinate(expression))
            .is_some()
        {
            Err("coordinate collides with a scoped value")
        } else {
            Ok(())
        }
    }

    pub(in crate::hierarchy) fn insert_parameter(
        &mut self,
        name: String,
        parameter: ResolvedParameter,
    ) -> Option<ResolvedParameter> {
        self.values
            .insert(name, ScopedValue::Static(parameter))
            .and_then(|value| match value {
                ScopedValue::Static(value) => Some(value),
                ScopedValue::Runtime { .. } | ScopedValue::Coordinate(_) => None,
            })
    }

    pub(in crate::hierarchy) fn parameter(&self, name: &str) -> Option<&ResolvedParameter> {
        match self.values.get(name) {
            Some(ScopedValue::Static(value)) => Some(value),
            _ => None,
        }
    }

    pub(in crate::hierarchy) fn symbolic_parameters(&self) -> SymbolicParameterMap {
        self.values
            .iter()
            .filter_map(|(name, value)| {
                let ScopedValue::Static(value) = value else {
                    return None;
                };
                Some((
                    name.clone(),
                    SymbolicParameterValue {
                        value: Some(value.value.clone()),
                        value_type: value.value.value_type().clone(),
                        expression: Some(value.expression.clone()),
                        lineage: Some(value.lineage.clone()),
                    },
                ))
            })
            .collect()
    }

    pub(in crate::hierarchy) fn value_expression(&self, name: &str) -> Option<LoweringExpression> {
        match self.values.get(name)? {
            ScopedValue::Static(_) => Some(self.parameter_expression(name)),
            ScopedValue::Runtime { expression, .. } | ScopedValue::Coordinate(expression) => {
                Some(expression.clone())
            }
        }
    }

    pub(in crate::hierarchy) fn value_activation(
        &self,
        name: &str,
    ) -> Option<DependencyActivation> {
        self.values.get(name).map(|value| match value {
            ScopedValue::Static(_) => DependencyActivation::Static,
            ScopedValue::Runtime { activation, .. } => activation.clone(),
            ScopedValue::Coordinate(_) => DependencyActivation::Static,
        })
    }

    pub(in crate::hierarchy) fn insert_runtime_let(
        &mut self,
        name: String,
        expression: LoweringExpression,
        activation: DependencyActivation,
    ) -> Result<(), &'static str> {
        if self
            .values
            .insert(
                name,
                ScopedValue::Runtime {
                    expression,
                    activation,
                },
            )
            .is_some()
        {
            Err("runtime let alias collides with a scoped value")
        } else {
            Ok(())
        }
    }

    pub(in crate::hierarchy) fn insert_let(
        &mut self,
        name: String,
        value: SymbolicParameterValue,
    ) -> Result<(), &'static str> {
        let (Some(scalar), Some(expression)) = (value.value, value.expression) else {
            return Err("static let alias did not resolve to a closed expression");
        };
        let resolved = ResolvedParameter {
            value: scalar,
            expression,
            lineage: value.lineage.unwrap_or(ParameterLineage::Derived),
        };
        if self
            .values
            .insert(name, ScopedValue::Static(resolved))
            .is_some()
        {
            Err("static let alias collides with a compile-time value")
        } else {
            Ok(())
        }
    }
}
