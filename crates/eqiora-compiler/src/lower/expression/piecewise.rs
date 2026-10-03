//! Emit the shared bounded recipes into the ordinary expression DAG.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_piecewise(
        &mut self,
        expression: &LoweringExpression,
        name: &str,
        arguments: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        use crate::math::piecewise::{self, Primitive};
        let operands = arguments
            .iter()
            .map(|value| self.lower(value))
            .collect::<Result<Vec<_>, _>>()?;
        let dimension = operands
            .first()
            .map_or(DimExponents::DIMENSIONLESS, |value| value.dimension);
        piecewise::emit(name, &operands, dimension, |primitive| {
            let (result, dimension) = match primitive {
                Primitive::Constant(value, dimension) => (
                    self.builder.constant(
                        eqiora_core::ValueLiteral::from_real(
                            eqiora_core::ValueType::scalar(
                                eqiora_core::ScalarDomain::Real,
                                dimension,
                            )
                            .expect("admitted numeric scalar type"),
                            f64::from(value),
                        )
                        .expect("small exact real coefficient"),
                    ),
                    dimension,
                ),
                Primitive::Compare(op, left, right) => (
                    self.builder.compare(op, left.id, right.id),
                    DimExponents::DIMENSIONLESS,
                ),
                Primitive::Select {
                    condition,
                    then_value,
                    else_value,
                } => (
                    self.builder
                        .select(condition.id, then_value.id, else_value.id),
                    then_value.dimension,
                ),
                Primitive::Require { condition, value } => (
                    self.builder.require(condition.id, value.id),
                    value.dimension,
                ),
            };
            result
                .map(|id| TypedExpression { id, dimension })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic))
        })
        .ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "invalid piecewise builtin arity",
            )
        })?
    }
}
