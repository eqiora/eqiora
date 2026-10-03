//! Emit nominal finite operations after shared type admission.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_finite(
        &mut self,
        expression: &LoweringExpression,
        operation: crate::math::finite::Operation,
        arguments: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        let operands = arguments
            .iter()
            .map(|value| self.lower(value))
            .collect::<Result<Vec<_>, _>>()?;
        let dimension = match operands.as_slice() {
            [value] => value.dimension,
            [left, right] => left
                .dimension
                .mul(right.dimension)
                .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "finite operation has incorrect arity",
                ));
            }
        };
        operation
            .emit(
                &mut self.builder,
                &operands.iter().map(|value| value.id).collect::<Vec<_>>(),
            )
            .map(|id| TypedExpression { id, dimension })
            .map_err(|error| self.builder_error(expression, error))
    }
}
