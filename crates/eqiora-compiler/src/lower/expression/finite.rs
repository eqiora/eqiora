//! Emit nominal finite operations after shared type admission.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_finite(
        &mut self,
        expression: &LoweringExpression,
        operation: crate::math::finite::Operation,
        arguments: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        let types = arguments
            .iter()
            .map(|argument| {
                expression_type(self.file, argument, self.bindings, self.support.as_ref())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(definition) = operation.spatial_definition(&types) {
            let definition = definition.map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    error.to_string(),
                )
            })?;
            return self.lower_pure_operator(expression, &definition, arguments);
        }
        let dimension = operation
            .result_type(&types)
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    error.to_string(),
                )
            })?
            .value_type
            .dimension();
        let operands = arguments
            .iter()
            .map(|value| self.lower(value))
            .collect::<Result<Vec<_>, _>>()?;
        operation
            .emit(
                &mut self.builder,
                &operands.iter().map(|value| value.id).collect::<Vec<_>>(),
            )
            .map(|id| TypedExpression { id, dimension })
            .map_err(|error| self.builder_error(expression, error))
    }
}
