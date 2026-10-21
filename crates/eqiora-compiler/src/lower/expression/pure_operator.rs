//! Lower retained typed pointwise operator applications.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_pure_operator(
        &mut self,
        expression: &LoweringExpression,
        definition: &PureOperatorDefinition,
        arguments: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        let arguments = arguments
            .iter()
            .map(|argument| self.lower(argument))
            .collect::<Result<Vec<_>, _>>()?;
        let dimension = definition
            .result_dimension(
                &arguments
                    .iter()
                    .map(|argument| argument.dimension)
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    error.to_string(),
                )
            })?;
        self.builder
            .pure_operator(definition, arguments.iter().map(|argument| argument.id))
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }
}

impl ExpressionLowerer<'_> {
    pub(super) fn lower_tensor(
        &mut self,
        expression: &LoweringExpression,
        operation: &crate::math::tensor::Operation,
        arguments: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        let types = arguments
            .iter()
            .map(|argument| {
                expression_type(self.file, argument, self.bindings, self.support.as_ref())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let definition = operation.definition(&types).map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                error.to_string(),
            )
        })?;
        self.lower_pure_operator(expression, &definition, arguments)
    }
}
