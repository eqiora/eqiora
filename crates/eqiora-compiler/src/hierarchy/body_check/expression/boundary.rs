//! Resolve authored boundary selectors without adding interface laws.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn check_boundary(
        &mut self,
        expression: &Expr,
        operation: crate::math::boundary::Operation,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let error = |message: String| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                message,
            )
        };
        let selected =
            crate::math::boundary::source(arguments).map_err(|message| error(message.into()))?;
        if self.intrinsic && selected.on.is_none() {
            return Err(error(
                "let alias has no support context for this operator; supply on = boundary".into(),
            ));
        }
        if operation == crate::math::boundary::Operation::Trace
            && self.is_boundary_port_selection(selected.value)
        {
            return Err(error(
                "physical Port quantities require their declared `port.member` name".into(),
            ));
        }
        let explicit = selected
            .on
            .map(|name| {
                self.scope
                    .spatial_support(name)
                    .ok_or_else(|| error(format!("unknown boundary support `{name}`")))
            })
            .transpose()?;
        let from = selected
            .from
            .map(|name| {
                self.scope
                    .spatial_support(name)
                    .ok_or_else(|| error(format!("unknown parent support `{name}`")))
            })
            .transpose()?;
        let target = explicit.or_else(|| self.relation_support.clone());
        let operand = self.check(selected.value)?;
        operation
            .result_type(&operand, target.as_ref(), from.as_ref())
            .map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    message,
                )
            })
    }
}
