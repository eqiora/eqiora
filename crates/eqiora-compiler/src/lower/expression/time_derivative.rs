//! Total time differentiation composes the shared formal partial transform.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_time_derivative(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        if self.sampling || (self.allow_discrete_symbols && !self.initial) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "smooth time differentiation requires a continuous evolution context",
            ));
        }
        let mut evolving = Vec::new();
        for name in value.referenced_names() {
            if name == "time" {
                evolving.push(name);
                continue;
            }
            match self.bindings.get(&name) {
                Some(Binding::Parameter(..)) => {}
                Some(Binding::Field(_, contract))
                    if self.eligible_evolution("derivative", contract) =>
                {
                    evolving.push(name)
                }
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "time derivative requires declared continuous states, fixed Parameters, and the enclosing time coordinate",
                    ));
                }
            }
        }
        // The time coordinate also supplies the quotient dimension for a fixed expression.
        if evolving.is_empty() {
            evolving.push("time".to_owned());
        }
        let mut total: Option<TypedExpression> = None;
        for name in evolving {
            let partial = self.lower_partial(expression, value, &name)?;
            let term = if name == "time" {
                partial
            } else {
                let argument = LoweringExpression::name(name, expression.range());
                let rate = self.lower_call(expression, "derivative", &argument)?;
                let dimension = partial
                    .dimension
                    .mul(rate.dimension)
                    .ok_or_else(|| dimension_overflow(self.file, expression.range()))?;
                let id = self
                    .builder
                    .mul(partial.id, rate.id)
                    .map_err(|failure| self.builder_error(expression, failure))?;
                TypedExpression { id, dimension }
            };
            total = Some(match total {
                Some(previous) => TypedExpression {
                    id: self
                        .builder
                        .add(previous.id, term.id)
                        .map_err(|failure| self.builder_error(expression, failure))?,
                    dimension: term.dimension,
                },
                None => term,
            });
        }
        Ok(total.expect("at least the fixed-time partial"))
    }
}
