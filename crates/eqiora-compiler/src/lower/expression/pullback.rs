//! Source and native expressions share the Kernel pullback typing rule.
use super::*;

pub(super) fn infer_factor<'a>(
    file: &str,
    expression: &LoweringExpression,
    factor: eqiora_schema::kernel::CoordinateMapFactor,
    source: &'a [LoweringExpression],
    at: &'a [(LoweringExpression, LoweringExpression)],
    infer: &mut impl FnMut(&'a LoweringExpression) -> Result<ExpressionType<RawId>, Diagnostic>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let source = source
        .iter()
        .map(&mut *infer)
        .collect::<Result<Vec<_>, _>>()?;
    let at = at
        .iter()
        .map(|(coordinate, mapped)| Ok((infer(coordinate)?, infer(mapped)?)))
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    factor.result_type(&source, &at).map_err(|error| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            error.to_string(),
        )
    })
}

pub(super) fn infer<'a>(
    file: &str,
    expression: &LoweringExpression,
    value: &'a LoweringExpression,
    source: &'a [LoweringExpression],
    at: &'a [(LoweringExpression, LoweringExpression)],
    infer: &mut impl FnMut(&'a LoweringExpression) -> Result<ExpressionType<RawId>, Diagnostic>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let value = infer(value)?;
    let source = source
        .iter()
        .map(&mut *infer)
        .collect::<Result<Vec<_>, _>>()?;
    let at = at
        .iter()
        .map(|(coordinate, mapped)| Ok((infer(coordinate)?, infer(mapped)?)))
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    value.pullback(&source, &at).map_err(|error| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            error.to_string(),
        )
    })
}

impl ExpressionLowerer<'_> {
    pub(super) fn lower_coordinate_map_factor_action(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        parameter: &LoweringExpression,
        directions: &[LoweringExpression],
    ) -> Result<TypedExpression, Diagnostic> {
        let ty = types::expression_type(self.file, expression, self.bindings, None)?;
        let value = self.lower(value)?.id;
        let parameter = self.lower(parameter)?.id;
        let directions = directions
            .iter()
            .map(|value| self.lower(value).map(|value| value.id))
            .collect::<Result<Vec<_>, _>>()?;
        let id = self
            .builder
            .coordinate_map_factor_action(value, parameter, directions)
            .map_err(|error| self.builder_error(expression, error))?;
        Ok(TypedExpression {
            id,
            dimension: ty.dimension(),
        })
    }
    pub(super) fn lower_coordinate_map_factor(
        &mut self,
        expression: &LoweringExpression,
        factor: eqiora_schema::kernel::CoordinateMapFactor,
        source: &[LoweringExpression],
        at: &[(LoweringExpression, LoweringExpression)],
    ) -> Result<TypedExpression, Diagnostic> {
        let ty = types::expression_type(self.file, expression, self.bindings, None)?;
        validate_selectors(self.file, source, at)?;
        let source = source
            .iter()
            .map(|coordinate| self.lower(coordinate).map(|value| value.id))
            .collect::<Result<Vec<_>, _>>()?;
        let at = at
            .iter()
            .map(|(coordinate, mapped)| Ok((self.lower(coordinate)?.id, self.lower(mapped)?.id)))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let id = self
            .builder
            .coordinate_map_factor(factor, source, at)
            .map_err(|failure| self.builder_error(expression, failure))?;
        Ok(TypedExpression {
            id,
            dimension: ty.dimension(),
        })
    }
    pub(super) fn lower_pullback(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        source: &[LoweringExpression],
        at: &[(LoweringExpression, LoweringExpression)],
    ) -> Result<TypedExpression, Diagnostic> {
        let ty = types::expression_type(self.file, expression, self.bindings, None)?;
        validate_selectors(self.file, source, at)?;
        let value = self.lower(value)?.id;
        let source = source
            .iter()
            .map(|coordinate| self.lower(coordinate).map(|value| value.id))
            .collect::<Result<Vec<_>, _>>()?;
        let at = at
            .iter()
            .map(|(coordinate, mapped)| Ok((self.lower(coordinate)?.id, self.lower(mapped)?.id)))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let id = self
            .builder
            .pullback(value, source, at)
            .map_err(|failure| self.builder_error(expression, failure))?;
        Ok(TypedExpression {
            id,
            dimension: ty.dimension(),
        })
    }
}

fn validate_selectors(
    file: &str,
    source: &[LoweringExpression],
    at: &[(LoweringExpression, LoweringExpression)],
) -> Result<(), Diagnostic> {
    for selectors in [
        source.iter().collect::<Vec<_>>(),
        at.iter().map(|(selector, _)| selector).collect(),
    ] {
        let mut seen = BTreeSet::new();
        for selector in selectors {
            let LoweringExpressionNode::Coordinate { factor, axis, .. } = selector.node.as_ref()
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    selector.range(),
                    "pullback selector must name an exact coordinate",
                ));
            };
            if !seen.insert((factor, axis)) {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    selector.range(),
                    "pullback repeats an exact coordinate through another binding",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn infer_action<'a>(
    file: &str,
    expression: &LoweringExpression,
    value: &'a LoweringExpression,
    parameter: &'a LoweringExpression,
    infer: &mut impl FnMut(&'a LoweringExpression) -> Result<ExpressionType<RawId>, Diagnostic>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    super::partial::result_type(&infer(value)?, &infer(parameter)?).map_err(|message| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    })
}
