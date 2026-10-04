//! Coordinate binding retains its expression until an admitted representation evaluates it.
use super::*;

pub(crate) fn result_type<I: Clone + Eq>(
    value: &ExpressionType<I>,
    at: &[(ExpressionType<I>, ExpressionType<I>)],
    side: bool,
) -> Result<ExpressionType<I>, &'static str> {
    let error = "evaluate requires every exact coordinate once, lumped real points with matching units, and a one-dimensional support for a side";
    let Some((coordinate, _)) = at.first() else {
        return Err(error);
    };
    let Some(support @ (SpatialSupport::Coordinates { .. } | SpatialSupport::Volume { .. })) =
        coordinate.support.as_ref()
    else {
        return Err(error);
    };
    if at.len() > 64
        || at.len() != support.intrinsic_dimensions()
        || (side && at.len() != 1)
        || value.support.as_ref().is_some_and(|value| value != support)
    {
        return Err(error);
    }
    for (coordinate, point) in at {
        if coordinate.support.as_ref() != Some(support)
            || point.support.is_some()
            || point.dimension() != coordinate.dimension()
            || !point.shape().is_scalar()
            || point.value_type.array_rank() != 0
            || point.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
            || point.value_type.frame() != eqiora_core::ValueFrame::Invariant
        {
            return Err(error);
        }
    }
    Ok(ExpressionType::new(value.value_type.clone(), None))
}

// Keep the pair inventory out of every recursive ordinary-expression frame.
pub(super) fn infer<'a>(
    file: &str,
    expression: &LoweringExpression,
    value: &'a LoweringExpression,
    at: &'a [(LoweringExpression, LoweringExpression)],
    side: bool,
    infer: &mut impl FnMut(&'a LoweringExpression) -> Result<ExpressionType<RawId>, Diagnostic>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let value = infer(value)?;
    let points = at
        .iter()
        .map(|(coordinate, point)| Ok((infer(coordinate)?, infer(point)?)))
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    result_type(&value, &points, side).map_err(|message| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    })
}

impl ExpressionLowerer<'_> {
    pub(super) fn lower_evaluate(
        &mut self,
        expression: &LoweringExpression,
        value: &LoweringExpression,
        at: &[(LoweringExpression, LoweringExpression)],
        side: Option<eqiora_schema::kernel::BoundarySide>,
    ) -> Result<TypedExpression, Diagnostic> {
        let ty = types::expression_type(self.file, expression, self.bindings, None)?;
        let mut axes = BTreeSet::new();
        for (coordinate, _) in at {
            let LoweringExpressionNode::Coordinate { factor, axis, .. } = coordinate.node.as_ref()
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "evaluation selector must name an exact coordinate",
                ));
            };
            if !axes.insert((factor, axis)) {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "evaluation repeats an exact coordinate through another binding",
                ));
            }
        }
        let value = self.lower(value)?.id;
        let points = at
            .iter()
            .map(|(coordinate, point)| Ok((self.lower(coordinate)?.id, self.lower(point)?.id)))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let id = self
            .builder
            .evaluate_at(value, points, side)
            .map_err(|failure| self.builder_error(expression, failure))?;
        Ok(TypedExpression {
            id,
            dimension: ty.dimension(),
        })
    }
}
