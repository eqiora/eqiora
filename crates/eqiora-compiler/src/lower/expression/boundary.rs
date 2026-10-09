//! Resolve boundary selectors through the existing Domain bindings.
use super::*;
use crate::math::boundary::Operation;

pub(super) fn selected_supports(
    file: &str,
    range: TextRange,
    on: Option<&str>,
    from: Option<&str>,
    bindings: &BTreeMap<String, Binding>,
    implicit: Option<&SpatialSupport<RawId>>,
) -> Result<(SpatialSupport<RawId>, Option<SpatialSupport<RawId>>), Diagnostic> {
    let target = on
        .map(|name| types::relation_support(file, range, name, bindings))
        .transpose()?
        .or_else(|| implicit.cloned())
        .filter(|support| {
            matches!(
                support,
                SpatialSupport::Boundary { .. } | SpatialSupport::PhysicalInterface { .. }
            )
        })
        .ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "boundary operator requires an exact boundary through on or its Relation scope",
            )
        })?;
    let from = from
        .map(|name| types::relation_support(file, range, name, bindings))
        .transpose()?;
    Ok((target, from))
}

// Resolve selectors outside the recursive operand-typing frame.
pub(super) fn infer<'a>(
    file: &str,
    expression: &'a LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
    support: Option<&SpatialSupport<RawId>>,
    infer_operand: &mut impl FnMut(&'a LoweringExpression) -> Result<ExpressionType<RawId>, Diagnostic>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let LoweringExpressionNode::Boundary {
        operation,
        argument,
        on,
        from,
    } = expression.node.as_ref()
    else {
        unreachable!("boundary dispatch")
    };
    let (target, from) = selected_supports(
        file,
        expression.range(),
        on.as_deref(),
        from.as_deref(),
        bindings,
        support,
    )?;
    operation
        .result_type(&infer_operand(argument)?, Some(&target), from.as_ref())
        .map_err(|message| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                message,
            )
        })
}

impl ExpressionLowerer<'_> {
    pub(super) fn lower_boundary(
        &mut self,
        expression: &LoweringExpression,
        operation: Operation,
        argument: &LoweringExpression,
        on: Option<&str>,
        from: Option<&str>,
    ) -> Result<TypedExpression, Diagnostic> {
        let (target, from) = selected_supports(
            self.file,
            expression.range(),
            on,
            from,
            self.bindings,
            self.support.as_ref(),
        )?;
        let operand_type =
            types::expression_type(self.file, argument, self.bindings, self.support.as_ref())?;
        let result_type = operation
            .result_type(&operand_type, Some(&target), from.as_ref())
            .map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    message,
                )
            })?;
        let on = target.domain().downcast::<kinds::Domain>().ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "boundary selection has a non-Domain identity",
            )
        })?;
        let operand = self.lower(argument)?;
        let id = match operation {
            Operation::Trace => self.builder.trace(operand.id, on),
            Operation::Normal => self.builder.normal_component(operand.id, on),
            Operation::Tangential => {
                let (definition, _) = crate::math::oriented::Operation::TangentialTrace
                    .definition(&operand_type)
                    .map_err(|message| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.file,
                            expression.range(),
                            message,
                        )
                    })?;
                let lifted = self
                    .builder
                    .pure_operator(&definition, [operand.id])
                    .map_err(|error| self.builder_error(expression, error))?;
                self.builder.normal_component(lifted, on)
            }
        }
        .map_err(|error| self.builder_error(expression, error))?;
        Ok(TypedExpression {
            id,
            dimension: result_type.dimension(),
        })
    }
}
