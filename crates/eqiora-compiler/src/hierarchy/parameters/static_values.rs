//! Closed values and exact structural selectors share the Parameter evaluator.
use super::*;

/// Evaluate a closed declaration through the same typed static expression owner.
pub(crate) fn closed_value(
    file: &str,
    expression: &Expr,
    target: ValueType,
) -> Result<ValueLiteral, Diagnostic> {
    closed_value_with_frames(file, expression, target, &mut |_| None)
}

/// Infer a closed value through the ordinary static evaluator, without a nominal cast.
pub(crate) fn infer_closed_value(
    file: &str,
    expression: &Expr,
) -> Result<ValueLiteral, Diagnostic> {
    let evaluated = expression_eval::evaluate_parameter_expression(
        file,
        expression,
        ExpressionContext::Let,
        &mut |name, range| {
            Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("closed value cannot depend on `{name}`"),
            ))
        },
        &mut |_| None,
        &mut |_| None,
    )?;
    evaluated.value.ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "closed value remained symbolic",
        )
    })
}

pub(in crate::hierarchy) fn closed_value_with_frames(
    file: &str,
    expression: &Expr,
    target: ValueType,
    resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
) -> Result<ValueLiteral, Diagnostic> {
    let evaluated = evaluate_initializer(
        file,
        expression,
        ExpressionContext::Let,
        &mut |name, range| {
            Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("closed value cannot depend on `{name}`"),
            ))
        },
        target.clone(),
        "declared value",
        (&mut |_| None, &mut *resolve_frame),
    )?;
    let value = coerce_parameter_with_label(
        file,
        expression.range(),
        evaluated,
        target,
        "declared value",
        true,
    )?;
    value.value.ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "closed value remained symbolic",
        )
    })
}

pub(in crate::hierarchy) fn static_index(
    file: &str,
    expression: &Expr,
    values: &SymbolicParameterMap,
) -> Result<u32, Diagnostic> {
    structural_index(file, expression, values)?
        .map(|(index, _)| index)
        .ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                "channel index requires an exact static integer value",
            )
        })
}

pub(in crate::hierarchy) fn static_slice(
    file: &str,
    lower: &Expr,
    upper: &Expr,
    values: &SymbolicParameterMap,
) -> Result<(u32, u32), Diagnostic> {
    let start = static_index(file, lower, values)?;
    let end = static_index(file, upper, values)?;
    if end
        .checked_sub(start)
        .is_none_or(|count| count == 0 || count > 65_536)
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            upper.range(),
            "slice requires increasing exact bounds and at most 65536 channels",
        ));
    }
    Ok((start, end))
}

pub(in crate::hierarchy) fn structural_extent(
    file: &str,
    expression: &Expr,
    values: &SymbolicParameterMap,
) -> Result<Option<(u32, Vec<String>)>, Diagnostic> {
    let value = structural_index(file, expression, values)?;
    if value.as_ref().is_some_and(|(extent, _)| *extent == 0) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "index set extent requires a positive exact integer",
        ));
    }
    Ok(value)
}

pub(in crate::hierarchy) fn index_set_extent(
    file: &str,
    declaration: &eqiora_lang::NamedDefinitionDecl,
    values: &SymbolicParameterMap,
) -> Result<Option<(u32, Vec<String>)>, Diagnostic> {
    let invalid = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "index set requires only range(extent), without alias assertions",
        )
    };
    let ExprKind::Call {
        callee,
        arguments: eqiora_lang::CallArguments::Positional(arguments),
    } = declaration.value().kind()
    else {
        return Err(invalid());
    };
    let [extent] = arguments.as_slice() else {
        return Err(invalid());
    };
    if callee.as_str() != "range"
        || declaration.value_type().is_some()
        || declaration.domain().is_some()
        || declaration.activation().is_some()
    {
        return Err(invalid());
    }
    structural_extent(file, extent, values)
}

pub(in crate::hierarchy) fn structural_index(
    file: &str,
    expression: &Expr,
    values: &SymbolicParameterMap,
) -> Result<Option<(u32, Vec<String>)>, Diagnostic> {
    let evaluated = expression_eval::evaluate_with_domain(
        file,
        expression,
        ExpressionContext::Let,
        &mut |name, range| {
            values.get(name).cloned().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "index set extent depends on an unknown or runtime value",
                )
            })
        },
        &mut |_| None,
        &mut |_| None,
        Some(ScalarDomain::Integer),
    )?;
    let Some(value) = evaluated.value else {
        return Ok(None);
    };
    let extent = value
        .integer_scalar_value()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                "structural index requires a nonnegative exact integer within u32 bounds",
            )
        })?;
    let dependencies = evaluated
        .expression
        .map(|expression| expression.referenced_names().into_iter().collect())
        .unwrap_or_default();
    Ok(Some((extent, dependencies)))
}
