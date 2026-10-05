//! Target-directed initializer typing with explicit value demand.
use super::*;

pub(in crate::hierarchy::parameters) fn evaluate_initializer(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    target: ValueType,
    label: &str,
    (resolve_clock, resolve_frame): StaticContexts<'_>,
) -> Result<EvaluatedParameter, Diagnostic> {
    evaluate_initializer_mode(
        file,
        expression,
        context,
        resolve,
        target,
        (label, true),
        (resolve_clock, resolve_frame),
    )
}

pub(in crate::hierarchy::parameters) fn evaluate_initializer_mode(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    target: ValueType,
    (label, evaluate_values): (&str, bool),
    (resolve_clock, resolve_frame): StaticContexts<'_>,
) -> Result<EvaluatedParameter, Diagnostic> {
    let mut evaluated = if matches!(expression.kind(), ExprKind::Call { callee, .. } if matches!(callee.as_str(), "coordinates" | "linear_map" | "identity"))
        && target.finite_bases().next().is_some()
        && matches!(
            target.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
        let value = crate::nominal::contextual_literal(file, expression, &target)?;
        EvaluatedParameter {
            value: evaluate_values.then(|| value.clone()),
            value_type: EvaluatedType::Known(target.clone()),
            bare_literal: false,
            expression: Some(LoweringExpression::literal(value, expression.range())),
            lineage: Some(ParameterLineage::Constant),
        }
    } else if matches!(expression.kind(), ExprKind::Call { callee, .. } if callee.as_str() == "tensor_value")
    {
        super::super::tensor_values::evaluate(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            Some(&target),
            evaluate_values,
        )?
    } else if matches!(expression.kind(), ExprKind::Array(_))
        || matches!(expression.kind(), ExprKind::Call { callee, .. } if callee.as_str() == "math.complex")
    {
        super::super::value_expressions::evaluate_mode(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            Some(&target),
            evaluate_values,
        )?
    } else {
        evaluate_mode(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            (target.scalar_domain() == ScalarDomain::Integer).then_some(ScalarDomain::Integer),
            evaluate_values,
        )?
    };
    if !evaluate_values && evaluated.bare_literal {
        // Literal projection supplies its exact authored value for contextual typing;
        // no arithmetic subtree is evaluated on this path.
        evaluated = evaluate_mode(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            (target.scalar_domain() == ScalarDomain::Integer).then_some(ScalarDomain::Integer),
            true,
        )?;
    }
    let declaration_initializer = !matches!(
        context,
        ExpressionContext::Binding | ExpressionContext::IndexedBinding(_)
    );
    let mut result: EvaluatedParameter = coerce_parameter_with_label(
        file,
        expression.range(),
        evaluated,
        target,
        label,
        declaration_initializer,
    )?
    .into();
    if !evaluate_values {
        result.value = None;
    }
    Ok(result)
}

pub(in crate::hierarchy::parameters) fn coerce_parameter(
    file: &str,
    range: TextRange,
    evaluated: EvaluatedParameter,
    target: ValueType,
) -> Result<SymbolicParameterValue, Diagnostic> {
    coerce_parameter_with_label(file, range, evaluated, target, "Parameter binding", false)
}

pub(in crate::hierarchy::parameters) fn coerce_parameter_with_label(
    file: &str,
    range: TextRange,
    evaluated: EvaluatedParameter,
    target: ValueType,
    label: &str,
    declaration_initializer: bool,
) -> Result<SymbolicParameterValue, Diagnostic> {
    if evaluated.bare_literal
        && (declaration_initializer
            || evaluated.value.as_ref().is_some_and(ValueLiteral::is_zero)
            || target.dimension() == DimExponents::DIMENSIONLESS)
    {
        let value = evaluated
            .value
            .as_ref()
            .expect("bare literals have a known value");
        let literal = if target.scalar_domain() == ScalarDomain::Integer {
            let integer = value.integer_scalar_value().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "integer context requires an exact integer literal",
                )
            })?;
            ValueLiteral::from_integer(target.clone(), integer)
        } else {
            let real = value.real_scalar_value().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "integer/real conversion must be explicit",
                )
            })?;
            ValueLiteral::from_real(target.clone(), real.value())
        }
        .map_err(|error| {
            source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
        })?;
        return Ok(SymbolicParameterValue {
            value: Some(literal.clone()),
            value_type: target,
            expression: Some(
                LoweringExpression::literal(literal, range).with_structural_parameters(
                    evaluated
                        .expression
                        .iter()
                        .flat_map(LoweringExpression::structural_parameters),
                ),
            ),
            lineage: evaluated.lineage,
        });
    }
    match &evaluated.value_type {
        EvaluatedType::Known(actual) if actual.dimension() != target.dimension() => {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!(
                    "{label} has dimension [{}], expected [{}]",
                    actual.dimension(),
                    target.dimension()
                ),
            ));
        }
        EvaluatedType::Known(actual) | EvaluatedType::Deferred(actual)
            if target.scalar_domain() == ScalarDomain::Real
                && target.shape().is_scalar()
                && (actual.scalar_domain() != ScalarDomain::Real
                    || !actual.shape().is_scalar()) =>
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                format!("{label} requires a real scalar type"),
            ));
        }
        _ => {}
    }
    use eqiora_schema::kernel::typing::ExpressionType;
    let actual = evaluated
        .value_type
        .value_type()
        .clone()
        .with_dimension(target.dimension())
        .map_err(|error| {
            source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
        })?;
    let common = ExpressionType::<()>::new(actual.clone(), None)
        .equation(ExpressionType::new(target.clone(), None))
        .map_err(|error| {
            source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
        })?;
    if common.value_type != target {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("{label} cannot implicitly convert complex values to real values"),
        ));
    }
    let embedding = actual.scalar_domain() != target.scalar_domain();
    let expression = evaluated.expression.map(|expression| {
        if embedding {
            expression.embed_complex()
        } else {
            expression
        }
    });
    let lineage = if embedding {
        transform_lineage(evaluated.lineage)
    } else {
        evaluated.lineage
    };
    Ok(SymbolicParameterValue {
        value: evaluated
            .value
            .map(|value| {
                crate::typed_values::retype(&value, target.clone()).map_err(|message| {
                    source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message)
                })
            })
            .transpose()?,
        value_type: target,
        expression,
        lineage,
    })
}
