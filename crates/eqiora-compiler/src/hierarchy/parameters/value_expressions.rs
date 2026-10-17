//! Complete static values retain their authored expression dependencies.

use super::*;
use eqiora_schema::kernel::typing::ExpressionType;

pub(super) fn evaluate_mode(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    (resolve_clock, resolve_frame): StaticContexts<'_>,
    target: Option<&ValueType>,
    evaluate_values: bool,
) -> Result<EvaluatedParameter, Diagnostic> {
    let error = |message: String| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };

    let (operands, value_type, lowered, value) = match expression.kind() {
        ExprKind::Array(elements) => {
            let element_target = target
                .filter(|target| target.array_rank() > 0)
                .map(|target| {
                    ExpressionType::index(ExpressionType::<()>::new(target.clone(), None), 0)
                        .expect("checked array")
                        .value_type
                });
            let operands = elements
                .iter()
                .map(|element| match &element_target {
                    Some(target) => super::expression_eval::evaluate_initializer_mode(
                        file,
                        element,
                        context,
                        resolve,
                        target.clone(),
                        ("declaration initializer", evaluate_values),
                        (&mut *resolve_clock, &mut *resolve_frame),
                    ),
                    None => super::expression_eval::evaluate_mode(
                        file,
                        element,
                        context,
                        resolve,
                        (&mut *resolve_clock, &mut *resolve_frame),
                        None,
                        evaluate_values,
                    ),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let types = operands
                .iter()
                .map(|value| ExpressionType::<()>::new(value.value_type.value_type().clone(), None))
                .collect::<Vec<_>>();
            let value_type = ExpressionType::array(&types)
                .map_err(|violation| error(violation.to_string()))?
                .value_type;
            crate::typed_values::check_type(&value_type).map_err(error)?;
            let lowered = operands
                .iter()
                .map(|value| value.expression.clone())
                .collect::<Option<Vec<_>>>()
                .map(|elements| LoweringExpression::array(elements, expression.range()));
            let value = operands
                .iter()
                .map(|value| value.value.as_ref())
                .collect::<Option<Vec<_>>>()
                .map(|values| {
                    if value_type.scalar_domain() == ScalarDomain::Integer {
                        ValueLiteral::integer(
                            value_type.clone(),
                            values.iter().flat_map(|value| {
                                value.integer_components().expect("checked integer array")
                            }),
                        )
                    } else {
                        ValueLiteral::new(
                            value_type.clone(),
                            values.iter().flat_map(|value| {
                                value.components().expect("checked real/complex array")
                            }),
                        )
                    }
                    .map_err(|violation| error(violation.to_string()))
                })
                .transpose()?;
            (operands, value_type, lowered, value)
        }
        ExprKind::Index { value, index }
        | ExprKind::Slice {
            value,
            lower: index,
            ..
        } => {
            let operand = super::expression_eval::evaluate_mode(
                file,
                value,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )?;
            let index_value = super::expression_eval::evaluate_with_domain(
                file,
                index,
                context,
                resolve,
                &mut *resolve_clock,
                &mut *resolve_frame,
                Some(ScalarDomain::Integer),
            )?;
            let index = checked_index(file, index.range(), &index_value)?;
            let mut dependencies = index_value
                .expression
                .as_ref()
                .map(LoweringExpression::referenced_names)
                .unwrap_or_default();
            let end = if let ExprKind::Slice { upper, .. } = expression.kind() {
                let upper_value = super::expression_eval::evaluate_with_domain(
                    file,
                    upper,
                    context,
                    resolve,
                    &mut *resolve_clock,
                    &mut *resolve_frame,
                    Some(ScalarDomain::Integer),
                )?;
                let end = checked_index(file, upper.range(), &upper_value)?;
                if let Some(expression) = upper_value.expression {
                    dependencies.extend(expression.referenced_names());
                }
                if end
                    .checked_sub(index)
                    .is_none_or(|width| width == 0 || width > 65_536)
                {
                    return Err(error(
                        "slice requires increasing exact bounds and at most 65536 channels".into(),
                    ));
                }
                Some(end)
            } else {
                None
            };
            let element_type = ExpressionType::index(
                ExpressionType::<()>::new(operand.value_type.value_type().clone(), None),
                end.map_or(index, |end| end - 1),
            )
            .map_err(|violation| error(violation.to_string()))?
            .value_type;
            let value_type = match end {
                Some(end) => element_type
                    .clone()
                    .array(end - index)
                    .map_err(|violation| error(violation.to_string()))?,
                None => element_type.clone(),
            };
            let lowered = operand
                .expression
                .clone()
                .map(|value| match end {
                    Some(end) => LoweringExpression::array(
                        (index..end)
                            .map(|index| {
                                LoweringExpression::index(value.clone(), index, expression.range())
                            })
                            .collect(),
                        expression.range(),
                    ),
                    None => LoweringExpression::index(value, index, expression.range()),
                })
                .map(|value| value.with_structural_parameters(dependencies));
            let value = operand
                .value
                .as_ref()
                .map(|value| {
                    let count = value_type
                        .shape()
                        .component_count()
                        .expect("checked element type");
                    let offset = index as usize
                        * element_type
                            .shape()
                            .component_count()
                            .expect("checked element type");
                    if value_type.scalar_domain() == ScalarDomain::Integer {
                        ValueLiteral::integer(
                            value_type.clone(),
                            value
                                .integer_components()
                                .expect("checked integer array")
                                .skip(offset)
                                .take(count),
                        )
                    } else {
                        ValueLiteral::new(
                            value_type.clone(),
                            value
                                .components()
                                .expect("checked real/complex array")
                                .skip(offset)
                                .take(count),
                        )
                    }
                    .map_err(|violation| error(violation.to_string()))
                })
                .transpose()?;
            (vec![operand], value_type, lowered, value)
        }
        ExprKind::Path(path) if path.as_str() == "math.i" => {
            let value_type = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .expect("valid numeric scalar type");
            let value =
                ValueLiteral::new(value_type.clone(), [(0.0, 1.0)]).expect("imaginary unit");
            (
                vec![],
                value_type,
                Some(LoweringExpression::literal(
                    value.clone(),
                    expression.range(),
                )),
                Some(value),
            )
        }
        ExprKind::Call { arguments, .. } => {
            let Some([real, imag]) = arguments.positional() else {
                return Err(error(
                    "math.complex requires exactly two real-valued arguments".into(),
                ));
            };
            let real_target =
                target.and_then(|target| target.clone().with_scalar_domain(ScalarDomain::Real));
            let mut evaluate = |value| match &real_target {
                Some(target) => super::expression_eval::evaluate_initializer_mode(
                    file,
                    value,
                    context,
                    resolve,
                    target.clone(),
                    ("declaration initializer", evaluate_values),
                    (&mut *resolve_clock, &mut *resolve_frame),
                ),
                None => super::expression_eval::evaluate_mode(
                    file,
                    value,
                    context,
                    resolve,
                    (&mut *resolve_clock, &mut *resolve_frame),
                    None,
                    evaluate_values,
                ),
            };
            let real = evaluate(real)?;
            let imag = evaluate(imag)?;
            let value_type = ExpressionType::complex(
                ExpressionType::<()>::new(real.value_type.value_type().clone(), None),
                ExpressionType::new(imag.value_type.value_type().clone(), None),
            )
            .map_err(|violation| error(violation.to_string()))?
            .value_type;
            let lowered = real
                .expression
                .clone()
                .zip(imag.expression.clone())
                .map(|(real, imag)| LoweringExpression::complex(real, imag, expression.range()));
            let value = real
                .value
                .as_ref()
                .zip(imag.value.as_ref())
                .map(|(real, imag)| {
                    ValueLiteral::new(
                        value_type.clone(),
                        real.components()
                            .expect("checked real components")
                            .zip(imag.components().expect("checked real components"))
                            .map(|(real, imag)| (real.0, imag.0)),
                    )
                    .map_err(|violation| error(violation.to_string()))
                })
                .transpose()?;
            (vec![real, imag], value_type, lowered, value)
        }
        _ => unreachable!("only complete value constructors reach this owner"),
    };
    let known = operands
        .iter()
        .all(|operand| matches!(operand.value_type, EvaluatedType::Known(_)));
    let mut lineage = Some(ParameterLineage::Constant);
    for operand in &operands {
        lineage = combine_lineages(lineage, operand.lineage.clone());
    }
    Ok(EvaluatedParameter {
        value: if evaluate_values { value } else { None },
        value_type: if known {
            EvaluatedType::Known(value_type)
        } else {
            EvaluatedType::Deferred(value_type)
        },
        bare_literal: false,
        expression: lowered,
        lineage,
    })
}

pub(super) fn checked_index(
    file: &str,
    range: TextRange,
    value: &EvaluatedParameter,
) -> Result<u32, Diagnostic> {
    let invalid = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "channel index requires an exact static dimensionless nonnegative integer",
        )
    };
    let index = value
        .value
        .as_ref()
        .and_then(ValueLiteral::integer_scalar_value)
        .ok_or_else(invalid)?;
    u32::try_from(index).map_err(|_| invalid())
}
