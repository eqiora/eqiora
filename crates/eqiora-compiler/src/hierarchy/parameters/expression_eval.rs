mod case;
mod finite;
mod initializers;
pub(super) use initializers::{
    coerce_parameter, coerce_parameter_with_label, evaluate_initializer, evaluate_initializer_mode,
};
mod piecewise;
use super::*;
use eqiora_schema::kernel::typing::SpatialSupport;

#[derive(Debug, Clone, Copy)]
pub(super) enum ExpressionContext<'a> {
    IndexedBinding(&'a str),
    Binding,
    Default,
    Let,
}

impl ExpressionContext<'_> {
    pub(super) fn unknown_name_message(self, name: &str) -> String {
        match self {
            Self::Binding | Self::IndexedBinding(_) => {
                format!("unknown Parameter `{name}` in compile-time binding")
            }
            Self::Default => format!("unknown component Parameter `{name}`"),
            Self::Let => format!("unknown Parameter or let alias `{name}`"),
        }
    }

    pub(super) fn qualified_name_message(self, path: &impl std::fmt::Display) -> String {
        match self {
            Self::Binding | Self::IndexedBinding(_) => format!(
                "qualified name `{path}` is not allowed in a compile-time Parameter binding"
            ),
            Self::Default => {
                format!("qualified name `{path}` is not allowed in a Parameter default")
            }
            Self::Let => format!("qualified name `{path}` is not allowed in a let alias"),
        }
    }

    pub(super) fn call_message(self, callee: &str) -> String {
        match self {
            Self::Binding | Self::IndexedBinding(_) => {
                format!("operator `{callee}(...)` is not allowed in a compile-time binding")
            }
            Self::Default => {
                format!("operator `{callee}(...)` is not allowed in a Parameter default")
            }
            Self::Let => format!(
                "static let operator `{callee}(...)` is not supported by compile-time evaluation"
            ),
        }
    }

    pub(super) const fn unsupported_message(self) -> &'static str {
        match self {
            Self::Binding | Self::IndexedBinding(_) => {
                "binding expression syntax is newer than this compiler"
            }
            Self::Default => "Parameter default syntax is newer than this compiler",
            Self::Let => "let expression syntax is newer than this compiler",
        }
    }
}

pub(super) fn evaluate_parameter_expression(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
) -> Result<EvaluatedParameter, Diagnostic> {
    evaluate_with_domain(
        file,
        expression,
        context,
        resolve,
        resolve_clock,
        resolve_frame,
        None,
    )
}

pub(super) fn evaluate_with_domain(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
    expected: Option<ScalarDomain>,
) -> Result<EvaluatedParameter, Diagnostic> {
    evaluate_mode(
        file,
        expression,
        context,
        resolve,
        (&mut *resolve_clock, &mut *resolve_frame),
        expected,
        true,
    )
}

pub(super) fn evaluate_mode(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    (resolve_clock, resolve_frame): StaticContexts<'_>,
    expected: Option<ScalarDomain>,
    evaluate_values: bool,
) -> Result<EvaluatedParameter, Diagnostic> {
    if expression.resolved_enum().is_some() || matches!(expression.kind(), ExprKind::Case { .. }) {
        return case::evaluate(
            file,
            expression,
            expected,
            evaluate_values,
            |value, expected, demand| {
                evaluate_mode(
                    file,
                    value,
                    context,
                    resolve,
                    (&mut *resolve_clock, &mut *resolve_frame),
                    expected,
                    demand,
                )
            },
        );
    }
    if let ExprKind::Select {
        condition,
        then_value,
        else_value,
    } = expression.kind()
    {
        let condition = evaluate_mode(
            file,
            condition,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            None,
            evaluate_values,
        )?;
        let demand = condition.value.as_ref().and_then(ValueLiteral::as_bool);
        let then_value = evaluate_mode(
            file,
            then_value,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            expected,
            evaluate_values && demand == Some(true),
        )?;
        let else_value = evaluate_mode(
            file,
            else_value,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            expected,
            evaluate_values && demand == Some(false),
        )?;
        return piecewise::select(file, expression.range(), condition, then_value, else_value);
    }
    if let ExprKind::Call { callee, arguments } = expression.kind()
        && crate::math::finite::Operation::named(callee.as_str()).is_some()
    {
        return finite::evaluate(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            (callee.as_str(), arguments),
            evaluate_values,
        );
    }
    if let ExprKind::Call { callee, arguments } = expression.kind()
        && let Some(arity) = crate::math::piecewise::arity(callee.as_str())
    {
        let arguments = arguments
            .positional()
            .filter(|values| values.len() == arity)
            .ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("{callee} requires {arity} positional operands"),
                )
            })?;
        return piecewise::sugar(
            file,
            expression.range(),
            callee.as_str(),
            arity,
            evaluate_values,
            |index, demand| {
                evaluate_mode(
                    file,
                    &arguments[index],
                    context,
                    resolve,
                    (&mut *resolve_clock, &mut *resolve_frame),
                    None,
                    demand,
                )
            },
        );
    }
    if matches!(expression.kind(), ExprKind::Reduction { .. }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "finite reductions are admitted in Relations and runtime let aliases, not Parameter or extent initializers",
        ));
    }
    if let ExpressionContext::IndexedBinding(member) = context
        && let ExprKind::Call { callee, arguments } = expression.kind()
        && callee.as_str() == "ordinal"
        && let Some([argument]) = arguments.positional()
        && matches!(argument.kind(), ExprKind::Name(name) if name == member)
    {
        return Ok(EvaluatedParameter {
            value: None,
            value_type: EvaluatedType::Known(
                ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                    .expect("valid numeric scalar type"),
            ),
            expression: None,
            lineage: None,
            bare_literal: false,
        });
    }
    if matches!(expression.kind(), ExprKind::Call { callee, .. } if callee.as_str() == "tensor_value")
    {
        return super::tensor_values::evaluate(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            None,
            evaluate_values,
        );
    }
    if expression.resolved_nominal().is_some() {
        let value = crate::nominal::literal(file, expression)?;
        return Ok(EvaluatedParameter {
            expression: Some(LoweringExpression::literal(
                value.clone(),
                expression.range(),
            )),
            value_type: EvaluatedType::Known(value.value_type().clone()),
            value: evaluate_values.then_some(value),
            bare_literal: false,
            lineage: Some(ParameterLineage::Constant),
        });
    }
    if matches!(
        expression.kind(),
        ExprKind::Array(_) | ExprKind::Index { .. } | ExprKind::Slice { .. }
    ) || matches!(expression.kind(), ExprKind::Path(path) if path.as_str() == "math.i")
        || matches!(expression.kind(), ExprKind::Call { callee, .. } if callee.as_str() == "math.complex")
    {
        return super::value_expressions::evaluate_mode(
            file,
            expression,
            context,
            resolve,
            (&mut *resolve_clock, &mut *resolve_frame),
            None,
            evaluate_values,
        );
    }
    if expected == Some(ScalarDomain::Integer)
        && let Some(literal) = exact_signed_literal(expression)
    {
        let value = literal.map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                error.message(),
            )
        })?;
        let value_type = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid numeric scalar type");
        let value =
            ValueLiteral::from_integer(value_type.clone(), value).expect("checked integer scalar");
        return Ok(EvaluatedParameter {
            expression: Some(LoweringExpression::literal(
                value.clone(),
                expression.range(),
            )),
            value: evaluate_values.then_some(value),
            value_type: EvaluatedType::Known(value_type),
            bare_literal: true,
            lineage: Some(ParameterLineage::Constant),
        });
    }
    let mut evaluated = match expression.kind() {
        ExprKind::Boolean(value) => {
            let literal = ValueLiteral::boolean(*value);
            EvaluatedParameter {
                value: Some(literal.clone()),
                value_type: EvaluatedType::Known(ValueType::boolean()),
                bare_literal: false,
                expression: Some(LoweringExpression::literal(literal, expression.range())),
                lineage: Some(ParameterLineage::Constant),
            }
        }
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => super::predicates::negate(
            file,
            expression.range(),
            evaluate_mode(
                file,
                value,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )?,
        )?,
        ExprKind::Binary { op, left, right } if matches!(op, BinaryOp::And | BinaryOp::Or) => {
            let left = evaluate_mode(
                file,
                left,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )?;
            let demand_right = left
                .value
                .as_ref()
                .and_then(ValueLiteral::as_bool)
                .is_some_and(|value| if *op == BinaryOp::And { value } else { !value });
            let right = evaluate_mode(
                file,
                right,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values && demand_right,
            )?;
            super::predicates::combine(file, expression.range(), *op, left, right)?
        }
        ExprKind::Number(literal) => {
            let value = literal.to_f64().map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    error.message(),
                )
            })?;
            EvaluatedParameter {
                value: Some(
                    ValueLiteral::from_real(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                            .expect("valid numeric scalar type"),
                        normalize_zero(value),
                    )
                    .expect("finite source literal"),
                ),
                value_type: EvaluatedType::Known(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("valid numeric scalar type"),
                ),
                bare_literal: true,
                expression: Some(LoweringExpression::quantity(
                    DynQuantity::new(normalize_zero(value), DimExponents::DIMENSIONLESS),
                    expression.range(),
                )),
                lineage: Some(ParameterLineage::Constant),
            }
        }
        ExprKind::Quantity { value, unit } => {
            let quantity = crate::units::quantity(value, unit).map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    message,
                )
            })?;
            EvaluatedParameter {
                value: Some(ValueLiteral::try_from(quantity).expect("finite quantity")),
                value_type: EvaluatedType::Known(
                    ValueType::scalar(ScalarDomain::Real, quantity.dim())
                        .expect("valid numeric scalar type"),
                ),
                bare_literal: false,
                expression: Some(LoweringExpression::quantity(quantity, expression.range())),
                lineage: Some(ParameterLineage::Constant),
            }
        }
        ExprKind::Call { callee, arguments }
            if crate::lower::IntegerBuiltin::named(callee.as_str()).is_some() =>
        {
            let operator = crate::lower::IntegerBuiltin::named(callee.as_str()).unwrap();
            let arguments = arguments.positional().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "static integer builtins require positional arguments",
                )
            })?;
            if arguments.len() != operator.arity() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("{callee} requires {} operands", operator.arity()),
                ));
            }
            let operands = arguments
                .iter()
                .map(|argument| {
                    evaluate_mode(
                        file,
                        argument,
                        context,
                        resolve,
                        (&mut *resolve_clock, &mut *resolve_frame),
                        Some(operator.operand_domain()),
                        evaluate_values,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let types = operands
                .iter()
                .map(|operand| {
                    eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                        operand.value_type.value_type().clone(),
                        None,
                    )
                })
                .collect::<Vec<_>>();
            let value_type = operator
                .infer(&types)
                .map_err(|error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        error.to_string(),
                    )
                })?
                .value_type;
            let value = operands
                .iter()
                .map(|operand| operand.value.clone())
                .collect::<Option<Vec<_>>>()
                .map(|values| {
                    operator.evaluate(&values).map_err(|error| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            expression.range(),
                            error.to_string(),
                        )
                    })
                })
                .transpose()?;
            let lowered = operands
                .iter()
                .map(|operand| operand.expression.clone())
                .collect::<Option<Vec<_>>>()
                .map(|arguments| {
                    LoweringExpression::integer_call(operator, arguments, expression.range())
                });
            let lineage = operands
                .iter()
                .try_fold(ParameterLineage::Constant, |lineage, operand| {
                    combine_lineages(Some(lineage), operand.lineage.clone())
                });
            EvaluatedParameter {
                value,
                value_type: EvaluatedType::Known(value_type),
                bare_literal: false,
                expression: lowered,
                lineage: transform_lineage(lineage),
            }
        }
        ExprKind::Call { callee, arguments } if callee.as_str() == "period" => {
            let Some([argument]) = arguments.positional() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "period requires exactly one declared Clock",
                ));
            };
            let ExprKind::Name(name) = argument.kind() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    "period requires a direct declared Clock name",
                ));
            };
            let period = resolve_clock(name).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    format!("`{name}` is not a declared Clock"),
                )
            })?;
            let value_type =
                ValueType::scalar(ScalarDomain::Real, crate::dimensions::time_dimension())
                    .expect("valid numeric scalar type");
            EvaluatedParameter {
                value: period.map(|period| {
                    ValueLiteral::from_real(value_type.clone(), period.as_seconds_f64())
                        .expect("bounded rational period")
                }),
                value_type: EvaluatedType::Known(value_type),
                bare_literal: false,
                expression: period.map(|period| {
                    LoweringExpression::quantity(
                        DynQuantity::new(
                            period.as_seconds_f64(),
                            crate::dimensions::time_dimension(),
                        ),
                        expression.range(),
                    )
                }),
                lineage: Some(ParameterLineage::Constant),
            }
        }
        ExprKind::Name(name) => resolve(name, expression.range())?.into(),
        ExprKind::Path(path) => match crate::math::constant(path) {
            Some(value) => EvaluatedParameter {
                value: Some(
                    ValueLiteral::from_real(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                            .expect("valid numeric scalar type"),
                        value,
                    )
                    .expect("finite math constant"),
                ),
                value_type: EvaluatedType::Known(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("valid numeric scalar type"),
                ),
                bare_literal: false,
                expression: Some(LoweringExpression::quantity(
                    DynQuantity::new(value, DimExponents::DIMENSIONLESS),
                    expression.range(),
                )),
                lineage: Some(ParameterLineage::Constant),
            },
            None => resolve(path.as_str(), path.range())?.into(),
        },
        ExprKind::Call { callee, arguments }
            if crate::math::unary_function(callee.as_str()).is_some() =>
        {
            let Some([argument]) = arguments.positional() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "static scalar mathematics requires exactly one argument",
                ));
            };
            let operand = evaluate_mode(
                file,
                argument,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )?;
            let EvaluatedType::Known(value_type) = &operand.value_type else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "static scalar mathematics requires a known operand dimension",
                ));
            };
            let function = crate::math::unary_function(callee.as_str())
                .expect("compiler-owned unary mathematics was checked");
            let inferred = eqiora_schema::kernel::typing::unary_math(
                function,
                &eqiora_schema::kernel::typing::ExpressionType::<()>::new(value_type.clone(), None),
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    error.to_string(),
                )
            })?;
            let value = operand
                .value
                .map(|value| {
                    function.evaluate(&value).map_err(|error| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            expression.range(),
                            error.message(),
                        )
                    })
                })
                .transpose()?;
            EvaluatedParameter {
                value,
                value_type: EvaluatedType::Known(inferred.value_type),
                bare_literal: false,
                expression: operand.expression.map(|argument| {
                    LoweringExpression::call(
                        callee.as_str().to_owned(),
                        argument,
                        expression.range(),
                    )
                }),
                lineage: Some(ParameterLineage::Derived),
            }
        }
        ExprKind::Call { callee, .. } => {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                context.call_message(callee.as_str()),
            ));
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => {
            let operand = evaluate_mode(
                file,
                value,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                expected,
                evaluate_values,
            )?;
            if operand.value_type.value_type().scalar_domain() == ScalarDomain::Boolean {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "Boolean values do not admit arithmetic negation",
                ));
            }
            let negated = operand
                .value
                .map(|value| {
                    crate::typed_values::negate(&value).map_err(|message| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            expression.range(),
                            message,
                        )
                    })
                })
                .transpose()?;
            EvaluatedParameter {
                value: negated,
                value_type: operand.value_type,
                bare_literal: operand.bare_literal,
                expression: operand
                    .expression
                    .map(|value| LoweringExpression::neg(value, expression.range())),
                lineage: transform_lineage(operand.lineage),
            }
        }
        ExprKind::Binary { op, left, right }
            if crate::lower::comparison_operator(*op).is_some() =>
        {
            let left_type = evaluate_mode(
                file,
                left,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                false,
            )?;
            let right_type = evaluate_mode(
                file,
                right,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                false,
            )?;
            let operand_domain = [
                left_type.value_type.value_type(),
                right_type.value_type.value_type(),
            ]
            .iter()
            .any(|value| value.scalar_domain() == ScalarDomain::Integer)
            .then_some(ScalarDomain::Integer);
            let left = evaluate_mode(
                file,
                left,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                operand_domain,
                evaluate_values,
            )?;
            let right = evaluate_mode(
                file,
                right,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                operand_domain,
                evaluate_values,
            )?;
            super::predicates::combine(file, expression.range(), *op, left, right)?
        }
        ExprKind::Binary { op, left, right } => {
            let expected = if crate::lower::comparison_operator(*op).is_some() {
                None
            } else {
                expected
            };
            let right_ast = right;
            let mut right = evaluate_mode(
                file,
                right,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                expected,
                evaluate_values || *op == BinaryOp::Pow,
            )?;
            let contextual = expected.or_else(|| {
                (right.value_type.value_type().scalar_domain() == ScalarDomain::Integer)
                    .then_some(ScalarDomain::Integer)
            });
            let left = evaluate_mode(
                file,
                left,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                contextual,
                evaluate_values,
            )?;
            if expected.is_none()
                && left.value_type.value_type().scalar_domain() == ScalarDomain::Integer
                && (right.bare_literal || super::predicates::untyped_numeric_tree(right_ast))
            {
                right = evaluate_mode(
                    file,
                    right_ast,
                    context,
                    resolve,
                    (&mut *resolve_clock, &mut *resolve_frame),
                    Some(ScalarDomain::Integer),
                    evaluate_values,
                )?;
            }
            if crate::lower::comparison_operator(*op).is_some() {
                super::predicates::combine(file, expression.range(), *op, left, right)?
            } else {
                combine_parameters(file, expression.range(), *op, left, right)?
            }
        }
        _ => {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                expression.range(),
                context.unsupported_message(),
            ));
        }
    };
    if !evaluate_values {
        evaluated.value = None;
    }
    Ok(evaluated)
}

pub(crate) fn exact_signed_literal(
    expression: &Expr,
) -> Option<Result<i64, eqiora_lang::AstConstructionError>> {
    match expression.kind() {
        ExprKind::Number(value) => Some(value.to_i64()),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => match value.kind() {
            ExprKind::Number(value) => {
                let text = value.canonical_text();
                let signed = if value.is_negative() {
                    text.trim_start_matches('-').to_owned()
                } else {
                    format!("-{text}")
                };
                Some(eqiora_lang::DecimalLiteral::parse(&signed).and_then(|value| value.to_i64()))
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn infer_parameter_with_label(
    file: &str,
    range: TextRange,
    evaluated: EvaluatedParameter,
    label: &str,
) -> Result<SymbolicParameterValue, Diagnostic> {
    let EvaluatedType::Known(value_type) = evaluated.value_type else {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!(
                "{label} dimension cannot be inferred from this expression; add an explicit dimension annotation"
            ),
        ));
    };
    Ok(SymbolicParameterValue {
        value: evaluated.value,
        value_type,
        expression: evaluated.expression,
        lineage: evaluated.lineage,
    })
}

#[cfg(test)]
#[path = "expression_tests.rs"]
mod expression_tests;
