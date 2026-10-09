//! Lossless source expression projection before contextual value admission.
use super::*;

pub(in crate::lower) fn from_source(expression: &Expr) -> LoweringExpression {
    if let Some(value) = expression.resolved_enum() {
        return LoweringExpression::literal(value.clone(), expression.range());
    }
    let kind = match expression.kind() {
        ExprKind::Call { callee, arguments }
            if crate::math::boundary::Operation::named(callee.as_str()).is_some() =>
        {
            match crate::math::boundary::source(arguments) {
                Ok(selection) => LoweringExpressionNode::Boundary {
                    operation: crate::math::boundary::Operation::named(callee.as_str()).unwrap(),
                    argument: from_source(selection.value),
                    on: selection.on.map(str::to_owned),
                    from: selection.from.map(str::to_owned),
                },
                Err(message) => LoweringExpressionNode::InvalidValue(message),
            }
        }
        ExprKind::Call { callee, arguments } if crate::math::tensor::named(callee.as_str()) => {
            match crate::math::tensor::source(callee.as_str(), arguments) {
                Ok((operation, arguments)) => LoweringExpressionNode::Tensor {
                    operation,
                    arguments: arguments
                        .into_iter()
                        .map(LoweringExpression::from_source)
                        .collect(),
                },
                Err(message) => LoweringExpressionNode::InvalidValue(message),
            }
        }

        ExprKind::Array(elements) => LoweringExpressionNode::Array(
            elements
                .iter()
                .map(LoweringExpression::from_source)
                .collect(),
        ),
        ExprKind::Index { value, index } => match crate::hierarchy::closed_index(index) {
            Ok(index) => LoweringExpressionNode::Index {
                value: LoweringExpression::from_source(value),
                index,
            },
            Err(_) => LoweringExpressionNode::InvalidValue(
                "channel index requires a constant nonnegative integer",
            ),
        },
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            match crate::hierarchy::closed_index(lower)
                .ok()
                .zip(crate::hierarchy::closed_index(upper).ok())
            {
                Some((start, end))
                    if end
                        .checked_sub(start)
                        .is_some_and(|width| width > 0 && width <= 65_536) =>
                {
                    let value = from_source(value);
                    LoweringExpressionNode::Array(
                        (start..end)
                            .map(|index| {
                                LoweringExpression::index(value.clone(), index, expression.range())
                            })
                            .collect(),
                    )
                }
                _ => LoweringExpressionNode::InvalidValue(
                    "slice requires explicit increasing constant integer bounds",
                ),
            }
        }
        ExprKind::Path(path) if path.as_str() == "math.i" => {
            return LoweringExpression::literal(
                eqiora_core::ValueLiteral::new(
                    eqiora_core::ValueType::scalar(
                        eqiora_core::ScalarDomain::Complex,
                        DimExponents::DIMENSIONLESS,
                    )
                    .expect("admitted numeric scalar type"),
                    [(0.0, 1.0)],
                )
                .expect("imaginary unit"),
                expression.range(),
            );
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if callee.as_str() == "math.complex" => match arguments.as_slice() {
            [real, imag] => LoweringExpressionNode::Complex {
                real: LoweringExpression::from_source(real),
                imag: LoweringExpression::from_source(imag),
            },
            _ => LoweringExpressionNode::InvalidValue(
                "math.complex requires exactly two real scalar arguments",
            ),
        },
        ExprKind::Case { value, arms } => {
            let arms = arms
                .iter()
                .map(|arm| {
                    arm.resolved_pattern()
                        .cloned()
                        .map(|pattern| (pattern, from_source(arm.value())))
                })
                .collect::<Option<Vec<_>>>();
            match arms {
                Some(arms) => LoweringExpressionNode::Case {
                    value: from_source(value),
                    arms,
                },
                None => {
                    LoweringExpressionNode::InvalidValue("case pattern requires exact enum binding")
                }
            }
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => LoweringExpressionNode::Select {
            condition: LoweringExpression::from_source(condition),
            then_value: LoweringExpression::from_source(then_value),
            else_value: LoweringExpression::from_source(else_value),
        },
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if crate::math::finite::Operation::named(callee.as_str()).is_some() => {
            match crate::math::finite::Operation::source(callee.as_str(), arguments) {
                Ok((operation, arguments)) => LoweringExpressionNode::Finite {
                    operation,
                    arguments: arguments.iter().map(from_source).collect(),
                },
                Err(message) => LoweringExpressionNode::InvalidValue(message),
            }
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if crate::math::piecewise::arity(callee.as_str()).is_some() => {
            LoweringExpressionNode::Piecewise {
                name: callee.as_str().to_owned(),
                arguments: arguments.iter().map(from_source).collect(),
            }
        }
        ExprKind::Quantity { value, unit } => match crate::units::quantity(value, unit) {
            Ok(value) => return LoweringExpression::quantity(value, expression.range()),
            Err(message) => LoweringExpressionNode::InvalidValue(message),
        },
        ExprKind::Boolean(value) => {
            LoweringExpressionNode::Literal(eqiora_core::ValueLiteral::boolean(*value))
        }
        ExprKind::Number(value) => LoweringExpressionNode::Number(value.clone()),
        ExprKind::Path(path) => match crate::math::constant(path) {
            Some(value) => {
                return LoweringExpression::quantity(
                    DynQuantity::new(value, DimExponents::DIMENSIONLESS),
                    expression.range(),
                );
            }
            None if crate::math::is_namespaced(path) => {
                LoweringExpressionNode::UnknownMath(path.as_str().to_owned())
            }
            None => LoweringExpressionNode::Unsupported,
        },
        ExprKind::Name(name) => LoweringExpressionNode::Name(name.clone()),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => return LoweringExpression::neg(from_source(value), expression.range()),
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => LoweringExpressionNode::Not(from_source(value)),
        ExprKind::Binary { op, left, right } => LoweringExpressionNode::Binary {
            operator: *op,
            left: from_source(left),
            right: from_source(right),
        },
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if super::IntegerBuiltin::named(callee.as_str()).is_some() => {
            let operator = super::IntegerBuiltin::named(callee.as_str()).unwrap();
            if arguments.len() != operator.arity() {
                LoweringExpressionNode::InvalidValue("invalid integer operation arity")
            } else {
                LoweringExpressionNode::IntegerCall {
                    operator,
                    arguments: arguments.iter().map(from_source).collect(),
                }
            }
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if callee.as_str() == "sample" => match arguments.as_slice() {
            [value, clock] => match clock.kind() {
                ExprKind::Name(clock) => LoweringExpressionNode::Sample {
                    value: from_source(value),
                    clock: clock.clone(),
                },
                _ => LoweringExpressionNode::InvalidValue("sample requires one clock name"),
            },
            _ => LoweringExpressionNode::InvalidValue("sample requires a value and one clock name"),
        },
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if (!callee.is_qualified() || crate::math::is_namespaced(callee))
            && arguments.len() == 1 =>
        {
            LoweringExpressionNode::Call {
                callee: callee.as_str().to_owned(),
                argument: from_source(&arguments[0]),
            }
        }
        _ => LoweringExpressionNode::Unsupported,
    };
    LoweringExpression {
        node: Arc::new(kind),
        range: expression.range(),
        structural_parameters: None,
    }
}
