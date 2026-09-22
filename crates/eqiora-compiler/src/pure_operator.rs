//! Compiler-owned translation from exact source syntax to Kernel definitions.
//!
//! This is the single semantic conversion path used by local compilation,
//! resolved-package analysis, and source identity.  Keeping it here prevents
//! source adapters from inventing a second interpretation of the bounded
//! pure calculus.

pub(crate) mod property;
use std::collections::BTreeMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents};
use eqiora_lang::{
    BinaryOp, CallArguments, DecimalLiteral, Expr, ExprKind, PureOperatorDecl,
    PureValueClassSyntax, TextRange, UnaryOp,
};
use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, ExactRational, PureOperatorDefinition,
    PureOperatorError, PureValueClass, ResultAxis,
};

use crate::diagnostics::source_error;

/// Whether a call path denotes the closed source builtin vocabulary rather
/// than a package-resolved pure definition.
pub(crate) fn is_builtin_operator(path: &eqiora_lang::NamePath) -> bool {
    crate::math::is_function(path)
        || (!path.is_qualified()
            && matches!(
                path.as_str(),
                "trace"
                    | "coordinate"
                    | "grad"
                    | "div"
                    | "symmetric_part"
                    | "isotropic_lift"
                    | "normal"
                    | "time"
                    | "derivative"
                    | "partial"
                    | "pre"
                    | "next"
                    | "sample"
                    | "hold"
                    | "period"
            ))
}

/// Compile one lexical source unit, checking cycles before calculus expansion.
pub(crate) fn compile_definitions(
    file: &str,
    document: &eqiora_lang::Document,
) -> Result<BTreeMap<String, PureOperatorDefinition>, Diagnostic> {
    if document.pure_operators().is_empty() {
        return Ok(BTreeMap::new());
    }
    let document = crate::dimensions::elaborate_dimension_aliases(file, document)
        .map_err(|errors| errors.into_iter().next().expect("dimension error"))?;
    let declarations = document.pure_operators();
    let mut sources = BTreeMap::new();
    for declaration in declarations {
        if sources.insert(declaration.name(), declaration).is_some() {
            return Err(pure_error(
                file,
                declaration.range(),
                "duplicate operator declaration",
            ));
        }
    }
    let mut compiled = BTreeMap::new();
    let mut active = Vec::new();
    for name in sources.keys() {
        compile_local(file, name, &sources, &mut compiled, &mut active)?;
    }
    Ok(compiled)
}

fn compile_local(
    file: &str,
    name: &str,
    sources: &BTreeMap<&str, &PureOperatorDecl>,
    compiled: &mut BTreeMap<String, PureOperatorDefinition>,
    active: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    if compiled.contains_key(name) {
        return Ok(());
    }
    let declaration = sources[name];
    if active.iter().any(|value| value == name) {
        return Err(pure_error(
            file,
            declaration.range(),
            "recursive operator composition is not admitted",
        ));
    }
    if active.len() >= eqiora_schema::kernel::pure_operator::MAX_DEPTH {
        return Err(pure_error(
            file,
            declaration.range(),
            "operator composition exceeds the calculus depth limit",
        ));
    }
    active.push(name.to_owned());
    let mut callees = Vec::new();
    let mut pending = vec![(declaration.body(), 1usize)];
    let mut work = 0usize;
    while let Some((expression, depth)) = pending.pop() {
        work += match expression.kind() {
            ExprKind::Call { callee, .. } => {
                crate::math::piecewise::cost(callee.as_str()).map_or(1, |cost| cost.0)
            }
            _ => 1,
        };
        if work > eqiora_schema::kernel::pure_operator::MAX_NODES
            || depth > eqiora_schema::kernel::pure_operator::MAX_DEPTH
        {
            return Err(pure_error(
                file,
                expression.range(),
                "operator body exceeds the calculus work or depth limit",
            ));
        }
        match expression.kind() {
            ExprKind::Partial { value, .. } => pending.push((value, depth + 1)),
            ExprKind::Call { callee, arguments } => {
                if declaration
                    .formals()
                    .iter()
                    .any(|formal| formal.name() == callee.as_str())
                {
                    return Err(pure_error(
                        file,
                        expression.range(),
                        "a lexical formal cannot be called as an operator",
                    ));
                }
                if !matches!(
                    callee.as_str(),
                    "component" | "rational" | "delta" | "math.sqrt" | "math.sin"
                ) && crate::math::piecewise::arity(callee.as_str()).is_none()
                {
                    callees.push((callee.clone(), expression.range()));
                }
                pending.extend(arguments.expressions().map(|value| (value, depth + 1)));
            }
            ExprKind::Unary { value, .. } => pending.push((value, depth + 1)),
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                let exponent = power_exponent(file, right)?;
                work = work.saturating_add(exponent.saturating_sub(1));
                if work > eqiora_schema::kernel::pure_operator::MAX_NODES {
                    return Err(pure_error(
                        file,
                        expression.range(),
                        "polynomial powers exceed the calculus work limit",
                    ));
                }
                pending.push((left, depth + 1));
            }
            ExprKind::Binary { left, right, .. } => {
                pending.push((left, depth + 1));
                pending.push((right, depth + 1));
            }
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => pending.extend([
                (condition.as_ref(), depth + 1),
                (then_value.as_ref(), depth + 1),
                (else_value.as_ref(), depth + 1),
            ]),
            ExprKind::Name(_)
            | ExprKind::Number(_)
            | ExprKind::Boolean(_)
            | ExprKind::Quantity { .. } => {}
            _ => {
                return Err(pure_error(
                    file,
                    expression.range(),
                    "operator bodies require exact polynomial expressions over lexical formals",
                ));
            }
        }
    }
    for (callee, range) in callees {
        if callee.is_qualified() || !sources.contains_key(callee.as_str()) {
            return Err(pure_error(
                file,
                range,
                "operator bodies admit only local lexical polynomial operators",
            ));
        }
        compile_local(file, callee.as_str(), sources, compiled, active)?;
    }
    let mut formal_names = BTreeMap::new();
    let mut formal_rules = Vec::new();
    for (slot, formal) in declaration.formals().iter().enumerate() {
        let slot = u16::try_from(slot)
            .map_err(|_| pure_error(file, formal.range(), "too many operator formals"))?;
        if formal_names.insert(formal.name(), slot).is_some() {
            return Err(pure_error(
                file,
                formal.range(),
                "duplicate operator formal",
            ));
        }
        formal_rules.push(value_class(file, formal.range(), formal.value_class())?);
    }
    let mut builder = CalculusBuilder::new(
        formal_rules,
        value_class(file, declaration.range(), declaration.result())?,
    )
    .map_err(|error| kernel_error(file, declaration.range(), error))?;
    let root = compile_expression(
        file,
        declaration.body(),
        &formal_names,
        sources,
        compiled,
        &mut builder,
    )?;
    let definition = builder
        .finish(root)
        .map_err(|error| kernel_error(file, declaration.range(), error))?;
    compiled.insert(name.to_owned(), definition);
    active.pop();
    Ok(())
}

pub(crate) fn ordered_arguments<'a>(
    file: &str,
    range: TextRange,
    names: impl IntoIterator<Item = &'a str>,
    arguments: &'a CallArguments,
) -> Result<Vec<&'a Expr>, Diagnostic> {
    let Some(bindings) = arguments.named() else {
        return Err(pure_error(
            file,
            range,
            "operator calls require named arguments",
        ));
    };
    let mut remaining = BTreeMap::new();
    for binding in bindings {
        if remaining.insert(binding.name(), binding.value()).is_some() {
            return Err(pure_error(
                file,
                binding.range(),
                "duplicate operator argument",
            ));
        }
    }
    let ordered = names
        .into_iter()
        .map(|name| {
            remaining.remove(name).ok_or_else(|| {
                pure_error(file, range, format!("missing operator argument `{name}`"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some((name, value)) = remaining.first_key_value() {
        return Err(pure_error(
            file,
            value.range(),
            format!("unknown operator argument `{name}`"),
        ));
    }
    Ok(ordered)
}

fn value_class(
    file: &str,
    range: TextRange,
    syntax: &PureValueClassSyntax,
) -> Result<PureValueClass, Diagnostic> {
    match syntax {
        PureValueClassSyntax::Scalar => Ok(PureValueClass::invariant_scalar()),
        PureValueClassSyntax::Spatial { rank } => PureValueClass::spatial_tensor(
            u16::try_from(rank.value())
                .map_err(|_| pure_error(file, rank.range(), "spatial rank exceeds u16"))?,
        )
        .map_err(|error| kernel_error(file, range, error)),
        PureValueClassSyntax::Typed(syntax) => {
            let value_type = crate::value_types::lower_scalar_type(file, syntax)?;
            if value_type.scalar_domain() != eqiora_core::ScalarDomain::Real {
                return Err(pure_error(
                    file,
                    range,
                    "operator scalar contracts require ordinary real scalar types",
                ));
            }
            PureValueClass::invariant_scalar()
                .with_dimension(value_type.dimension())
                .with_scalar_domain(eqiora_core::ScalarDomain::Real)
                .map_err(|error| kernel_error(file, range, error))
        }
        _ => Err(pure_error(file, range, "unsupported operator value class")),
    }
}

fn compile_expression(
    file: &str,
    expression: &Expr,
    formals: &BTreeMap<&str, u16>,
    sources: &BTreeMap<&str, &PureOperatorDecl>,
    compiled: &BTreeMap<String, PureOperatorDefinition>,
    builder: &mut CalculusBuilder,
) -> Result<CalculusNodeId, Diagnostic> {
    let node = match expression.kind() {
        ExprKind::Partial {
            value,
            wrt,
            holding,
        } => {
            let selected = *formals.get(wrt.as_str()).ok_or_else(|| {
                pure_error(
                    file,
                    wrt.range(),
                    "partial wrt must name an independent lexical formal",
                )
            })?;
            let mut seen = std::collections::BTreeSet::new();
            for binding in holding {
                if !formals.contains_key(binding.as_str())
                    || binding.as_str() == wrt.as_str()
                    || !seen.insert(binding.as_str())
                {
                    return Err(pure_error(
                        file,
                        binding.range(),
                        "partial holding requires distinct other independent formals",
                    ));
                }
            }
            let root = compile_expression(file, value, formals, sources, compiled, builder)?;
            return builder.partial(root, selected).map_err(|error| {
                pure_error(
                    file,
                    expression.range(),
                    format!("unsupported partial of explicit real scalar polynomial: {error}"),
                )
            });
        }
        ExprKind::Boolean(value) => CalculusNode::Boolean(*value),
        ExprKind::Quantity { value, unit } => {
            let (dimension, shift) = crate::units::exact_unit(unit)
                .map_err(|message| pure_error(file, unit.range(), message))?;
            CalculusNode::Rational {
                value: decimal_signed(file, expression.range(), value, false, shift)?,
                dimension,
            }
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => CalculusNode::Select {
            condition: compile_expression(file, condition, formals, sources, compiled, builder)?,
            then_value: compile_expression(file, then_value, formals, sources, compiled, builder)?,
            else_value: compile_expression(file, else_value, formals, sources, compiled, builder)?,
        },
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => CalculusNode::Not(compile_expression(
            file, value, formals, sources, compiled, builder,
        )?),
        ExprKind::Number(value) => CalculusNode::Rational {
            value: decimal(file, expression.range(), value)?,
            dimension: DimExponents::DIMENSIONLESS,
        },
        ExprKind::Name(name) => CalculusNode::FormalComponent {
            formal: *formals.get(name.as_str()).ok_or_else(|| {
                pure_error(
                    file,
                    expression.range(),
                    format!("unknown operator formal `{name}`"),
                )
            })?,
            axes: Box::new([]),
        },
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } if matches!(value.kind(), ExprKind::Number(_)) => {
            let ExprKind::Number(number) = value.kind() else {
                unreachable!()
            };
            CalculusNode::Rational {
                value: decimal_signed(file, expression.range(), number, true, 0)?,
                dimension: DimExponents::DIMENSIONLESS,
            }
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => CalculusNode::Neg(compile_expression(
            file, value, formals, sources, compiled, builder,
        )?),
        ExprKind::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            let exponent = power_exponent(file, right)?;
            let value = compile_expression(file, left, formals, sources, compiled, builder)?;
            let mut product = value;
            for _ in 1..exponent {
                product = builder
                    .push(CalculusNode::Mul(product, value))
                    .map_err(|error| kernel_error(file, expression.range(), error))?;
            }
            return Ok(product);
        }
        ExprKind::Binary { op, left, right }
            if matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::And | BinaryOp::Or
            ) || crate::lower::comparison_operator(*op).is_some() =>
        {
            let left = compile_expression(file, left, formals, sources, compiled, builder)?;
            let mut right = compile_expression(file, right, formals, sources, compiled, builder)?;
            if *op == BinaryOp::Sub {
                right = builder
                    .push(CalculusNode::Neg(right))
                    .map_err(|error| kernel_error(file, expression.range(), error))?;
            }
            match op {
                BinaryOp::Mul => CalculusNode::Mul(left, right),
                BinaryOp::And => CalculusNode::And(left, right),
                BinaryOp::Or => CalculusNode::Or(left, right),
                op if crate::lower::comparison_operator(*op).is_some() => CalculusNode::Compare(
                    crate::lower::comparison_operator(*op).unwrap(),
                    left,
                    right,
                ),
                _ => CalculusNode::Add(left, right),
            }
        }
        ExprKind::Call { callee, arguments }
            if matches!(callee.as_str(), "math.sqrt" | "math.sin")
                || crate::math::piecewise::arity(callee.as_str()).is_some() =>
        {
            let arguments = arguments.positional().ok_or_else(|| {
                pure_error(
                    file,
                    expression.range(),
                    "mathematical builtins require positional arguments",
                )
            })?;
            let arguments = arguments
                .iter()
                .map(|argument| {
                    compile_expression(file, argument, formals, sources, compiled, builder)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(callee.as_str(), "math.sqrt" | "math.sin") {
                let [value] = arguments.as_slice() else {
                    return Err(pure_error(
                        file,
                        expression.range(),
                        "unary mathematical function requires one operand",
                    ));
                };
                CalculusNode::UnaryMath(
                    if callee.as_str() == "math.sin" {
                        eqiora_schema::kernel::UnaryMathFunction::Sin
                    } else {
                        eqiora_schema::kernel::UnaryMathFunction::Sqrt
                    },
                    *value,
                )
            } else {
                return piecewise_calculus(
                    file,
                    expression.range(),
                    callee.as_str(),
                    &arguments,
                    builder,
                );
            }
        }
        ExprKind::Call { callee, arguments } if !callee.is_qualified() => {
            if let Some(definition) = compiled.get(callee.as_str()) {
                let declaration = sources[callee.as_str()];
                let ordered = ordered_arguments(
                    file,
                    expression.range(),
                    declaration.formals().iter().map(|formal| formal.name()),
                    arguments,
                )?;
                let arguments = ordered
                    .into_iter()
                    .map(|value| {
                        compile_expression(file, value, formals, sources, compiled, builder)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return builder
                    .apply_scalar(definition, &arguments)
                    .map_err(|error| kernel_error(file, expression.range(), error));
            }
            let args = arguments.positional().ok_or_else(|| {
                pure_error(
                    file,
                    expression.range(),
                    "calculus constructors require positional arguments",
                )
            })?;
            match (callee.as_str(), args) {
                ("rational", [numerator, denominator]) => CalculusNode::Rational {
                    value: ExactRational::new(
                        integer(file, numerator)?,
                        integer(file, denominator)?,
                    )
                    .map_err(|error| kernel_error(file, expression.range(), error))?,
                    dimension: DimExponents::DIMENSIONLESS,
                },
                ("delta", [left, right]) => {
                    CalculusNode::KroneckerDelta(axis(file, left)?, axis(file, right)?)
                }
                ("component", [formal, axes @ ..]) => {
                    let ExprKind::Name(name) = formal.kind() else {
                        return Err(pure_error(
                            file,
                            formal.range(),
                            "component requires one lexical formal name",
                        ));
                    };
                    CalculusNode::FormalComponent {
                        formal: *formals.get(name.as_str()).ok_or_else(|| {
                            pure_error(
                                file,
                                formal.range(),
                                format!("unknown operator formal `{name}`"),
                            )
                        })?,
                        axes: axes
                            .iter()
                            .map(|value| axis(file, value))
                            .collect::<Result<Box<[_]>, _>>()?,
                    }
                }
                _ => {
                    return Err(pure_error(
                        file,
                        expression.range(),
                        "unsupported exact polynomial constructor",
                    ));
                }
            }
        }
        _ => {
            return Err(pure_error(
                file,
                expression.range(),
                "operator bodies require exact polynomial expressions over lexical formals",
            ));
        }
    };
    builder
        .push(node)
        .map_err(|error| kernel_error(file, expression.range(), error))
}

fn piecewise_calculus(
    file: &str,
    range: TextRange,
    name: &str,
    arguments: &[CalculusNodeId],
    builder: &mut CalculusBuilder,
) -> Result<CalculusNodeId, Diagnostic> {
    use crate::math::piecewise::{self, Primitive};
    let types = arguments
        .iter()
        .map(|id| {
            builder
                .value_type(*id)
                .map_err(|error| kernel_error(file, range, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operand_types = types
        .iter()
        .cloned()
        .map(|value| eqiora_schema::kernel::typing::ExpressionType::<()>::new(value, None))
        .collect::<Vec<_>>();
    piecewise::result_type(name, &operand_types).map_err(|_| {
        pure_error(
            file,
            range,
            "piecewise mathematical builtins require equal complete real scalar types",
        )
    })?;
    piecewise::emit(name, arguments, types[0].dimension(), |primitive| {
        let node = match primitive {
            Primitive::Constant(value, dimension) => CalculusNode::Rational {
                value: ExactRational::new(i64::from(value), 1).expect("small exact coefficient"),
                dimension,
            },
            Primitive::Neg(value) => CalculusNode::Neg(value),
            Primitive::Compare(op, left, right) => CalculusNode::Compare(op, left, right),
            Primitive::Select {
                condition,
                then_value,
                else_value,
            } => CalculusNode::Select {
                condition,
                then_value,
                else_value,
            },
            Primitive::Require { condition, value } => CalculusNode::Require { condition, value },
        };
        builder
            .push(node)
            .map_err(|error| kernel_error(file, range, error))
    })
    .ok_or_else(|| pure_error(file, range, "invalid piecewise builtin arity"))?
}

fn power_exponent(file: &str, expression: &Expr) -> Result<usize, Diagnostic> {
    let ExprKind::Number(value) = expression.kind() else {
        return Err(pure_error(
            file,
            expression.range(),
            "polynomial powers require a positive integer literal exponent",
        ));
    };
    value
        .to_i64()
        .ok()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| {
            *value > 0
                && *value <= usize::from(eqiora_schema::kernel::pure_operator::MAX_FORMAL_EXPONENT)
        })
        .ok_or_else(|| {
            pure_error(
                file,
                expression.range(),
                "polynomial exponent exceeds the positive bounded calculus range",
            )
        })
}

fn integer(file: &str, expression: &Expr) -> Result<i64, Diagnostic> {
    match expression.kind() {
        ExprKind::Number(value) => value
            .to_i64()
            .map_err(|_| pure_error(file, expression.range(), "expected exact i64 literal")),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } if matches!(value.kind(), ExprKind::Number(_)) => {
            let ExprKind::Number(number) = value.kind() else {
                unreachable!()
            };
            let signed = if number.is_negative() {
                number.canonical_text().trim_start_matches('-').to_owned()
            } else {
                format!("-{}", number.canonical_text())
            };
            DecimalLiteral::parse(&signed)
                .and_then(|value| value.to_i64())
                .map_err(|_| pure_error(file, expression.range(), "expected exact i64 literal"))
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => integer(file, value)?
            .checked_neg()
            .ok_or_else(|| pure_error(file, expression.range(), "integer literal overflow")),
        _ => Err(pure_error(
            file,
            expression.range(),
            "expected exact integer literal",
        )),
    }
}

fn axis(file: &str, expression: &Expr) -> Result<ResultAxis, Diagnostic> {
    u16::try_from(integer(file, expression)?)
        .map(ResultAxis::new)
        .map_err(|_| pure_error(file, expression.range(), "result axis exceeds u16"))
}

fn decimal(
    file: &str,
    range: TextRange,
    value: &DecimalLiteral,
) -> Result<ExactRational, Diagnostic> {
    decimal_signed(file, range, value, false, 0)
}

fn decimal_signed(
    file: &str,
    range: TextRange,
    value: &DecimalLiteral,
    negate: bool,
    exponent_shift: i32,
) -> Result<ExactRational, Diagnostic> {
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use num_traits::ToPrimitive;
    let error = || {
        pure_error(
            file,
            range,
            "exact decimal exceeds the rational i64 coefficient range",
        )
    };
    if value.is_zero() {
        return ExactRational::new(0, 1).map_err(|error| kernel_error(file, range, error));
    }
    let exponent = value
        .exponent10()
        .checked_add(i64::from(exponent_shift))
        .ok_or_else(error)?;
    // Any smaller nonzero magnitude is below every representable i64 ratio.
    if exponent > 18 || exponent < -(value.coefficient().len() as i64 + 19) {
        return Err(error());
    }
    let mut numerator: BigInt = value.coefficient().parse().map_err(|_| error())?;
    if value.is_negative() != negate {
        numerator = -numerator;
    }
    let power = BigInt::from(10).pow(exponent.unsigned_abs() as u32);
    let rational = if exponent >= 0 {
        Ratio::from_integer(numerator * power)
    } else {
        Ratio::new(numerator, power)
    };
    ExactRational::new(
        rational.numer().to_i64().ok_or_else(error)?,
        rational.denom().to_i64().ok_or_else(error)?,
    )
    .map_err(|error| kernel_error(file, range, error))
}

fn kernel_error(file: &str, range: TextRange, error: PureOperatorError) -> Diagnostic {
    pure_error(file, range, format!("invalid pure operator: {error}"))
}

fn pure_error(file: &str, range: TextRange, message: impl Into<String>) -> Diagnostic {
    source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(source: &str) -> PureOperatorDefinition {
        let document = eqiora_lang::parse("operator.eqi", source)
            .into_document()
            .expect("source");
        compile_definitions("operator.eqi", &document)
            .expect("definition")
            .into_values()
            .next()
            .unwrap()
    }

    #[test]
    fn source_names_do_not_enter_definition_identity() {
        let first = definition(
            "operator dyadic(input a:spatial[1],input b:spatial[1]):spatial[2]=component(a,0)*component(b,1);",
        );
        let second = definition(
            "operator outer(input x:spatial[1],input y:spatial[1]):spatial[2]=component(x,0)*component(y,1);",
        );
        assert_eq!(first.digest(), second.digest());
    }

    #[test]
    fn decimal_coefficients_are_exact_reduced_rationals() {
        let decimal = definition("operator scale(input x:scalar):scalar=0.01*x;");
        let rational = definition("operator scale(input x:scalar):scalar=rational(1,100)*x;");
        assert_eq!(decimal.digest(), rational.digest());
        let minimum = definition("operator scale(input x:scalar):scalar=-9223372036854775808*x;");
        assert!(minimum.nodes().iter().any(|node| matches!(node, CalculusNode::Rational { value, .. } if *value == ExactRational::new(i64::MIN,1).unwrap())));
        let tiny = DecimalLiteral::parse("25e-20").unwrap();
        assert_eq!(
            super::decimal("exact.eqi", TextRange::new(0, 6), &tiny).unwrap(),
            ExactRational::new(1, 4_000_000_000_000_000_000).unwrap()
        );
        for value in ["1e999999", "1e-999999", "9223372036854775808"] {
            assert!(
                super::decimal(
                    "exact.eqi",
                    TextRange::new(0, 1),
                    &DecimalLiteral::parse(value).unwrap()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn unresolved_formals_fail_at_the_central_definition_boundary() {
        let document = eqiora_lang::parse(
            "invalid.eqi",
            "operator broken(input x:scalar):scalar=component(missing);",
        )
        .into_document()
        .unwrap();
        let error = compile_definitions("invalid.eqi", &document).unwrap_err();
        assert!(
            error
                .message()
                .contains("unknown operator formal `missing`")
        );
        assert!(error.source_span().is_some());
    }
}
