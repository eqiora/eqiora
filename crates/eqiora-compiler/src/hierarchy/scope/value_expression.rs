//! Rewrite source values in one exact occurrence scope.
use super::*;

pub(in crate::hierarchy) fn rewrite_expression_with_boundary_member(
    file: &str,
    expression: &Expr,
    scope: &Scope,
    active: Option<ActiveBoundaryMember<'_>>,
) -> Result<LoweringExpression, Diagnostic> {
    if expression.resolved_nominal().is_some() {
        return Ok(LoweringExpression::literal(
            crate::nominal::literal(file, expression)?,
            expression.range(),
        ));
    }
    if let Some((port, member)) = super::super::quantity_member::split(expression) {
        let symbol = match port.kind() {
            ExprKind::Name(name) => NamePath::from_segments([name.as_str()], port.range())
                .ok()
                .and_then(|path| scope.resolve_symbol(&path)),
            ExprKind::Path(path) => scope.resolve_symbol(path),
            ExprKind::Member { .. } => scope.indexed_port(file, &port).ok(),
            ExprKind::BoundaryPortSelection { port, selector } => {
                resolve_boundary_family_selection(file, port, selector, scope, active).ok()
            }
            _ => None,
        };
        if let Some(FlatSymbol {
            internal_name,
            kind:
                SymbolKind::Port {
                    quantities: Some(quantities),
                    ..
                },
            ..
        }) = symbol
        {
            let role = quantities.role(member).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("Port has no declared quantity `{member}`"),
                )
            })?;
            return Ok(LoweringExpression::call(
                role.to_owned(),
                LoweringExpression::name(internal_name.clone(), port.range()),
                expression.range(),
            ));
        }
    }
    let lowered = match expression.kind() {
        ExprKind::Partial {
            value,
            wrt,
            holding,
        } => {
            let independent = |binding: &NamePath| -> Result<String, Diagnostic> {
                // The definition checker already excludes let aliases. Component
                // Parameter slots are terms, and a direct binding preserves the
                // exact parent Parameter rather than allocating another entity.
                if let Some(parameter) = scope.parameter(binding.as_str()) {
                    return match parameter.lineage {
                        super::super::parameters::ParameterLineage::Parameter(identity) => {
                            Ok(super::super::expand::names::internal_name(identity))
                        }
                        _ => Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            binding.range(),
                            "partial binding must retain an independent Parameter; a specialized literal or derived term is not a new independent value",
                        )),
                    };
                }
                let symbol = resolve_expression_symbol(file, binding, scope)?;
                if !matches!(symbol.kind, SymbolKind::Parameter | SymbolKind::Field) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        binding.range(),
                        "partial binding must name a declared independent Parameter or continuous state Field",
                    ));
                }
                Ok(symbol.internal_name.clone())
            };
            let selected = independent(wrt)?;
            let mut seen = std::collections::BTreeSet::from([selected.clone()]);
            for binding in holding {
                if !seen.insert(independent(binding)?) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        binding.range(),
                        "partial holding requires distinct other independent bindings",
                    ));
                }
            }
            LoweringExpression::partial(
                rewrite_expression_with_boundary_member(file, value, scope, active)?,
                selected,
                expression.range(),
            )
        }
        ExprKind::Case { value, arms } => {
            let value = rewrite_expression_with_boundary_member(file, value, scope, active)?;
            let arms = arms
                .iter()
                .map(|arm| {
                    let pattern = arm.resolved_pattern().cloned().ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            arm.range(),
                            "case pattern requires exact enum binding",
                        )
                    })?;
                    Ok((
                        pattern,
                        rewrite_expression_with_boundary_member(file, arm.value(), scope, active)?,
                    ))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            LoweringExpression::case(value, arms, expression.range())
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => LoweringExpression::select(
            rewrite_expression_with_boundary_member(file, condition, scope, active)?,
            rewrite_expression_with_boundary_member(file, then_value, scope, active)?,
            rewrite_expression_with_boundary_member(file, else_value, scope, active)?,
            expression.range(),
        ),
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if crate::math::piecewise::arity(callee.as_str()).is_some() => {
            LoweringExpression::piecewise(
                callee.as_str().to_owned(),
                arguments
                    .iter()
                    .map(|argument| {
                        rewrite_expression_with_boundary_member(file, argument, scope, active)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                expression.range(),
            )
        }
        ExprKind::Reduction { .. } => return reductions::rewrite(file, expression, scope, active),
        ExprKind::Array(elements) => LoweringExpression::array(
            elements
                .iter()
                .map(|value| rewrite_expression_with_boundary_member(file, value, scope, active))
                .collect::<Result<Vec<_>, _>>()?,
            expression.range(),
        ),
        ExprKind::Index { value, index } => {
            let (index, dependencies) = super::super::parameters::structural_index(
                file,
                index,
                &scope.symbolic_parameters(),
            )?
            .ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    index.range(),
                    "channel index requires an exact static integer value",
                )
            })?;
            LoweringExpression::index(
                rewrite_expression_with_boundary_member(file, value, scope, active)?,
                index,
                expression.range(),
            )
            .with_structural_parameters(dependencies)
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            let (start, end) = super::super::parameters::static_slice(
                file,
                lower,
                upper,
                &scope.symbolic_parameters(),
            )?;
            let value = rewrite_expression_with_boundary_member(file, value, scope, active)?;
            let values = scope.symbolic_parameters();
            let dependencies = [lower.as_ref(), upper.as_ref()]
                .into_iter()
                .map(|bound| {
                    super::super::parameters::structural_index(file, bound, &values)
                        .map(|resolved| resolved.expect("checked slice bound").1)
                })
                .collect::<Result<Vec<_>, _>>()?;
            LoweringExpression::array(
                (start..end)
                    .map(|index| {
                        LoweringExpression::index(value.clone(), index, expression.range())
                    })
                    .collect(),
                expression.range(),
            )
            .with_structural_parameters(dependencies.into_iter().flatten())
        }
        ExprKind::Member { .. } => LoweringExpression::name(
            scope.indexed_port(file, expression)?.internal_name.clone(),
            expression.range(),
        ),
        ExprKind::Path(path) if path.as_str() == "math.i" => LoweringExpression::literal(
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
        ),
        ExprKind::Call { callee, .. } if callee.as_str() == "tensor_value" => {
            LoweringExpression::literal(
                super::super::parameters::frames::literal(
                    file,
                    expression,
                    &scope.symbolic_parameters(),
                    &mut |name| {
                        scope
                            .spatial_support(name)
                            .map(super::super::parameters::frames::occurrence)
                    },
                )?,
                expression.range(),
            )
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if callee.as_str() == "math.complex" => {
            let [real, imag] = arguments.as_slice() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "math.complex requires exactly two real scalar arguments",
                ));
            };
            LoweringExpression::complex(
                rewrite_expression_with_boundary_member(file, real, scope, active)?,
                rewrite_expression_with_boundary_member(file, imag, scope, active)?,
                expression.range(),
            )
        }
        ExprKind::Boolean(value) => LoweringExpression::literal(
            eqiora_core::ValueLiteral::boolean(*value),
            expression.range(),
        ),
        ExprKind::Number(value) => LoweringExpression::number(value.clone(), expression.range()),
        ExprKind::Quantity { .. } => LoweringExpression::from_source(expression),
        ExprKind::Call { callee, arguments } if callee.as_str() == "time" => {
            if arguments
                .positional()
                .is_none_or(|values| !values.is_empty())
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "time() accepts no arguments",
                ));
            }
            LoweringExpression::name("time".to_owned(), expression.range())
        }
        ExprKind::Name(name) => {
            if let Some(value) = scope.property_value(file, name, expression.range())? {
                return Ok(value);
            }
            if let Some(value) = scope.value_expression(name) {
                return Ok(value);
            }
            let path = NamePath::from_segments([name.clone()], expression.range()).map_err(
                |ast_error| {
                    source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        file,
                        expression.range(),
                        ast_error.message(),
                    )
                },
            )?;
            LoweringExpression::name(
                resolve_expression_symbol(file, &path, scope)?
                    .internal_name
                    .clone(),
                expression.range(),
            )
        }
        ExprKind::Path(path) if scope.value_expression(path.as_str()).is_some() => scope
            .value_expression(path.as_str())
            .expect("known Parameter expression")
            .clone(),
        ExprKind::Path(path) => match crate::math::constant(path) {
            Some(value) => LoweringExpression::quantity(
                DynQuantity::new(value, DimExponents::DIMENSIONLESS),
                expression.range(),
            ),
            None => LoweringExpression::name(
                resolve_expression_symbol(file, path, scope)?
                    .internal_name
                    .clone(),
                expression.range(),
            ),
        },
        ExprKind::BoundaryPortSelection { port, selector } => LoweringExpression::name(
            resolve_boundary_family_selection(file, port, selector, scope, active)?
                .internal_name
                .clone(),
            expression.range(),
        ),
        ExprKind::Unary {
            op: eqiora_lang::UnaryOp::Neg,
            value,
        } => LoweringExpression::neg(
            rewrite_expression_with_boundary_member(file, value, scope, active)?,
            expression.range(),
        ),
        ExprKind::Unary {
            op: eqiora_lang::UnaryOp::Not,
            value,
        } => LoweringExpression::logical_not(
            rewrite_expression_with_boundary_member(file, value, scope, active)?,
            expression.range(),
        ),
        ExprKind::Binary { op, left, right } => LoweringExpression::binary(
            *op,
            rewrite_expression_with_boundary_member(file, left, scope, active)?,
            rewrite_expression_with_boundary_member(file, right, scope, active)?,
            expression.range(),
        ),
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if callee.as_str() == "period" => {
            let [argument] = arguments.as_slice() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "period requires one clock name",
                ));
            };
            let ExprKind::Name(name) = argument.kind() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    argument.range(),
                    "period requires one clock name",
                ));
            };
            let clock = resolve_local_kind(
                file,
                argument.range(),
                scope,
                name,
                |kind| matches!(kind, SymbolKind::Clock(_)),
                "period clock",
            )?;
            let SymbolKind::Clock(period) = clock.kind else {
                unreachable!("checked clock")
            };
            LoweringExpression::quantity(
                DynQuantity::new(period.as_seconds_f64(), crate::dimensions::time_dimension()),
                expression.range(),
            )
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if callee.as_str() == "sample" => {
            let [value, clock] = arguments.as_slice() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "sample requires a value and one clock name",
                ));
            };
            let ExprKind::Name(clock_name) = clock.kind() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    clock.range(),
                    "sample requires one clock name",
                ));
            };
            let clock = resolve_local_kind(
                file,
                clock.range(),
                scope,
                clock_name,
                |kind| matches!(kind, SymbolKind::Clock(_)),
                "sample clock",
            )?;
            LoweringExpression::sample(
                rewrite_expression_with_boundary_member(file, value, scope, active)?,
                clock.internal_name.clone(),
                expression.range(),
            )
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if crate::lower::IntegerBuiltin::named(callee.as_str()).is_some() => {
            let operator =
                crate::lower::IntegerBuiltin::named(callee.as_str()).expect("named builtin guard");
            if arguments.len() != operator.arity() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("{callee} requires exactly {} arguments", operator.arity()),
                ));
            }
            LoweringExpression::integer_call(
                operator,
                arguments
                    .iter()
                    .map(|argument| {
                        rewrite_expression_with_boundary_member(file, argument, scope, active)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                expression.range(),
            )
        }
        ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } if is_builtin_operator(callee) => {
            let [argument] = arguments.as_slice() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    format!("builtin operator `{callee}` requires exactly one argument"),
                ));
            };
            LoweringExpression::call(
                callee.as_str().to_owned(),
                rewrite_expression_with_boundary_member(file, argument, scope, active)?,
                expression.range(),
            )
        }
        ExprKind::Call { callee, arguments } => {
            operators::rewrite(file, expression, callee, arguments, scope, active)?
        }
        _ => {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                expression.range(),
                "expression syntax is newer than hierarchy elaboration",
            ));
        }
    };
    Ok(lowered)
}
