//! Bounded source expansion for ordered finite scalar reductions.
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_lang::{DecimalLiteral, Expr, ExprKind, FamilyBinderSyntax, NamePath, SourceAstFactory};

use super::parameters::SymbolicParameterMap;
use crate::{diagnostics::source_error, source_identity::LocalSourceIdentityLimits};

fn error(file: &str, expression: &Expr, message: impl Into<String>) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        message,
    )
}

pub(super) fn preflight(
    file: &str,
    expression: &Expr,
    resolve_extent: &mut dyn FnMut(&str) -> Option<u32>,
    values: &SymbolicParameterMap,
    max_terms: usize,
) -> Result<(), Diagnostic> {
    expanded_nodes(file, expression, resolve_extent, values, max_terms).map(|_| ())
}

pub(super) fn expanded_nodes(
    file: &str,
    expression: &Expr,
    resolve_extent: &mut dyn FnMut(&str) -> Option<u32>,
    values: &SymbolicParameterMap,
    max_terms: usize,
) -> Result<usize, Diagnostic> {
    let limits = LocalSourceIdentityLimits::default();
    let (nodes, _) = measure(
        file,
        expression,
        resolve_extent,
        values,
        max_terms.min(limits.max_expression_nodes),
        1,
        &mut Vec::new(),
    )?;
    Ok(nodes)
}

fn measure<'a>(
    file: &str,
    expression: &'a Expr,
    resolve_extent: &mut dyn FnMut(&str) -> Option<u32>,
    values: &SymbolicParameterMap,
    budget: usize,
    source_depth: usize,
    binders: &mut Vec<&'a str>,
) -> Result<(usize, usize), Diagnostic> {
    let depth_limit = LocalSourceIdentityLimits::default().max_expression_depth;
    let exceeded = || {
        error(
            file,
            expression,
            "reduction expansion exceeds the expression work or depth limit",
        )
    };
    if source_depth > depth_limit {
        return Err(exceeded());
    }
    let result = if let ExprKind::Reduction {
        operation,
        binder,
        value,
    } = expression.kind()
    {
        if binders.contains(&binder.member()) {
            return Err(error(
                file,
                expression,
                "nested reduction cannot shadow an active member binder",
            ));
        }
        let extent = resolve_extent(binder.set().as_str())
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                error(
                    file,
                    expression,
                    "reduction requires an existing nonempty bounded IndexSet",
                )
            })? as usize;
        if has_slice(value) {
            // Slice widths can vary by member. Reuse capture-free ordinal
            // substitution, and measure each bounded temporary before any DAG
            // expansion. This is not a second static-expression evaluator.
            let fold_cost = if matches!(
                operation,
                eqiora_lang::ReductionOp::Min | eqiora_lang::ReductionOp::Max
            ) {
                2
            } else {
                1
            };
            let fold_work = (extent - 1).checked_mul(fold_cost).ok_or_else(exceeded)?;
            if extent
                .checked_add(fold_work)
                .is_none_or(|work| work > budget)
                || fold_work >= depth_limit
            {
                return Err(exceeded());
            }
            let mut nodes = fold_work;
            let mut depth = 0usize;
            for ordinal in 0..extent {
                check_copy_work(file, value, budget.saturating_sub(nodes))?;
                let term = instantiate(file, value, binder, ordinal as u32)?;
                let (work, term_depth) = measure(
                    file,
                    &term,
                    resolve_extent,
                    values,
                    budget.saturating_sub(nodes),
                    source_depth + 1,
                    &mut Vec::new(),
                )?;
                nodes = nodes
                    .checked_add(work)
                    .filter(|work| *work <= budget)
                    .ok_or_else(exceeded)?;
                depth = depth.max(term_depth.checked_add(fold_work).ok_or_else(exceeded)?);
                if depth > depth_limit {
                    return Err(exceeded());
                }
            }
            return Ok((nodes, depth));
        }
        binders.push(binder.member());
        let (nodes, depth) = measure(
            file,
            value,
            resolve_extent,
            values,
            budget,
            source_depth + 1,
            binders,
        )?;
        binders.pop();
        let fold_cost = if matches!(
            operation,
            eqiora_lang::ReductionOp::Min | eqiora_lang::ReductionOp::Max
        ) {
            2
        } else {
            1
        };
        let fold_work = (extent - 1).checked_mul(fold_cost).ok_or_else(exceeded)?;
        (
            nodes
                .checked_mul(extent)
                .and_then(|n| n.checked_add(fold_work))
                .ok_or_else(exceeded)?,
            depth.checked_add(fold_work).ok_or_else(exceeded)?,
        )
    } else {
        let (mut nodes, overhead) = match expression.kind() {
            ExprKind::Slice { .. } => (1, 2),
            ExprKind::Case { arms, .. } => {
                let folds = arms.len().saturating_sub(1).max(1);
                (
                    folds.checked_mul(3).ok_or_else(exceeded)?,
                    folds.checked_mul(2).ok_or_else(exceeded)?,
                )
            }

            ExprKind::Call { callee, .. } => {
                crate::math::piecewise::cost(callee.as_str()).unwrap_or((1, 1))
            }
            _ => (1, 1),
        };
        let mut depth = overhead;
        visit_children(expression, &mut |child| {
            let (child_nodes, child_depth) = measure(
                file,
                child,
                resolve_extent,
                values,
                budget,
                source_depth + 1,
                binders,
            )?;
            nodes = nodes.checked_add(child_nodes).ok_or_else(exceeded)?;
            depth = depth.max(child_depth + overhead);
            if nodes > budget {
                return Err(exceeded());
            }
            Ok(())
        })?;
        if let ExprKind::Slice { lower, upper, .. } = expression.kind() {
            // Check all bound-expression work before evaluating either bound.
            // Resolved bounds charge their exact generated array/index width.
            // Generic or reduction-member bounds reserve the admitted maximum;
            // ordinary typing still independently rejects invalid bounds.
            let width = if unresolved_bound(lower, values) || unresolved_bound(upper, values) {
                65_536
            } else {
                let (start, end) = super::parameters::static_slice(file, lower, upper, values)?;
                (end - start) as usize
            };
            nodes = nodes.checked_add(width).ok_or_else(exceeded)?;
        }
        (nodes, depth)
    };
    if result.0 > budget || result.1 > depth_limit {
        return Err(exceeded());
    }
    Ok(result)
}

fn has_slice(expression: &Expr) -> bool {
    let mut found = matches!(expression.kind(), ExprKind::Slice { .. });
    let _ = visit_children(expression, &mut |child| {
        found |= has_slice(child);
        Ok(())
    });
    found
}

fn check_copy_work(file: &str, expression: &Expr, budget: usize) -> Result<(), Diagnostic> {
    let mut pending = vec![expression];
    let mut nodes = 0usize;
    while let Some(value) = pending.pop() {
        nodes += 1;
        if nodes > budget {
            return Err(error(
                file,
                expression,
                "slice specialization exceeds the expression work limit",
            ));
        }
        visit_children(value, &mut |child| {
            pending.push(child);
            Ok(())
        })?;
    }
    Ok(())
}

fn unresolved_bound(expression: &Expr, values: &SymbolicParameterMap) -> bool {
    if expression.resolved_nominal().is_some() || expression.resolved_enum().is_some() {
        return false;
    }
    if let ExprKind::Name(name) = expression.kind() {
        return values.get(name).is_none_or(|value| value.value.is_none());
    }
    let mut unresolved = false;
    let _ = visit_children(expression, &mut |child| {
        unresolved |= unresolved_bound(child, values);
        Ok(())
    });
    unresolved
}

fn visit_children<'a>(
    expression: &'a Expr,
    visit: &mut impl FnMut(&'a Expr) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    match expression.kind() {
        ExprKind::Case { value, arms } => {
            visit(value)?;
            for arm in arms {
                visit(arm.value())?;
            }
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => {
            visit(condition)?;
            visit(then_value)?;
            visit(else_value)?;
        }
        ExprKind::Quantity { unit, .. } => visit(unit)?,
        ExprKind::Unary { value, .. }
        | ExprKind::Member { value, .. }
        | ExprKind::Reduction { value, .. } => visit(value)?,
        ExprKind::Binary { left, right, .. } => {
            visit(left)?;
            visit(right)?;
        }
        ExprKind::Index { value, index } => {
            visit(value)?;
            visit(index)?;
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            visit(value)?;
            visit(lower)?;
            visit(upper)?;
        }
        ExprKind::Call { arguments, .. } => {
            for argument in arguments.expressions() {
                visit(argument)?;
            }
        }
        ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
            for element in elements {
                visit(element)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn instantiate_instance(
    file: &str,
    instance: &eqiora_lang::InstanceDecl,
    ordinal: u32,
) -> Result<eqiora_lang::InstanceDecl, Diagnostic> {
    let Some(binder) = instance.family() else {
        return Ok(instance.clone());
    };
    let bindings = instance
        .bindings()
        .iter()
        .map(|binding| {
            let value = instantiate(file, binding.value(), binder, ordinal)?;
            SourceAstFactory::named_binding(binding.name(), value, binding.range()).map_err(
                |failure| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        binding.range(),
                        failure.to_string(),
                    )
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    SourceAstFactory::instance(
        instance.name(),
        instance.definition().clone(),
        None,
        bindings,
        instance.range(),
    )
    .map_err(|failure| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            instance.range(),
            failure.to_string(),
        )
    })
}

pub(super) fn instantiate(
    file: &str,
    value: &Expr,
    binder: &FamilyBinderSyntax,
    ordinal: u32,
) -> Result<Expr, Diagnostic> {
    substitute(file, value, binder.member(), ordinal, 1)
}

fn substitute(
    file: &str,
    expression: &Expr,
    member: &str,
    ordinal: u32,
    depth: usize,
) -> Result<Expr, Diagnostic> {
    if depth > LocalSourceIdentityLimits::default().max_expression_depth {
        return Err(error(
            file,
            expression,
            "reduction member expression exceeds the depth limit",
        ));
    }
    if expression.resolved_enum().is_some() {
        return Ok(expression.clone());
    }
    let construct = |kind| {
        SourceAstFactory::expression(kind, expression.range())
            .map_err(|failure| error(file, expression, failure.to_string()))
    };
    if let ExprKind::Call { callee, arguments } = expression.kind()
        && callee.as_str() == "ordinal"
        && matches!(arguments.positional(), Some([argument]) if matches!(argument.kind(), ExprKind::Name(name) if name == member))
    {
        let number = construct(ExprKind::Number(
            DecimalLiteral::parse(&ordinal.to_string())
                .map_err(|failure| error(file, expression, failure.to_string()))?,
        ))?;
        return construct(ExprKind::Call {
            callee: NamePath::from_segments(["to_integer"], expression.range())
                .map_err(|failure| error(file, expression, failure.to_string()))?,
            arguments: eqiora_lang::CallArguments::Positional(vec![number]),
        });
    }
    let mut child = |value: &Expr| substitute(file, value, member, ordinal, depth + 1);
    let kind = match expression.kind() {
        ExprKind::Name(name) if name == member => {
            return Err(error(
                file,
                expression,
                "reduction members require explicit ordinal(member) projection",
            ));
        }
        ExprKind::Reduction {
            operation,
            binder,
            value,
        } => {
            if binder.member() == member {
                return Err(error(
                    file,
                    expression,
                    "nested reduction cannot shadow an active member binder",
                ));
            }
            ExprKind::Reduction {
                operation: *operation,
                binder: binder.clone(),
                value: Box::new(child(value)?),
            }
        }
        ExprKind::Case { value, arms } => ExprKind::Case {
            value: Box::new(child(value)?),
            arms: arms
                .iter()
                .map(|arm| {
                    SourceAstFactory::case_arm_value(arm, child(arm.value())?)
                        .map_err(|failure| error(file, expression, failure.to_string()))
                })
                .collect::<Result<_, _>>()?,
        },
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => ExprKind::Select {
            condition: Box::new(child(condition)?),
            then_value: Box::new(child(then_value)?),
            else_value: Box::new(child(else_value)?),
        },
        ExprKind::Quantity { value, unit } => ExprKind::Quantity {
            value: value.clone(),
            unit: Box::new(child(unit)?),
        },
        ExprKind::Unary { op, value } => ExprKind::Unary {
            op: *op,
            value: Box::new(child(value)?),
        },
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(child(left)?),
            right: Box::new(child(right)?),
        },
        ExprKind::Array(elements) => {
            ExprKind::Array(elements.iter().map(&mut child).collect::<Result<_, _>>()?)
        }
        ExprKind::Tuple(elements) => {
            ExprKind::Tuple(elements.iter().map(&mut child).collect::<Result<_, _>>()?)
        }
        ExprKind::Index { value, index } => ExprKind::Index {
            value: Box::new(child(value)?),
            index: Box::new(child(index)?),
        },
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => ExprKind::Slice {
            value: Box::new(child(value)?),
            lower: Box::new(child(lower)?),
            upper: Box::new(child(upper)?),
        },
        ExprKind::Member { value, member } => ExprKind::Member {
            value: Box::new(child(value)?),
            member: member.clone(),
        },
        ExprKind::Call { callee, arguments } => ExprKind::Call {
            callee: callee.clone(),
            arguments: match arguments {
                eqiora_lang::CallArguments::Mixed { positional, named } => {
                    eqiora_lang::CallArguments::Mixed {
                        positional: positional
                            .iter()
                            .map(&mut child)
                            .collect::<Result<_, _>>()?,
                        named: named
                            .iter()
                            .map(|binding| {
                                SourceAstFactory::named_binding(
                                    binding.name(),
                                    child(binding.value())?,
                                    binding.range(),
                                )
                                .map_err(|failure| error(file, expression, failure.to_string()))
                            })
                            .collect::<Result<_, _>>()?,
                    }
                }
                eqiora_lang::CallArguments::Positional(values) => {
                    eqiora_lang::CallArguments::Positional(
                        values.iter().map(&mut child).collect::<Result<_, _>>()?,
                    )
                }
                eqiora_lang::CallArguments::Named(values) => eqiora_lang::CallArguments::Named(
                    values
                        .iter()
                        .map(|binding| {
                            SourceAstFactory::named_binding(
                                binding.name(),
                                child(binding.value())?,
                                binding.range(),
                            )
                            .map_err(|failure| error(file, expression, failure.to_string()))
                        })
                        .collect::<Result<_, _>>()?,
                ),
            },
        },
        _ => return Ok(expression.clone()),
    };
    let mut result = construct(kind)?;
    if let Some(value_type) = expression.resolved_nominal() {
        let ExprKind::Call {
            arguments: eqiora_lang::CallArguments::Positional(arguments),
            ..
        } = expression.kind()
        else {
            return Err(error(
                file,
                expression,
                "nominal metadata requires a constructor call",
            ));
        };
        let count = if value_type.map_bases().is_some() {
            2
        } else {
            1
        };
        let declarations = arguments
            .iter()
            .take(count)
            .map(|argument| {
                eqiora_lang::FiniteBasisSyntax::from_expression(argument)
                    .map(|basis| basis.name)
                    .ok_or_else(|| {
                        error(
                            file,
                            expression,
                            "nominal constructor requires exact declaration arguments",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        SourceAstFactory::bind_nominal_expression(&mut result, &declarations, value_type.clone())
            .map_err(|failure| error(file, expression, failure.to_string()))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_lang::Item;

    fn expression(text: &str) -> Expr {
        let source = format!("model M() {{ let value = {text}; }}");
        let document = eqiora_lang::parse("reduce.eqi", &source)
            .into_document()
            .unwrap();
        let Item::Let(value) = &document.models()[0].items()[0] else {
            panic!("alias")
        };
        value.value().clone()
    }

    #[test]
    fn slice_budget_counts_generated_indices_and_preserves_bound_diagnostics() {
        let slice = expression("data[0:2]");
        // Array construction + operand + two source bounds + two emitted indices.
        assert_eq!(
            expanded_nodes(
                "slice",
                &slice,
                &mut |_| None,
                &SymbolicParameterMap::new(),
                6
            )
            .unwrap(),
            6
        );
        assert!(
            expanded_nodes(
                "slice",
                &slice,
                &mut |_| None,
                &SymbolicParameterMap::new(),
                5
            )
            .is_err()
        );
        let error = expanded_nodes(
            "slice",
            &expression("data[2:1]"),
            &mut |_| None,
            &SymbolicParameterMap::new(),
            100,
        )
        .unwrap_err();
        assert!(!error.message().contains("work"), "{error:?}");
        assert!(error.source_span().is_some());
    }

    #[test]
    fn extrema_charge_comparison_and_selection_before_copying() {
        let value = expression("min(1, over=(i in A))");
        assert!(
            preflight(
                "test",
                &value,
                &mut |_| Some(3),
                &SymbolicParameterMap::new(),
                7
            )
            .is_ok()
        );
        assert!(
            preflight(
                "test",
                &value,
                &mut |_| Some(3),
                &SymbolicParameterMap::new(),
                6
            )
            .is_err()
        );
        assert!(
            preflight(
                "test",
                &value,
                &mut |_| Some(128),
                &SymbolicParameterMap::new(),
                1000
            )
            .is_ok()
        );
        assert!(
            preflight(
                "test",
                &value,
                &mut |_| Some(129),
                &SymbolicParameterMap::new(),
                1000
            )
            .is_err()
        );
    }

    #[test]
    fn expansion_counts_nested_work_and_left_fold_depth_before_copying() {
        let nested = expression("sum(product(1, over = (j in B)), over = (i in A))");
        // Inner three leaves + two product nodes; outer two copies + one sum.
        let mut extent = |name: &str| match name {
            "A" => Some(2),
            "B" => Some(3),
            _ => None,
        };
        assert!(
            preflight(
                "test",
                &nested,
                &mut extent,
                &SymbolicParameterMap::new(),
                11
            )
            .is_ok()
        );
        assert!(
            preflight(
                "test",
                &nested,
                &mut extent,
                &SymbolicParameterMap::new(),
                10
            )
            .is_err()
        );
        let flat = expression("sum(1, over = (i in A))");
        assert!(
            preflight(
                "test",
                &flat,
                &mut |_| Some(256),
                &SymbolicParameterMap::new(),
                1000
            )
            .is_ok()
        );
        assert!(
            preflight(
                "test",
                &flat,
                &mut |_| Some(257),
                &SymbolicParameterMap::new(),
                1000
            )
            .is_err()
        );
        assert!(
            preflight(
                "test",
                &flat,
                &mut |_| Some(0),
                &SymbolicParameterMap::new(),
                1000
            )
            .is_err()
        );
        assert!(
            preflight(
                "test",
                &flat,
                &mut |_| None,
                &SymbolicParameterMap::new(),
                1000
            )
            .is_err()
        );
        assert!(
            preflight(
                "test",
                &nested,
                &mut |_| Some(u32::MAX),
                &SymbolicParameterMap::new(),
                usize::MAX
            )
            .is_err()
        );
    }

    #[test]
    fn substitution_is_capture_aware_and_keeps_integer_projection() {
        let outer =
            expression("sum(product(ordinal(i) + ordinal(j), over = (j in B)), over = (i in A))");
        let ExprKind::Reduction { binder, value, .. } = outer.kind() else {
            panic!("reduction")
        };
        let result = instantiate("test", value, binder, u32::MAX).unwrap();
        let ExprKind::Reduction {
            binder: inner,
            value,
            ..
        } = result.kind()
        else {
            panic!("nested")
        };
        assert_eq!(inner.member(), "j");
        let ExprKind::Binary { left, right, .. } = value.kind() else {
            panic!("sum")
        };
        let ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } = left.kind()
        else {
            panic!("conversion")
        };
        assert_eq!(callee.as_str(), "to_integer");
        let ExprKind::Number(number) = arguments[0].kind() else {
            panic!("number")
        };
        assert_eq!(number.to_i64().unwrap(), i64::from(u32::MAX));
        let ExprKind::Call { callee, .. } = right.kind() else {
            panic!("inner ordinal")
        };
        assert_eq!(callee.as_str(), "ordinal");
        assert!(instantiate("test", &expression("i"), binder, 0).is_err());
        let collision = expression("sum(ordinal(i), over = (i in A))");
        assert!(instantiate("test", &collision, binder, 0).is_err());
        assert!(
            preflight(
                "test",
                &expression("sum(sum(1, over = (i in B)), over = (i in A))"),
                &mut |_| Some(1),
                &SymbolicParameterMap::new(),
                100
            )
            .is_err()
        );
    }
    #[test]
    fn indexed_selection_retains_exact_nominal_constructor_metadata() {
        let reduction = expression("sum(ordinal(i), over = (i in Stages))");
        let ExprKind::Reduction { binder, .. } = reduction.kind() else {
            panic!("reduction")
        };
        let mut constructor = expression("index(Stages, ordinal(i))");
        let exact_type = eqiora_core::ValueType::index(eqiora_core::Id::new(), 4).unwrap();
        let declaration = NamePath::from_segments(["Stages"], constructor.range()).unwrap();
        SourceAstFactory::bind_nominal_expression(
            &mut constructor,
            std::slice::from_ref(&declaration),
            exact_type.clone(),
        )
        .unwrap();
        let result = instantiate("test", &constructor, binder, 3).unwrap();
        assert_eq!(result.resolved_nominal(), Some(&exact_type));
        assert_eq!(result.range(), constructor.range());
        let ExprKind::Call {
            arguments: eqiora_lang::CallArguments::Positional(arguments),
            ..
        } = result.kind()
        else {
            panic!("index")
        };
        assert_eq!(
            arguments[0],
            match constructor.kind() {
                ExprKind::Call {
                    arguments: eqiora_lang::CallArguments::Positional(arguments),
                    ..
                } => arguments[0].clone(),
                _ => unreachable!(),
            }
        );
    }
}
