use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{ComponentDecl, ComponentItem, Item, ModelDecl, NamedDefinitionDecl};

use crate::diagnostics::{source_error, stable_sort};
use eqiora_schema::kernel::typing::SpatialSupport;

use super::expression_eval::{
    ExpressionContext, coerce_parameter_with_label, evaluate_parameter_expression,
    infer_parameter_with_label,
};
use super::{
    ExpressionDefinition, ParameterLineage, SymbolicParameterMap, expression_cycles,
    expression_evaluation_order,
};

pub(in crate::hierarchy) fn resolve_model_lets(
    file: &str,
    model: &ModelDecl,
    values: &mut SymbolicParameterMap,
    mut resolve_clock: impl FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
) -> Result<(), Vec<Diagnostic>> {
    resolve_lets(
        file,
        model.items().iter().filter_map(|item| match item {
            Item::Let(declaration) => Some(declaration),
            _ => None,
        }),
        values,
        &mut resolve_clock,
        &super::super::supports::model_spatial_supports(file, model)?,
    )
}

pub(in crate::hierarchy) fn resolve_component_lets(
    file: &str,
    component: &ComponentDecl,
    values: &mut SymbolicParameterMap,
    mut resolve_clock: impl FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
) -> Result<(), Vec<Diagnostic>> {
    resolve_lets(
        file,
        component.items().iter().filter_map(|item| match item {
            ComponentItem::Let(declaration) => Some(declaration),
            _ => None,
        }),
        values,
        &mut resolve_clock,
        &super::super::supports::component_spatial_supports(file, component)?,
    )
}

fn resolve_lets<'a>(
    file: &str,
    declarations: impl Iterator<Item = &'a NamedDefinitionDecl>,
    values: &mut SymbolicParameterMap,
    resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    frames: &BTreeMap<String, SpatialSupport<String>>,
) -> Result<(), Vec<Diagnostic>> {
    let declarations = declarations
        .map(|declaration| (declaration.name().to_owned(), declaration))
        .collect::<BTreeMap<_, _>>();
    let order = alias_order(file, declarations.values().copied())?;
    let mut diagnostics = Vec::new();
    for declaration in order {
        let name = declaration.name().to_owned();
        // Runtime expressions are validated after Field/support interfaces are bound.
        if !is_static_expression(declaration.value(), values) {
            continue;
        }
        let target = match declaration
            .value_type()
            .map(|ty| {
                specialize_type(file, ty, values).and_then(|ty| {
                    super::frames::parameter_type(file, &ty, Some(declaration.value()), frames)
                })
            })
            .transpose()
        {
            Ok(target) => target,
            Err(error) => {
                diagnostics.push(error);
                continue;
            }
        };
        let mut resolve = |dependency: &str, range| {
            values.get(dependency).cloned().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    ExpressionContext::Let.unknown_name_message(dependency),
                )
            })
        };
        let evaluated = match &target {
            Some(target) => super::expression_eval::evaluate_initializer(
                file,
                declaration.value(),
                ExpressionContext::Let,
                &mut resolve,
                target.clone(),
                "let alias",
                (&mut *resolve_clock, &mut |name| frames.get(name).cloned()),
            ),
            None => evaluate_parameter_expression(
                file,
                declaration.value(),
                ExpressionContext::Let,
                &mut resolve,
                resolve_clock,
                &mut |name| frames.get(name).cloned(),
            ),
        };
        let range = declaration.range();
        match evaluated.and_then(|value| match &target {
            Some(target) => {
                coerce_parameter_with_label(file, range, value, target.clone(), "let alias", true)
            }
            None => infer_parameter_with_label(file, range, value, "let alias"),
        }) {
            Ok(mut value) => {
                if let Some(syntax) = declaration.value_type() {
                    let mut dependencies = BTreeSet::new();
                    for extent in extent_expressions(syntax) {
                        if let Some((_, names)) =
                            structural_extent(file, extent, values).map_err(|error| vec![error])?
                        {
                            dependencies.extend(names);
                        }
                    }
                    value.expression = value
                        .expression
                        .map(|expression| expression.with_structural_parameters(dependencies));
                }
                if !matches!(value.lineage, Some(ParameterLineage::Constant)) {
                    value.lineage = Some(ParameterLineage::Derived);
                }
                values.insert(name, value);
            }
            Err(error) => diagnostics.push(error),
        }
    }
    stable_sort(&mut diagnostics);
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(diagnostics)
    }
}

/// One bounded dependency graph for both static and runtime aliases.
pub(in crate::hierarchy) fn alias_order<'a>(
    file: &str,
    declarations: impl Iterator<Item = &'a NamedDefinitionDecl>,
) -> Result<Vec<&'a NamedDefinitionDecl>, Vec<Diagnostic>> {
    let declarations = declarations
        .map(|d| (d.name().to_owned(), d))
        .collect::<BTreeMap<_, _>>();
    let definitions = declarations
        .iter()
        .map(|(name, d)| {
            let mut dependencies = BTreeMap::new();
            let mut pending = vec![d.value()];
            if let Some(syntax) = d.value_type() {
                pending.extend(extent_expressions(syntax));
            }
            while let Some(e) = pending.pop() {
                if e.resolved_enum().is_some() {
                    continue;
                }
                if e.resolved_nominal().is_some() {
                    continue;
                }
                match e.kind() {
                    eqiora_lang::ExprKind::Name(n) => {
                        dependencies.entry(n.clone()).or_insert(e.range());
                    }
                    eqiora_lang::ExprKind::Reduction { value, .. } => pending.push(value),
                    eqiora_lang::ExprKind::Case { value, arms } => {
                        pending.push(value.as_ref());
                        pending.extend(arms.iter().map(eqiora_lang::CaseArm::value));
                    }
                    eqiora_lang::ExprKind::Select {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        pending.extend([
                            condition.as_ref(),
                            then_value.as_ref(),
                            else_value.as_ref(),
                        ]);
                    }
                    eqiora_lang::ExprKind::Array(elements) => pending.extend(elements),
                    eqiora_lang::ExprKind::Slice {
                        value,
                        lower,
                        upper,
                    } => pending.extend([value.as_ref(), lower.as_ref(), upper.as_ref()]),
                    eqiora_lang::ExprKind::Index { value, index } => {
                        pending.extend([value.as_ref(), index.as_ref()])
                    }
                    eqiora_lang::ExprKind::Unary { value, .. }
                    | eqiora_lang::ExprKind::Partial { value, .. } => pending.push(value),
                    eqiora_lang::ExprKind::Binary { left, right, .. } => {
                        pending.push(right);
                        pending.push(left);
                    }
                    eqiora_lang::ExprKind::Call { callee, arguments }
                        if callee.as_str() == "tensor_value" =>
                    {
                        if let Some((_, components)) = super::tensor_values::arguments(arguments) {
                            pending.push(components);
                        } else {
                            pending.extend(arguments.expressions());
                        }
                    }
                    eqiora_lang::ExprKind::Call { arguments, .. } => pending.extend(
                        arguments
                            .expressions()
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev(),
                    ),
                    _ => {}
                }
            }
            (
                name.clone(),
                ExpressionDefinition {
                    expression: Some(d.value()),
                    dependencies,
                    valid: true,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let cycles = expression_cycles(&definitions);
    if !cycles.is_empty() {
        return Err(cycles
            .into_iter()
            .map(|cycle| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    cycle.range,
                    format!("let alias dependency cycle: {}", cycle.path.join(" -> ")),
                )
            })
            .collect());
    }
    Ok(expression_evaluation_order(&definitions, &BTreeSet::new())
        .into_iter()
        .map(|name| declarations[&name])
        .collect())
}

fn is_static_expression(expression: &eqiora_lang::Expr, values: &SymbolicParameterMap) -> bool {
    let mut pending = vec![expression];
    while let Some(e) = pending.pop() {
        if e.resolved_enum().is_some() {
            continue;
        }
        match e.kind() {
            eqiora_lang::ExprKind::Boolean(_)
            | eqiora_lang::ExprKind::Number(_)
            | eqiora_lang::ExprKind::Quantity { .. } => {}
            eqiora_lang::ExprKind::Name(n) if values.contains_key(n) => {}
            eqiora_lang::ExprKind::Path(p)
                if (crate::math::constant(p).is_some() || p.as_str() == "math.i") => {}
            eqiora_lang::ExprKind::Case { value, arms } => {
                pending.push(value.as_ref());
                pending.extend(arms.iter().map(eqiora_lang::CaseArm::value));
            }
            eqiora_lang::ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => {
                pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
            }
            eqiora_lang::ExprKind::Array(elements) => pending.extend(elements),
            eqiora_lang::ExprKind::Slice {
                value,
                lower,
                upper,
            } => pending.extend([value.as_ref(), lower.as_ref(), upper.as_ref()]),
            eqiora_lang::ExprKind::Index { value, index } => {
                pending.extend([value.as_ref(), index.as_ref()])
            }
            eqiora_lang::ExprKind::Unary { value, .. } => pending.push(value),
            eqiora_lang::ExprKind::Binary { left, right, .. } => {
                pending.push(left);
                pending.push(right);
            }
            eqiora_lang::ExprKind::Call { callee, .. } if callee.as_str() == "period" => {}
            eqiora_lang::ExprKind::Call { callee, arguments }
                if callee.as_str() == "tensor_value" =>
            {
                let Some((_, components)) = super::tensor_values::arguments(arguments) else {
                    return false;
                };
                pending.push(components);
            }
            eqiora_lang::ExprKind::Call { callee, arguments }
                if !matches!(
                    callee.as_str(),
                    "derivative" | "pre" | "next" | "coordinate" | "time"
                ) =>
            {
                pending.extend(arguments.expressions())
            }
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests;
use super::{extent_expressions, specialize_type, structural_extent};
