//! Fixed-factor integral partials reuse ordinary polynomial differentiation and reduction.
use super::*;
use std::borrow::Cow;

pub(in crate::lower) fn expand<'a>(
    file: &str,
    model: &'a LoweringModel,
    bindings: &mut BTreeMap<String, Binding>,
) -> Result<Cow<'a, LoweringModel>, Vec<Diagnostic>> {
    let declarations = model
        .items
        .iter()
        .filter_map(|item| match item {
            LoweringItem::Observable {
                name,
                value,
                reduction,
                domain,
                ..
            } => Some((name.as_str(), (value, reduction, domain))),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut replacements = BTreeMap::new();
    for (name, (value, reduction, output)) in &declarations {
        if reduction.is_some() {
            continue;
        }
        let LoweringExpressionNode::Partial {
            value: operand,
            wrt,
        } = value.node.as_ref()
        else {
            continue;
        };
        let LoweringExpressionNode::Name(source) = operand.node.as_ref() else {
            continue;
        };
        let Some((density, Some(measure), original_output)) = declarations.get(source.as_str())
        else {
            continue;
        };
        let invalid = |message| {
            vec![source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                value.range,
                message,
            )]
        };
        if output != original_output {
            return Err(invalid(
                if original_output.is_none()
                    && matches!(wrt.node.as_ref(), LoweringExpressionNode::Coordinate { .. })
                {
                    "integrated coordinates are bound, not independent output coordinates"
                } else {
                    "fixed integral partial must retain the original output support"
                },
            ));
        }
        if measure.limits.is_some() {
            return Err(invalid(
                "explicit-limit integral partials require admitted Leibniz endpoint contributions",
            ));
        }
        if measure.measure.is_some() {
            return Err(invalid(
                "fixed integral partial requires the admitted Cartesian factor measure",
            ));
        }
        require_fixed_factors(file, value.range, &measure.domain, model)
            .map_err(|error| vec![error])?;
        let measure_support = relation_support(file, value.range, &measure.domain, bindings)
            .map_err(|error| vec![error])?;
        let density = contextual::typed_value(
            file,
            density,
            bindings,
            Some(&measure_support),
            eqiora_core::ScalarDomain::Real,
        )
        .map_err(|error| vec![error])?;
        // A spatial Field declaration does not establish the regularity needed
        // to interchange its derivative and an integral.
        if density.referenced_names().iter().any(|name| {
            matches!(bindings.get(name), Some(Binding::Field(_, contract)) if contract.domain.is_some())
        }) {
            return Err(invalid(
                "fixed integral partial of a spatial Field requires admitted differentiation-under-integral regularity",
            ));
        }
        let inferred = expression_type(file, &density, bindings, Some(&measure_support))
            .map_err(|error| vec![error])?;
        let input = inferred.support.unwrap_or(measure_support);
        let selected = match wrt.node.as_ref() {
            LoweringExpressionNode::Coordinate { factor, axis, .. } => {
                let Some(output) = output.as_ref() else {
                    return Err(invalid(
                        "integrated coordinates are bound, not independent output coordinates",
                    ));
                };
                let output = expression::relation_support(file, value.range, output, bindings)
                    .map_err(|error| vec![error])?;
                let Some(Binding::Domain(factor_id, _)) = bindings.get(factor) else {
                    return Err(invalid("partial coordinate factor is unavailable"));
                };
                eqiora_schema::kernel::typing::ExpressionType::coordinate(
                    &factor_id.erase(),
                    *axis,
                    Some(&output),
                )
                .map_err(|_| {
                    invalid("partial coordinate is not an exact remaining integral factor")
                })?;
                let input_name = bindings
                    .iter()
                    .find_map(|(name, binding)| match binding {
                        Binding::Domain(id, _) if id.erase() == *input.domain() => {
                            Some(name.clone())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| invalid("integral input Domain is unavailable"))?;
                LoweringExpression::coordinate(input_name, factor.clone(), *axis, wrt.range)
            }
            LoweringExpressionNode::Name(name)
                if matches!(bindings.get(name), Some(Binding::Parameter(..))) =>
            {
                wrt.clone()
            }
            _ => {
                return Err(invalid(
                    "fixed integral partial requires an independent Parameter or remaining coordinate",
                ));
            }
        };
        // The existing partial lowerer admits the polynomial profile independently; it
        // rejects unsupported operations rather than treating them as constants.
        replacements.insert(
            (*name).to_owned(),
            (
                LoweringExpression::partial(density, selected, value.range),
                measure.clone(),
                (*source).clone(),
            ),
        );
    }
    if replacements.is_empty() {
        return Ok(Cow::Borrowed(model));
    }
    let mut expanded = model.clone();
    for item in &mut expanded.items {
        if let LoweringItem::Observable {
            name,
            value,
            reduction,
            ..
        } = item
            && let Some((derived, measure, source)) = replacements.remove(name)
        {
            if let Some(dependencies) = model.structural_dependencies.get(&source) {
                expanded
                    .structural_dependencies
                    .entry(name.clone())
                    .or_default()
                    .extend(dependencies.iter().cloned());
            }
            *value = derived;
            *reduction = Some(measure.clone());
            if let Some(Binding::Observable(_, _, _, bound_measure)) = bindings.get_mut(name) {
                *bound_measure = Some(measure);
            }
        }
    }
    Ok(Cow::Owned(expanded))
}

fn require_fixed_factors(
    file: &str,
    range: TextRange,
    measure: &str,
    model: &LoweringModel,
) -> Result<(), Diagnostic> {
    let domains = model
        .items
        .iter()
        .filter_map(|item| match item {
            LoweringItem::Domain { name, contract, .. } => Some((name.as_str(), contract)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut pending = vec![measure];
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "integral partial requires distinct fixed measure factors",
            ));
        }
        match domains.get(name) {
            Some(LoweringDomainContract::CoordinateInterval(_)) => {}
            Some(LoweringDomainContract::Source(eqiora_lang::DomainSyntax::Product {
                factors,
            })) => pending.extend(factors.iter().map(String::as_str)),
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "integral partial requires fixed bounded coordinate factors; moving bounds and shape derivatives are unsupported",
                ));
            }
        }
    }
    Ok(())
}
