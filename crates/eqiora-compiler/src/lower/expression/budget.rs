//! Charge substituted expression DAGs before any Model expression DAG is emitted.
use super::*;

pub(in crate::lower) fn check_expanded_budget(
    file: &str,
    model: &LoweringModel,
    bindings: &BTreeMap<String, Binding>,
    limit: usize,
) -> Result<(), Diagnostic> {
    let mut remaining = limit;
    for item in &model.items {
        let mut roots = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut add = |values: Vec<LoweringExpression>| {
            charge_with_seen(values.iter(), &mut remaining, &mut seen).map_err(|()| {
                source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    file,
                    model.range,
                    format!("expanded expression DAG nodes exceed the configured {limit} limit"),
                )
            })?;
            // Retain pointer-key owners throughout this consumer.
            roots.extend(values);
            Ok::<_, Diagnostic>(())
        };
        match item {
            LoweringItem::Relation {
                body: LoweringRelationBody::Equations(equations),
                domain,
                ..
            } => {
                let support = domain
                    .as_deref()
                    .map(|name| relation_support(file, model.range, name, bindings))
                    .transpose()?;
                for equation in equations {
                    let (left, right) = contextual::equation(
                        file,
                        &equation.left,
                        &equation.right,
                        bindings,
                        support.as_ref(),
                    )?;
                    add(vec![left, right])?;
                }
            }
            LoweringItem::Relation { body, .. } => {
                for value in body.expressions() {
                    add(vec![contextual::value(file, value, bindings)?])?;
                }
            }
            LoweringItem::Observable {
                values, reduction, ..
            } => {
                for value in values.iter().chain(
                    reduction
                        .iter()
                        .filter_map(|value| value.limits())
                        .flatten(),
                ) {
                    add(vec![contextual::value(file, value, bindings)?])?;
                }
            }
            LoweringItem::Event { guard, .. } => {
                add(vec![contextual::value(file, guard, bindings)?])?
            }
            LoweringItem::RecordInstance { members, .. } => add(members.clone())?,
            _ => continue,
        }
    }
    Ok(())
}

#[cfg(test)]
fn charge<'a>(
    roots: impl Iterator<Item = &'a LoweringExpression>,
    remaining: &mut usize,
) -> Result<(), ()> {
    charge_with_seen(roots, remaining, &mut std::collections::HashSet::new())
}

fn charge_with_seen<'a>(
    roots: impl Iterator<Item = &'a LoweringExpression>,
    remaining: &mut usize,
    seen: &mut std::collections::HashSet<(usize, bool)>,
) -> Result<(), ()> {
    let mut pending = roots.map(|value| (value, false)).collect::<Vec<_>>();
    while let Some((expression, sampling)) = pending.pop() {
        let key = (Arc::as_ptr(&expression.node) as usize, sampling);
        if seen.contains(&key) {
            continue;
        }
        *remaining = remaining.checked_sub(1).ok_or(())?;
        seen.insert(key);
        let mut children = Vec::new();
        expression.node.push_children(&mut children);
        let sampling = sampling
            || matches!(
                expression.node.as_ref(),
                LoweringExpressionNode::Sample { .. }
            );
        pending.extend(children.into_iter().map(|value| (value, sampling)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_nodes_are_charged_once_per_consumer_and_sampling_context() {
        let range = TextRange::new(0, 0);
        let leaf = LoweringExpression::coordinate("body".into(), "body".into(), 0, range);
        let diamond = LoweringExpression::binary(BinaryOp::Add, leaf.clone(), leaf.clone(), range);
        let mut remaining = 2;
        charge([&diamond].into_iter(), &mut remaining).unwrap();
        assert_eq!(remaining, 0);
        assert!(charge([&diamond].into_iter(), &mut remaining).is_err());
        let sampled = LoweringExpression::sample(leaf.clone(), "tick".into(), range);
        let mut remaining = 3;
        charge([&leaf, &sampled].into_iter(), &mut remaining).unwrap();
        assert_eq!(remaining, 0);
        assert!(charge([&leaf, &sampled].into_iter(), &mut 2).is_err());
    }
    #[test]
    fn separate_equation_normalization_is_charged_without_false_arc_sharing() {
        let range = TextRange::new(0, 0);
        let leaf =
            LoweringExpression::number(eqiora_lang::DecimalLiteral::parse("1").unwrap(), range);
        let shared = LoweringExpression::binary(BinaryOp::Add, leaf.clone(), leaf, range);
        let left = LoweringExpression::binary(BinaryOp::Add, shared.clone(), shared, range);
        let right =
            LoweringExpression::number(eqiora_lang::DecimalLiteral::parse("0").unwrap(), range);
        let bindings = BTreeMap::new();
        let mut roots = Vec::new();
        for _ in 0..2 {
            let (left, right) =
                contextual::equation("equations.eqi", &left, &right, &bindings, None).unwrap();
            roots.extend([left, right]);
        }
        // Each normalization owns two Add nodes and two contextual literals.
        // Repeated operands share within each equation, not between resolvers.
        let mut remaining = 8;
        charge(roots.iter(), &mut remaining).unwrap();
        assert_eq!(remaining, 0);
        assert!(charge(roots.iter(), &mut 7).is_err());
    }
}
