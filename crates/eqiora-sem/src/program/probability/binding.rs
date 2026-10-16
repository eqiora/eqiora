//! Exact law contexts and explicit point realizations remain distinct from moments.
use super::*;

pub(in crate::program) fn validate_bindings(
    owner: RawId,
    expression: &ExprDag,
    nodes: &BTreeMap<RawId, KernelNode>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let bound = match nodes.get(&owner) {
        Some(KernelNode::Observable(observable)) => observable.reduction().distribution(),
        _ => None,
    };
    let mut pending = expression
        .roots()
        .iter()
        .map(|id| (*id, None))
        .collect::<Vec<_>>();
    let mut visited = BTreeSet::new();
    while let Some((id, point)) = pending.pop() {
        if !visited.insert((id.index(), point)) {
            continue;
        }
        if visited.len() > 1_000_000 {
            diagnostics.push(kernel_error(
                owner,
                "probability binding exceeds its scoped expression work budget",
            ));
            return;
        }
        let node = expression.node(id).expect("validated expression operand");
        if let ExprNode::Symbol(symbol) = node {
            let required = required_law(*symbol, nodes);
            let realized = required.is_some_and(|law| {
                point.is_some() && domain(law, nodes).ok().flatten().map(Id::erase) == point
            });
            if required.is_some() && bound != required && (bound.is_some() || !realized) {
                diagnostics.push(kernel_error(owner, "random-variable expressions require their exact probability-law context or an explicit point on that law's support; a sample or ordinary value is not an expectation"));
            }
        }
        if let ExprNode::Evaluate { value, at, .. } = node {
            let support = at
                .first()
                .and_then(|(selector, _)| match expression.node(*selector) {
                    Some(ExprNode::Symbol(SymbolRef::Coordinate { support, .. })) => {
                        Some(support.erase())
                    }
                    _ => None,
                });
            pending.push((*value, support));
            // Coordinate expressions retain the outer binding; only the value
            // receives the new complete point. Typing checks exact completeness.
            for (selector, mapped) in at {
                pending.push((*selector, point));
                pending.push((*mapped, point));
            }
        } else {
            node.try_for_each_operand(|operand| {
                pending.push((operand, point));
                Ok::<_, std::convert::Infallible>(())
            })
            .expect("infallible operand traversal");
        }
    }
}

fn required_law(
    symbol: SymbolRef,
    nodes: &BTreeMap<RawId, KernelNode>,
) -> Option<Id<kinds::Relation>> {
    match symbol {
        SymbolRef::Outcome(distribution) => Some(distribution),
        SymbolRef::Observable(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Observable(observable)) => match observable.reduction() {
                eqiora_schema::kernel::ObservableReduction::RandomVariable { distribution } => {
                    Some(distribution)
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}
