//! Follow realized values through the existing expression and connection graph.
use super::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Dependency {
    Value(SymbolRef),
    Condition(RawId, usize),
}

/// Conditions have separate dependency hubs, keeping construction linear in
/// their operands. Across/through and trace/flux quantities stay distinct, and
/// an unrelated condition in the same Relation cannot transfer provenance.
pub(in crate::program) fn validate_sample_dependencies(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    sampled_fields: impl Iterator<Item = RawId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut sampled = sampled_fields
        .map(|id| Dependency::Value(SymbolRef::Field(id.downcast().expect("State"))))
        .collect::<HashSet<_>>();
    let mut consumers = HashMap::<_, Vec<_>>::new();
    for (&id, node) in nodes {
        match node {
            KernelNode::Observable(observable) => {
                for symbol in dependencies(observable.expression(), observable.expression().roots())
                {
                    consumers
                        .entry(Dependency::Value(symbol))
                        .or_default()
                        .push(Dependency::Value(SymbolRef::Observable(observable.id())));
                }
            }
            KernelNode::Relation(relation) => {
                let mut add_condition = |index, roots: &[eqiora_schema::kernel::ExprId]| {
                    let condition = Dependency::Condition(id, index);
                    for activation in edges.iter().filter_map(|edge| {
                        (edge.kind() == EdgeKind::Activates && edge.to() == id)
                            .then_some(edge.from())
                    }) {
                        consumers
                            .entry(Dependency::Condition(activation, 0))
                            .or_default()
                            .push(condition);
                    }
                    for symbol in dependencies(relation.expression(), roots) {
                        let value = Dependency::Value(symbol);
                        consumers.entry(value).or_default().push(condition);
                        if !matches!(symbol, SymbolRef::Observable(_)) {
                            consumers.entry(condition).or_default().push(value);
                        }
                    }
                };
                match relation.meaning() {
                    RelationMeaning::Conditions(_) => {
                        for (index, roots) in relation
                            .expression()
                            .roots()
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .enumerate()
                        {
                            add_condition(index, roots);
                        }
                    }
                    RelationMeaning::Conservation(_) => {
                        add_condition(0, relation.expression().roots())
                    }
                    _ => {} // Probability/process laws are not implicit equations.
                }
            }
            KernelNode::Activation(activation) => {
                let guard = match activation.kind() {
                    ActivationKind::Event { guard, .. } | ActivationKind::Guard { guard } => guard,
                    _ => continue,
                };
                for symbol in dependencies(guard, guard.roots()) {
                    consumers
                        .entry(Dependency::Value(symbol))
                        .or_default()
                        .push(Dependency::Condition(id, 0));
                }
            }
            KernelNode::Connection(connection) => {
                let ports = edge_targets(edges, id, EdgeKind::Connects);
                match connection.semantics() {
                    ConnectionSemantics::Signal { driver } => {
                        for port in ports {
                            let Some(port) = port.downcast() else {
                                continue;
                            };
                            consumers
                                .entry(Dependency::Value(SymbolRef::Port(driver)))
                                .or_default()
                                .push(Dependency::Value(SymbolRef::Port(port)));
                        }
                    }
                    ConnectionSemantics::Conserving | ConnectionSemantics::SpatialPeriodic => {
                        for port in ports {
                            let Some(port) = port.downcast() else {
                                continue;
                            };
                            // Equality and signed conservation both propagate value
                            // dependence within their own quantity, never across it.
                            for (index, symbol) in [
                                SymbolRef::Across(port),
                                SymbolRef::Through(port),
                                SymbolRef::PortTrace(port),
                                SymbolRef::PortFlux(port),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                let condition = Dependency::Condition(id, index);
                                let value = Dependency::Value(symbol);
                                consumers.entry(value).or_default().push(condition);
                                consumers.entry(condition).or_default().push(value);
                            }
                        }
                    }
                    _ => {} // Unsupported connection semantics already fail admission.
                }
            }
            _ => {}
        }
    }
    let mut pending = sampled.iter().copied().collect::<Vec<_>>();
    while let Some(dependency) = pending.pop() {
        for &consumer in consumers.get(&dependency).into_iter().flatten() {
            if sampled.insert(consumer) {
                pending.push(consumer);
            }
        }
    }
    // Model ordering, rather than hash traversal order, owns diagnostic order.
    for (&id, node) in nodes {
        if let KernelNode::Observable(observable) = node
            && observable.reduction().distribution().is_some()
            && sampled.contains(&Dependency::Value(SymbolRef::Observable(observable.id())))
        {
            diagnostics.push(kernel_error(id,
                "a realized stochastic State cannot define an exact probability moment; sample estimators require explicit ensemble provenance"));
        }
    }
}

fn dependencies(
    expression: &ExprDag,
    roots: &[eqiora_schema::kernel::ExprId],
) -> HashSet<SymbolRef> {
    let mut dependencies = HashSet::new();
    let mut pending = roots.to_vec();
    let mut visited = HashSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = expression.node(id).expect("admitted expression operand");
        match node {
            ExprNode::Symbol(
                SymbolRef::Field(id)
                | SymbolRef::Pre(id)
                | SymbolRef::Next(id)
                | SymbolRef::Derivative(id, _),
            ) => {
                dependencies.insert(SymbolRef::Field(*id));
            }
            ExprNode::Symbol(
                symbol @ (SymbolRef::Observable(_)
                | SymbolRef::Port(_)
                | SymbolRef::Across(_)
                | SymbolRef::Through(_)
                | SymbolRef::PortTrace(_)
                | SymbolRef::PortFlux(_)),
            ) => {
                dependencies.insert(*symbol);
            }
            _ => {}
        }
        node.try_for_each_operand(|operand| {
            pending.push(operand);
            Ok::<_, std::convert::Infallible>(())
        })
        .expect("infallible operand traversal");
    }
    dependencies
}
