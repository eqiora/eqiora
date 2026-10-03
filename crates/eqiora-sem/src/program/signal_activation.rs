//! Exact signal activation and explicit transition admission after graph replay.
use super::*;
use eqiora_schema::kernel::{ClockKind, ExprId, FieldRole};

pub(super) fn validate(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&id, node) in nodes {
        if let KernelNode::Port(port) = node
            && port.signal_contract().is_some()
        {
            let clocks = edge_targets(edges, id, EdgeKind::ClockedBy);
            if clocks.len() > 1 || clocks.iter().any(|clock| !matches!(nodes.get(clock), Some(KernelNode::ClockDomain(c)) if matches!(c.kind(), ClockKind::Periodic { .. }))) {
                diagnostics.push(kernel_error(id, "signal Port activation requires at most one exact periodic ClockDomain"));
            }
            let supports = edge_targets(edges, id, EdgeKind::DefinedOn);
            if supports.len() > 1
                || supports
                    .iter()
                    .any(|support| !matches!(nodes.get(support), Some(KernelNode::Domain(_))))
            {
                diagnostics.push(kernel_error(
                    id,
                    "signal Port requires at most one exact spatial Domain",
                ));
            }
        }
        let KernelNode::Relation(relation) = node else {
            continue;
        };
        let activation = edges
            .iter()
            .find(|e| e.kind() == EdgeKind::Activates && e.to() == id)
            .map(Edge::from);
        let clock =
            activation.and_then(|a| edge_targets(edges, a, EdgeKind::ClockedBy).first().copied());
        let mut pending = relation
            .expression()
            .roots()
            .iter()
            .map(|root| (*root, false))
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        while let Some((index, sampling)) = pending.pop() {
            if !seen.insert((index, sampling)) {
                continue;
            }
            let node = &relation.expression().nodes()[index.index() as usize];
            match node {
                ExprNode::Sample {
                    value,
                    clock: supplied,
                } => {
                    if relation.is_initial()
                        || clock != Some(supplied.erase())
                        || sampling
                        || !matches!(nodes.get(&supplied.erase()), Some(KernelNode::ClockDomain(c)) if matches!(c.kind(), ClockKind::Periodic { .. }))
                    {
                        diagnostics.push(kernel_error(
                            id,
                            "Sample requires the exact active periodic ClockDomain",
                        ));
                    }
                    pending.push((*value, true));
                }
                ExprNode::Hold(value) => {
                    let Some(ExprNode::Symbol(SymbolRef::Field(field))) =
                        relation.expression().nodes().get(value.index() as usize)
                    else {
                        diagnostics
                            .push(kernel_error(id, "Hold requires one direct clocked State"));
                        continue;
                    };
                    let field_id = field.erase();
                    if !matches!(nodes.get(&field_id), Some(KernelNode::Field(f)) if f.role() == FieldRole::State)
                        || edge_targets(edges, field_id, EdgeKind::ClockedBy).len() != 1
                    {
                        diagnostics
                            .push(kernel_error(id, "Hold requires one direct clocked State"));
                    }
                    let has_update = nodes.values().any(|node| matches!(node, KernelNode::Relation(r) if !r.is_initial() && r.expression().nodes().iter().any(|n| matches!(n, ExprNode::Symbol(SymbolRef::Next(f)) if f == field))));
                    if !has_update {
                        diagnostics.push(kernel_error(
                            id,
                            "Hold memory requires an explicit periodic update",
                        ));
                    }
                    // Fresh initialization owns whether all memory values are determined.
                    // Hold erases the dependency's update activation, not its state identity.
                }
                ExprNode::Symbol(SymbolRef::Port(port)) => {
                    let port_clock = edge_targets(edges, port.erase(), EdgeKind::ClockedBy)
                        .first()
                        .copied();
                    if (sampling && port_clock.is_some())
                        || (!sampling && port_clock != clock)
                        || (relation.is_initial() && port_clock.is_some())
                    {
                        diagnostics.push(kernel_error(
                            id,
                            "signal expression crosses activation without explicit Sample or Hold",
                        ));
                    }
                }
                ExprNode::Symbol(SymbolRef::Field(field))
                    if matches!(nodes.get(&field.erase()), Some(KernelNode::Field(definition)) if definition.role() == FieldRole::Variable)
                        && !edge_targets(edges, field.erase(), EdgeKind::ClockedBy).is_empty() =>
                {
                    let field_clocks = edge_targets(edges, field.erase(), EdgeKind::ClockedBy);
                    if sampling || relation.is_initial() || field_clocks.first().copied() != clock {
                        diagnostics.push(kernel_error(id, "clocked Variable requires its exact periodic activation and has no initial or continuous value"));
                    }
                }
                ExprNode::Symbol(
                    SymbolRef::Field(field) | SymbolRef::Pre(field) | SymbolRef::Next(field),
                ) if sampling => {
                    if !edge_targets(edges, field.erase(), EdgeKind::ClockedBy).is_empty()
                        || !matches!(node, ExprNode::Symbol(SymbolRef::Field(_)))
                    {
                        diagnostics.push(kernel_error(id, "Sample operand must be continuous"));
                    }
                }
                _ => pending.extend(operands(node).into_iter().map(|value| (value, sampling))),
            }
        }
    }
}

pub(crate) fn operands(node: &ExprNode) -> Vec<ExprId> {
    match node {
        ExprNode::Select {
            condition,
            then_value,
            else_value,
        } => vec![*condition, *then_value, *else_value],
        ExprNode::Require { condition, value } => vec![*condition, *value],
        ExprNode::Array { elements } => elements.clone(),
        ExprNode::Sample { value, .. }
        | ExprNode::Hold(value)
        | ExprNode::Index { value, .. }
        | ExprNode::Not(value)
        | ExprNode::ToReal(value)
        | ExprNode::ToInteger(value)
        | ExprNode::Ordinal(value)
        | ExprNode::Neg(value)
        | ExprNode::PowI(value, _)
        | ExprNode::UnaryMath(_, value)
        | ExprNode::FiniteUnary(_, value)
        | ExprNode::Gradient(value)
        | ExprNode::Divergence(value)
        | ExprNode::SymmetricPart(value)
        | ExprNode::IsotropicLift(value)
        | ExprNode::Trace(value)
        | ExprNode::NormalComponent(value) => vec![*value],
        ExprNode::Complex { real: a, imag: b }
        | ExprNode::FiniteBinary(_, a, b)
        | ExprNode::Compare(_, a, b)
        | ExprNode::And(a, b)
        | ExprNode::Or(a, b)
        | ExprNode::Quotient(a, b)
        | ExprNode::Remainder(a, b)
        | ExprNode::Add(a, b)
        | ExprNode::Sub(a, b)
        | ExprNode::Mul(a, b)
        | ExprNode::Div(a, b) => vec![*a, *b],
        ExprNode::PureOperatorApplication(application) => application.arguments().to_vec(),
        _ => Vec::new(),
    }
}
