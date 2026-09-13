//! Relation dependency, support, activation and state ownership admission.

use super::*;

pub(super) fn validate_relations(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    spatial_supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&id, node) in nodes {
        let KernelNode::Relation(relation) = node else {
            continue;
        };

        let scopes = edge_targets(edges, id, EdgeKind::AppliesOn);
        if relation.is_initial() && !scopes.is_empty() {
            diagnostics.push(kernel_error(
                id,
                "initial Relation roots own their support and must not have AppliesOn edges",
            ));
        }
        if scopes.len() > 1 {
            diagnostics.push(kernel_error(
                id,
                format!(
                    "Relation may apply on at most one Domain, found {}",
                    scopes.len()
                ),
            ));
        }
        let scope = (scopes.len() == 1).then(|| *scopes.first().expect("one scope was checked"));
        let symbols = validate_expression(
            relation.expression(),
            id,
            TypingEnvironment {
                nodes,
                edges,
                spatial_supports,
            },
            scope,
            if relation.has_constraints() {
                RootContract::RelationOperands
            } else if relation.is_initial() {
                RootContract::InitialConditions
            } else {
                RootContract::EquationSides
            },
            diagnostics,
        );
        let dependencies = edge_targets(edges, id, EdgeKind::DependsOn);
        if symbols != dependencies {
            diagnostics.push(kernel_error(
                id,
                format!(
                    "Relation symbol set {symbols:?} differs from DependsOn targets {dependencies:?}"
                ),
            ));
        }

        let activations = edges
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Activates && edge.to() == id)
            .map(Edge::from)
            .collect::<Vec<_>>();
        if relation.is_initial() {
            if !activations.is_empty() {
                diagnostics.push(kernel_error(
                    id,
                    "initial Relation must not have an Activation",
                ));
            }
            if relation
                .expression()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Next(_))))
            {
                diagnostics.push(kernel_error(
                    id,
                    "initial Relation cannot read Next symbols",
                ));
            }
            for node in relation.expression().nodes() {
                if let ExprNode::Symbol(SymbolRef::Pre(field)) = node
                    && edge_targets(edges, field.erase(), EdgeKind::ClockedBy).len() != 1
                {
                    diagnostics.push(kernel_error(id, "initial Pre requires a clocked state"));
                }
            }
            continue;
        }
        if activations.len() == 1 {
            let relation_clocks = edge_targets(edges, activations[0], EdgeKind::ClockedBy);
            let event_reset = matches!(nodes.get(&activations[0]), Some(KernelNode::Activation(activation)) if matches!(activation.kind(), ActivationKind::Event { .. }));
            for node in relation.expression().nodes() {
                if let ExprNode::Symbol(SymbolRef::Pre(field) | SymbolRef::Next(field)) = node {
                    let clocks = edge_targets(edges, field.erase(), EdgeKind::ClockedBy);
                    if !(event_reset && clocks.is_empty())
                        && (clocks.len() != 1 || clocks != relation_clocks)
                    {
                        diagnostics.push(kernel_error(
                            id,
                            "Pre/Next state must own the exact Relation ClockDomain",
                        ));
                    }
                }
            }
        }
        if activations.len() != 1 {
            diagnostics.push(kernel_error(
                id,
                format!(
                    "Relation requires exactly one Activation, found {}",
                    activations.len()
                ),
            ));
        } else if matches!(
            nodes.get(&activations[0]),
            Some(KernelNode::Activation(activation))
                if matches!(activation.kind(), ActivationKind::Continuous)
        ) && relation.expression().nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Symbol(SymbolRef::Pre(_) | SymbolRef::Next(_))
            )
        }) {
            diagnostics.push(kernel_error(
                id,
                "continuous Relation cannot read Pre or Next symbols",
            ));
        } else if matches!(nodes.get(&activations[0]), Some(KernelNode::Activation(activation)) if matches!(activation.kind(), ActivationKind::Periodic))
            && relation
                .expression()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Derivative(_))))
        {
            diagnostics.push(kernel_error(
                id,
                "clocked Relation cannot read Derivative symbols",
            ));
        }
    }
}
