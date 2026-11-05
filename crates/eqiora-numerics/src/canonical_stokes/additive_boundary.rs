//! Additive admission views for steady-Stokes boundary roles.

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

use crate::additive_residual::AdditiveResidualView;

use super::support::is_field;

pub(super) fn additive_prescribed_complete_velocity_parts(
    expression: &ExprDag,
    root: ExprId,
    velocity: RawId,
    owner: RawId,
) -> Result<Option<ExprId>, Diagnostic> {
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    let [first, second] = view.leaves() else {
        return Ok(None);
    };
    if !first.sign().is_opposite(second.sign()) {
        return Ok(None);
    }
    for (velocity_trace, potential_trace) in [(first, second), (second, first)] {
        let Some(ExprNode::Trace {
            value: velocity_value,
            ..
        }) = expression.node(velocity_trace.value())
        else {
            continue;
        };
        if !is_field(expression, *velocity_value, velocity) {
            continue;
        }
        let Some(ExprNode::Trace {
            value: gradient, ..
        }) = expression.node(potential_trace.value())
        else {
            continue;
        };
        let Some(ExprNode::Gradient(potential)) = expression.node(*gradient) else {
            continue;
        };
        if matches!(
            expression.node(*potential),
            Some(ExprNode::Symbol(SymbolRef::Field(_)))
        ) {
            return Ok(Some(*potential));
        }
    }
    Ok(None)
}

pub(super) fn additive_prescribed_velocity_parts(
    expression: &ExprDag,
    root: ExprId,
    velocity: RawId,
    relation: RawId,
) -> Result<Option<(ExprId, f64)>, Diagnostic> {
    // Preserve authored datum grouping and its arithmetic domain. Only the
    // additive path containing the trace may be projected into signed leaves.
    let view = AdditiveResidualView::derive_preserving(expression, root, relation, &|id| {
        !contains_velocity_or_port_trace(expression, id, velocity)
    })?;
    let [first, second] = view.leaves() else {
        return Ok(None);
    };
    for (trace, datum) in [(first, second), (second, first)] {
        if is_velocity_or_port_trace(expression, trace.value(), velocity)
            && !contains_velocity_or_port_trace(expression, datum.value(), velocity)
        {
            return Ok(Some((
                datum.value(),
                if trace.sign() == datum.sign() {
                    -1.
                } else {
                    1.
                },
            )));
        }
    }
    Ok(None)
}

fn contains_velocity_or_port_trace(expression: &ExprDag, root: ExprId, velocity: RawId) -> bool {
    let mut pending = vec![root];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.index()) {
            continue;
        }
        if is_velocity_or_port_trace(expression, id, velocity) {
            return true;
        }
        if let Some(node) = expression.node(id) {
            let _: Result<(), std::convert::Infallible> = node.try_for_each_operand(|operand| {
                pending.push(operand);
                Ok(())
            });
        }
    }
    false
}

fn is_velocity_or_port_trace(expression: &ExprDag, value: ExprId, velocity: RawId) -> bool {
    match expression.node(value) {
        Some(ExprNode::Trace { value: field, .. }) => is_field(expression, *field, velocity),
        Some(ExprNode::Symbol(SymbolRef::PortTrace(_))) => true,
        _ => false,
    }
}
