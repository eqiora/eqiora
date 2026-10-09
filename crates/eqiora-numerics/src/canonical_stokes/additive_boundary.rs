//! Additive admission views for steady-Stokes boundary roles.

use eqiora_core::{Diagnostic, RawId};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::additive_residual::AdditiveResidualView;

use super::support::{is_field, lowering_error, typed_relation};

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

pub(super) fn additive_prescribed_normal_velocity_parts(
    program: &KernelProgram,
    expression: &ExprDag,
    root: ExprId,
    velocity: RawId,
    relation: RawId,
) -> Result<Option<(ExprId, f64)>, Diagnostic> {
    let view = AdditiveResidualView::derive(expression, root, relation)?;
    let [first, second] = view.leaves() else {
        return Ok(None);
    };
    for (trace, normal) in [(first, second), (second, first)] {
        if !is_velocity_or_port_trace(expression, trace.value(), velocity) {
            continue;
        }
        let Some(ExprNode::NormalComponent { value: tensor, .. }) = expression.node(normal.value())
        else {
            continue;
        };
        let typed = typed_relation(program, relation)?;
        let Some(proof) =
            OperatorApplicationProof::classify(&typed, *tensor, StandardPureOperator::IsotropicLift)
                .map_err(|error| {
                    lowering_error(
                        relation,
                        format!(
                            "prescribed velocity isotropic-lift proof failed at expression node {}: {error}",
                            tensor.index()
                        ),
                    )
                })?
        else {
            continue;
        };
        let sign = if trace.sign() == normal.sign() {
            -1.0
        } else {
            1.0
        };
        return Ok(Some((proof.operand(), sign)));
    }
    Ok(None)
}

fn is_velocity_or_port_trace(expression: &ExprDag, value: ExprId, velocity: RawId) -> bool {
    match expression.node(value) {
        Some(ExprNode::Trace { value: field, .. }) => is_field(expression, *field, velocity),
        Some(ExprNode::Symbol(SymbolRef::PortTrace(_))) => true,
        _ => false,
    }
}
