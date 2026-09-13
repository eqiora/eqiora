//! Shared source meaning of an explicit nonnegative complementarity operand.
use super::{LoweringExpression, LoweringExpressionNode};
use eqiora_lang::{BinaryOp, Expr, ExprKind, UnaryOp};

pub(crate) fn source_operand(predicate: &Expr) -> Result<&Expr, &'static str> {
    match predicate.kind() {
        ExprKind::Binary {
            op: BinaryOp::LessEqual,
            left,
            right,
        } if source_zero(left) => Ok(right),
        ExprKind::Binary {
            op: BinaryOp::GreaterEqual,
            left,
            right,
        } if source_zero(right) => Ok(left),
        _ => Err("complementarity requires two explicit nonnegativity predicates: 0 <= operand"),
    }
}
fn source_zero(value: &Expr) -> bool {
    match value.kind() {
        ExprKind::Number(value) | ExprKind::Quantity { value, .. } => value.is_zero(),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => source_zero(value),
        _ => false,
    }
}

pub(super) fn lowered_operand(
    predicate: &LoweringExpression,
) -> Result<LoweringExpression, &'static str> {
    let operand = match predicate.node.as_ref() {
        LoweringExpressionNode::Binary {
            operator: BinaryOp::LessEqual,
            left,
            right,
        } if lowered_zero(left) => right,
        LoweringExpressionNode::Binary {
            operator: BinaryOp::GreaterEqual,
            left,
            right,
        } if lowered_zero(right) => left,
        _ => {
            return Err(
                "complementarity requires two explicit nonnegativity predicates: 0 <= operand",
            );
        }
    };
    Ok(operand
        .clone()
        .with_structural_parameters(predicate.structural_parameters()))
}
fn lowered_zero(value: &LoweringExpression) -> bool {
    match value.node.as_ref() {
        LoweringExpressionNode::Number(value) => value.is_zero(),
        LoweringExpressionNode::Literal(value) => value.is_zero(),
        LoweringExpressionNode::Neg(value) => lowered_zero(value),
        _ => false,
    }
}
