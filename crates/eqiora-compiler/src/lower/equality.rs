//! The one checked authored-equality boundary, shared by body checking and lowering.

use eqiora_lang::{Expr, ExprKind, UnaryOp};
use eqiora_schema::kernel::typing::{ExpressionType, TypeViolation};

/// Parentheses are already represented by expression ranges. Only a literal
/// zero, optionally beneath unary negation, is contextual; named or computed
/// values never gain a contextual type, even when a later substitution is zero.
pub(crate) fn is_contextual_zero(mut expression: &Expr) -> bool {
    loop {
        match expression.kind() {
            ExprKind::Number(value) => return value.is_zero(),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
            } => expression = value,
            _ => return false,
        }
    }
}

pub(crate) struct CheckedEquality<I> {
    pub(crate) left: ExpressionType<I>,
    pub(crate) right: ExpressionType<I>,
    pub(crate) equation_type: ExpressionType<I>,
}

/// Both operands must already have passed their ordinary expression checks.
/// Contextual literal zero inherits a mathematical type, never a new spatial
/// support. Explicit typed literals retain every part of their declared type.
pub(crate) fn check<I: Clone + Eq>(
    mut left: ExpressionType<I>,
    mut right: ExpressionType<I>,
    contextual_left_zero: bool,
    contextual_right_zero: bool,
) -> Result<CheckedEquality<I>, TypeViolation<I>> {
    if (contextual_left_zero
        && matches!(
            right.value_type.scalar_domain(),
            eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
        ))
        || (contextual_right_zero
            && matches!(
                left.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
            ))
    {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if contextual_left_zero && !contextual_right_zero {
        left = ExpressionType::new(right.value_type.clone(), None);
    }
    if contextual_right_zero && !contextual_left_zero {
        right = ExpressionType::new(left.value_type.clone(), None);
    }
    let equation_type = left.clone().equation(right.clone())?;
    Ok(CheckedEquality {
        left,
        right,
        equation_type,
    })
}

#[cfg(test)]
mod tests;
