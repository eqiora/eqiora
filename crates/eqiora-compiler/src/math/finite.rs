//! Source names for the shared nominal finite algebra.
use eqiora_core::Diagnostic;
use eqiora_schema::kernel::{
    ExprDagBuilder, ExprId, FiniteBinaryOperation, FiniteUnaryOperation,
    typing::{ExpressionType, TypeViolation},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Operation {
    Unary(FiniteUnaryOperation),
    Binary(FiniteBinaryOperation),
}

impl Operation {
    pub(crate) fn named(name: &str) -> Option<Self> {
        Some(match name {
            "transpose" => Self::Unary(FiniteUnaryOperation::Transpose),
            "adjoint" => Self::Unary(FiniteUnaryOperation::Adjoint),
            "apply" => Self::Binary(FiniteBinaryOperation::Apply),
            "compose" => Self::Binary(FiniteBinaryOperation::Compose),
            "pair" => Self::Binary(FiniteBinaryOperation::Pair),
            _ => return None,
        })
    }

    pub(crate) fn arity(self) -> usize {
        match self {
            Self::Unary(_) => 1,
            Self::Binary(_) => 2,
        }
    }

    pub(crate) fn result_type<I: Clone + Eq>(
        self,
        operands: &[ExpressionType<I>],
    ) -> Result<ExpressionType<I>, TypeViolation<I>> {
        if operands.len() != self.arity() {
            return Err(TypeViolation::FiniteBasisMismatch);
        }
        match (self, operands) {
            (Self::Unary(op), [value]) => value.clone().finite_unary(op),
            (Self::Binary(op), [left, right]) => left.clone().finite_binary(op, right.clone()),
            _ => Err(TypeViolation::FiniteBasisMismatch),
        }
    }

    pub(crate) fn emit(
        self,
        builder: &mut ExprDagBuilder,
        operands: &[ExprId],
    ) -> Result<ExprId, Diagnostic> {
        match (self, operands) {
            (Self::Unary(op), [value]) => builder.finite_unary(op, *value),
            (Self::Binary(op), [left, right]) => builder.finite_binary(op, *left, *right),
            _ => Err(Diagnostic::error(
                eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                "finite operation has incorrect arity",
            )),
        }
    }
}
