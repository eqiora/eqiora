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
            "tensor_product" => Self::Binary(FiniteBinaryOperation::TensorProduct),
            "permute_factors" => Self::Unary(FiniteUnaryOperation::PermuteFactors([0, 1])),
            "pair" => Self::Binary(FiniteBinaryOperation::Pair),
            _ => return None,
        })
    }

    pub(crate) fn source<'a>(
        name: &str,
        arguments: &'a [eqiora_lang::Expr],
    ) -> Result<(Self, &'a [eqiora_lang::Expr]), &'static str> {
        let operation = Self::named(name).ok_or("unknown finite operation")?;
        if name != "permute_factors" {
            return Ok((operation, arguments));
        }
        let [_, permutation] = arguments else {
            return Err("permute_factors requires a value and a two-factor permutation");
        };
        let eqiora_lang::ExprKind::Array(entries) = permutation.kind() else {
            return Err("factor permutation must be a literal array [0,1] or [1,0]");
        };
        let order = entries
            .iter()
            .map(|entry| match entry.kind() {
                eqiora_lang::ExprKind::Number(value) => {
                    value.to_i64().ok().and_then(|n| u8::try_from(n).ok())
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("factor indices must be exact nonnegative integers")?;
        let order: [u8; 2] = order
            .try_into()
            .map_err(|_| "factor permutation requires two indices")?;
        if !matches!(order, [0, 1] | [1, 0]) {
            return Err("factor permutation must contain each factor exactly once");
        }
        Ok((
            Self::Unary(FiniteUnaryOperation::PermuteFactors(order)),
            &arguments[..1],
        ))
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
