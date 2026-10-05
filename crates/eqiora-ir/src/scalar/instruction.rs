//! Scalar SSA vocabulary shared by typed and numerical execution.
use super::SymbolSlot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ValueId(pub(super) u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Instruction {
    Constant(eqiora_core::DynQuantity),
    TypedConstant(u32),
    Array {
        start: u32,
        len: u32,
    },
    PureOperator {
        definition: u32,
        start: u32,
        len: u32,
    },
    Index(ValueId, u32),
    Quotient(ValueId, ValueId),
    Remainder(ValueId, ValueId),
    ToReal(ValueId),
    ToInteger(ValueId),
    Ordinal(ValueId),
    Compare(eqiora_schema::kernel::ComparisonOp, ValueId, ValueId),
    Select {
        condition: ValueId,
        then_value: ValueId,
        else_value: ValueId,
    },
    Require {
        condition: ValueId,
        value: ValueId,
    },
    Sin(ValueId),
    Exp(ValueId),
    Sqrt(ValueId),
    Not(ValueId),
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Read(SymbolSlot),
    Neg(ValueId),
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    ComplexDiv {
        operands: [ValueId; 4],
        imaginary: bool,
    },
    PowI(ValueId, i32),
    // Contiguous row-major numerical operands; None selects the determinant.
    MapInvariant {
        start: ValueId,
        extent: u32,
        component: Option<u32>,
    },
}

impl Instruction {
    pub(super) fn map_scalar_operands(
        self,
        mut at: impl FnMut(ValueId) -> ValueId,
    ) -> Option<Self> {
        Some(match self {
            Self::Constant(_) | Self::TypedConstant(_) | Self::Read(_) => self,
            Self::Neg(a) => Self::Neg(at(a)),
            Self::Sin(a) => Self::Sin(at(a)),
            Self::Exp(a) => Self::Exp(at(a)),
            Self::Sqrt(a) => Self::Sqrt(at(a)),
            Self::Not(a) => Self::Not(at(a)),
            Self::ToReal(a) => Self::ToReal(at(a)),
            Self::ToInteger(a) => Self::ToInteger(at(a)),
            Self::Ordinal(a) => Self::Ordinal(at(a)),
            Self::Index(a, n) => Self::Index(at(a), n),
            Self::PowI(a, n) => Self::PowI(at(a), n),
            Self::Add(a, b) => Self::Add(at(a), at(b)),
            Self::Sub(a, b) => Self::Sub(at(a), at(b)),
            Self::Mul(a, b) => Self::Mul(at(a), at(b)),
            Self::Div(a, b) => Self::Div(at(a), at(b)),
            Self::ComplexDiv {
                operands,
                imaginary,
            } => Self::ComplexDiv {
                operands: operands.map(at),
                imaginary,
            },
            Self::MapInvariant { .. } => return None,
            Self::Quotient(a, b) => Self::Quotient(at(a), at(b)),
            Self::Remainder(a, b) => Self::Remainder(at(a), at(b)),
            Self::And(a, b) => Self::And(at(a), at(b)),
            Self::Or(a, b) => Self::Or(at(a), at(b)),
            Self::Compare(op, a, b) => Self::Compare(op, at(a), at(b)),
            Self::Select {
                condition,
                then_value,
                else_value,
            } => Self::Select {
                condition: at(condition),
                then_value: at(then_value),
                else_value: at(else_value),
            },
            Self::Require { condition, value } => Self::Require {
                condition: at(condition),
                value: at(value),
            },
            Self::Array { .. } | Self::PureOperator { .. } => return None,
        })
    }
}
