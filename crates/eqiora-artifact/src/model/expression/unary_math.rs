//! Closed unary-function vocabulary for the current Model wire contract.
use crate::invalid_artifact;
use eqiora_core::Diagnostic;
use eqiora_schema::kernel::UnaryMathFunction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireUnaryMath {
    Sin,
    Sqrt,
    Cos,
    Exp,
    Log,
    Conj,
    Real,
    Imag,
    Abs,
    Abs2,
    Arg,
}

impl WireUnaryMath {
    pub(crate) fn encode(value: UnaryMathFunction) -> Result<Self, Diagnostic> {
        match value {
            UnaryMathFunction::Sin => Ok(Self::Sin),
            UnaryMathFunction::Sqrt => Ok(Self::Sqrt),
            UnaryMathFunction::Cos => Ok(Self::Cos),
            UnaryMathFunction::Exp => Ok(Self::Exp),
            UnaryMathFunction::Log => Ok(Self::Log),
            UnaryMathFunction::Conj => Ok(Self::Conj),
            UnaryMathFunction::Real => Ok(Self::Real),
            UnaryMathFunction::Imag => Ok(Self::Imag),
            UnaryMathFunction::Abs => Ok(Self::Abs),
            UnaryMathFunction::Abs2 => Ok(Self::Abs2),
            UnaryMathFunction::Arg => Ok(Self::Arg),
            _ => Err(invalid_artifact(
                "unary math function is unsupported by the current Model contract",
            )),
        }
    }

    pub(crate) const fn decode(self) -> UnaryMathFunction {
        match self {
            Self::Sin => UnaryMathFunction::Sin,
            Self::Sqrt => UnaryMathFunction::Sqrt,
            Self::Cos => UnaryMathFunction::Cos,
            Self::Exp => UnaryMathFunction::Exp,
            Self::Log => UnaryMathFunction::Log,
            Self::Conj => UnaryMathFunction::Conj,
            Self::Real => UnaryMathFunction::Real,
            Self::Imag => UnaryMathFunction::Imag,
            Self::Abs => UnaryMathFunction::Abs,
            Self::Abs2 => UnaryMathFunction::Abs2,
            Self::Arg => UnaryMathFunction::Arg,
        }
    }
}
