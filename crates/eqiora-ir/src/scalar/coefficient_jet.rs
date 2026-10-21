//! Function and solve coefficients keep exponents until the requested output.
//!
//! Coefficient k multiplies the product of the nilpotent directions in k's
//! bitmask. Solves use B*Q=A; analytic functions compose their Taylor coefficients;
//! repeated directions therefore retain the ordinary mixed-derivative factors.
use super::*;
mod affine;
mod complex_math;
mod typed;
pub(super) use affine::evaluate as affine;
pub(super) use typed::{dimension, evaluate as typed};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    Solve {
        extent: u32,
        component: u32,
    },
    ComplexMath {
        function: eqiora_schema::kernel::UnaryMathFunction,
        imaginary: bool,
    },
}

impl Shape {
    pub(super) const SCALAR: Self = Self::Solve {
        extent: 1,
        component: 0,
    };

    pub(super) fn width(self) -> usize {
        match self {
            Self::Solve { extent, .. } => (extent as usize).pow(2),
            Self::ComplexMath { .. } => 1,
        }
    }

    pub(super) fn admit(self, length: usize) -> Result<usize, Diagnostic> {
        match self {
            Self::Solve { extent, component } => {
                let count = coefficient_solve::admit(length, extent as usize)?;
                if component as usize >= self.width() {
                    return Err(ir_builder_error(
                        "coefficient solve output is outside its extent",
                    ));
                }
                Ok(count)
            }
            Self::ComplexMath { function, .. } => complex_math::admit(function, length),
        }
    }

    pub(super) fn jet(self, a: &[Option<f64>], b: &[Option<f64>]) -> Result<f64, Diagnostic> {
        let a = a.iter().map(|v| v.map(Scaled::new)).collect::<Vec<_>>();
        let b = b.iter().map(|v| v.map(Scaled::new)).collect::<Vec<_>>();
        self.scaled_jet(&a, &b).map(Scaled::value)
    }

    pub(super) fn scaled_jet(
        self,
        a: &[Option<Scaled>],
        b: &[Option<Scaled>],
    ) -> Result<Scaled, Diagnostic> {
        self.admit(a.len())?;
        require_length(b, a.len(), "coefficient jet operands")?;
        match self {
            Self::Solve { extent, component } => {
                let q = coefficient_solve::coefficients(a, b, extent as usize)?;
                Ok(q[q.len() - self.width() + component as usize])
            }
            Self::ComplexMath {
                function,
                imaginary,
            } => {
                let coefficients = complex_math::coefficients(function, a, b)?;
                Ok(coefficients[coefficients.len() - 1][usize::from(imaginary)])
            }
        }
    }
}

/// A missing coefficient is structural zero. It keeps its absence through
/// subsequent differentiation, unlike an active coefficient evaluated at zero.
type Coefficients<T> = Vec<Option<T>>;

pub(super) struct Operands {
    pub(super) shape: Shape,
    pub(super) a: Vec<Option<ValueId>>,
    pub(super) b: Vec<Option<ValueId>>,
}

impl Operands {
    pub(super) fn iter(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.a.iter().chain(&self.b).flatten().copied()
    }

    pub(super) fn values<T>(
        &self,
        mut read: impl FnMut(ValueId) -> Result<T, Diagnostic>,
    ) -> Result<(Coefficients<T>, Coefficients<T>), Diagnostic> {
        let mut collect = |ids: &[Option<ValueId>]| {
            ids.iter()
                .map(|id| id.map(&mut read).transpose())
                .collect::<Result<Vec<_>, _>>()
        };
        Ok((collect(&self.a)?, collect(&self.b)?))
    }
}

fn decode(
    start: ValueId,
    len: u32,
    shape: Shape,
    available: usize,
    mut flag: impl FnMut(usize) -> Result<f64, Diagnostic>,
) -> Result<Operands, Diagnostic> {
    let n = len as usize;
    shape.admit(n)?;
    let end = (start.0 as usize)
        .checked_add(n.checked_mul(4).ok_or_else(ir_size_error)?)
        .filter(|&end| end <= available)
        .ok_or_else(ir_size_error)?;
    let mut ids = Vec::with_capacity(2 * n);
    for index in (start.0 as usize..end).step_by(2) {
        ids.push(match flag(index)? {
            0. => None,
            1. => Some(ValueId((index + 1) as u32)),
            _ => return Err(ir_builder_error("invalid solve coefficient presence")),
        });
    }
    let b = ids.split_off(n);
    Ok(Operands { a: ids, b, shape })
}

pub(super) fn operands(
    start: ValueId,
    len: u32,
    shape: Shape,
    instructions: &[Instruction],
) -> Result<Operands, Diagnostic> {
    decode(
        start,
        len,
        shape,
        instructions.len(),
        |index| match instructions[index] {
            Instruction::Constant(flag) => Ok(flag.value()),
            _ => Err(ir_builder_error(
                "solve coefficient presence must be literal",
            )),
        },
    )
}

pub(super) fn numerical_operands(
    start: ValueId,
    len: u32,
    shape: Shape,
    values: &[f64],
) -> Result<Operands, Diagnostic> {
    decode(start, len, shape, values.len(), |index| Ok(values[index]))
}

impl Operands {
    pub(super) fn first(
        &self,
        values: &[f64],
        mut direction: impl FnMut(ValueId) -> Result<Scaled, Diagnostic>,
    ) -> Result<Scaled, Diagnostic> {
        let (mut a, mut b) = self.values(|id| read(values, id, values.len()).map(Scaled::new))?;
        let (da, db) = self.values(&mut direction)?;
        a.extend(da);
        b.extend(db);
        self.shape.scaled_jet(&a, &b)
    }

    pub(super) fn second(
        &self,
        values: &[f64],
        left: &[Scaled],
        right: &[Scaled],
    ) -> Result<Scaled, Diagnostic> {
        let (mut a, mut b) = self.values(|id| read(values, id, values.len()).map(Scaled::new))?;
        for direction in [left, right] {
            let (da, db) = self.values(|id| read(direction, id, direction.len()))?;
            a.extend(da);
            b.extend(db);
        }
        a.extend(std::iter::repeat_n(None, self.a.len()));
        b.extend(std::iter::repeat_n(None, self.b.len()));
        self.shape.scaled_jet(&a, &b)
    }
}

pub(super) fn gather(
    instructions: &mut Vec<Instruction>,
    a: &[Option<ValueId>],
    b: &[Option<ValueId>],
    shape: Shape,
) -> Result<Instruction, Diagnostic> {
    shape.admit(a.len())?;
    require_length(b, a.len(), "solve coefficient operands")?;
    let before = instructions.len();
    if a.iter()
        .chain(b)
        .flatten()
        .any(|id| id.0 as usize >= before)
    {
        return Err(ir_builder_error(
            "solve coefficients must precede their gather",
        ));
    }
    let end = before
        .checked_add(a.len().checked_mul(4).ok_or_else(ir_size_error)?)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(ir_size_error)?;
    u32::try_from(end).map_err(|_| ir_size_error())?;
    let quantity =
        |value| eqiora_core::DynQuantity::new(value, eqiora_core::DimExponents::DIMENSIONLESS);
    let one = ValueId(before as u32);
    instructions.push(Instruction::Constant(quantity(1.)));
    let start = ValueId(one.0 + 1);
    for id in a.iter().chain(b) {
        instructions.push(Instruction::Constant(quantity(f64::from(id.is_some()))));
        instructions.push(match id {
            Some(id) => Instruction::Mul(*id, one),
            None => Instruction::Constant(quantity(0.)),
        });
    }
    Ok(Instruction::CoefficientJet {
        shape,
        start,
        len: a.len() as u32,
    })
}

impl ScalarInputIrBuilder {
    pub(crate) fn complex_math(
        &mut self,
        function: eqiora_schema::kernel::UnaryMathFunction,
        operands: [ScalarInputValueId; 2],
        imaginary: bool,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.coefficient_jet(
            Shape::ComplexMath {
                function,
                imaginary,
            },
            &[Some(operands[0])],
            &[Some(operands[1])],
        )
    }

    pub(super) fn coefficient_jet(
        &mut self,
        shape: Shape,
        a: &[Option<ScalarInputValueId>],
        b: &[Option<ScalarInputValueId>],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let a = a.iter().map(|id| id.map(|id| id.0)).collect::<Vec<_>>();
        let b = b.iter().map(|id| id.map(|id| id.0)).collect::<Vec<_>>();
        let instruction = gather(&mut self.instructions, &a, &b, shape)?;
        self.push(instruction)
    }
}

#[cfg(test)]
mod tests;
