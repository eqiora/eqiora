//! Mixed quotient coefficients keep binary exponents until the requested output.
//!
//! Coefficient k multiplies the product of the nilpotent directions in k's
//! bitmask. The identity B*Q=A gives every order through the same recurrence;
//! repeated directions therefore retain the ordinary mixed-derivative factors.
use super::*;
mod affine;
mod typed;
pub(super) use affine::evaluate as affine;
pub(super) use typed::evaluate as typed;

fn admit(n: usize) -> Result<(), Diagnostic> {
    if !n.is_power_of_two()
        || 3usize
            .checked_pow(n.trailing_zeros())
            .is_none_or(|work| work > MAX_COMPONENT_PRODUCTS)
    {
        return Err(ir_builder_error(
            "quotient coefficient work exceeds scalar product budget",
        ));
    }
    Ok(())
}

pub(super) fn jet(a: &[Option<f64>], b: &[Option<f64>]) -> Result<f64, Diagnostic> {
    let a = a.iter().map(|v| v.map(Scaled::new)).collect::<Vec<_>>();
    let b = b.iter().map(|v| v.map(Scaled::new)).collect::<Vec<_>>();
    scaled_jet(&a, &b).map(Scaled::value)
}

pub(super) fn scaled_jet(a: &[Option<Scaled>], b: &[Option<Scaled>]) -> Result<Scaled, Diagnostic> {
    admit(a.len())?;
    require_length(b, a.len(), "quotient coefficient denominator")?;
    let base = b[0].ok_or_else(|| ir_builder_error("quotient requires a base denominator"))?;
    if base.is_zero()
        || !base.is_finite()
        || a.iter().chain(b).flatten().any(|value| !value.is_finite())
    {
        return Ok(Scaled::new(f64::NAN));
    }
    let mut q = Vec::with_capacity(a.len());
    for mask in 0..a.len() {
        let mut value = a[mask].unwrap_or(Scaled::ZERO);
        let mut subset = mask;
        while subset != 0 {
            value = value.sub(b[subset].unwrap_or(Scaled::ZERO).mul(q[mask ^ subset]));
            subset = (subset - 1) & mask;
        }
        q.push(value.div(base));
    }
    Ok(*q.last().expect("nonempty coefficients"))
}

/// A missing coefficient is structural zero. It keeps its absence through
/// subsequent differentiation, unlike an active coefficient evaluated at zero.
type Coefficients<T> = Vec<Option<T>>;

pub(super) struct Operands {
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
    available: usize,
    mut flag: impl FnMut(usize) -> Result<f64, Diagnostic>,
) -> Result<Operands, Diagnostic> {
    let n = len as usize;
    admit(n)?;
    let end = (start.0 as usize)
        .checked_add(n.checked_mul(4).ok_or_else(ir_size_error)?)
        .filter(|&end| end <= available)
        .ok_or_else(ir_size_error)?;
    let mut ids = Vec::with_capacity(2 * n);
    for index in (start.0 as usize..end).step_by(2) {
        ids.push(match flag(index)? {
            0. => None,
            1. => Some(ValueId((index + 1) as u32)),
            _ => return Err(ir_builder_error("invalid quotient coefficient presence")),
        });
    }
    let b = ids.split_off(n);
    Ok(Operands { a: ids, b })
}

pub(super) fn operands(
    start: ValueId,
    len: u32,
    instructions: &[Instruction],
) -> Result<Operands, Diagnostic> {
    decode(start, len, instructions.len(), |index| {
        match instructions[index] {
            Instruction::Constant(flag) => Ok(flag.value()),
            _ => Err(ir_builder_error(
                "quotient coefficient presence must be literal",
            )),
        }
    })
}

pub(super) fn numerical_operands(
    start: ValueId,
    len: u32,
    values: &[f64],
) -> Result<Operands, Diagnostic> {
    decode(start, len, values.len(), |index| Ok(values[index]))
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
        scaled_jet(&a, &b)
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
        scaled_jet(&a, &b)
    }
}

pub(super) fn gather(
    instructions: &mut Vec<Instruction>,
    a: &[Option<ValueId>],
    b: &[Option<ValueId>],
) -> Result<Instruction, Diagnostic> {
    admit(a.len())?;
    require_length(b, a.len(), "quotient coefficient operands")?;
    let before = instructions.len();
    if a.iter()
        .chain(b)
        .flatten()
        .any(|id| id.0 as usize >= before)
    {
        return Err(ir_builder_error(
            "quotient coefficients must precede their gather",
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
    Ok(Instruction::QuotientJet {
        start,
        len: a.len() as u32,
    })
}

impl ScalarInputIrBuilder {
    pub(super) fn quotient_jet(
        &mut self,
        a: &[Option<ScalarInputValueId>],
        b: &[Option<ScalarInputValueId>],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let a = a.iter().map(|id| id.map(|id| id.0)).collect::<Vec<_>>();
        let b = b.iter().map(|id| id.map(|id| id.0)).collect::<Vec<_>>();
        let instruction = gather(&mut self.instructions, &a, &b)?;
        self.push(instruction)
    }
}

#[cfg(test)]
mod tests;
