use super::*;

#[derive(Clone)]
pub(super) struct Polynomial(BTreeMap<Vec<u32>, f64>);
impl Polynomial {
    pub(super) fn constant(variables: usize, value: f64) -> Self {
        Self(BTreeMap::from([(vec![0; variables], value)]))
    }
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = (&Vec<u32>, &f64)> {
        self.0.iter()
    }
    fn insert(&mut self, powers: Vec<u32>, value: f64, budget: usize) -> Result<(), Diagnostic> {
        if !self.0.contains_key(&powers) && self.len() >= budget {
            return Err(ir_builder_error(
                "polynomial pencil term resource budget exhausted",
            ));
        }
        let coefficient = self.0.entry(powers).or_default();
        *coefficient += value;
        if !coefficient.is_finite() {
            return Err(ir_builder_error(
                "polynomial coefficient arithmetic is nonfinite",
            ));
        }
        Ok(())
    }
    fn map(&self, operation: impl Fn(f64) -> f64) -> Result<Self, Diagnostic> {
        let mut output = Self(BTreeMap::new());
        for (powers, value) in &self.0 {
            output.insert(powers.clone(), operation(*value), self.len())?;
        }
        Ok(output)
    }
    fn add(&self, other: &Self, sign: f64, budget: usize) -> Result<Self, Diagnostic> {
        let mut output = self.clone();
        for (powers, value) in &other.0 {
            output.insert(powers.clone(), sign * value, budget)?;
        }
        Ok(output)
    }
    fn multiply(&self, other: &Self, budget: usize) -> Result<Self, Diagnostic> {
        let mut output = Self(BTreeMap::new());
        for (left, a) in &self.0 {
            for (right, b) in &other.0 {
                let powers = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| {
                        a.checked_add(*b)
                            .ok_or_else(|| ir_builder_error("polynomial exponent overflow"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                output.insert(powers, a * b, budget)?;
            }
        }
        Ok(output)
    }
}

pub(super) fn evaluate(
    instruction: Instruction,
    polynomials: &[Polynomial],
    values: &[f64],
    variables: usize,
    slots: &[ScalarInputSlot],
    coordinates: &HashMap<&ScalarSymbolCoordinate, usize>,
    budget: usize,
) -> Result<Polynomial, Diagnostic> {
    let at = |id: ValueId| {
        polynomials
            .get(id.0 as usize)
            .ok_or_else(|| ir_builder_error("invalid polynomial SSA operand"))
    };
    Ok(match instruction {
        Instruction::Read(slot) => {
            let source = slots
                .get(slot.0 as usize)
                .ok_or_else(|| ir_builder_error("invalid polynomial input slot"))?
                .source();
            let index = coordinates
                .get(source)
                .ok_or_else(|| ir_builder_error("missing polynomial coordinate"))?;
            let mut powers = vec![0; variables];
            powers[*index] = 1;
            Polynomial(BTreeMap::from([(powers, 1.)]))
        }
        Instruction::Neg(a) => at(a)?.map(|value| -value)?,
        Instruction::Add(a, b) => at(a)?.add(at(b)?, 1., budget)?,
        Instruction::Sub(a, b) => at(a)?.add(at(b)?, -1., budget)?,
        Instruction::Mul(a, b) => at(a)?.multiply(at(b)?, budget)?,
        Instruction::Div(a, b) => at(a)?.map(|value| value / values[b.0 as usize])?,
        Instruction::ComplexDiv {
            operands: [a, b, c, d],
            imaginary,
        } => {
            let mut powers = at(a)?
                .0
                .keys()
                .chain(at(b)?.0.keys())
                .cloned()
                .collect::<Vec<_>>();
            powers.sort();
            powers.dedup();
            let mut output = Polynomial(BTreeMap::new());
            for power in powers {
                let numerator = [
                    at(a)?.0.get(&power).copied().unwrap_or(0.),
                    at(b)?.0.get(&power).copied().unwrap_or(0.),
                ];
                let value = complex_quotient::quotient(
                    numerator,
                    [values[c.0 as usize], values[d.0 as usize]],
                )[usize::from(imaginary)];
                output.insert(power, value, budget)?;
            }
            output
        }
        Instruction::PowI(a, exponent) if exponent > 0 => {
            let mut power = exponent as u32;
            let mut base = at(a)?.clone();
            let mut output = Polynomial::constant(variables, 1.);
            while power > 0 {
                if power % 2 == 1 {
                    output = output.multiply(&base, budget)?;
                }
                power /= 2;
                if power > 0 {
                    base = base.multiply(&base, budget)?;
                }
            }
            output
        }
        _ => {
            return Err(ir_builder_error(
                "dependent instruction is outside the polynomial pencil profile",
            ));
        }
    })
}
