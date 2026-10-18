//! The same arithmetic differential is evaluated numerically or emitted as SSA.
use super::*;

pub(super) trait Arithmetic {
    type Value: Copy;
    fn math(
        &mut self,
        function: eqiora_schema::kernel::UnaryMathFunction,
        value: Self::Value,
    ) -> Result<Self::Value, Diagnostic>;
    fn positive(&mut self, value: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn constant(&mut self, value: f64) -> Result<Self::Value, Diagnostic>;
    fn neg(&mut self, value: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn add(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn sub(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn mul(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn div(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic>;
    fn powi(&mut self, value: Self::Value, exponent: i32) -> Result<Self::Value, Diagnostic>;
    fn quotient_jet(
        &mut self,
        numerator: &[Option<Self::Value>],
        denominator: &[Option<Self::Value>],
    ) -> Result<Self::Value, Diagnostic>;
    fn complex_quotient(
        &mut self,
        numerator: [Self::Value; 2],
        denominator: [Self::Value; 2],
    ) -> Result<[Self::Value; 2], Diagnostic>;
}

/// The quotient coefficient owner retains exponent range through the complete
/// action. Absence is structural independence, never a sampled zero.
pub(super) fn quotient_direction<A: Arithmetic>(
    algebra: &mut A,
    numerator: A::Value,
    denominator: A::Value,
    numerator_direction: Option<A::Value>,
    denominator_direction: Option<A::Value>,
) -> Result<Option<A::Value>, Diagnostic> {
    if denominator_direction.is_none() {
        return numerator_direction
            .map(|direction| algebra.div(direction, denominator))
            .transpose();
    }
    algebra
        .quotient_jet(
            &[Some(numerator), numerator_direction],
            &[Some(denominator), denominator_direction],
        )
        .map(Some)
}

pub(super) fn power_slope<A: Arithmetic>(
    algebra: &mut A,
    base: A::Value,
    exponent: i32,
) -> Result<A::Value, Diagnostic> {
    if exponent == 0 {
        return algebra.constant(0.);
    }
    let coefficient = algebra.constant(f64::from(exponent))?;
    let power = algebra.powi(base, exponent.checked_sub(1).unwrap_or(exponent))?;
    let slope = algebra.mul(coefficient, power)?;
    if exponent == i32::MIN {
        algebra.div(slope, base)
    } else {
        Ok(slope)
    }
}

/// None is structural independence, never a sampled numerical zero.
/// Primal domain obligations remain with the caller's program.
pub(super) fn apply<A: Arithmetic>(
    algebra: &mut A,
    instruction: Instruction,
    value: impl Fn(ValueId) -> Result<A::Value, Diagnostic>,
    direction: impl Fn(ValueId) -> Result<Option<A::Value>, Diagnostic>,
) -> Result<Option<A::Value>, Diagnostic> {
    let result = match instruction {
        Instruction::Neg(a) => direction(a)?.map(|d| algebra.neg(d)).transpose()?,
        Instruction::Add(a, b) | Instruction::Sub(a, b) => {
            let subtract = matches!(instruction, Instruction::Sub(..));
            match (direction(a)?, direction(b)?) {
                (None, None) => None,
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(if subtract { algebra.neg(b)? } else { b }),
                (Some(a), Some(b)) => Some(if subtract {
                    algebra.sub(a, b)?
                } else {
                    algebra.add(a, b)?
                }),
            }
        }
        Instruction::Mul(a, b) => {
            let left = direction(a)?
                .map(|d| algebra.mul(d, value(b)?))
                .transpose()?;
            let right = direction(b)?
                .map(|d| algebra.mul(value(a)?, d))
                .transpose()?;
            match (left, right) {
                (None, None) => None,
                (Some(d), None) | (None, Some(d)) => Some(d),
                (Some(a), Some(b)) => Some(algebra.add(a, b)?),
            }
        }
        Instruction::Div(a, b) => {
            quotient_direction(algebra, value(a)?, value(b)?, direction(a)?, direction(b)?)?
        }
        Instruction::PowI(a, exponent) => direction(a)?
            .map(|d| {
                let slope = power_slope(algebra, value(a)?, exponent)?;
                algebra.mul(slope, d)
            })
            .transpose()?,
        _ => {
            return Err(ir_builder_error(
                "instruction has no arithmetic differential",
            ));
        }
    };
    Ok(result)
}

pub(super) struct Numerical;
impl Arithmetic for Numerical {
    type Value = Scaled;
    fn quotient_jet(
        &mut self,
        numerator: &[Option<Scaled>],
        denominator: &[Option<Scaled>],
    ) -> Result<Scaled, Diagnostic> {
        coefficient_jet::Shape::SCALAR.scaled_jet(numerator, denominator)
    }
    fn math(
        &mut self,
        function: eqiora_schema::kernel::UnaryMathFunction,
        value: Scaled,
    ) -> Result<Scaled, Diagnostic> {
        real_math::value(function, value.value()).map(Scaled::new)
    }
    fn positive(&mut self, value: Scaled) -> Result<Scaled, Diagnostic> {
        if value.is_positive() {
            Ok(value)
        } else {
            Err(ir_builder_error(
                "scalar derivative requires a positive argument",
            ))
        }
    }
    fn complex_quotient(
        &mut self,
        numerator: [Scaled; 2],
        denominator: [Scaled; 2],
    ) -> Result<[Scaled; 2], Diagnostic> {
        Ok(complex_quotient::scaled_quotient(numerator, denominator))
    }
    fn constant(&mut self, value: f64) -> Result<Scaled, Diagnostic> {
        Ok(Scaled::new(value))
    }
    fn neg(&mut self, value: Scaled) -> Result<Scaled, Diagnostic> {
        Ok(-value)
    }
    fn add(&mut self, a: Scaled, b: Scaled) -> Result<Scaled, Diagnostic> {
        Ok(a + b)
    }
    fn sub(&mut self, a: Scaled, b: Scaled) -> Result<Scaled, Diagnostic> {
        Ok(a - b)
    }
    fn mul(&mut self, a: Scaled, b: Scaled) -> Result<Scaled, Diagnostic> {
        Ok(a * b)
    }
    fn div(&mut self, a: Scaled, b: Scaled) -> Result<Scaled, Diagnostic> {
        Ok(a / b)
    }
    fn powi(&mut self, value: Scaled, exponent: i32) -> Result<Scaled, Diagnostic> {
        Ok(value.powi(exponent))
    }
}

impl Arithmetic for ScalarInputIrBuilder {
    type Value = ScalarInputValueId;
    fn quotient_jet(
        &mut self,
        numerator: &[Option<Self::Value>],
        denominator: &[Option<Self::Value>],
    ) -> Result<Self::Value, Diagnostic> {
        self.coefficient_jet(coefficient_jet::Shape::SCALAR, numerator, denominator)
    }
    fn math(
        &mut self,
        function: eqiora_schema::kernel::UnaryMathFunction,
        value: Self::Value,
    ) -> Result<Self::Value, Diagnostic> {
        self.unary_math(function, value)
    }
    fn positive(&mut self, value: Self::Value) -> Result<Self::Value, Diagnostic> {
        let zero = self.sub(value, value)?;
        let condition = self.compare(eqiora_schema::kernel::ComparisonOp::Greater, value, zero)?;
        self.require(condition, value)
    }
    fn complex_quotient(
        &mut self,
        numerator: [Self::Value; 2],
        denominator: [Self::Value; 2],
    ) -> Result<[Self::Value; 2], Diagnostic> {
        let operands = [numerator[0], numerator[1], denominator[0], denominator[1]];
        Ok([
            self.complex_div(operands, false)?,
            self.complex_div(operands, true)?,
        ])
    }
    fn constant(&mut self, value: f64) -> Result<Self::Value, Diagnostic> {
        self.constant(eqiora_core::DynQuantity::new(
            value,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))
    }
    fn neg(&mut self, value: Self::Value) -> Result<Self::Value, Diagnostic> {
        self.neg(value)
    }
    fn add(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic> {
        self.add(a, b)
    }
    fn sub(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic> {
        self.sub(a, b)
    }
    fn mul(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic> {
        self.mul(a, b)
    }
    fn div(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Diagnostic> {
        self.div(a, b)
    }
    fn powi(&mut self, value: Self::Value, exponent: i32) -> Result<Self::Value, Diagnostic> {
        self.powi(value, exponent)
    }
}

#[cfg(test)]
mod tests;
