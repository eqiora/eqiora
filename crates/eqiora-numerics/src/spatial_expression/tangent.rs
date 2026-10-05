//! Common real-direction forward evaluation for spatial coefficients.
use super::*;

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    /// Evaluate the physical-coordinate gradient while holding Parameters fixed.
    pub(crate) fn evaluate_gradient<const D: usize>(
        &self,
        coordinates: &[f64],
    ) -> Result<[S; D], Diagnostic> {
        self.validate_coordinates(coordinates)?;
        if D != self.coordinate_dimension {
            return Err(input_mismatch(
                "spatial gradient dimension differs from its coordinate space",
            ));
        }
        let zero = <S as From<f64>>::from(0.0);
        let zero_parameters = vec![zero; self.parameter_fields.len()];
        let mut gradient = [zero; D];
        for axis in 0..D {
            let mut direction = [0.0; D];
            direction[axis] = 1.0;
            gradient[axis] = self
                .evaluate_tangent(coordinates, &direction, &zero_parameters)?
                .1;
        }
        Ok(gradient)
    }

    pub(crate) fn evaluate_tangent(
        &self,
        coordinates: &[f64],
        coordinate_tangent: &[f64],
        parameter_tangent: &[S],
    ) -> Result<(S, S), Diagnostic> {
        let scalar = <S as From<f64>>::from;
        let zero = scalar(0.0);
        self.validate_coordinates(coordinates)?;
        if coordinate_tangent.len() != self.coordinate_dimension
            || parameter_tangent.len() != self.parameter_fields.len()
        {
            return Err(input_mismatch(format!(
                "spatial expression expects {}/{} coordinate/Parameter tangents, received {}/{}",
                self.coordinate_dimension,
                self.parameter_fields.len(),
                coordinate_tangent.len(),
                parameter_tangent.len()
            )));
        }
        if coordinate_tangent.iter().any(|value| !value.is_finite())
            || parameter_tangent.iter().any(|value| !value.is_finite())
        {
            return Err(nonfinite(
                "spatial coordinate or Parameter tangent is non-finite",
            ));
        }
        let mut values: Vec<S> = Vec::with_capacity(self.instructions.len());
        let mut tangents: Vec<S> = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let (value, tangent) = match *instruction {
                Instruction::Constant(value) => (value, zero),
                Instruction::Parameter(parameter) => (
                    self.parameter_values[parameter],
                    parameter_tangent[parameter],
                ),
                Instruction::Coordinate(axis) => {
                    (scalar(coordinates[axis]), scalar(coordinate_tangent[axis]))
                }
                Instruction::Neg(value) => (-values[value], -tangents[value]),
                Instruction::Conjugate(value) => (values[value].conj(), tangents[value].conj()),
                Instruction::Add(left, right) => (
                    values[left] + values[right],
                    tangents[left] + tangents[right],
                ),
                Instruction::Sub(left, right) => (
                    values[left] - values[right],
                    tangents[left] - tangents[right],
                ),
                Instruction::Mul(left, right) => (
                    values[left] * values[right],
                    tangents[left] * values[right] + values[left] * tangents[right],
                ),
                Instruction::Div(left, right) => (
                    values[left] / values[right],
                    (tangents[left] * values[right] - values[left] * tangents[right])
                        / values[right].powi(2),
                ),
                Instruction::PowI(base, exponent) => {
                    let value = values[base].powi(exponent);
                    let tangent = if exponent == 0 {
                        zero
                    } else {
                        scalar(f64::from(exponent))
                            * preceding_power(values[base], exponent)
                            * tangents[base]
                    };
                    (value, tangent)
                }
                Instruction::Sin(value) => {
                    (values[value].sin(), values[value].cos() * tangents[value])
                }
                Instruction::Sqrt(value) => {
                    let root = if S::DOMAIN == eqiora_core::ScalarDomain::Real {
                        scalar(real_sqrt(values[value].re())?)
                    } else {
                        if values[value].im() == 0.0 && values[value].re() < 0.0 {
                            return Err(nonfinite(
                                "complex square-root derivative is not admitted on the principal branch cut",
                            ));
                        }
                        values[value].sqrt()
                    };
                    if root == zero {
                        return Err(nonfinite("square-root derivative is undefined at zero"));
                    }
                    (root, tangents[value] / (scalar(2.0) * root))
                }
            };
            if !value.is_finite() || !tangent.is_finite() {
                return Err(nonfinite(
                    "scalar spatial expression produced a non-finite primal or tangent value",
                ));
            }
            values.push(value);
            tangents.push(tangent);
        }
        Ok((values[self.root], tangents[self.root]))
    }
}

// Preserve the integer exponent domain even when n-1 lies outside i32.
// For the minimum (negative) exponent, x^(n-1) = x^n / x; x=0
// remains outside the primal domain and is rejected by the finite checks.
pub(super) fn preceding_power<S: ComplexFloat>(value: S, exponent: i32) -> S {
    match exponent.checked_sub(1) {
        Some(previous) => value.powi(previous),
        None => value.powi(exponent) / value,
    }
}
