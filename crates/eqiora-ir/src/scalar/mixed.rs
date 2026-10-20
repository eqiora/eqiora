//! Mixed products reuse the same primal and first-direction evaluator.
use super::*;

#[cfg(test)]
mod tests;

impl ScalarLinearization<'_> {
    /// Evaluate `D²R[first, second] + DR[mixed]` at this bound point.
    /// The third input describes mixed derivatives of the input coordinates;
    /// supply zero directions for the ordinary Hessian bilinear product.
    /// Frozen inputs stay fixed and active-branch admission is unchanged.
    ///
    /// # Errors
    /// Rejects wrong direction cardinalities, nonfinite arithmetic and undefined derivatives.
    pub fn mixed_jvp(
        &self,
        first: RelationTangent<'_, f64>,
        second: RelationTangent<'_, f64>,
        mixed: RelationTangent<'_, f64>,
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        require_length(output, self.residual_dimension(), "mixed residual tangent")?;
        self.require_mixed_contracts(&first, &second)?;
        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        let active = self.derivative_demand(&values)?;
        let mut maps = map_evaluation::MapCache::default();
        let left = self.forward(first, &values, &active, &mut maps, None)?;
        let right = self.forward(second, &values, &active, &mut maps, None)?;
        let mixed = self.forward(mixed, &values, &active, &mut maps, Some((&left, &right)))?;
        write_roots(&self.roots, &mixed, output)
    }
}

pub(super) fn curvature(
    instruction: Instruction,
    values: &[f64],
    left: &[Scaled],
    right: &[Scaled],
    index: usize,
    maps: &mut map_evaluation::MapCache,
) -> Result<Scaled, Diagnostic> {
    let l = |id| read(left, id, index);
    let r = |id| read(right, id, index);
    Ok(match instruction {
        Instruction::Constant(_)
        | Instruction::DerivativeLimit { .. }
        | Instruction::Read(_)
        | Instruction::Neg(_)
        | Instruction::Add(_, _)
        | Instruction::Sub(_, _) => Scaled::ZERO,
        Instruction::Mul(a, b) => l(a)? * r(b)? + r(a)? * l(b)?,
        Instruction::Div(a, b) => coefficient_jet::Shape::SCALAR.scaled_jet(
            &[
                Some(Scaled::new(read(values, a, index)?)),
                Some(l(a)?),
                Some(r(a)?),
                None,
            ],
            &[
                Some(Scaled::new(read(values, b, index)?)),
                Some(l(b)?),
                Some(r(b)?),
                None,
            ],
        )?,
        Instruction::PowI(a, power) => {
            let value = read(values, a, index)?;
            let derivative = match power {
                0 | 1 => Scaled::ZERO,
                2 => Scaled::new(2.),
                n if value == 0. && n > 2 => Scaled::ZERO,
                n => {
                    differential::power_slope(&mut differential::Numerical, Scaled::new(value), n)?
                        * (f64::from(n) - 1.)
                        / value
                }
            };
            derivative * l(a)? * r(a)?
        }
        Instruction::Math(function, a) => {
            real_math::second_action(function, read(values, a, index)?, l(a)?, r(a)?)?
        }
        Instruction::ComplexDiv {
            operands,
            imaginary,
        } => complex_quotient::curvature(operands, values, left, right, index, imaginary)?,
        Instruction::CoefficientJet { start, len, shape } => {
            coefficient_jet::numerical_operands(start, len, shape, &values[..index])?
                .second(values, left, right)?
        }
        Instruction::MapInvariant {
            start,
            extent,
            output,
        } => {
            let range = map_evaluation::operand_range(start, extent, index)?;
            maps.get(start, extent, &values[..index])?.curvature(
                output,
                &left[range.clone()],
                &right[range],
            )?
        }
        _ => {
            return Err(ir_builder_error(
                "discrete operations cannot be differentiated",
            ));
        }
    })
}
