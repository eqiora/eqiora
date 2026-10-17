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
        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        let mut maps = map_evaluation::MapCache::default();
        let left = self.forward(first, &values, &mut maps, None)?;
        let right = self.forward(second, &values, &mut maps, None)?;
        let mixed = self.forward(mixed, &values, &mut maps, Some((&left, &right)))?;
        write_roots(&self.roots, &mixed, output)
    }
}

pub(super) fn curvature(
    instruction: Instruction,
    values: &[f64],
    left: &[f64],
    right: &[f64],
    index: usize,
    maps: &mut map_evaluation::MapCache,
) -> Result<f64, Diagnostic> {
    let l = |id| read(left, id, index);
    let r = |id| read(right, id, index);
    Ok(match instruction {
        Instruction::Constant(_)
        | Instruction::Read(_)
        | Instruction::Neg(_)
        | Instruction::Add(_, _)
        | Instruction::Sub(_, _) => 0.,
        Instruction::Mul(a, b) => l(a)? * r(b)? + r(a)? * l(b)?,
        Instruction::Div(a, b) => {
            let denominator = read(values, b, index)?;
            let quotient = read(values, a, index)? / denominator;
            let dl = (l(a)? - quotient * l(b)?) / denominator;
            let dr = (r(a)? - quotient * r(b)?) / denominator;
            -(dl * r(b)? + dr * l(b)?) / denominator
        }
        Instruction::PowI(a, power) => {
            let value = read(values, a, index)?;
            let derivative = match power {
                0 | 1 => 0.,
                2 => 2.,
                n if value == 0. && n > 2 => 0.,
                n => powi_derivative(value, n) * (f64::from(n) - 1.) / value,
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
        Instruction::MapInvariant {
            start,
            extent,
            component,
        } => {
            let range = map_evaluation::operand_range(start, extent, index)?;
            maps.get(start, extent, &values[..index])?.curvature(
                component,
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
