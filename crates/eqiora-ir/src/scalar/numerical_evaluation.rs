//! Strict real numerical evaluation of scalar SSA.
use super::*;

pub(super) fn evaluate_instructions(
    instructions: &[Instruction],
    inputs: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let mut values = Vec::with_capacity(instructions.len());
    let mut maps = map_evaluation::MapCache::default();
    for (index, instruction) in instructions.iter().enumerate() {
        let value = match *instruction {
            Instruction::Constant(value) => value.value(),
            Instruction::Read(slot) => {
                inputs
                    .get(slot_index(slot, index)?)
                    .copied()
                    .ok_or_else(|| {
                        Diagnostic::error(
                            codes::INVALID_OPERATOR_IR,
                            "scalar input slot is outside the supplied input inventory",
                        )
                        .with_graph_path(ir_path(index))
                    })?
            }
            Instruction::Math(function, value) => {
                real_math::value(function, read(&values, value, index)?)?
            }
            Instruction::Neg(value) => -read(&values, value, index)?,
            Instruction::Add(left, right) => {
                read(&values, left, index)? + read(&values, right, index)?
            }
            Instruction::Sub(left, right) => {
                read(&values, left, index)? - read(&values, right, index)?
            }
            Instruction::Mul(left, right) => {
                read(&values, left, index)? * read(&values, right, index)?
            }
            Instruction::Div(left, right) => {
                read(&values, left, index)? / read(&values, right, index)?
            }
            Instruction::ComplexDiv {
                operands,
                imaginary,
            } => complex_quotient::evaluate(operands, &values, index)?[usize::from(imaginary)],
            Instruction::CoefficientJet { start, len, shape } => {
                let operands =
                    coefficient_jet::operands(start, len, shape, &instructions[..index])?;
                let (a, b) = operands.values(|id| read(&values, id, index))?;
                shape.jet(&a, &b)?
            }
            Instruction::MapInvariant {
                start,
                extent,
                output,
            } => maps.get(start, extent, &values)?.value(output)?,
            Instruction::PowI(base, exponent) => read(&values, base, index)?.powi(exponent),
            _ => {
                return Err(ir_builder_error(
                    "discrete operations and retained pure applications require typed execution",
                ));
            }
        };
        if !value.is_finite() {
            return Err(Diagnostic::error(
                codes::NONFINITE_EVALUATION,
                format!("scalar Operator IR instruction {index} evaluated to {value}"),
            )
            .with_graph_path(ir_path(index)));
        }
        values.push(value);
    }
    Ok(values)
}
