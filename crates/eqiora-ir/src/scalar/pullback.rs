//! Scoped SSA substitution; mapped coordinates retain their outer-context values.
use super::*;

pub(super) fn substitute(
    root: ValueId,
    bindings: &HashMap<SymbolRef, ValueId>,
    symbols: &[SymbolRef],
    instructions: &mut Vec<Instruction>,
    operands: &mut Vec<ValueId>,
) -> Result<ValueId, Diagnostic> {
    let count = root.0 as usize + 1;
    if instructions
        .len()
        .checked_add(count)
        .is_none_or(|n| n > 1_000_000)
    {
        return Err(ir_builder_error(
            "coordinate substitution exceeds scalar work budget",
        ));
    }
    let mut mapped = Vec::with_capacity(count);
    for index in 0..count {
        let instruction = instructions[index];
        if let Instruction::Read(slot) = instruction {
            mapped.push(
                bindings
                    .get(&symbols[slot.0 as usize])
                    .copied()
                    .unwrap_or(ValueId(index as u32)),
            );
            continue;
        }
        let at = |id: ValueId| mapped[id.0 as usize];
        let instruction = match instruction {
            Instruction::Array { start, len } | Instruction::PureOperator { start, len, .. } => {
                if operands
                    .len()
                    .checked_add(len as usize)
                    .is_none_or(|n| n > 1_000_000)
                {
                    return Err(ir_builder_error(
                        "coordinate substitution exceeds operand work budget",
                    ));
                }
                let values = operands[start as usize..start as usize + len as usize]
                    .iter()
                    .copied()
                    .map(at)
                    .collect::<Vec<_>>();
                let start = u32::try_from(operands.len()).map_err(|_| ir_size_error())?;
                operands.extend(values);
                match instruction {
                    Instruction::Array { .. } => Instruction::Array { start, len },
                    Instruction::PureOperator { definition, .. } => Instruction::PureOperator {
                        definition,
                        start,
                        len,
                    },
                    _ => unreachable!(),
                }
            }
            _ => instruction.map_scalar_operands(at).ok_or_else(|| {
                ir_builder_error("coordinate substitution requires scalar operands")
            })?,
        };
        mapped.push(ValueId(
            u32::try_from(instructions.len()).map_err(|_| ir_size_error())?,
        ));
        instructions.push(instruction);
    }
    Ok(mapped[root.0 as usize])
}
