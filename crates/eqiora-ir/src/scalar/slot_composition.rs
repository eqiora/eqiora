//! Substitute scalar inputs without evaluating values or losing demanded domains.
use super::*;

/// Relocate scalar operands, regathering primitives whose inputs must be contiguous.
/// Guard projection and input composition use the same relocation rules.
pub(super) fn remap_instruction(
    source: &[Instruction],
    index: usize,
    instructions: &mut Vec<Instruction>,
    map_entries: &mut HashMap<(u32, u32), ValueId>,
    at: impl Fn(ValueId) -> ValueId,
) -> Result<Instruction, Diagnostic> {
    Ok(match source[index] {
        Instruction::CoefficientJet { start, len, shape } => {
            let operands = coefficient_jet::operands(start, len, shape, &source[..index])?;
            let (a, b) = operands.values(|id| Ok(at(id)))?;
            coefficient_jet::gather(instructions, &a, &b, shape)?
        }
        Instruction::MapInvariant {
            start,
            extent,
            output,
        } => {
            let start = if let Some(&mapped) = map_entries.get(&(start.0, extent)) {
                mapped
            } else {
                let entries = map_evaluation::operand_range(start, extent, index)?
                    .map(|operand| at(ValueId(operand as u32)));
                let mapped = map_evaluation::gather_entries(instructions, entries)?;
                map_entries.insert((start.0, extent), mapped);
                mapped
            };
            Instruction::MapInvariant {
                start,
                extent,
                output,
            }
        }
        instruction => instruction
            .map_scalar_operands(at)
            .ok_or_else(|| ir_builder_error("scalar composition requires scalar operands"))?,
    })
}

impl ScalarInputIrBuilder {
    pub(crate) fn compose(
        &mut self,
        source: &ScalarInputOperatorIr,
        inputs: &[ScalarInputValueId],
        mut charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarInputValueId>, Diagnostic> {
        require_length(inputs, source.slots.len(), "scalar composition inputs")?;
        for &input in inputs {
            self.require_prior(input)?;
        }
        // Include every copied instruction and the worst-case primitive gathers.
        // Charge before allocation/emission against the caller's existing budget.
        let work = source
            .instructions
            .iter()
            .try_fold(0usize, |work, instruction| {
                let extra = match instruction {
                    Instruction::CoefficientJet { len, .. } => (*len as usize)
                        .checked_mul(4)
                        .and_then(|n| n.checked_add(1)),
                    Instruction::MapInvariant { extent, .. } => (*extent as usize)
                        .checked_mul(*extent as usize)
                        .and_then(|n| n.checked_add(1)),
                    _ => Some(0),
                };
                work.checked_add(extra?).and_then(|n| n.checked_add(1))
            })
            .ok_or_else(ir_size_error)?;
        charge(work)?;
        let mut mapped = Vec::with_capacity(source.instructions.len());
        let mut map_entries = HashMap::new();
        for (index, instruction) in source.instructions.iter().enumerate() {
            let value = if let Instruction::Read(slot) = instruction {
                inputs[slot.0 as usize]
            } else {
                let instruction = remap_instruction(
                    &source.instructions,
                    index,
                    &mut self.instructions,
                    &mut map_entries,
                    |id| mapped[id.0 as usize],
                )?;
                self.push(instruction)?
            };
            mapped.push(value.0);
        }
        self.primal_checks
            .extend(source.primal_checks.iter().map(|id| mapped[id.0 as usize]));
        Ok(source
            .roots
            .iter()
            .map(|id| ScalarInputValueId(mapped[id.0 as usize]))
            .collect())
    }
}
