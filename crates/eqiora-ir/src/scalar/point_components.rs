//! Component selection at an already evaluated typed point. This projects
//! coordinates within scalar IR; the existing derivative executor owns calculus.
use super::*;
use eqiora_core::ValueLiteral;

struct PointComponents {
    instructions: Vec<Instruction>,
    typed_constants: Vec<ValueLiteral>,
    roots: Vec<ValueId>,
}

impl ScalarOperatorIr {
    pub(super) fn point_components(
        &self,
        trace: &[Option<ValueLiteral>],
    ) -> Result<Self, Diagnostic> {
        let point = point_components(
            &self.instructions,
            &self.array_operands,
            self.symbols.len(),
            &self.roots,
            trace,
        )?;
        Ok(Self {
            typed_inputs: self.typed_inputs.clone(),
            source_values: Vec::new(),
            instructions: point.instructions,
            typed_constants: point.typed_constants,
            roots: point.roots,
            symbols: self.symbols.clone(),
            array_operands: Vec::new(),
            definitions: Vec::new(),
        })
    }
}

impl ScalarInputOperatorIr {
    pub(super) fn point_components(
        &self,
        trace: &[Option<ValueLiteral>],
    ) -> Result<Self, Diagnostic> {
        let mut point = point_components(
            &self.instructions,
            &[],
            self.slots.len(),
            &self.evaluation_roots(),
            trace,
        )?;
        let roots = point.roots.split_off(self.primal_checks.len());
        Ok(Self {
            primal_checks: point.roots,
            slots: self.slots.clone(),
            instructions: point.instructions,
            typed_constants: point.typed_constants,
            roots,
        })
    }
}

fn point_components(
    source_instructions: &[Instruction],
    array_operands: &[ValueId],
    input_count: usize,
    roots: &[ValueId],
    trace: &[Option<ValueLiteral>],
) -> Result<PointComponents, Diagnostic> {
    // Scalar points also need live-branch projection. Otherwise a downstream
    // term walk can demand an inactive arm that has no evaluated value.
    let mut instructions = (0..input_count)
        .map(|index| Instruction::Read(SymbolSlot(index as u32)))
        .collect::<Vec<_>>();
    let symbols = (0..input_count)
        .map(|index| ValueId(index as u32))
        .collect::<Vec<_>>();
    let mut typed_constants = Vec::new();
    let mut mapped: Vec<Vec<ValueId>> = vec![Vec::new(); source_instructions.len()];
    let mut map_entries = HashMap::new();
    for (index, instruction) in source_instructions.iter().copied().enumerate() {
        let Some(value) = &trace[index] else { continue };
        let count = value
            .value_type()
            .shape()
            .component_count()
            .ok_or_else(|| ir_builder_error("point component count is unavailable"))?;
        let at = |id: ValueId, component: usize| {
            let values = &mapped[id.0 as usize];
            values[if values.len() == 1 { 0 } else { component }]
        };
        let mut components = Vec::with_capacity(count);
        match instruction {
            Instruction::Array { start, len } => {
                for id in &array_operands[start as usize..start as usize + len as usize] {
                    components.extend_from_slice(&mapped[id.0 as usize]);
                }
            }
            Instruction::Index(operand, channel) => {
                let start = channel as usize * count;
                components.extend_from_slice(&mapped[operand.0 as usize][start..start + count]);
            }
            _ => {
                for component in 0..count {
                    if let Instruction::Read(slot) = instruction {
                        components.push(symbols[slot.0 as usize]);
                        continue;
                    }
                    if let Instruction::And(a, b) | Instruction::Or(a, b) = instruction
                        && trace[b.0 as usize].is_none()
                    {
                        components.push(at(a, component));
                        continue;
                    }
                    let instruction = match instruction {
                        Instruction::Constant(_) | Instruction::TypedConstant(_) => {
                            let extents = value.value_type().shape().extents();
                            let mut remaining = component;
                            let mut coordinates = vec![0; extents.len()];
                            for (axis, extent) in extents.iter().enumerate().rev() {
                                coordinates[axis] = (remaining % extent.get() as usize) as u32;
                                remaining /= extent.get() as usize;
                            }
                            let mut selected = value.clone();
                            for coordinate in coordinates {
                                selected = selected
                                    .index(coordinate)
                                    .map_err(|error| ir_builder_error(error.to_string()))?;
                            }
                            let constant = u32::try_from(typed_constants.len())
                                .map_err(|_| ir_size_error())?;
                            typed_constants.push(selected);
                            Instruction::TypedConstant(constant)
                        }
                        Instruction::Select {
                            condition,
                            then_value,
                            else_value,
                        } => {
                            let branch = if trace[condition.0 as usize]
                                .as_ref()
                                .and_then(ValueLiteral::as_bool)
                                .expect("evaluated condition")
                            {
                                then_value
                            } else {
                                else_value
                            };
                            let selected = at(branch, component);
                            // Retain the live predicate's comparison boundary.
                            Instruction::Select {
                                condition: at(condition, 0),
                                then_value: selected,
                                else_value: selected,
                            }
                        }
                        Instruction::Require { condition, value } => Instruction::Require {
                            condition: at(condition, 0),
                            value: at(value, component),
                        },
                        _ => slot_composition::remap_instruction(
                            source_instructions,
                            index,
                            &mut instructions,
                            &mut map_entries,
                            |id| at(id, component),
                        )?,
                    };
                    components.push(ValueId(
                        u32::try_from(instructions.len()).map_err(|_| ir_size_error())?,
                    ));
                    instructions.push(instruction);
                }
            }
        }
        if components.len() != count {
            return Err(ir_builder_error("point component projection changed shape"));
        }
        mapped[index] = components;
    }
    let roots = roots
        .iter()
        .map(|id| {
            let values = &mapped[id.0 as usize];
            if values.len() == 1 {
                Ok(values[0])
            } else {
                Err(ir_builder_error("point derivative roots must be scalar"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PointComponents {
        instructions,
        typed_constants,
        roots,
    })
}
