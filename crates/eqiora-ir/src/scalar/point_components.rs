//! Component selection at an already evaluated typed point. This projects
//! coordinates only; the existing scalar derivative executor owns calculus.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_schema::kernel::{ExprDagBuilder, ExprId};

impl ScalarOperatorIr {
    pub(super) fn point_components(
        &self,
        trace: &[Option<ValueLiteral>],
    ) -> Result<(Self, Vec<ExprId>), Diagnostic> {
        let mut builder = ExprDagBuilder::new();
        // Preserve all input slots, including inputs absent from the selected
        // component. Their values and differentiation roles keep their order.
        let symbols = self
            .symbols
            .iter()
            .map(|&symbol| builder.symbol(symbol))
            .collect::<Result<Vec<_>, _>>()?;
        let mut mapped: Vec<Vec<ExprId>> = vec![Vec::new(); self.instructions.len()];
        for (index, instruction) in self.instructions.iter().copied().enumerate() {
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
                    for id in &self.array_operands[start as usize..start as usize + len as usize] {
                        components.extend_from_slice(&mapped[id.0 as usize]);
                    }
                }
                Instruction::Index(operand, channel) => {
                    // Index removes the outer channel, exactly as the existing
                    // component scalarizer's (channel, inner) coordinate rule.
                    let start = channel as usize * count;
                    components.extend_from_slice(&mapped[operand.0 as usize][start..start + count]);
                }
                Instruction::Constant(_) | Instruction::TypedConstant(_) => {
                    let literal = builder.constant(value.clone())?;
                    let extents = value.value_type().shape().extents();
                    for flat in 0..count {
                        let mut remaining = flat;
                        let mut coordinates = vec![0; extents.len()];
                        for (axis, extent) in extents.iter().enumerate().rev() {
                            coordinates[axis] = (remaining % extent.get() as usize) as u32;
                            remaining /= extent.get() as usize;
                        }
                        let mut selected = literal;
                        for coordinate in coordinates {
                            selected = builder.index(selected, coordinate)?;
                        }
                        components.push(selected);
                    }
                }
                _ => {
                    for component in 0..count {
                        let id = match instruction {
                            Instruction::Read(slot) => symbols[slot.0 as usize],
                            Instruction::And(a, b) | Instruction::Or(a, b)
                                if trace[b.0 as usize].is_none() =>
                            {
                                at(a, component)
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
                                // Retain the predicate so active comparison ties
                                // keep the existing derivative-boundary rejection.
                                builder.select(at(condition, 0), selected, selected)?
                            }
                            Instruction::Require { condition, value } => {
                                builder.require(at(condition, 0), at(value, component))?
                            }
                            _ => self.append_instruction(&mut builder, instruction, |id| {
                                at(id, component)
                            })?,
                        };
                        components.push(id);
                    }
                }
            }
            if components.len() != count {
                return Err(ir_builder_error("point component projection changed shape"));
            }
            mapped[index] = components;
        }
        let roots = self
            .roots
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
        let projected = Self::lower(&builder.finish(roots.iter().copied())?)?;
        if projected.symbols != self.symbols {
            return Err(ir_builder_error(
                "point component projection changed input slots",
            ));
        }
        Ok((projected, roots))
    }
}
