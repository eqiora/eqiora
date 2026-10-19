//! Directional scalar SSA emission; source coordinates are supplied by the caller.
use super::*;

impl ScalarInputIrBuilder {
    fn derivative_dependencies(&self, root: ScalarInputValueId) -> Result<Vec<bool>, Diagnostic> {
        self.require_prior(root)?;
        let mut needed = vec![false; root.0.0 as usize + 1];
        needed[root.0.0 as usize] = true;
        for index in (0..needed.len()).rev() {
            if !needed[index] {
                continue;
            }
            let instruction = self.instructions[index];
            if let Instruction::CoefficientJet { start, len, shape } = instruction {
                for operand in
                    coefficient_jet::operands(start, len, shape, &self.instructions[..index])?
                        .iter()
                {
                    needed[operand.0 as usize] = true;
                }
            } else if let Instruction::MapInvariant { start, extent, .. } = instruction {
                for operand in map_evaluation::operand_range(start, extent, index)? {
                    needed[operand] = true;
                }
            } else if instruction
                .map_scalar_operands(|id| {
                    needed[id.0 as usize] = true;
                    id
                })
                .is_none()
            {
                return Err(ir_builder_error(
                    "directional instruction has no scalar operands",
                ));
            }
        }
        Ok(needed)
    }

    pub(crate) fn directional_inputs(
        &self,
        root: ScalarInputValueId,
    ) -> Result<Vec<ScalarInputSlot>, Diagnostic> {
        let needed = self.derivative_dependencies(root)?;
        let mut slots = vec![false; self.slots.len()];
        for (instruction, needed) in self.instructions.iter().zip(needed) {
            if let Instruction::Read(slot) = instruction {
                slots[slot.0 as usize] |= needed;
            }
        }
        Ok(self
            .slots
            .iter()
            .zip(slots)
            .filter(|(_, needed)| *needed)
            .map(|(slot, _)| slot.clone())
            .collect())
    }

    fn derivative_condition(
        &mut self,
        condition: ValueId,
        active: &[bool],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if !active[condition.0 as usize] {
            return Ok(ScalarInputValueId(condition));
        }
        match self.instructions[condition.0 as usize] {
            Instruction::Compare(_, a, b) => {
                let regular = self.compare(
                    eqiora_schema::kernel::ComparisonOp::NotEqual,
                    ScalarInputValueId(a),
                    ScalarInputValueId(b),
                )?;
                self.and(ScalarInputValueId(condition), regular)
            }
            Instruction::And(a, b) => {
                let a = self.derivative_condition(a, active)?;
                let b = self.derivative_condition(b, active)?;
                self.and(a, b)
            }
            _ => Err(ir_builder_error(
                "spatial derivative condition requires scalar comparisons and conjunctions",
            )),
        }
    }

    fn direction_pair(
        &mut self,
        real: Option<ScalarInputValueId>,
        imaginary: Option<ScalarInputValueId>,
    ) -> Result<Option<[ScalarInputValueId; 2]>, Diagnostic> {
        Ok(match (real, imaginary) {
            (None, None) => None,
            (Some(real), Some(imaginary)) => Some([real, imaginary]),
            (Some(real), None) => Some([real, self.sub(real, real)?]),
            (None, Some(imaginary)) => Some([self.sub(imaginary, imaginary)?, imaginary]),
        })
    }

    pub(crate) fn directional(
        &mut self,
        root: ScalarInputValueId,
        seeds: &[Option<ScalarInputValueId>],
        mut charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        let needed = self.derivative_dependencies(root)?;
        self.primal_checks.push(root.0);
        let instructions = self.instructions[..=root.0.0 as usize].to_vec();
        let bindings = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if seeds.get(index).is_some_and(Option::is_some) {
                    InputBinding::Unknown(index)
                } else {
                    InputBinding::Frozen
                }
            })
            .collect::<Vec<_>>();
        let active = activity::numerical_activity(&instructions, &bindings)?;
        let mut directions = Vec::with_capacity(instructions.len());
        for ((instruction, needed), is_active) in instructions
            .into_iter()
            .zip(needed)
            .zip(active.iter().copied())
        {
            if !needed || !is_active {
                directions.push(None);
                continue;
            }
            let direction = match instruction {
                Instruction::Constant(_)
                | Instruction::PowI(_, 0)
                | Instruction::Compare(..)
                | Instruction::And(..) => None,
                Instruction::Math(function, value) => directions[value.0 as usize]
                    .map(|direction| {
                        real_math::action_in(self, function, ScalarInputValueId(value), direction)
                    })
                    .transpose()?,
                Instruction::Require { condition, value } => {
                    let condition = self.derivative_condition(condition, &active)?;
                    let checked = self.require(condition, ScalarInputValueId(value))?;
                    self.primal_checks.push(checked.0);
                    directions[value.0 as usize]
                        .map(|direction| self.require(condition, direction))
                        .transpose()?
                }
                Instruction::ValueOnly { input, value } => {
                    require_inactive_value_only_input(active[input.0 as usize])?;
                    directions[value.0 as usize]
                        .map(|direction| self.value_only(ScalarInputValueId(input), direction))
                        .transpose()?
                }
                Instruction::Read(slot) => seeds.get(slot.0 as usize).copied().flatten(),
                Instruction::CoefficientJet { start, len, shape } => {
                    let operands = coefficient_jet::operands(
                        start,
                        len,
                        shape,
                        &self.instructions[..directions.len()],
                    )?;
                    let (mut a, mut b) = operands.values(|id| Ok(ScalarInputValueId(id)))?;
                    let (da, db) = operands.values(|id| Ok(directions[id.0 as usize]))?;
                    let da = da.into_iter().map(Option::flatten).collect::<Vec<_>>();
                    let db = db.into_iter().map(Option::flatten).collect::<Vec<_>>();
                    if da.iter().chain(&db).all(Option::is_none) {
                        None
                    } else {
                        a.extend(da);
                        b.extend(db);
                        if shape != coefficient_jet::Shape::SCALAR {
                            charge(
                                a.len()
                                    .checked_mul(4)
                                    .and_then(|n| n.checked_add(2))
                                    .ok_or_else(ir_size_error)?,
                            )?;
                        }
                        Some(self.coefficient_jet(shape, &a, &b)?)
                    }
                }
                Instruction::ComplexDiv {
                    operands: [a, b, c, d],
                    imaginary,
                } => {
                    let da =
                        self.direction_pair(directions[a.0 as usize], directions[b.0 as usize])?;
                    let db =
                        self.direction_pair(directions[c.0 as usize], directions[d.0 as usize])?;
                    complex_quotient::direction_in(
                        self,
                        [ScalarInputValueId(a), ScalarInputValueId(b)],
                        [ScalarInputValueId(c), ScalarInputValueId(d)],
                        da,
                        db,
                    )?
                    .map(|d| d[usize::from(imaginary)])
                }
                Instruction::MapInvariant {
                    start,
                    extent,
                    output,
                } => self.map_direction(start, extent, output, &directions, &mut charge)?,
                Instruction::Neg(_)
                | Instruction::Add(..)
                | Instruction::Sub(..)
                | Instruction::Mul(..)
                | Instruction::Div(..)
                | Instruction::PowI(..) => differential::apply(
                    self,
                    instruction,
                    |id| Ok(ScalarInputValueId(id)),
                    |id| {
                        directions
                            .get(id.0 as usize)
                            .copied()
                            .ok_or_else(|| ir_builder_error("directional operand is unavailable"))
                    },
                )?,
                _ => {
                    return Err(ir_builder_error(
                        "retained derivative instruction is not yet lowered",
                    ));
                }
            };
            directions.push(direction);
        }
        Ok(directions[root.0.0 as usize])
    }
}
