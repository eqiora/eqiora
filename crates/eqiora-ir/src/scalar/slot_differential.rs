//! Directional scalar SSA emission; source coordinates are supplied by the caller.
use super::*;
mod dimensions;
mod validity;

impl ScalarInputIrBuilder {
    pub(crate) fn directional_inputs(
        &self,
        root: ScalarInputValueId,
    ) -> Result<Vec<ScalarInputSlot>, Diagnostic> {
        let needed = slot_composition::required_instructions(&self.instructions, &[root.0])?;
        let mut slots = vec![false; self.slots.len()];
        for (instruction, needed) in self.instructions.iter().zip(needed) {
            if let Instruction::Read(slot) = instruction {
                slots[slot.0 as usize] |= needed;
            }
        }
        self.slots
            .iter()
            .zip(slots)
            .filter(|(_, needed)| *needed)
            .map(|(slot, _)| match slot {
                BuilderInput::Bound(slot) => Ok(slot.clone()),
                BuilderInput::Formal(_) => Err(ir_builder_error(
                    "formal scalar inputs have no Semantic coordinate",
                )),
            })
            .collect()
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
        dimension: eqiora_core::DimExponents,
        charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        self.directional_in(root, seeds, dimension, false, charge)
    }

    /// Formal partials retain authored validity predicates without changing their boundary.
    /// The source value itself is not demanded when its derivative does not need it.
    pub(crate) fn formal_directional(
        &mut self,
        root: ScalarInputValueId,
        seeds: &[Option<ScalarInputValueId>],
        dimension: eqiora_core::DimExponents,
        charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.directional_in(root, seeds, dimension, true, charge)?
            .ok_or_else(|| ir_builder_error("formal derivative lacks its typed result"))
    }

    fn directional_in(
        &mut self,
        root: ScalarInputValueId,
        seeds: &[Option<ScalarInputValueId>],
        dimension: eqiora_core::DimExponents,
        formal: bool,
        mut charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        let needed = if formal {
            slot_composition::required_derivative_instructions(&self.instructions, &[root.0])?
        } else {
            slot_composition::required_instructions(&self.instructions, &[root.0])?
        };
        let mut validity = Vec::new();
        let mut derivative_limits = Vec::new();
        if !formal {
            self.primal_checks.push(root.0);
        }
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
        let mut dimensions = Vec::new();
        for ((instruction, needed), is_active) in instructions
            .into_iter()
            .zip(needed)
            .zip(active.iter().copied())
        {
            if needed
                && let Instruction::DerivativeLimit {
                    start, len, order, ..
                } = instruction
            {
                let inputs = derivative_limit::inputs(start, len, directions.len())?;
                let requested = u32::from(inputs.clone().any(|input| active[input]));
                derivative_limit::require_order(order, requested)?;
                derivative_limits.push((
                    inputs
                        .map(|input| ScalarInputValueId(ValueId(input as u32)))
                        .collect::<Vec<_>>(),
                    order - requested,
                ));
            }
            validity.push(if needed && formal {
                self.formal_validity(directions.len(), &validity)?
            } else {
                None
            });
            if needed
                && formal
                && let Instruction::Select { condition, .. } = instruction
                && active[condition.0 as usize]
            {
                return Err(ir_builder_error(
                    "formal branch selection must be independent of the selected input",
                ));
            }
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
                    if formal {
                        directions[value.0 as usize]
                    } else {
                        let condition = self.derivative_condition(condition, &active)?;
                        let checked = self.require(condition, ScalarInputValueId(value))?;
                        self.primal_checks.push(checked.0);
                        directions[value.0 as usize]
                            .map(|direction| self.require(condition, direction))
                            .transpose()?
                    }
                }
                Instruction::DerivativeLimit { value, .. } => directions[value.0 as usize],
                Instruction::Select {
                    condition,
                    then_value,
                    else_value,
                } if formal => {
                    match (
                        directions[then_value.0 as usize],
                        directions[else_value.0 as usize],
                    ) {
                        (None, None) => None,
                        (Some(then_direction), Some(else_direction)) => Some(self.select(
                            ScalarInputValueId(condition),
                            then_direction,
                            else_direction,
                        )?),
                        (then_direction, else_direction) => {
                            let witness = then_direction.or(else_direction).expect("active branch");
                            let zero = self.zero_like(witness, &mut dimensions)?;
                            Some(self.select(
                                ScalarInputValueId(condition),
                                then_direction.unwrap_or(zero),
                                else_direction.unwrap_or(zero),
                            )?)
                        }
                    }
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
        let direction = directions[root.0.0 as usize];
        if formal || !derivative_limits.is_empty() {
            let mut value = match direction {
                Some(value) => value,
                None => self.constant(eqiora_core::DynQuantity::new(0., dimension))?,
            };
            // Keep validity on the returned value, so an inactive caller branch
            // never acquires an eager global check from a nested formal partial.
            if let Some(condition) = validity[root.0.0 as usize] {
                value = self.require(condition, value)?;
            }
            for (inputs, order) in derivative_limits.into_iter().rev() {
                value = self.derivative_limit(&inputs, value, order)?;
            }
            Ok(Some(value))
        } else {
            Ok(direction)
        }
    }
}

#[cfg(test)]
mod tests;
