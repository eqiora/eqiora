//! Substitute scalar inputs without evaluating values or losing demanded domains.
use super::*;

/// Gather scalar coordinates without changing units or signed zero. Both initial
/// lowering and active-program projection use this contiguous operand convention.
pub(super) fn gather_entries(
    instructions: &mut Vec<Instruction>,
    entries: impl ExactSizeIterator<Item = ValueId> + Clone,
) -> Result<ValueId, Diagnostic> {
    let end = instructions.len();
    let limit = end
        .checked_add(entries.len())
        .and_then(|n| n.checked_add(1))
        .ok_or_else(ir_size_error)?;
    u32::try_from(limit).map_err(|_| ir_size_error())?;
    if entries.clone().any(|entry| entry.0 as usize >= end) {
        return Err(ir_builder_error(
            "scalar gather requires prior scalar operands",
        ));
    }
    let one = ValueId(end as u32);
    instructions.push(Instruction::Constant(eqiora_core::DynQuantity::new(
        1.,
        eqiora_core::DimExponents::DIMENSIONLESS,
    )));
    let start = ValueId(one.0 + 1);
    instructions.extend(entries.map(|entry| Instruction::Mul(entry, one)));
    Ok(start)
}

/// Structural operand closure shared by directional emission and substitution.
pub(super) fn required_instructions(
    instructions: &[Instruction],
    roots: &[ValueId],
) -> Result<Vec<bool>, Diagnostic> {
    required_with_conditions(instructions, roots, true)
}

/// Formal differentiation retains a Require predicate verbatim, without traversing
/// its inactive Boolean branches as separate derivative/domain obligations.
pub(super) fn required_derivative_instructions(
    instructions: &[Instruction],
    roots: &[ValueId],
) -> Result<Vec<bool>, Diagnostic> {
    required_with_conditions(instructions, roots, false)
}

fn required_with_conditions(
    instructions: &[Instruction],
    roots: &[ValueId],
    include_conditions: bool,
) -> Result<Vec<bool>, Diagnostic> {
    let mut needed = vec![false; instructions.len()];
    for root in roots {
        *needed
            .get_mut(root.0 as usize)
            .ok_or_else(|| ir_builder_error("scalar root is unavailable"))? = true;
    }
    for index in (0..needed.len()).rev() {
        if !needed[index] {
            continue;
        }
        let instruction = instructions[index];
        if !include_conditions && let Instruction::Require { value, .. } = instruction {
            needed[value.0 as usize] = true;
        } else {
            visit_instruction_operands(instructions, index, |operand| {
                needed[operand.0 as usize] = true;
            })?;
        }
    }
    Ok(needed)
}

/// Visit all operands, including contiguous primitive input tuples.
/// Dependency walks and relocation consumers share this structural authority.
pub(super) fn visit_instruction_operands(
    instructions: &[Instruction],
    index: usize,
    mut visit: impl FnMut(ValueId),
) -> Result<(), Diagnostic> {
    match instructions[index] {
        Instruction::DerivativeLimit {
            start, len, value, ..
        } => {
            for input in derivative_limit::inputs(start, len, index)? {
                visit(ValueId(input as u32));
            }
            visit(value);
        }
        Instruction::CoefficientJet { start, len, shape } => {
            for operand in
                coefficient_jet::operands(start, len, shape, &instructions[..index])?.iter()
            {
                visit(operand);
            }
        }
        Instruction::MapInvariant { start, extent, .. } => {
            for input in map_evaluation::operand_range(start, extent, index)? {
                visit(ValueId(input as u32));
            }
        }
        instruction => {
            instruction
                .map_scalar_operands(|id| {
                    visit(id);
                    id
                })
                .ok_or_else(|| ir_builder_error("scalar instruction has no scalar operands"))?;
        }
    }
    Ok(())
}

/// Conservative allocation work for relocation, including primitive gathers.
pub(super) fn relocation_work(instructions: &[Instruction]) -> Result<usize, Diagnostic> {
    instructions
        .iter()
        .try_fold(0usize, |work, instruction| {
            let extra = match instruction {
                Instruction::DerivativeLimit { len, .. } => (*len as usize).checked_add(1),
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
        .ok_or_else(ir_size_error)
}

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
        Instruction::DerivativeLimit {
            start,
            len,
            value,
            order,
        } => {
            let inputs =
                derivative_limit::inputs(start, len, index)?.map(|input| at(ValueId(input as u32)));
            derivative_limit::gather(instructions, inputs, at(value), order)?
        }
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
                let mapped = slot_composition::gather_entries(instructions, entries)?;
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
        charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarInputValueId>, Diagnostic> {
        require_length(inputs, source.slots.len(), "scalar composition inputs")?;
        for &input in inputs {
            self.require_prior(input)?;
        }
        self.compose_parts(
            &source.instructions,
            &source.typed_constants,
            &source.primal_checks,
            &source.roots,
            inputs,
            charge,
        )
    }

    /// Create an input local to a calculus scope, without a fabricated Semantic identity.
    pub(crate) fn scoped_input(
        &mut self,
        dimension: eqiora_core::DimExponents,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let index = u32::try_from(self.slots.len()).map_err(|_| ir_size_error())?;
        self.slots.push(BuilderInput::Formal(dimension));
        self.push(Instruction::Read(SymbolSlot(index)))
    }

    /// Bind a completed scope only after its independent formal actions are emitted.
    pub(crate) fn compose_scoped(
        &mut self,
        source: Self,
        roots: &[ScalarInputValueId],
        inputs: &[(ScalarInputValueId, eqiora_core::DimExponents)],
        charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarInputValueId>, Diagnostic> {
        require_length(inputs, source.slots.len(), "scoped scalar inputs")?;
        for (slot, &(input, dimension)) in source.slots.iter().zip(inputs) {
            match slot {
                BuilderInput::Formal(expected) if *expected == dimension => {}
                _ => {
                    return Err(ir_builder_error(
                        "scoped scalar input has a different type or owner",
                    ));
                }
            }
            self.require_prior(input)?;
        }
        for &root in roots {
            source.require_prior(root)?;
        }
        let roots = roots.iter().map(|id| id.0).collect::<Vec<_>>();
        let inputs = inputs.iter().map(|(value, _)| *value).collect::<Vec<_>>();
        self.compose_parts(
            &source.instructions,
            &source.typed_constants,
            &source.primal_checks,
            &roots,
            &inputs,
            charge,
        )
    }

    fn compose_parts(
        &mut self,
        instructions: &[Instruction],
        typed_constants: &[eqiora_core::ValueLiteral],
        primal_checks: &[ValueId],
        roots: &[ValueId],
        inputs: &[ScalarInputValueId],
        mut charge: impl FnMut(usize) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarInputValueId>, Diagnostic> {
        // Include every copied instruction and the worst-case primitive gathers.
        // Charge before allocation/emission against the caller's existing budget.
        let work = relocation_work(instructions)?;
        charge(work)?;
        let demands = roots
            .iter()
            .chain(primal_checks)
            .copied()
            .collect::<Vec<_>>();
        let needed = required_instructions(instructions, &demands)?;
        let mut mapped = Vec::with_capacity(instructions.len());
        let mut map_entries = HashMap::new();
        for (index, instruction) in instructions.iter().enumerate() {
            if !needed[index] {
                mapped.push(None);
                continue;
            }
            let value = if let Instruction::Read(slot) = instruction {
                inputs[slot.0 as usize]
            } else if let Instruction::TypedConstant(literal) = instruction {
                self.typed_constant(typed_constants[*literal as usize].clone())?
            } else {
                let instruction = remap_instruction(
                    instructions,
                    index,
                    &mut self.instructions,
                    &mut map_entries,
                    |id| mapped[id.0 as usize].expect("demanded scalar operand"),
                )?;
                self.push(instruction)?
            };
            mapped.push(Some(value.0));
        }
        self.primal_checks.extend(
            primal_checks
                .iter()
                .map(|id| mapped[id.0 as usize].expect("demanded primal check")),
        );
        Ok(roots
            .iter()
            .map(|id| ScalarInputValueId(mapped[id.0 as usize].expect("demanded scalar root")))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::UnaryMathFunction;

    #[test]
    fn scoped_partial_precedes_aliased_input_substitution_and_keeps_scaled_division() {
        let dimension = DimExponents::DIMENSIONLESS;
        let mut scope = ScalarInputIrBuilder::new();
        let numerator = scope.scoped_input(dimension).unwrap();
        let denominator = scope.scoped_input(dimension).unwrap();
        let quotient = scope.div(numerator, denominator).unwrap();
        let one = scope.constant(DynQuantity::new(1., dimension)).unwrap();
        let partial = scope
            .formal_directional(quotient, &[None, Some(one)], dimension, |_| Ok(()))
            .unwrap();

        let mut destination = ScalarInputIrBuilder::new();
        let same_value = destination
            .constant(DynQuantity::new(1e-200, dimension))
            .unwrap();
        let roots = destination
            .compose_scoped(
                scope,
                &[partial],
                &[(same_value, dimension), (same_value, dimension)],
                |_| Ok(()),
            )
            .unwrap();
        let result = destination.finish(roots).unwrap().evaluate(&[]).unwrap();
        // The independent partial d(a/b)/db is -1e200 at a=b=1e-200.
        // Substituting a=b before differentiation would incorrectly give zero;
        // materializing b² would underflow and lose the representable result.
        assert!((result[0] / -1e200 - 1.).abs() <= 16. * f64::EPSILON);
    }

    #[test]
    fn scoped_inputs_cannot_escape_or_acquire_a_different_dimension() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let mut scope = ScalarInputIrBuilder::new();
        let input = scope.scoped_input(length).unwrap();
        assert!(scope.finish([input]).is_err());

        let mut scope = ScalarInputIrBuilder::new();
        let input = scope.scoped_input(length).unwrap();
        let mut destination = ScalarInputIrBuilder::new();
        let value = destination
            .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
            .unwrap();
        assert!(
            destination
                .compose_scoped(
                    scope,
                    &[input],
                    &[(value, DimExponents::DIMENSIONLESS)],
                    |_| Ok(())
                )
                .is_err()
        );
    }

    #[test]
    fn scoped_substitution_omits_unrequested_overflow_but_retains_explicit_checks() {
        for demand_original in [false, true] {
            let dimension = DimExponents::DIMENSIONLESS;
            let mut scope = ScalarInputIrBuilder::new();
            let x = scope.scoped_input(dimension).unwrap();
            let square = scope.mul(x, x).unwrap();
            let two = scope.constant(DynQuantity::new(2., dimension)).unwrap();
            let partial = scope.mul(two, x).unwrap();
            if demand_original {
                scope.primal_checks.push(square.0);
            }
            let mut destination = ScalarInputIrBuilder::new();
            let x = destination
                .constant(DynQuantity::new(1e200, dimension))
                .unwrap();
            let roots = destination
                .compose_scoped(scope, &[partial], &[(x, dimension)], |_| Ok(()))
                .unwrap();
            let result = destination.finish(roots).unwrap().evaluate(&[]);
            if demand_original {
                assert!(result.is_err());
            } else {
                // d(x²)/dx=2x is finite here, although x² is not representable.
                assert_eq!(result.unwrap(), [2e200]);
            }
        }
    }

    #[test]
    fn composition_relocates_boolean_constants_and_keeps_inactive_domains_lazy() {
        let mut source = ScalarInputIrBuilder::new();
        let condition = source.boolean(false).unwrap();
        let minus_one = source
            .constant(DynQuantity::new(-1., DimExponents::DIMENSIONLESS))
            .unwrap();
        let invalid = source
            .unary_math(UnaryMathFunction::Sqrt, minus_one)
            .unwrap();
        let seven = source
            .constant(DynQuantity::new(7., DimExponents::DIMENSIONLESS))
            .unwrap();
        let root = source.select(condition, invalid, seven).unwrap();
        let source = source.finish([root]).unwrap();
        assert_eq!(source.evaluate(&[]).unwrap(), [7.]);

        let mut destination = ScalarInputIrBuilder::new();
        // Index zero in the destination deliberately has the opposite truth value.
        let valid = destination.boolean(true).unwrap();
        let root = destination.compose(&source, &[], |_| Ok(())).unwrap()[0];
        let root = destination.require(valid, root).unwrap();
        let program = destination.finish([root]).unwrap();
        assert_eq!(program.evaluate(&[]).unwrap(), [7.]);
        assert!(program.linearize(&[], &[]).is_ok());
    }
}
