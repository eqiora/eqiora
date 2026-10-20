//! One total derivative budget for a complete independent-input tuple.
use super::*;

pub(super) fn inputs(
    start: ValueId,
    len: u32,
    end: usize,
) -> Result<std::ops::Range<usize>, Diagnostic> {
    let start = start.0 as usize;
    let limit = start
        .checked_add(len as usize)
        .filter(|limit| *limit <= end)
        .ok_or_else(|| ir_builder_error("derivative contract inputs are unavailable"))?;
    Ok(start..limit)
}

pub(super) fn gather(
    instructions: &mut Vec<Instruction>,
    inputs: impl ExactSizeIterator<Item = ValueId> + Clone,
    value: ValueId,
    order: u32,
) -> Result<Instruction, Diagnostic> {
    let len = u32::try_from(inputs.len()).map_err(|_| ir_size_error())?;
    if value.0 as usize >= instructions.len() {
        return Err(ir_builder_error("derivative contract value is unavailable"));
    }
    let start = slot_composition::gather_entries(instructions, inputs)?;
    Ok(Instruction::DerivativeLimit {
        start,
        len,
        value,
        order,
    })
}

pub(super) fn require_order(available: u32, requested: u32) -> Result<(), Diagnostic> {
    if requested <= available {
        Ok(())
    } else if available == 0 {
        Err(ir_builder_error(
            "value-only contract does not admit independent-input derivative products",
        ))
    } else {
        Err(ir_builder_error(format!(
            "property contract admits derivative order {available}, requested {requested}"
        )))
    }
}

pub(super) fn require_first(
    instructions: &[Instruction],
    active: &[bool],
) -> Result<(), Diagnostic> {
    for (index, instruction) in instructions.iter().enumerate() {
        if let Instruction::DerivativeLimit {
            start, len, order, ..
        } = *instruction
        {
            require_order(
                order,
                u32::from(inputs(start, len, index)?.any(|input| active[input])),
            )?;
        }
    }
    Ok(())
}

impl ScalarLinearization<'_> {
    pub(super) fn require_mixed_contracts(
        &self,
        first: &RelationTangent<'_, f64>,
        second: &RelationTangent<'_, f64>,
    ) -> Result<(), Diagnostic> {
        if !self
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::DerivativeLimit { .. }))
        {
            return Ok(());
        }
        let dependencies = |direction: &RelationTangent<'_, f64>| {
            let bindings = self
                .bindings
                .iter()
                .map(|binding| match (direction, binding) {
                    (
                        RelationTangent::Unknown(_) | RelationTangent::Both { .. },
                        InputBinding::Unknown(_),
                    )
                    | (
                        RelationTangent::Parameter(_) | RelationTangent::Both { .. },
                        InputBinding::Parameter(_),
                    ) => *binding,
                    _ => InputBinding::Frozen,
                })
                .collect::<Vec<_>>();
            activity::numerical_activity(&self.instructions, &bindings)
        };
        let first = dependencies(first)?;
        let second = dependencies(second)?;
        for (index, instruction) in self.instructions.iter().enumerate() {
            if let Instruction::DerivativeLimit {
                start, len, order, ..
            } = *instruction
            {
                let inputs = inputs(start, len, index)?;
                let active = inputs.clone().any(|input| first[input])
                    && inputs.clone().any(|input| second[input]);
                if active {
                    require_order(order, 2)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_symbolic_partial_keeps_its_consumed_derivative_permission() {
        let dimension = eqiora_core::DimExponents::DIMENSIONLESS;
        let mut builder = ScalarInputIrBuilder::new();
        let x = builder.scoped_input(dimension).unwrap();
        let one = builder
            .constant(eqiora_core::DynQuantity::new(1., dimension))
            .unwrap();
        let constant = builder
            .constant(eqiora_core::DynQuantity::new(9., dimension))
            .unwrap();
        let root = builder.derivative_limit(&[x], constant, 1).unwrap();
        let first = builder
            .formal_directional(root, &[Some(one)], dimension, |_| Ok(()))
            .unwrap();
        let error = builder
            .formal_directional(first, &[Some(one)], dimension, |_| Ok(()))
            .unwrap_err();
        assert!(error.message().contains("value-only contract"));
    }
}
