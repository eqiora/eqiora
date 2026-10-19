//! Point-bound active-program selection through the existing typed demand evaluator.
use super::*;
use eqiora_core::ValueLiteral;

impl ScalarOperatorIr {
    /// Bind a typed point, rejecting demanded active-input boundaries.
    /// Inactive branches are type-checked but never evaluated or differentiated.
    /// Active inputs must be real scalars. Frozen inputs retain their exact typed
    /// values (including nominal discrete outcomes) and have no tangent space.
    /// Active input symbols and roles retain their original order.
    ///
    /// # Errors
    /// Rejects wrong input types, invalid active domains, comparison ties, square
    /// root derivatives at zero, and operations outside the scalar derivative profile.
    pub fn linearize_typed(
        &self,
        inputs: &[ValueLiteral],
        roles: &[DifferentiationRole],
    ) -> Result<ScalarLinearization<'_>, Diagnostic> {
        if roles.len() != self.symbols.len() || inputs.len() != self.symbols.len() {
            return Err(ir_builder_error(
                "typed linearization point/role count differs from symbols",
            ));
        }
        let frozen = roles.contains(&DifferentiationRole::Frozen);
        let bound = if frozen {
            std::borrow::Cow::Owned(self.freeze_inputs(inputs, roles)?)
        } else {
            std::borrow::Cow::Borrowed(self)
        };
        let active_inputs = inputs
            .iter()
            .zip(roles)
            .filter(|(_, role)| **role != DifferentiationRole::Frozen)
            .map(|(value, _)| value.clone())
            .collect::<Vec<_>>();
        let roles = roles
            .iter()
            .copied()
            .filter(|role| *role != DifferentiationRole::Frozen)
            .collect::<Vec<_>>();
        let (projected, roots) = bound.project_point(&active_inputs, &bound.roots)?;
        let point = bound
            .symbols
            .iter()
            .copied()
            .zip(active_inputs.iter().cloned())
            .collect::<HashMap<_, _>>();
        let (_, trace) =
            projected.evaluate_trace(&roots, &mut |symbol| point.get(&symbol).cloned())?;
        // Validate all demanded typed values before choosing components: an
        // Array eagerly evaluates even its nonselected elements.
        let (projected, roots) = projected.point_components(&trace)?;
        let (_, trace) =
            projected.evaluate_trace(&roots, &mut |symbol| point.get(&symbol).cloned())?;
        let dependencies = projected.active_input_dependencies(&trace, &roles)?;
        projected
            .typed_instructions()
            .require_regular_active_trace(&trace, &dependencies)?;
        let active = projected.active_program(&trace, &dependencies)?;
        let point = active_inputs
            .iter()
            .map(|value| {
                value
                    .real_scalar_value()
                    .expect("checked real input")
                    .value()
            })
            .collect::<Vec<_>>();
        let bound = active.linearize(&point, &roles)?;
        let ScalarLinearization {
            inputs,
            bindings,
            active: activity,
            unknown_dimension,
            parameter_dimension,
            ..
        } = bound;
        Ok(ScalarLinearization {
            instructions: std::borrow::Cow::Owned(active.instructions),
            roots: std::borrow::Cow::Owned(active.roots),
            inputs,
            bindings,
            active: activity,
            unknown_dimension,
            parameter_dimension,
        })
    }

    // Use the canonical typed expression projection, never an ordinal or a
    // fabricated floating-point placeholder for a frozen discrete input.
    fn freeze_inputs(
        &self,
        inputs: &[ValueLiteral],
        roles: &[DifferentiationRole],
    ) -> Result<Self, Diagnostic> {
        self.projection_cost()?;
        let mut builder = eqiora_schema::kernel::ExprDagBuilder::new();
        let mut mapped = Vec::with_capacity(self.instructions.len());
        for &instruction in &self.instructions {
            let value = match instruction {
                Instruction::Read(slot)
                    if roles[slot.0 as usize] == DifferentiationRole::Frozen =>
                {
                    builder.constant(inputs[slot.0 as usize].clone())?
                }
                _ => {
                    self.append_instruction(&mut builder, instruction, |id| mapped[id.0 as usize])?
                }
            };
            mapped.push(value);
        }
        let bound =
            Self::lower(&builder.finish(self.roots.iter().map(|id| mapped[id.0 as usize]))?)?;
        if !bound.symbols.iter().copied().eq(self
            .symbols
            .iter()
            .copied()
            .zip(roles)
            .filter(|(_, role)| **role != DifferentiationRole::Frozen)
            .map(|(symbol, _)| symbol))
        {
            return Err(ir_builder_error(
                "frozen input projection changed active input order",
            ));
        }
        Ok(bound)
    }

    pub(super) fn active_input_dependencies(
        &self,
        trace: &[Option<ValueLiteral>],
        roles: &[DifferentiationRole],
    ) -> Result<Vec<bool>, Diagnostic> {
        self.typed_instructions()
            .active_input_dependencies(trace, roles)
    }

    fn active_program(
        &self,
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<Self, Diagnostic> {
        let (instructions, roots) =
            self.typed_instructions()
                .active_program(&self.roots, trace, dependencies)?;
        Ok(Self {
            typed_inputs: None,
            source_values: (0..instructions.len()).map(|i| ValueId(i as u32)).collect(),
            typed_constants: Vec::new(),
            array_operands: Vec::new(),
            definitions: Vec::new(),
            symbols: self.symbols.clone(),
            roots,
            instructions,
        })
    }
}

impl typed::TypedInstructions<'_> {
    pub(super) fn require_regular_active_trace(
        &self,
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<(), Diagnostic> {
        for (index, node) in self.instructions.iter().enumerate() {
            if trace[index].is_none() {
                continue;
            }
            match *node {
                Instruction::ValueOnly { input, .. } => {
                    require_inactive_value_only_input(dependencies[input.0 as usize])?;
                }
                Instruction::Compare(_, a, b) => {
                    let a = trace[a.0 as usize].as_ref().expect("demanded comparison");
                    let b = trace[b.0 as usize].as_ref().expect("demanded comparison");
                    if dependencies[index]
                        && a.real_scalar_value().is_some()
                        && b.real_scalar_value().is_some()
                        && a.checked_equal(b)
                            .map_err(|e| ir_builder_error(e.to_string()))?
                    {
                        return Err(ir_builder_error(
                            "derivative is undefined at a demanded comparison boundary",
                        ));
                    }
                }
                Instruction::Math(eqiora_schema::kernel::UnaryMathFunction::Sqrt, a)
                    if dependencies[index]
                        && trace[a.0 as usize]
                            .as_ref()
                            .and_then(ValueLiteral::real_scalar_value)
                            .is_some_and(|value| value.value() == 0.) =>
                {
                    return Err(ir_builder_error(
                        "square-root derivative is undefined at zero",
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    // A branch boundary constrains derivatives only when its predicate can
    // vary with this admitted input map. Frozen values still pass ordinary
    // domain validation in evaluate_trace, but contribute no derivative inputs.
    pub(super) fn active_input_dependencies(
        &self,
        trace: &[Option<ValueLiteral>],
        roles: &[DifferentiationRole],
    ) -> Result<Vec<bool>, Diagnostic> {
        let mut dependencies = Vec::with_capacity(self.instructions.len());
        for (index, node) in self.instructions.iter().enumerate() {
            let depends = if trace[index].is_none() {
                false
            } else {
                let at = |id: ValueId| dependencies[id.0 as usize];
                match *node {
                    Instruction::ComplexDiv { operands, .. } => operands.into_iter().any(at),
                    Instruction::CoefficientJet { start, len, shape } => {
                        coefficient_jet::operands(start, len, shape, &self.instructions[..index])?
                            .iter()
                            .any(at)
                    }
                    Instruction::MapInvariant { start, extent, .. } => {
                        map_evaluation::operand_range(start, extent, index)?
                            .any(|operand| dependencies[operand])
                    }
                    Instruction::Constant(_) | Instruction::TypedConstant(_) => false,
                    Instruction::Read(slot) => {
                        roles[slot.0 as usize] != DifferentiationRole::Frozen
                    }
                    Instruction::Neg(a)
                    | Instruction::PowI(a, _)
                    | Instruction::Math(_, a)
                    | Instruction::Not(a) => at(a),
                    Instruction::Add(a, b)
                    | Instruction::Sub(a, b)
                    | Instruction::Mul(a, b)
                    | Instruction::Div(a, b)
                    | Instruction::Compare(_, a, b)
                    | Instruction::And(a, b)
                    | Instruction::Or(a, b) => at(a) || at(b),
                    Instruction::Require { condition, value } => at(condition) || at(value),
                    Instruction::ValueOnly { input, value } => at(input) || at(value),
                    Instruction::Select {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        at(condition)
                            || at(
                                if trace[condition.0 as usize]
                                    .as_ref()
                                    .and_then(ValueLiteral::as_bool)
                                    .expect("typed demanded selector")
                                {
                                    then_value
                                } else {
                                    else_value
                                },
                            )
                    }
                    Instruction::Index(a, _)
                    | Instruction::ToReal(a)
                    | Instruction::ToInteger(a)
                    | Instruction::Ordinal(a) => at(a),
                    Instruction::Quotient(a, b) | Instruction::Remainder(a, b) => at(a) || at(b),
                    Instruction::Array { start, len }
                    | Instruction::PureOperator { start, len, .. } => self.array_operands
                        [start as usize..start as usize + len as usize]
                        .iter()
                        .any(|id| at(*id)),
                }
            };
            dependencies.push(depends);
        }
        Ok(dependencies)
    }

    pub(super) fn active_program(
        &self,
        roots: &[ValueId],
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<(Vec<Instruction>, Vec<ValueId>), Diagnostic> {
        let chosen = |node: Instruction| -> Option<ValueId> {
            match node {
                Instruction::Select {
                    condition,
                    then_value,
                    else_value,
                } => Some(if trace[condition.0 as usize].as_ref()?.as_bool()? {
                    then_value
                } else {
                    else_value
                }),
                Instruction::Require { value, .. } | Instruction::ValueOnly { value, .. } => {
                    Some(value)
                }
                _ => None,
            }
        };
        let mut needed = vec![false; self.instructions.len()];
        let mut pending = roots.to_vec();
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut needed[id.0 as usize], true) {
                continue;
            }
            // Frozen demanded values use the typed evaluator's exact result.
            // Stop here: their typed operands need no derivative projection.
            if !dependencies[id.0 as usize] {
                continue;
            }
            let node = self.instructions[id.0 as usize];
            if let Some(value) = chosen(node) {
                pending.push(value);
                continue;
            }
            match node {
                Instruction::Neg(a) | Instruction::PowI(a, _) | Instruction::Math(_, a) => {
                    pending.push(a)
                }
                Instruction::Add(a, b)
                | Instruction::Sub(a, b)
                | Instruction::Mul(a, b)
                | Instruction::Div(a, b) => pending.extend([a, b]),
                Instruction::ComplexDiv { operands, .. } => pending.extend(operands),
                Instruction::CoefficientJet { start, len, shape } => pending.extend(
                    coefficient_jet::operands(
                        start,
                        len,
                        shape,
                        &self.instructions[..id.0 as usize],
                    )?
                    .iter(),
                ),
                Instruction::MapInvariant { start, extent, .. } => {
                    pending.extend(
                        map_evaluation::operand_range(start, extent, id.0 as usize)?
                            .map(|operand| ValueId(operand as u32)),
                    );
                }
                Instruction::Constant(_) | Instruction::Read(_) => {}
                _ => {
                    return Err(ir_builder_error(
                        "active derivative value is outside the real scalar profile",
                    ));
                }
            }
        }
        let mut mapped = vec![None; self.instructions.len()];
        let mut instructions = Vec::new();
        let mut map_entries = HashMap::new();
        for (index, node) in self.instructions.iter().copied().enumerate() {
            if !needed[index] {
                continue;
            }
            if !dependencies[index] {
                let value = trace[index]
                    .as_ref()
                    .and_then(ValueLiteral::real_scalar_value)
                    .ok_or_else(|| {
                        ir_builder_error("active residual requires a real scalar value")
                    })?;
                mapped[index] = Some(ValueId(instructions.len() as u32));
                instructions.push(Instruction::Constant(value));
                continue;
            }
            let at = |id: ValueId| mapped[id.0 as usize].expect("prior active operand");
            if let Some(value) = chosen(node) {
                mapped[index] = Some(at(value));
                continue;
            }
            let node = slot_composition::remap_instruction(
                self.instructions,
                index,
                &mut instructions,
                &mut map_entries,
                at,
            )?;
            mapped[index] = Some(ValueId(instructions.len() as u32));
            instructions.push(node);
        }
        let roots = roots
            .iter()
            .map(|id| mapped[id.0 as usize].expect("active root"))
            .collect();
        Ok((instructions, roots))
    }
}
