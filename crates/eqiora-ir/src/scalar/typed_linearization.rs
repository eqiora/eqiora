//! Point-bound active-program selection through the existing typed demand evaluator.
use super::*;
use eqiora_core::ValueLiteral;

impl ScalarOperatorIr {
    /// Bind a typed real scalar point, rejecting demanded active-input boundaries.
    /// Inactive branches are type-checked but never evaluated or differentiated.
    /// All input symbols and differentiation roles retain their original order.
    ///
    /// # Errors
    /// Rejects wrong input types, invalid active domains, comparison ties, square
    /// root derivatives at zero, and operations outside the scalar derivative profile.
    pub fn linearize_typed(
        &self,
        inputs: &[ValueLiteral],
        roles: &[DifferentiationRole],
    ) -> Result<ScalarLinearization<'_>, Diagnostic> {
        if roles.len() != self.symbols.len() {
            return Err(ir_builder_error(
                "typed linearization role count differs from symbols",
            ));
        }
        let (projected, roots) = self.project_point(inputs, &self.roots)?;
        let point = self
            .symbols
            .iter()
            .copied()
            .zip(inputs.iter().cloned())
            .collect::<HashMap<_, _>>();
        let (_, trace) =
            projected.evaluate_trace(&roots, &mut |symbol| point.get(&symbol).cloned())?;
        let dependencies = projected.active_input_dependencies(&trace, roles)?;
        for (index, node) in projected.instructions.iter().enumerate() {
            if trace[index].is_none() {
                continue;
            }
            match *node {
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
                Instruction::Sqrt(a)
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
        let active = projected.active_program(&trace, &dependencies)?;
        let point = inputs
            .iter()
            .map(|value| {
                value
                    .real_scalar_value()
                    .expect("checked real input")
                    .value()
            })
            .collect::<Vec<_>>();
        let bound = active.linearize(&point, roles)?;
        let ScalarLinearization {
            inputs,
            bindings,
            unknown_dimension,
            parameter_dimension,
            ..
        } = bound;
        Ok(ScalarLinearization {
            ir: std::borrow::Cow::Owned(active),
            inputs,
            bindings,
            unknown_dimension,
            parameter_dimension,
        })
    }

    // A branch boundary constrains derivatives only when its predicate can
    // vary with this admitted input map. Frozen values still pass ordinary
    // domain validation in evaluate_trace, but contribute no derivative inputs.
    fn active_input_dependencies(
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
                    Instruction::Constant(_) | Instruction::TypedConstant(_) => false,
                    Instruction::Read(slot) => {
                        roles[slot.0 as usize] != DifferentiationRole::Frozen
                    }
                    Instruction::Neg(a)
                    | Instruction::PowI(a, _)
                    | Instruction::Sin(a)
                    | Instruction::Sqrt(a)
                    | Instruction::Not(a) => at(a),
                    Instruction::Add(a, b)
                    | Instruction::Sub(a, b)
                    | Instruction::Mul(a, b)
                    | Instruction::Div(a, b)
                    | Instruction::Compare(_, a, b)
                    | Instruction::And(a, b)
                    | Instruction::Or(a, b) => at(a) || at(b),
                    Instruction::Require { condition, value } => at(condition) || at(value),
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

    fn active_program(
        &self,
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<Self, Diagnostic> {
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
                Instruction::Require { value, .. } => Some(value),
                _ => None,
            }
        };
        let mut needed = vec![false; self.instructions.len()];
        let mut pending = self.roots.clone();
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
                Instruction::Neg(a)
                | Instruction::PowI(a, _)
                | Instruction::Sin(a)
                | Instruction::Sqrt(a) => pending.push(a),
                Instruction::Add(a, b)
                | Instruction::Sub(a, b)
                | Instruction::Mul(a, b)
                | Instruction::Div(a, b) => pending.extend([a, b]),
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
            let node = match node {
                Instruction::Neg(a) => Instruction::Neg(at(a)),
                Instruction::PowI(a, n) => Instruction::PowI(at(a), n),
                Instruction::Sin(a) => Instruction::Sin(at(a)),
                Instruction::Sqrt(a) => Instruction::Sqrt(at(a)),
                Instruction::Add(a, b) => Instruction::Add(at(a), at(b)),
                Instruction::Sub(a, b) => Instruction::Sub(at(a), at(b)),
                Instruction::Mul(a, b) => Instruction::Mul(at(a), at(b)),
                Instruction::Div(a, b) => Instruction::Div(at(a), at(b)),
                Instruction::Read(_) | Instruction::Constant(_) => node,
                _ => return Err(ir_builder_error("active scalar instruction is unsupported")),
            };
            mapped[index] = Some(ValueId(instructions.len() as u32));
            instructions.push(node);
        }
        Ok(Self {
            source_values: (0..instructions.len()).map(|i| ValueId(i as u32)).collect(),
            typed_constants: Vec::new(),
            array_operands: Vec::new(),
            definitions: Vec::new(),
            symbols: self.symbols.clone(),
            roots: self
                .roots
                .iter()
                .map(|id| mapped[id.0 as usize].expect("active root"))
                .collect(),
            instructions,
        })
    }
}
