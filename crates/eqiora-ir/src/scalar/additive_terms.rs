//! Algebraic term projection; differentiation remains in the scalar AD owner.
use super::*;
use eqiora_core::ValueLiteral;
use num_rational::BigRational;
use num_traits::{One, Zero};

impl ScalarOperatorIr {
    /// Express the selected-symbol dependence as an additive sum of opaque
    /// scalar terms, each with a coefficient for every original root.
    /// Other inputs are frozen at the typed point. Only addition, subtraction,
    /// negation, and factors independent of the selected symbol are expanded.
    /// Structurally equal terms have equal canonical IR, including across owners.
    /// Active guards remain inside terms; this performs no differentiation.
    /// Original demanded primal/domain validation occurs before cancellation.
    ///
    /// # Errors
    /// Rejects an invalid point, unsupported scalar projection, or exceeded
    /// exact-coefficient budget. Coefficients are exact binary rationals,
    /// including their sums and products. This does not evaluate derivatives.
    pub fn additive_terms(
        &self,
        inputs: &[ValueLiteral],
        selected: SymbolRef,
    ) -> Result<Vec<(Self, Vec<BigRational>)>, Diagnostic> {
        let (projected, roots) = self.project_point(inputs, &self.roots)?;
        let point = self
            .symbols
            .iter()
            .copied()
            .zip(inputs.iter().cloned())
            .collect::<HashMap<_, _>>();
        let (_, trace) =
            projected.evaluate_trace(&roots, &mut |symbol| point.get(&symbol).cloned())?;
        let (projected, roots) = projected.point_components(&trace)?;
        let (_, trace) =
            projected.evaluate_trace(&roots, &mut |symbol| point.get(&symbol).cloned())?;
        let roles = projected
            .symbols
            .iter()
            .map(|&symbol| {
                if symbol == selected {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let dependencies = projected.active_input_dependencies(&trace, &roles)?;
        // A finite binary64 literal needs at most 1076 numerator/denominator
        // bits. Bound each exact coefficient linearly in the admitted projection
        // size; shared squaring must not create exponentially large integers.
        let budget = projected
            .projection_cost()?
            .checked_mul(1076)
            .ok_or_else(|| ir_builder_error("exact coefficient budget overflow"))?;
        let mut frozen_values = vec![None; projected.instructions.len()];
        let mut terms: Vec<(Self, Vec<BigRational>)> = Vec::new();
        for (row, &root) in projected.roots.iter().enumerate() {
            let mut weights = vec![BigRational::zero(); projected.instructions.len()];
            weights[root.0 as usize] = BigRational::one();
            // Accumulate on DAG nodes before visiting their operands. Expanding
            // each path separately is exponential for a shared doubling chain.
            for index in (0..projected.instructions.len()).rev() {
                let scale = std::mem::take(&mut weights[index]);
                if !dependencies[index] || scale.is_zero() {
                    continue;
                }
                let mut frozen = |id: ValueId| {
                    projected.frozen_coefficient(id, &trace, budget, &mut frozen_values)
                };
                let mut push = |id: ValueId, coefficient: BigRational| -> Result<(), Diagnostic> {
                    check_coefficient(&coefficient, budget)?;
                    weights[id.0 as usize] += coefficient;
                    check_coefficient(&weights[id.0 as usize], budget)
                };
                match projected.instructions[index] {
                    Instruction::Add(a, b) => {
                        push(a, scale.clone())?;
                        push(b, scale)?;
                    }
                    Instruction::Sub(a, b) => {
                        push(a, scale.clone())?;
                        push(b, -scale)?;
                    }
                    Instruction::Neg(a) => push(a, -scale)?,
                    Instruction::PowI(a, 1) => push(a, scale)?,
                    Instruction::PowI(_, 0) => {}
                    Instruction::Mul(a, b) if !dependencies[a.0 as usize] => {
                        push(b, scale * frozen(a)?)?
                    }
                    Instruction::Mul(a, b) if !dependencies[b.0 as usize] => {
                        push(a, scale * frozen(b)?)?
                    }
                    Instruction::Div(a, b) if !dependencies[b.0 as usize] => {
                        let denominator = frozen(b)?;
                        if denominator.is_zero() {
                            return Err(ir_builder_error("exact coefficient denominator is zero"));
                        }
                        push(a, scale / denominator)?;
                    }
                    _ => {
                        let atom = projected.canonical_term(
                            ValueId(index as u32),
                            selected,
                            &trace,
                            &dependencies,
                        )?;
                        let at = if let Some(at) =
                            terms.iter().position(|(existing, _)| *existing == atom)
                        {
                            at
                        } else {
                            terms.push((atom, vec![BigRational::zero(); projected.roots.len()]));
                            terms.len() - 1
                        };
                        terms[at].1[row] += scale;
                        check_coefficient(&terms[at].1[row], budget)?;
                    }
                }
            }
        }
        Ok(terms)
    }

    // Exact arithmetic within a frozen coefficient must agree with arithmetic
    // outside the atom. Nonalgebraic primitives and bound inputs retain their
    // ordinary admitted point values; no second calculus is introduced.
    fn frozen_coefficient(
        &self,
        root: ValueId,
        trace: &[Option<ValueLiteral>],
        budget: usize,
        values: &mut [Option<BigRational>],
    ) -> Result<BigRational, Diagnostic> {
        let mut pending = vec![(root, false)];
        while let Some((id, exit)) = pending.pop() {
            let index = id.0 as usize;
            if values[index].is_some() {
                continue;
            }
            let node = self.instructions[index];
            if !exit {
                // Only algebraic coefficient operands need exact values. Opaque
                // primitives and guards retain their already validated trace.
                let children = match node {
                    Instruction::Add(a, b)
                    | Instruction::Sub(a, b)
                    | Instruction::Mul(a, b)
                    | Instruction::Div(a, b) => vec![a, b],
                    Instruction::Neg(a) | Instruction::PowI(a, _) => vec![a],
                    Instruction::Require { value, .. } => vec![value],
                    Instruction::Select {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        match trace[condition.0 as usize]
                            .as_ref()
                            .and_then(ValueLiteral::as_bool)
                        {
                            Some(true) => vec![then_value],
                            Some(false) => vec![else_value],
                            None => Vec::new(),
                        }
                    }
                    _ => Vec::new(),
                };
                if !children.is_empty() {
                    pending.push((id, true));
                    pending.extend(children.into_iter().rev().map(|child| (child, false)));
                    continue;
                }
            }
            let value = {
                let at = |id: ValueId| values[id.0 as usize].as_ref();
                let arithmetic = match node {
                    Instruction::Neg(a) => at(a).map(|value| -value),
                    Instruction::Add(a, b) => at(a).zip(at(b)).map(|(a, b)| a + b),
                    Instruction::Sub(a, b) => at(a).zip(at(b)).map(|(a, b)| a - b),
                    Instruction::Mul(a, b) => at(a).zip(at(b)).map(|(a, b)| a * b),
                    Instruction::Div(a, b) => match at(a).zip(at(b)) {
                        Some((_, b)) if b.is_zero() => {
                            return Err(ir_builder_error("exact coefficient denominator is zero"));
                        }
                        Some((a, b)) => Some(a / b),
                        _ => None,
                    },
                    Instruction::PowI(a, n) => match at(a) {
                        Some(a) => {
                            if n < 0 && a.is_zero() {
                                return Err(ir_builder_error(
                                    "exact coefficient denominator is zero",
                                ));
                            }
                            if !a.is_zero()
                                && a != &BigRational::one()
                                && a != &-BigRational::one()
                                && (a.numer().bits() + a.denom().bits())
                                    .saturating_mul(i64::from(n).unsigned_abs())
                                    > budget as u64
                            {
                                return Err(ir_builder_error("exact coefficient budget exceeded"));
                            }
                            Some(a.pow(n))
                        }
                        _ => None,
                    },
                    Instruction::Require { value, .. } => at(value).cloned(),
                    Instruction::Select {
                        condition,
                        then_value,
                        else_value,
                    } => trace[condition.0 as usize]
                        .as_ref()
                        .and_then(ValueLiteral::as_bool)
                        .and_then(|condition| {
                            at(if condition { then_value } else { else_value }).cloned()
                        }),
                    _ => None,
                };
                arithmetic.or_else(|| {
                    trace[index]
                        .as_ref()?
                        .real_scalar_value()
                        .and_then(|value| BigRational::from_float(value.value()))
                })
            };
            if let Some(value) = &value {
                check_coefficient(value, budget)?;
            }
            values[index] = value;
        }
        values[root.0 as usize]
            .clone()
            .ok_or_else(|| ir_builder_error("exact coefficient requires a demanded scalar value"))
    }

    fn canonical_term(
        &self,
        root: ValueId,
        selected: SymbolRef,
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<Self, Diagnostic> {
        let mut mapped = vec![None; self.instructions.len()];
        let mut pending = vec![(root, false)];
        let mut instructions = Vec::new();
        let mut literals = Vec::new();
        while let Some((id, exit)) = pending.pop() {
            let index = id.0 as usize;
            if mapped[index].is_some() {
                continue;
            }
            let node = if !dependencies[index] {
                let value = trace[index]
                    .as_ref()
                    .ok_or_else(|| ir_builder_error("additive term requires a demanded value"))?;
                if let Some(value) = value.real_scalar_value() {
                    Instruction::Constant(value)
                } else {
                    let slot =
                        if let Some(slot) = literals.iter().position(|literal| literal == value) {
                            slot
                        } else {
                            literals.push(value.clone());
                            literals.len() - 1
                        };
                    Instruction::TypedConstant(slot as u32)
                }
            } else if let Instruction::Read(_) = self.instructions[index] {
                Instruction::Read(SymbolSlot(0))
            } else if !exit {
                let mut children = Vec::new();
                self.instructions[index]
                    .map_scalar_operands(|child| {
                        children.push(child);
                        child
                    })
                    .ok_or_else(|| {
                        ir_builder_error("additive term is outside scalar projection")
                    })?;
                pending.push((id, true));
                pending.extend(children.into_iter().rev().map(|child| (child, false)));
                continue;
            } else {
                self.instructions[index]
                    .map_scalar_operands(|child| {
                        mapped[child.0 as usize].expect("prior term operand")
                    })
                    .ok_or_else(|| ir_builder_error("additive term is outside scalar projection"))?
            };
            let position = if let Some(position) =
                instructions.iter().position(|existing| *existing == node)
            {
                position
            } else {
                instructions.push(node);
                instructions.len() - 1
            };
            mapped[index] = Some(ValueId(position as u32));
        }
        Ok(Self {
            source_values: (0..instructions.len())
                .map(|index| ValueId(index as u32))
                .collect(),
            symbols: vec![selected],
            instructions,
            typed_constants: literals,
            roots: vec![mapped[root.0 as usize].expect("term root")],
            array_operands: Vec::new(),
            definitions: Vec::new(),
        })
    }
}

fn check_coefficient(value: &BigRational, budget: usize) -> Result<(), Diagnostic> {
    if value.numer().bits() + value.denom().bits() > budget as u64 {
        return Err(ir_builder_error("exact coefficient budget exceeded"));
    }
    Ok(())
}
