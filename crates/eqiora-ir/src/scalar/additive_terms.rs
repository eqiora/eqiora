//! Algebraic term projection; differentiation remains in the scalar AD owner.
use super::*;
use eqiora_core::ValueLiteral;
use num_rational::BigRational;
use num_traits::{One, Zero};

impl ScalarOperatorIr {
    /// Express the selected-symbol dependence as an additive sum of opaque
    /// scalar terms, each with a coefficient for every original root.
    /// Other inputs are frozen at the typed point. Only addition, subtraction,
    /// negation, and factors independent of the selected symbols are expanded.
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
        selected: &[SymbolRef],
    ) -> Result<Vec<(Self, Vec<BigRational>)>, Diagnostic> {
        let projected = self.project_point(inputs, &self.roots)?;
        let point = self
            .symbols
            .iter()
            .copied()
            .zip(inputs.iter().cloned())
            .collect::<HashMap<_, _>>();
        let (_, trace) = projected
            .evaluate_value_trace(&projected.roots, &mut |symbol| point.get(&symbol).cloned())?;
        let projected = projected.point_components(&trace)?;
        let (_, trace) = projected
            .evaluate_value_trace(&projected.roots, &mut |symbol| point.get(&symbol).cloned())?;
        let roles = projected
            .symbols
            .iter()
            .map(|&symbol| {
                if selected.contains(&symbol) {
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
        let inputs = projected
            .symbols
            .iter()
            .copied()
            .zip(
                projected
                    .typed_inputs
                    .as_ref()
                    .ok_or_else(|| {
                        ir_builder_error("additive projection requires admitted input types")
                    })?
                    .iter()
                    .cloned(),
            )
            .collect::<Vec<_>>();
        AdditiveProgram {
            instructions: &projected.instructions,
            roots: &projected.roots,
            inputs: &inputs,
        }
        .project(&trace, &dependencies, budget)?
        .into_iter()
        .map(|(atom, coefficients)| {
            let (symbols, types) = atom.inputs.into_iter().unzip();
            Ok((
                Self {
                    typed_inputs: Some(types),
                    source_values: (0..atom.instructions.len())
                        .map(|index| Some(ValueId(index as u32)))
                        .collect(),
                    symbols,
                    instructions: atom.instructions,
                    typed_constants: atom.literals,
                    roots: vec![atom.root],
                    array_operands: Vec::new(),
                    definitions: Vec::new(),
                },
                coefficients,
            ))
        })
        .collect()
    }
}

impl ScalarInputOperatorIr {
    pub(crate) fn additive_terms(
        &self,
        values: &[f64],
        selected: &[ScalarSymbolCoordinate],
    ) -> Result<Vec<(Self, BigRational)>, Diagnostic> {
        if self.roots.len() != 1 {
            return Err(ir_builder_error(
                "component additive projection requires one root",
            ));
        }
        let point = self.typed_point(values)?;
        let program = self.typed_instructions();
        let (_, trace) = program.evaluate(&self.evaluation_roots(), &mut |slot| {
            point.get(slot.0 as usize).cloned()
        })?;
        let source = self.point_components(&trace)?;
        let program = source.typed_instructions();
        let (_, trace) = program.evaluate(&source.evaluation_roots(), &mut |slot| {
            point.get(slot.0 as usize).cloned()
        })?;
        let roles = source
            .slots
            .iter()
            .map(|slot| {
                if selected.contains(slot.source()) {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let dependencies = program.active_input_dependencies(&trace, &roles)?;
        let work = slot_composition::relocation_work(&source.instructions)?;
        if work > 1_000_000 {
            return Err(ir_builder_error("typed projection work budget exceeded"));
        }
        let budget = work
            .checked_mul(1076)
            .ok_or_else(|| ir_builder_error("exact coefficient budget overflow"))?;
        let inputs = source
            .slots
            .iter()
            .map(|slot| {
                Ok((
                    slot.source().clone(),
                    eqiora_core::ValueType::scalar(
                        eqiora_core::ScalarDomain::Real,
                        slot.dimension(),
                    )
                    .map_err(|error| ir_builder_error(error.to_string()))?,
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(AdditiveProgram {
            instructions: &source.instructions,
            roots: &source.roots,
            inputs: &inputs,
        }
        .project(&trace, &dependencies, budget)?
        .into_iter()
        .map(|(atom, coefficients)| {
            let slots = atom
                .inputs
                .into_iter()
                .enumerate()
                .map(|(index, (source, ty))| {
                    ScalarInputSlot::new(index as u32, source, ty.dimension())
                })
                .collect();
            (
                Self {
                    primal_checks: Vec::new(),
                    slots,
                    instructions: atom.instructions,
                    typed_constants: atom.literals,
                    roots: vec![atom.root],
                },
                coefficients
                    .into_iter()
                    .next()
                    .expect("single component root"),
            )
        })
        .collect())
    }
}

#[derive(PartialEq)]
struct ProjectedAtom<K> {
    inputs: Vec<(K, eqiora_core::ValueType)>,
    instructions: Vec<Instruction>,
    literals: Vec<ValueLiteral>,
    root: ValueId,
}

type ProjectedTerms<K> = Vec<(ProjectedAtom<K>, Vec<BigRational>)>;

struct AdditiveProgram<'a, K> {
    inputs: &'a [(K, eqiora_core::ValueType)],
    instructions: &'a [Instruction],
    roots: &'a [ValueId],
}

impl<K: Clone + PartialEq> AdditiveProgram<'_, K> {
    fn project(
        &self,
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
        budget: usize,
    ) -> Result<ProjectedTerms<K>, Diagnostic> {
        let mut frozen_values = vec![None; self.instructions.len()];
        let mut terms: Vec<(ProjectedAtom<K>, Vec<BigRational>)> = Vec::new();
        for (row, &root) in self.roots.iter().enumerate() {
            let mut weights = vec![BigRational::zero(); self.instructions.len()];
            weights[root.0 as usize] = BigRational::one();
            // Accumulate on DAG nodes before visiting their operands. Expanding
            // each path separately is exponential for a shared doubling chain.
            for index in (0..self.instructions.len()).rev() {
                let scale = std::mem::take(&mut weights[index]);
                if !dependencies[index] || scale.is_zero() {
                    continue;
                }
                let mut frozen =
                    |id: ValueId| self.frozen_coefficient(id, trace, budget, &mut frozen_values);
                let mut push = |id: ValueId, coefficient: BigRational| -> Result<(), Diagnostic> {
                    check_coefficient(&coefficient, budget)?;
                    weights[id.0 as usize] += coefficient;
                    check_coefficient(&weights[id.0 as usize], budget)
                };
                match self.instructions[index] {
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
                        let atom =
                            self.canonical_term(ValueId(index as u32), trace, dependencies)?;
                        let at = if let Some(at) =
                            terms.iter().position(|(existing, _)| *existing == atom)
                        {
                            at
                        } else {
                            terms.push((atom, vec![BigRational::zero(); self.roots.len()]));
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
                    Instruction::Require { value, .. } | Instruction::Demand { value, .. } => {
                        vec![value]
                    }
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
                    Instruction::Require { value, .. } | Instruction::Demand { value, .. } => {
                        at(value).cloned()
                    }
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
        trace: &[Option<ValueLiteral>],
        dependencies: &[bool],
    ) -> Result<ProjectedAtom<K>, Diagnostic> {
        let mut inputs = Vec::new();
        let mut mapped = vec![None; self.instructions.len()];
        let mut pending = vec![(root, false)];
        let mut instructions = Vec::new();
        let mut literals = Vec::new();
        let mut map_entries = HashMap::new();
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
            } else if let Instruction::Read(slot) = self.instructions[index] {
                let input = self
                    .inputs
                    .get(slot.0 as usize)
                    .ok_or_else(|| ir_builder_error("additive term has no admitted input type"))?;
                let position =
                    if let Some(position) = inputs.iter().position(|existing| existing == input) {
                        position
                    } else {
                        inputs.push(input.clone());
                        inputs.len() - 1
                    };
                Instruction::Read(SymbolSlot(position as u32))
            } else if !exit {
                let mut children = Vec::new();
                slot_composition::visit_instruction_operands(self.instructions, index, |child| {
                    children.push(child)
                })?;
                pending.push((id, true));
                pending.extend(children.into_iter().rev().map(|child| (child, false)));
                continue;
            } else {
                slot_composition::remap_instruction(
                    self.instructions,
                    index,
                    &mut instructions,
                    &mut map_entries,
                    |child| mapped[child.0 as usize].expect("prior term operand"),
                )?
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
        Ok(ProjectedAtom {
            inputs,
            instructions,
            literals,
            root: mapped[root.0 as usize].expect("term root"),
        })
    }
}

fn check_coefficient(value: &BigRational, budget: usize) -> Result<(), Diagnostic> {
    if value.numer().bits() + value.denom().bits() > budget as u64 {
        return Err(ir_builder_error("exact coefficient budget exceeded"));
    }
    Ok(())
}
