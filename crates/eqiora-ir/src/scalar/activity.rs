//! Bound dependencies and derivative demand keep primal-domain obligations separate.
use super::*;

fn operands(
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
            for id in coefficient_jet::operands(start, len, shape, &instructions[..index])?.iter() {
                visit(id);
            }
        }
        Instruction::MapInvariant { start, extent, .. } => {
            for index in map_evaluation::operand_range(start, extent, index)? {
                visit(ValueId(index as u32));
            }
        }
        instruction => {
            instruction
                .map_scalar_operands(|id| {
                    visit(id);
                    id
                })
                .ok_or_else(|| invalid_linearization("activity requires scalar instructions"))?;
        }
    }
    Ok(())
}

pub(super) fn numerical_activity(
    instructions: &[Instruction],
    bindings: &[InputBinding],
) -> Result<Vec<bool>, Diagnostic> {
    dependencies(instructions, bindings, None)
}

fn dependencies(
    instructions: &[Instruction],
    bindings: &[InputBinding],
    values: Option<&[f64]>,
) -> Result<Vec<bool>, Diagnostic> {
    let mut active = Vec::with_capacity(instructions.len());
    for (index, instruction) in instructions.iter().enumerate() {
        let frozen_zero = |id: ValueId| {
            active.get(id.0 as usize) == Some(&false)
                && values.is_some_and(|values| values.get(id.0 as usize) == Some(&0.))
        };
        let depends = match *instruction {
            Instruction::Read(slot) => !matches!(
                bindings.get(slot.0 as usize).ok_or_else(|| {
                    invalid_linearization("activity input is outside its exact binding")
                })?,
                InputBinding::Frozen
            ),
            // Only a value proven independent of every admitted derivative
            // input can annihilate a product. An active sampled zero cannot.
            Instruction::Mul(a, b) if frozen_zero(a) || frozen_zero(b) => false,
            _ => {
                let mut depends = false;
                let mut invalid = false;
                operands(instructions, index, |id| {
                    if let Some(&value) = active.get(id.0 as usize) {
                        depends |= value;
                    } else {
                        invalid = true;
                    }
                })?;
                if invalid {
                    return Err(invalid_linearization(
                        "activity operand is outside its prior SSA values",
                    ));
                }
                depends
            }
        };
        active.push(depends);
    }
    Ok(active)
}

impl ScalarLinearization<'_> {
    pub(super) fn derivative_demand(&self, values: &[f64]) -> Result<Vec<bool>, Diagnostic> {
        derivative_limit::require_first(&self.instructions, &self.active)?;
        // The complete original primal has already been evaluated. Retain its
        // active unary regularity even when a frozen zero removes a derivative
        // contribution. The ordinary unary owner admits zero directions too.
        for (index, instruction) in self.instructions.iter().enumerate() {
            if self.active[index]
                && let Instruction::Math(function, operand) = *instruction
            {
                real_math::action(function, read(values, operand, index)?, 0.)?;
            }
        }
        let depends = dependencies(&self.instructions, &self.bindings, Some(values))?;
        let mut needed = vec![false; self.instructions.len()];
        for root in self.roots.iter() {
            *needed.get_mut(root.0 as usize).ok_or_else(|| {
                invalid_linearization("derivative root is outside its scalar program")
            })? = true;
        }
        for index in (0..needed.len()).rev() {
            needed[index] &= depends[index];
            if needed[index] {
                operands(&self.instructions, index, |id| needed[id.0 as usize] = true)?;
            }
        }
        Ok(needed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_schema::kernel::UnaryMathFunction;

    #[test]
    fn zero_weight_keeps_active_domains_and_sampled_zero_keeps_curvature() {
        // Zero weighting can remove a smooth contribution, but cannot admit
        // sqrt outside its primal domain or at its active derivative boundary.
        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            Instruction::Read(SymbolSlot(1)),
            Instruction::Math(UnaryMathFunction::Sqrt, ValueId(0)),
            Instruction::Mul(ValueId(1), ValueId(2)),
        ];
        let roots = [ValueId(3)];
        let roles = [DifferentiationRole::Parameter, DifferentiationRole::Frozen];
        for point in [-1., 0., 4.] {
            let action =
                ScalarLinearization::bind(&instructions, &roots, 2, &[point, 0.], &roles).unwrap();
            let mut output = [0.];
            assert_eq!(action.primal(&mut output).is_ok(), point >= 0.);
            assert_eq!(
                action
                    .jvp(RelationTangent::Parameter(&[1.]), &mut output)
                    .is_ok(),
                point > 0.
            );
            assert_eq!(
                action
                    .vjp(&[1.], RelationCotangent::Parameter(&mut output))
                    .is_ok(),
                point > 0.
            );
            assert_eq!(
                action
                    .mixed_jvp(
                        RelationTangent::Parameter(&[1.]),
                        RelationTangent::Parameter(&[1.]),
                        RelationTangent::Parameter(&[0.]),
                        &mut output,
                    )
                    .is_ok(),
                point > 0.
            );
            if point > 0. {
                assert_eq!(output, [0.]);
            }
        }

        // A sampled zero remains active: D²(p²)[2,3]=12 even at p=0.
        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            Instruction::Mul(ValueId(0), ValueId(0)),
        ];
        let roots = [ValueId(1)];
        let action = ScalarLinearization::bind(
            &instructions,
            &roots,
            1,
            &[0.],
            &[DifferentiationRole::Parameter],
        )
        .unwrap();
        let mut output = [0.];
        action
            .mixed_jvp(
                RelationTangent::Parameter(&[2.]),
                RelationTangent::Parameter(&[3.]),
                RelationTangent::Parameter(&[0.]),
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [12.]);
    }

    #[test]
    fn frozen_scalar_subgraphs_keep_primal_domains_without_derivative_domains() {
        // f(q,p,x)=sqrt(q)+p*x, with q frozen at zero. At (p,x)=(0,1/2),
        // df=dp/2 and D²f[(2,0),(0,3)]=6 independently of sqrt's derivative.
        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            Instruction::Read(SymbolSlot(1)),
            Instruction::Read(SymbolSlot(2)),
            Instruction::Math(UnaryMathFunction::Sqrt, ValueId(0)),
            Instruction::Mul(ValueId(1), ValueId(2)),
            Instruction::Add(ValueId(3), ValueId(4)),
        ];
        let roots = [ValueId(5)];
        let roles = [
            DifferentiationRole::Frozen,
            DifferentiationRole::Parameter,
            DifferentiationRole::Unknown,
        ];
        let action =
            ScalarLinearization::bind(&instructions, &roots, 3, &[0., 0., 0.5], &roles).unwrap();
        let mut output = [0.];
        action.primal(&mut output).unwrap();
        assert_eq!(output, [0.]);
        action
            .jvp(
                RelationTangent::Both {
                    unknown: &[3.],
                    parameter: &[2.],
                },
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [1.]);
        action
            .mixed_jvp(
                RelationTangent::Both {
                    unknown: &[0.],
                    parameter: &[2.],
                },
                RelationTangent::Both {
                    unknown: &[3.],
                    parameter: &[0.],
                },
                RelationTangent::Both {
                    unknown: &[0.],
                    parameter: &[0.],
                },
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [6.]);
        let (mut unknown, mut parameter) = ([0.], [0.]);
        action
            .vjp(
                &[4.],
                RelationCotangent::Both {
                    unknown: &mut unknown,
                    parameter: &mut parameter,
                },
            )
            .unwrap();
        assert_eq!(unknown, [0.]);
        assert_eq!(parameter, [2.]);
        let invalid =
            ScalarLinearization::bind(&instructions, &roots, 3, &[-1., 0., 0.5], &roles).unwrap();
        assert!(invalid.primal(&mut output).is_err());
        assert!(
            invalid
                .jvp(RelationTangent::Unknown(&[0.]), &mut output)
                .is_err()
        );

        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            Instruction::Mul(ValueId(0), ValueId(0)),
            Instruction::Math(UnaryMathFunction::Sqrt, ValueId(1)),
        ];
        let roots = [ValueId(2)];
        let cusp = ScalarLinearization::bind(
            &instructions,
            &roots,
            1,
            &[0.],
            &[DifferentiationRole::Unknown],
        )
        .unwrap();
        cusp.primal(&mut output).unwrap();
        assert!(
            cusp.jvp(RelationTangent::Unknown(&[0.]), &mut output)
                .is_err()
        );
        assert!(
            cusp.vjp(&[0.], RelationCotangent::Unknown(&mut unknown))
                .is_err()
        );
    }
}
