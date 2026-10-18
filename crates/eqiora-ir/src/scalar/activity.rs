//! Numerical SSA activity follows bound input roles, independently of their values.
use super::*;

pub(super) fn numerical_activity(
    instructions: &[Instruction],
    bindings: &[InputBinding],
) -> Result<Vec<bool>, Diagnostic> {
    let mut active = Vec::with_capacity(instructions.len());
    for (index, instruction) in instructions.iter().enumerate() {
        let depends = match *instruction {
            Instruction::Read(slot) => !matches!(
                bindings.get(slot.0 as usize).ok_or_else(|| {
                    invalid_linearization("activity input is outside its exact binding")
                })?,
                InputBinding::Frozen
            ),
            Instruction::MapInvariant { start, extent, .. } => active
                [map_evaluation::operand_range(start, extent, index)?]
            .iter()
            .any(|&value| value),
            _ => {
                let mut depends = false;
                let mut invalid = false;
                instruction
                    .map_scalar_operands(|id| {
                        if let Some(&value) = active.get(id.0 as usize) {
                            depends |= value;
                        } else {
                            invalid = true;
                        }
                        id
                    })
                    .ok_or_else(|| {
                        invalid_linearization("activity requires scalar instructions")
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

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_schema::kernel::UnaryMathFunction;

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
