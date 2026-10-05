//! Ordered affine dependency analysis of the scalar numerical instruction profile.
use super::*;

pub(super) enum SummaryFailure {
    Symbolic(SymbolicLinearityFailure),
    Numerical {
        instruction: usize,
        diagnostic: Diagnostic,
    },
}
impl From<SymbolicLinearityFailure> for SummaryFailure {
    fn from(value: SymbolicLinearityFailure) -> Self {
        Self::Symbolic(value)
    }
}

impl ScalarOperatorIr {
    pub(super) fn affine_summaries(
        &self,
        columns: &HashMap<SymbolRef, usize>,
        constants: Option<&HashMap<SymbolRef, f64>>,
    ) -> Result<Vec<AffineSummary>, SymbolicLinearityFailure> {
        let dimension = columns.len();
        summarize(&self.instructions, dimension, |slot, index| {
            let symbol =
                self.symbols
                    .get(usize::try_from(slot.0).map_err(|_| {
                        SymbolicLinearityFailure::InvalidProgram { instruction: index }
                    })?)
                    .copied()
                    .ok_or(SymbolicLinearityFailure::InvalidProgram { instruction: index })?;
            Ok(if let Some(column) = columns.get(&symbol).copied() {
                AffineSummary::variable(column, dimension)
            } else if let Some(value) = constants.and_then(|values| values.get(&symbol)) {
                AffineSummary::constant(*value, dimension)
            } else {
                AffineSummary::independent(dimension)
            })
        })
        .map_err(|failure| match failure {
            SummaryFailure::Symbolic(failure) => failure,
            SummaryFailure::Numerical { instruction, .. } => {
                SymbolicLinearityFailure::NonFiniteCoefficient { instruction }
            }
        })
    }
}

pub(super) fn summarize(
    instructions: &[Instruction],
    dimension: usize,
    resolve: impl Fn(SymbolSlot, usize) -> Result<AffineSummary, SymbolicLinearityFailure>,
) -> Result<Vec<AffineSummary>, SummaryFailure> {
    let mut summaries: Vec<AffineSummary> = Vec::with_capacity(instructions.len());
    for (index, instruction) in instructions.iter().copied().enumerate() {
        let summary = match instruction {
            Instruction::Sin(value) | Instruction::Exp(value) | Instruction::Sqrt(value) => {
                let argument = &summaries[summary_index(value, index)?];
                if argument.depends_on_selected() {
                    return Err(SymbolicLinearityFailure::Nonlinear { instruction: index }.into());
                }
                let constant = argument.constant.map(|value| match instruction {
                    Instruction::Sin(_) => value.sin(),
                    Instruction::Exp(_) => value.exp(),
                    Instruction::Sqrt(_) => value.sqrt(),
                    _ => unreachable!(),
                });
                AffineSummary::finite(constant, vec![0.0; dimension], index)?
            }
            Instruction::Select { .. }
            | Instruction::Require { .. }
            | Instruction::PureOperator { .. }
            | Instruction::Compare(_, _, _)
            | Instruction::Not(_)
            | Instruction::And(_, _)
            | Instruction::Or(_, _)
            | Instruction::Array { .. }
            | Instruction::Index(_, _)
            | Instruction::TypedConstant(_)
            | Instruction::Quotient(_, _)
            | Instruction::Remainder(_, _)
            | Instruction::ToReal(_)
            | Instruction::ToInteger(_)
            | Instruction::Ordinal(_) => {
                return Err(SymbolicLinearityFailure::InvalidProgram { instruction: index }.into());
            }
            Instruction::Constant(value) => AffineSummary::constant(value.value(), dimension),
            Instruction::Read(slot) => resolve(slot, index)?,
            Instruction::Neg(value) => {
                summaries[summary_index(value, index)?].scaled(-1.0, index)?
            }
            Instruction::Add(left, right) => AffineSummary::sum(
                &summaries[summary_index(left, index)?],
                &summaries[summary_index(right, index)?],
                1.0,
                index,
            )?,
            Instruction::Sub(left, right) => AffineSummary::sum(
                &summaries[summary_index(left, index)?],
                &summaries[summary_index(right, index)?],
                -1.0,
                index,
            )?,
            Instruction::Mul(left, right) => AffineSummary::product(
                &summaries[summary_index(left, index)?],
                &summaries[summary_index(right, index)?],
                index,
            )?,
            Instruction::Div(left, right) => AffineSummary::quotient(
                &summaries[summary_index(left, index)?],
                &summaries[summary_index(right, index)?],
                index,
            )?,
            Instruction::MapInvariant {
                start,
                extent,
                component,
            } => {
                let range = map_evaluation::operand_range(start, extent, index)
                    .map_err(|_| SymbolicLinearityFailure::InvalidProgram { instruction: index })?;
                let operands = &summaries[range];
                if operands.iter().any(AffineSummary::depends_on_selected) {
                    return Err(SymbolicLinearityFailure::Nonlinear { instruction: index }.into());
                }
                let constant = operands
                    .iter()
                    .map(|s| s.constant)
                    .collect::<Option<Vec<_>>>()
                    .map(|values| {
                        map_evaluation::MapEvaluation::new(&values, extent as usize)?
                            .value(component)
                    })
                    .transpose()
                    .map_err(|diagnostic| SummaryFailure::Numerical {
                        instruction: index,
                        diagnostic,
                    })?;
                AffineSummary::finite(constant, vec![0.0; dimension], index)?
            }
            Instruction::PowI(base, exponent) => {
                summaries[summary_index(base, index)?].integer_power(exponent, index)?
            }
        };
        summaries.push(summary);
    }
    Ok(summaries)
}
