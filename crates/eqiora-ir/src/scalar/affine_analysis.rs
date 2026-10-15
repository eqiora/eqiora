//! Ordered affine dependency analysis of the scalar numerical instruction profile.
use super::*;

mod branch;
pub(super) use branch::Summary;

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
    ) -> Result<Vec<Summary>, SymbolicLinearityFailure> {
        let dimension = columns.len();
        summarize(
            &self.instructions,
            &self.typed_constants,
            dimension,
            |slot, index| {
                let symbol = self
                    .symbols
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
            },
        )
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
    typed_constants: &[eqiora_core::ValueLiteral],
    dimension: usize,
    resolve: impl Fn(SymbolSlot, usize) -> Result<AffineSummary, SymbolicLinearityFailure>,
) -> Result<Vec<Summary>, SummaryFailure> {
    let mut summaries: Vec<Summary> = Vec::with_capacity(instructions.len());
    for (index, instruction) in instructions.iter().copied().enumerate() {
        if let Some(summary) = branch::summarize(instruction, &summaries, typed_constants, index)? {
            summaries.push(summary);
            continue;
        }
        let at = |id| summaries[summary_index(id, index)?].scalar(index);
        let summary = match instruction {
            Instruction::Sin(value) | Instruction::Exp(value) | Instruction::Sqrt(value) => {
                let argument = at(value)?;
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
            Instruction::Neg(value) => at(value)?.scaled(-1.0, index)?,
            Instruction::Add(left, right) => AffineSummary::sum(at(left)?, at(right)?, 1.0, index)?,
            Instruction::Sub(left, right) => {
                AffineSummary::sum(at(left)?, at(right)?, -1.0, index)?
            }
            Instruction::Mul(left, right) => AffineSummary::product(at(left)?, at(right)?, index)?,
            Instruction::Div(left, right) => AffineSummary::quotient(at(left)?, at(right)?, index)?,
            Instruction::ComplexDiv {
                operands,
                imaginary,
            } => {
                let [a, b, c, d] = operands;
                complex_quotient::affine([at(a)?, at(b)?, at(c)?, at(d)?], imaginary, index)?
            }
            Instruction::MapInvariant {
                start,
                extent,
                component,
            } => {
                let range = map_evaluation::operand_range(start, extent, index)
                    .map_err(|_| SymbolicLinearityFailure::InvalidProgram { instruction: index })?;
                let operands = summaries[range]
                    .iter()
                    .map(|value| value.scalar(index))
                    .collect::<Result<Vec<_>, _>>()?;
                if operands.iter().any(|value| value.depends_on_selected()) {
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
            Instruction::PowI(base, exponent) => at(base)?.integer_power(exponent, index)?,
        };
        summaries.push(Summary::Scalar(summary));
    }
    Ok(summaries)
}
