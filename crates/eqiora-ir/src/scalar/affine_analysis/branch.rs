//! A predicate may select an affine branch only independently of selected variables.
use super::*;
use eqiora_schema::kernel::ComparisonOp;

#[derive(Clone)]
pub(in crate::scalar) enum Summary {
    Scalar(AffineSummary),
    Predicate(Option<bool>),
    Failed(SummaryFailure),
}

impl Summary {
    pub(in crate::scalar) fn scalar(
        &self,
        instruction: usize,
    ) -> Result<&AffineSummary, SummaryFailure> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::Predicate(_) => {
                Err(SymbolicLinearityFailure::InvalidProgram { instruction }.into())
            }
            Self::Failed(failure) => Err(failure.clone()),
        }
    }

    fn predicate(&self, instruction: usize) -> Result<Option<bool>, SummaryFailure> {
        match self {
            Self::Predicate(value) => Ok(*value),
            Self::Scalar(_) => Err(SymbolicLinearityFailure::InvalidProgram { instruction }.into()),
            Self::Failed(failure) => Err(failure.clone()),
        }
    }
}

pub(super) fn summarize(
    instruction: Instruction,
    summaries: &[Summary],
    constants: &[eqiora_core::ValueLiteral],
    index: usize,
) -> Result<Option<Summary>, SummaryFailure> {
    let at = |id| Ok::<_, SymbolicLinearityFailure>(&summaries[summary_index(id, index)?]);
    let invalid_domain = || SummaryFailure::Numerical {
        instruction: index,
        diagnostic: ir_builder_error(
            "affine binding demands an invalid condition or derivative boundary",
        ),
    };
    let summary = match instruction {
        Instruction::TypedConstant(slot) => Summary::Predicate(Some(
            constants
                .get(slot as usize)
                .and_then(eqiora_core::ValueLiteral::as_bool)
                .ok_or(SymbolicLinearityFailure::InvalidProgram { instruction: index })?,
        )),
        Instruction::Compare(op, left, right) => {
            let left = at(left)?.scalar(index)?;
            let right = at(right)?.scalar(index)?;
            if left.depends_on_selected() || right.depends_on_selected() {
                return Err(SymbolicLinearityFailure::Nonlinear { instruction: index }.into());
            }
            if let Some((a, b)) = left.constant.zip(right.constant)
                && a.value == b.value
                && (a.tangent.is_some() || b.tangent.is_some())
            {
                return Err(invalid_domain());
            }
            Summary::Predicate(left.constant.zip(right.constant).map(|(a, b)| match op {
                ComparisonOp::Equal => a.value == b.value,
                ComparisonOp::NotEqual => a.value != b.value,
                ComparisonOp::Less => a.value < b.value,
                ComparisonOp::LessEqual => a.value <= b.value,
                ComparisonOp::Greater => a.value > b.value,
                ComparisonOp::GreaterEqual => a.value >= b.value,
            }))
        }
        Instruction::Not(value) => Summary::Predicate(at(value)?.predicate(index)?.map(|v| !v)),
        Instruction::And(left, right) | Instruction::Or(left, right) => {
            let left = at(left)?.predicate(index)?;
            if left == Some(matches!(instruction, Instruction::Or(..))) {
                return Ok(Some(Summary::Predicate(left)));
            }
            let right = at(right)?.predicate(index)?;
            Summary::Predicate(left.zip(right).map(|(a, b)| {
                if matches!(instruction, Instruction::And(..)) {
                    a && b
                } else {
                    a || b
                }
            }))
        }
        Instruction::Require { condition, value } => {
            if at(condition)?.predicate(index)? == Some(false) {
                return Err(invalid_domain());
            }
            at(value)?.clone()
        }
        Instruction::Demand {
            prerequisite,
            value,
        } => {
            if let Summary::Failed(failure) = at(prerequisite)? {
                return Err(failure.clone());
            }
            at(value)?.clone()
        }
        Instruction::Select {
            condition,
            then_value,
            else_value,
        } => {
            let condition = at(condition)?.predicate(index)?;
            if let Some(condition) = condition {
                return Ok(Some(
                    at(if condition { then_value } else { else_value })?.clone(),
                ));
            }
            if let (Summary::Predicate(left), Summary::Predicate(right)) =
                (at(then_value)?, at(else_value)?)
            {
                return Ok(Some(Summary::Predicate(
                    left.filter(|value| Some(*value) == *right),
                )));
            }
            let left = at(then_value)?.scalar(index)?;
            let right = at(else_value)?.scalar(index)?;
            Summary::Scalar({
                if left.coefficients != right.coefficients {
                    return Err(SymbolicLinearityFailure::VariableCoefficient {
                        instruction: index,
                    }
                    .into());
                }
                AffineSummary {
                    constant: left.constant.filter(|value| Some(*value) == right.constant),
                    coefficients: left.coefficients.clone(),
                }
            })
        }
        _ => return Ok(None),
    };
    Ok(Some(summary))
}
