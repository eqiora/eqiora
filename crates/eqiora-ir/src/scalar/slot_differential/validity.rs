//! Source domains follow the same lazy branches as their numerical values.
use super::*;

impl ScalarInputIrBuilder {
    pub(super) fn formal_validity(
        &mut self,
        index: usize,
        prior: &[Option<ScalarInputValueId>],
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        let instruction = self.instructions[index];
        let mut conditions = Vec::new();
        match instruction {
            Instruction::Require { condition, value } => {
                conditions.push(ScalarInputValueId(condition));
                conditions.extend(prior[value.0 as usize]);
            }
            Instruction::Select {
                condition,
                then_value,
                else_value,
            } => {
                // Even a zero derivative must evaluate the selector, whose own
                // domain may be invalid. Only its selected arm's domain is read.
                let valid = self.boolean(true)?;
                return self
                    .select(
                        ScalarInputValueId(condition),
                        prior[then_value.0 as usize].unwrap_or(valid),
                        prior[else_value.0 as usize].unwrap_or(valid),
                    )
                    .map(Some);
            }
            _ => slot_composition::visit_instruction_operands(&self.instructions, index, |id| {
                conditions.extend(prior[id.0 as usize]);
            })?,
        }
        let domain = match instruction {
            Instruction::Math(function, input) => {
                function.real_domain_condition().map(|op| (op, input))
            }
            Instruction::Div(_, denominator) => {
                Some((eqiora_schema::kernel::ComparisonOp::NotEqual, denominator))
            }
            Instruction::PowI(input, exponent) if exponent < 0 => {
                Some((eqiora_schema::kernel::ComparisonOp::NotEqual, input))
            }
            _ => None,
        };
        if let Some((op, input)) = domain {
            let input = ScalarInputValueId(input);
            let zero = self.sub(input, input)?;
            conditions.push(self.compare(op, input, zero)?);
        }
        conditions.into_iter().try_fold(None, |prior, condition| {
            prior
                .map_or(Ok(condition), |prior| self.and(prior, condition))
                .map(Some)
        })
    }
}
