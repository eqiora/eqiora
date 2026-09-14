//! Original condition typing, including explicit nonnegativity predicates.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn check_equation(
        &mut self,
        equation: &eqiora_lang::RelationCondition,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        for value in [equation.left(), equation.right()] {
            crate::hierarchy::reductions::preflight(
                self.scope.file,
                value,
                &mut |name| self.scope.index_sets.get(name).copied().flatten(),
                &self.scope.static_values,
                self.scope.elaborator.limits.max_parameter_terms,
            )?;
        }
        if equation.kind() == eqiora_schema::kernel::RelationConditionKind::Complementarity {
            let mut operands = Vec::new();
            for predicate in [equation.left(), equation.right()] {
                let operand =
                    crate::lower::constraints::source_operand(predicate).map_err(|message| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            predicate.range(),
                            message,
                        )
                    })?;
                // Check the complete predicate first: an explicit zero with wrong units
                // cannot disappear when the physical operand is retained in the Model.
                self.check(predicate)?;
                operands.push(self.check(operand)?);
            }
            return eqiora_schema::kernel::RelationConditionKind::Complementarity
                .check_operands(&operands[0], &operands[1])
                .map_err(|error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        equation.range(),
                        error.to_string(),
                    )
                });
        }
        let (left, right) = self.check_pair(equation.left(), equation.right())?;
        crate::lower::equality::check(
            left,
            right,
            crate::lower::equality::is_contextual_zero(equation.left()),
            crate::lower::equality::is_contextual_zero(equation.right()),
        )
        .and_then(|checked| {
            equation
                .kind()
                .check_operands(&checked.left, &checked.right)
        })
        .map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                equation.range(),
                error.to_string(),
            )
        })
    }
}
