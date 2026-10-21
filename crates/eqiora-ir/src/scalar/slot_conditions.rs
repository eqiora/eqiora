//! Typed conditions share the ordinary demand evaluator; Boolean values stay exact.
use super::*;

impl ScalarInputIrBuilder {
    pub(crate) fn demand(
        &mut self,
        prerequisite: ScalarInputValueId,
        value: ScalarInputValueId,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.binary(prerequisite, value, |prerequisite, value| {
            Instruction::Demand {
                prerequisite,
                value,
            }
        })
    }

    pub(super) fn typed_constant(
        &mut self,
        value: eqiora_core::ValueLiteral,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let index = u32::try_from(self.typed_constants.len()).map_err(|_| ir_size_error())?;
        self.typed_constants.push(value);
        self.push(Instruction::TypedConstant(index))
    }

    pub(crate) fn boolean(&mut self, value: bool) -> Result<ScalarInputValueId, Diagnostic> {
        self.typed_constant(eqiora_core::ValueLiteral::boolean(value))
    }

    pub(crate) fn not(
        &mut self,
        value: ScalarInputValueId,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.unary(value, Instruction::Not)
    }

    pub(crate) fn or(
        &mut self,
        left: ScalarInputValueId,
        right: ScalarInputValueId,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.binary(left, right, Instruction::Or)
    }

    pub(crate) fn select(
        &mut self,
        condition: ScalarInputValueId,
        then_value: ScalarInputValueId,
        else_value: ScalarInputValueId,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        for value in [condition, then_value, else_value] {
            self.require_prior(value)?;
        }
        self.push(Instruction::Select {
            condition: condition.0,
            then_value: then_value.0,
            else_value: else_value.0,
        })
    }
}
