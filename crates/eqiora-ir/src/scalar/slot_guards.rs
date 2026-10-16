//! Component slots reuse typed demand and active-branch differentiation rules.
use super::*;
use eqiora_core::{DynQuantity, ValueLiteral};

impl ScalarInputOperatorIr {
    pub(super) fn typed_instructions(&self) -> typed::TypedInstructions<'_> {
        typed::TypedInstructions {
            instructions: &self.instructions,
            typed_constants: &[],
            array_operands: &[],
            definitions: &[],
        }
    }

    pub(super) fn typed_point(&self, inputs: &[f64]) -> Result<Vec<ValueLiteral>, Diagnostic> {
        if inputs.len() != self.slots.len() || inputs.iter().any(|value| !value.is_finite()) {
            return Err(ir_builder_error(
                "component point requires complete finite inputs",
            ));
        }
        self.slots
            .iter()
            .zip(inputs)
            .map(|(slot, value)| {
                ValueLiteral::try_from(DynQuantity::new(*value, slot.dimension))
                    .map_err(|error| ir_builder_error(error.to_string()))
            })
            .collect()
    }

    pub(super) fn linearize_guarded(
        &self,
        inputs: &[f64],
        roles: &[DifferentiationRole],
    ) -> Result<ScalarLinearization<'_>, Diagnostic> {
        if roles.len() != self.slots.len() {
            return Err(ir_builder_error(
                "component derivative roles differ from input slots",
            ));
        }
        let point = self.typed_point(inputs)?;
        let program = self.typed_instructions();
        let (_, trace) =
            program.evaluate(&self.roots, &mut |slot| point.get(slot.0 as usize).cloned())?;
        let dependencies = program.active_input_dependencies(&trace, roles)?;
        program.require_regular_active_trace(&trace, &dependencies)?;
        let (instructions, roots) = program.active_program(&self.roots, &trace, &dependencies)?;
        let bound =
            ScalarLinearization::bind(&instructions, &roots, self.slots.len(), inputs, roles)?;
        let ScalarLinearization {
            inputs,
            bindings,
            unknown_dimension,
            parameter_dimension,
            ..
        } = bound;
        Ok(ScalarLinearization {
            instructions: std::borrow::Cow::Owned(instructions),
            roots: std::borrow::Cow::Owned(roots),
            inputs,
            bindings,
            unknown_dimension,
            parameter_dimension,
        })
    }
}
