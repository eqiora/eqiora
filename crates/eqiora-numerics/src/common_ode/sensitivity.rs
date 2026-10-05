//! Ordinary ODE sensitivity requests reuse the exact first-order lowering.
use super::*;

impl CommonOdePlan {
    pub(super) fn require_scalar_parameter_directions(&self) -> Result<(), Diagnostic> {
        let kernel = self
            .model_artifact()
            .to_program()
            .map_err(|errors| errors.into_iter().next().expect("failed Model replay"))?;
        for parameter in self.program.parameter_fields() {
            let value = kernel
                .typed_value(parameter.erase())
                .ok_or_else(|| invalid("Parameter has no exact typed value"))?;
            if !value.value_type().shape().is_scalar()
                || value.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
            {
                return Err(invalid(
                    "common forward sensitivity requires an admitted complete typed Parameter direction",
                ));
            }
        }
        Ok(())
    }

    /// Exact declared Parameter inventory; derivative controls establish coordinate admission.
    #[must_use]
    pub fn parameter_ids(&self) -> &[Id<kinds::Parameter>] {
        self.forward_parameter_ids
            .as_deref()
            .unwrap_or_else(|| self.program.parameter_fields())
    }
}

impl CommonOdeRunRequest {
    /// Request the existing forward Parameter sensitivity system at Model initial data.
    ///
    /// This profile requires Parameter-independent initial conditions and rejects
    /// resumed States until their initial tangent has its own admitted lineage.
    pub fn forward_sensitivity_problem(
        &self,
    ) -> Result<eqiora_time::ForwardSensitivityProblem<'_>, Diagnostic> {
        self.plan.require_scalar_parameter_directions()?;
        if self.plan.event_policy().is_some() {
            return Err(invalid(
                "registered-event sensitivity requires the canonical event sensitivity driver",
            ));
        }
        if self.state != self.plan.initial_state(self.state.time_s())? {
            return Err(invalid(
                "ODE Parameter sensitivity requires the exact Model initial State",
            ));
        }
        self.plan
            .program
            .forward_sensitivity_problem(self.state.time_s())
    }
}
