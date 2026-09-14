//! Ordinary ODE sensitivity requests reuse the exact first-order lowering.
use super::*;

impl CommonOdePlan {
    /// Exact declared Parameters in the lowering's derivative coordinate order.
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
        if self.plan.event_policy().is_some() {
            return Err(invalid(
                "registered-event sensitivity requires the canonical event sensitivity driver",
            ));
        }
        if self.state.time_s() != 0.0 || self.state.values() != self.plan.initial_state()?.values()
        {
            return Err(invalid(
                "ODE Parameter sensitivity requires the exact Model initial State",
            ));
        }
        self.plan.program.forward_sensitivity_problem()
    }
}
