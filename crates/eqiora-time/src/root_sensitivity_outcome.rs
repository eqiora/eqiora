//! Fixed-time sensitivity stencils through an uncommitted localized event.
use crate::diagnostic::invalid_sensitivity;
use crate::{AcceptedTimeHistory, TimeRootOutcome};
use eqiora_core::Diagnostic;

/// A primal root-search outcome and continuous fixed-time parameter sensitivities.
///
/// Parameter vectors are flattened in parameter-major, state-major order. Event
/// time derivatives and reset propagation belong to the canonical event owner.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeRootSensitivityOutcome {
    primal: TimeRootOutcome,
    parameter_dimension: usize,
    sensitivity_history: AcceptedTimeHistory,
}

impl TimeRootSensitivityOutcome {
    /// Accept sensitivity stencils on exactly the same smooth primal intervals.
    /// # Errors
    /// Rejects zero parameter dimension, inconsistent shape, events, or step times.
    pub fn accepted(
        primal: TimeRootOutcome,
        parameter_dimension: usize,
        sensitivity_history: AcceptedTimeHistory,
    ) -> Result<Self, Diagnostic> {
        let history = primal.history();
        if parameter_dimension == 0
            || history.dimension().checked_mul(parameter_dimension)
                != Some(sensitivity_history.dimension())
            || !sensitivity_history.events().is_empty()
            || history.steps().len() != sensitivity_history.steps().len()
            || history.steps().iter().zip(sensitivity_history.steps()).any(
                |(primal, sensitivity)| {
                    primal.start_time() != sensitivity.start_time()
                        || primal.end_time() != sensitivity.end_time()
                },
            )
        {
            return Err(invalid_sensitivity(
                "root sensitivity history must match the exact smooth primal accepted intervals and parameter layout",
            ));
        }
        Ok(Self {
            primal,
            parameter_dimension,
            sensitivity_history,
        })
    }

    /// Primal smooth prefix and uncommitted root, or completed horizon.
    #[must_use]
    pub const fn primal(&self) -> &TimeRootOutcome {
        &self.primal
    }
    /// Number of caller-supplied real parameter coordinates.
    #[must_use]
    pub const fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }
    /// Fixed-time sensitivities, including native interpolation at a localized root.
    #[must_use]
    pub const fn sensitivity_history(&self) -> &AcceptedTimeHistory {
        &self.sensitivity_history
    }
}
