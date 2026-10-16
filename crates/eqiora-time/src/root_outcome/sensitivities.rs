//! Fixed-time sensitivity stencils through an uncommitted localized event.
use crate::diagnostic::invalid_sensitivity;
use crate::solution::SensitivitySamples;
use crate::{AcceptedTimeHistory, TimeRootOutcome};
use eqiora_core::Diagnostic;

/// Continuous fixed-time parameter sensitivities of an accepted root-search outcome.
///
/// Parameter vectors are flattened in parameter-major, state-major order. Event
/// time derivatives and reset propagation belong to the canonical event owner.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RootSensitivities {
    sensitivities: SensitivitySamples,
    sensitivity_history: AcceptedTimeHistory,
}

impl TimeRootOutcome {
    /// Accept sensitivity stencils on exactly the same smooth primal intervals.
    /// # Errors
    /// Rejects zero parameter dimension, inconsistent sample/step shape, nonfinite
    /// products, events, or step times. Samples use parameter-major order.
    pub fn with_sensitivities(
        mut self,
        parameter_dimension: usize,
        sensitivity_history: AcceptedTimeHistory,
        sensitivities: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let history = self.history();
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
        let sensitivities = SensitivitySamples::accepted(
            parameter_dimension,
            self.samples().map_or(0, |samples| samples.times().len()),
            history.dimension(),
            sensitivities,
        )?;
        self.sensitivities = Some(RootSensitivities {
            sensitivities,
            sensitivity_history,
        });
        Ok(self)
    }

    /// Number of real parameter coordinates, or zero when sensitivities were not requested.
    #[must_use]
    pub fn parameter_dimension(&self) -> usize {
        self.sensitivities
            .as_ref()
            .map_or(0, |data| data.sensitivities.parameter_dimension())
    }
    /// Fixed-time sensitivity for one Parameter and one retained primal sample.
    /// No sample at the uncommitted root is implied by this lookup.
    #[must_use]
    pub fn sensitivity(&self, parameter: usize, sample: usize) -> Option<&[f64]> {
        self.sensitivities
            .as_ref()?
            .sensitivities
            .get(parameter, sample)
    }

    /// Requested fixed-time sensitivities, including interpolation at a localized root.
    #[must_use]
    pub fn sensitivity_history(&self) -> Option<&AcceptedTimeHistory> {
        self.sensitivities
            .as_ref()
            .map(|data| &data.sensitivity_history)
    }
}
