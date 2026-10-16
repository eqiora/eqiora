//! One checked parameter-major, sample-major, state-major sensitivity layout.
use crate::diagnostic::invalid_sensitivity;
use eqiora_core::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SensitivitySamples {
    parameters: usize,
    samples: usize,
    dimension: usize,
    values: Vec<f64>,
}

impl SensitivitySamples {
    pub(crate) fn accepted(
        parameters: usize,
        samples: usize,
        dimension: usize,
        values: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let expected = parameters
            .checked_mul(samples)
            .and_then(|n| n.checked_mul(dimension));
        if parameters == 0
            || dimension == 0
            || expected != Some(values.len())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_sensitivity(
                "time backend returned invalid forward-sensitivity shape or values",
            ));
        }
        Ok(Self {
            parameters,
            samples,
            dimension,
            values,
        })
    }

    pub(crate) const fn parameter_dimension(&self) -> usize {
        self.parameters
    }

    pub(crate) fn get(&self, parameter: usize, sample: usize) -> Option<&[f64]> {
        if parameter >= self.parameters || sample >= self.samples {
            return None;
        }
        let start = (parameter * self.samples + sample) * self.dimension;
        self.values.get(start..start + self.dimension)
    }
}
