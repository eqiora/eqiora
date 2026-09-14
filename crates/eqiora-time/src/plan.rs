//! Realization-owned integration and sensitivity plans.

use crate::diagnostic::invalid_plan;
use crate::problem::{ForwardSensitivityProblem, ImplicitDaeProblem, TimeProblem};
use eqiora_core::Diagnostic;

/// Integration algorithm selected by Realization, not model meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeMethod {
    /// Deterministic first-order backward Euler reference method.
    ImplicitEuler,
    /// Tsitouras 5(4) explicit Runge--Kutta for non-stiff ODEs.
    Tsitouras45,
    /// Variable-order backward differentiation formula for stiff/DAE systems.
    Bdf,
}

/// Complete adaptive integration and output-sampling policy for one run.
#[derive(Debug, Clone, PartialEq)]
pub struct TimePlan {
    method: TimeMethod,
    start_time: f64,
    initial_step: f64,
    relative_tolerance: f64,
    absolute_tolerances: Vec<f64>,
    output_times: Vec<f64>,
}

impl TimePlan {
    /// Construct a validated method/tolerance/output request.
    ///
    /// Absolute-tolerance cardinality is checked against a problem at adapter
    /// admission. Output times must be finite, strictly increasing, and later
    /// than the start time.
    ///
    /// # Errors
    /// Returns `EQ0501` for invalid times, steps, or tolerances.
    pub fn new(
        method: TimeMethod,
        start_time: f64,
        initial_step: f64,
        relative_tolerance: f64,
        absolute_tolerances: Vec<f64>,
        output_times: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if !start_time.is_finite() || !initial_step.is_finite() || initial_step <= 0.0 {
            return Err(invalid_plan(
                "time-plan start must be finite and initial step must be finite and positive",
            ));
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err(invalid_plan(
                "time-plan relative tolerance must be finite and positive",
            ));
        }
        if absolute_tolerances.is_empty()
            || absolute_tolerances
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid_plan(
                "time-plan absolute tolerances must be finite, positive, and non-empty",
            ));
        }
        if output_times.is_empty()
            || output_times.iter().any(|time| !time.is_finite())
            || output_times[0] <= start_time
            || output_times.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_plan(
                "time-plan outputs must be finite, strictly increasing, non-empty, and later than start",
            ));
        }
        Ok(Self {
            method,
            start_time,
            initial_step,
            relative_tolerance,
            absolute_tolerances,
            output_times,
        })
    }

    /// Validate state-shaped controls against a problem.
    ///
    /// # Errors
    /// Returns `EQ0501` unless one absolute tolerance exists per state.
    pub fn validate_for(&self, problem: &TimeProblem<'_>) -> Result<(), Diagnostic> {
        self.validate_dimension(problem.dimension())
    }

    /// Validate state-shaped controls against a general residual problem.
    ///
    /// # Errors
    /// Returns `EQ0501` unless one absolute tolerance exists per state.
    pub fn validate_for_implicit(
        &self,
        problem: &ImplicitDaeProblem<'_>,
    ) -> Result<(), Diagnostic> {
        self.validate_dimension(problem.dimension())
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), Diagnostic> {
        if self.absolute_tolerances.len() != dimension {
            return Err(invalid_plan(
                "time-plan must provide exactly one absolute tolerance per state",
            ));
        }
        Ok(())
    }

    /// Selected method.
    #[must_use]
    pub const fn method(&self) -> TimeMethod {
        self.method
    }

    /// Initial model time.
    #[must_use]
    pub const fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Initial adaptive step-size guess, or maximum step for a fixed reference
    /// method.
    #[must_use]
    pub const fn initial_step(&self) -> f64 {
        self.initial_step
    }

    /// Relative local-error tolerance.
    #[must_use]
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Per-state absolute local-error tolerances.
    #[must_use]
    pub fn absolute_tolerances(&self) -> &[f64] {
        &self.absolute_tolerances
    }

    /// Requested model-time samples, separate from internal adaptive steps.
    #[must_use]
    pub fn output_times(&self) -> &[f64] {
        &self.output_times
    }
}

/// Error-control policy for continuous forward sensitivities.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardSensitivityPlan {
    relative_tolerance: f64,
    absolute_tolerances: Vec<f64>,
}

impl ForwardSensitivityPlan {
    /// Construct positive finite sensitivity tolerances in parameter-major state order.
    ///
    /// # Errors
    /// Returns `EQ0501` for empty, non-positive, or non-finite controls.
    pub fn new(relative_tolerance: f64, absolute_tolerances: Vec<f64>) -> Result<Self, Diagnostic> {
        if !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
            || absolute_tolerances.is_empty()
            || absolute_tolerances
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid_plan(
                "forward-sensitivity tolerances must be finite, positive, and non-empty",
            ));
        }
        Ok(Self {
            relative_tolerance,
            absolute_tolerances,
        })
    }

    /// Validate one absolute tolerance per (parameter, state) pair.
    ///
    /// # Errors
    /// Returns `EQ0501` for a parameter/state shape mismatch.
    pub fn validate_for(&self, problem: &ForwardSensitivityProblem<'_>) -> Result<(), Diagnostic> {
        if problem
            .primal()
            .dimension()
            .checked_mul(problem.parameter_dimension())
            != Some(self.absolute_tolerances.len())
        {
            return Err(invalid_plan(
                "forward-sensitivity plan must provide one absolute tolerance per parameter and state",
            ));
        }
        Ok(())
    }

    /// Relative sensitivity error tolerance.
    #[must_use]
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Parameter-major absolute sensitivity tolerances, with contiguous state columns.
    #[must_use]
    pub fn absolute_tolerances(&self) -> &[f64] {
        &self.absolute_tolerances
    }
}
