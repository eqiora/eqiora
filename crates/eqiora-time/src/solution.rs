//! Accepted time-execution reports and owned solution samples.

use crate::AcceptedTimeHistory;
use crate::diagnostic::{invalid_sensitivity, time_solve_failed};
use crate::lowering::TimeEquationClass;
use crate::plan::TimeMethod;
use crate::problem::InitialConditionPolicy;
use eqiora_core::Diagnostic;

/// Atomic adapter identity attached to accepted time execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeBackendIdentity {
    id: &'static str,
    version: &'static str,
}

impl TimeBackendIdentity {
    /// Construct one inseparable backend-name and release pair.
    ///
    /// # Panics
    /// Panics during constant evaluation if either token violates the
    /// backend identity and version token contracts.
    #[must_use]
    pub const fn new(id: &'static str, version: &'static str) -> Self {
        assert!(is_backend_id(id), "invalid time backend identity");
        assert!(is_backend_version(version), "invalid time backend version");
        Self { id, version }
    }

    /// Stable namespaced adapter identity.
    #[must_use]
    pub const fn id(self) -> &'static str {
        self.id
    }

    /// Exact adapter/library release.
    #[must_use]
    pub const fn version(self) -> &'static str {
        self.version
    }
}

/// Backend and policy identity attached to an accepted time solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeExecutionReport {
    backend: TimeBackendIdentity,
    method: TimeMethod,
    equation_class: TimeEquationClass,
    initial_condition: InitialConditionPolicy,
}

impl TimeExecutionReport {
    /// Record the exact admitted backend, method, and equation class.
    #[must_use]
    pub const fn new(
        backend: TimeBackendIdentity,
        method: TimeMethod,
        equation_class: TimeEquationClass,
        initial_condition: InitialConditionPolicy,
    ) -> Self {
        Self {
            backend,
            method,
            equation_class,
            initial_condition,
        }
    }

    /// Adapter identity.
    #[must_use]
    pub const fn backend(self) -> &'static str {
        self.backend.id()
    }

    /// Exact release supplied by the adapter that produced this report.
    #[must_use]
    pub const fn backend_version(self) -> &'static str {
        self.backend.version()
    }

    /// Atomic adapter name and release identity.
    #[must_use]
    pub const fn backend_identity(self) -> TimeBackendIdentity {
        self.backend
    }

    /// Integration method.
    #[must_use]
    pub const fn method(self) -> TimeMethod {
        self.method
    }

    /// Admitted equation class.
    #[must_use]
    pub const fn equation_class(self) -> TimeEquationClass {
        self.equation_class
    }

    /// Initial-condition policy actually requested.
    #[must_use]
    pub const fn initial_condition(self) -> InitialConditionPolicy {
        self.initial_condition
    }
}

const fn is_backend_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.' || byte == b'-') {
            return false;
        }
        index += 1;
    }
    true
}

const fn is_backend_version(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !byte.is_ascii_graphic() {
            return false;
        }
        index += 1;
    }
    true
}

/// Dense field-local samples returned by a production time backend.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSolution {
    dimension: usize,
    times: Vec<f64>,
    values: Vec<f64>,
    report: TimeExecutionReport,
    history: Option<AcceptedTimeHistory>,
}

impl TimeSolution {
    /// Accept shape-checked, finite, state-major samples from an adapter.
    ///
    /// Values are flattened in time-major order. The solution owns all data;
    /// no backend vector or matrix lifetime escapes.
    ///
    /// # Errors
    /// Returns `EQ0802` for invalid shape, time order, or non-finite data.
    pub fn accepted(
        dimension: usize,
        times: Vec<f64>,
        values: Vec<f64>,
        report: TimeExecutionReport,
    ) -> Result<Self, Diagnostic> {
        if dimension == 0
            || times.is_empty()
            || values.len() != dimension.saturating_mul(times.len())
            || times.iter().any(|value| !value.is_finite())
            || times.windows(2).any(|pair| pair[0] >= pair[1])
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(time_solve_failed(
                "time backend returned invalid sample shape, order, or values",
            ));
        }
        Ok(Self {
            dimension,
            times,
            values,
            report,
            history: None,
        })
    }

    /// Attach native accepted history to the same requested output samples.
    /// # Errors
    /// Rejects incompatible dimensions, interval coverage, or non-finite samples.
    pub fn accepted_with_history(
        dimension: usize,
        times: Vec<f64>,
        values: Vec<f64>,
        report: TimeExecutionReport,
        history: AcceptedTimeHistory,
    ) -> Result<Self, Diagnostic> {
        let mut solution = Self::accepted(dimension, times, values, report)?;
        if history.dimension() != dimension
            || history.steps()[0].start_time() > solution.times[0]
            || history.steps().last().unwrap().end_time() != *solution.times.last().unwrap()
        {
            return Err(time_solve_failed(
                "accepted history does not cover the requested solution interval",
            ));
        }
        solution.history = Some(history);
        Ok(solution)
    }

    /// Native accepted stencils, absent for adapters supplying only output samples.
    #[must_use]
    pub const fn history(&self) -> Option<&AcceptedTimeHistory> {
        self.history.as_ref()
    }

    /// Scalar state dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Strictly increasing requested sample times.
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// State at one requested sample.
    #[must_use]
    pub fn state(&self, sample: usize) -> Option<&[f64]> {
        let start = sample.checked_mul(self.dimension)?;
        self.values.get(start..start + self.dimension)
    }

    /// Adapter/method/equation evidence.
    #[must_use]
    pub const fn report(&self) -> TimeExecutionReport {
        self.report
    }
}

/// Primal trajectory plus `dy/dp_j` samples for every declared parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardSensitivitySolution {
    primal: TimeSolution,
    parameter_dimension: usize,
    sensitivities: Vec<f64>,
    history: Option<AcceptedTimeHistory>,
}

impl ForwardSensitivitySolution {
    /// Accept finite parameter-major, time-major, state-major samples.
    ///
    /// # Errors
    /// Returns `EQ0704` for invalid parameter cardinality or sample shape.
    pub fn accepted(
        primal: TimeSolution,
        parameter_dimension: usize,
        sensitivities: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let expected = parameter_dimension
            .checked_mul(primal.times().len())
            .and_then(|value| value.checked_mul(primal.dimension()));
        if parameter_dimension == 0
            || expected != Some(sensitivities.len())
            || sensitivities.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_sensitivity(
                "time backend returned invalid forward-sensitivity shape or values",
            ));
        }
        Ok(Self {
            primal,
            parameter_dimension,
            sensitivities,
            history: None,
        })
    }

    /// Attach parameter-major native sensitivity stencils at the primal steps.
    /// # Errors
    /// Rejects dimension or accepted-step mismatches.
    pub fn accepted_with_history(
        primal: TimeSolution,
        parameter_dimension: usize,
        sensitivities: Vec<f64>,
        history: AcceptedTimeHistory,
    ) -> Result<Self, Diagnostic> {
        let mut solution = Self::accepted(primal, parameter_dimension, sensitivities)?;
        let Some(primal_history) = solution.primal.history() else {
            return Err(invalid_sensitivity(
                "sensitivity history requires primal accepted history",
            ));
        };
        if solution.primal.dimension().checked_mul(parameter_dimension) != Some(history.dimension())
            || primal_history.steps().len() != history.steps().len()
            || primal_history
                .steps()
                .iter()
                .zip(history.steps())
                .any(|(primal, sensitivity)| {
                    primal.start_time() != sensitivity.start_time()
                        || primal.end_time() != sensitivity.end_time()
                })
        {
            return Err(invalid_sensitivity(
                "sensitivity history must match the same smooth primal accepted steps",
            ));
        }
        solution.history = Some(history);
        Ok(solution)
    }

    /// Parameter-major sensitivity vectors at each native accepted stencil.
    #[must_use]
    pub const fn sensitivity_history(&self) -> Option<&AcceptedTimeHistory> {
        self.history.as_ref()
    }

    /// Accepted primal trajectory.
    #[must_use]
    pub const fn primal(&self) -> &TimeSolution {
        &self.primal
    }

    /// Number of parameter directions integrated as coordinate bases.
    #[must_use]
    pub const fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }

    /// State sensitivity for one parameter and output sample.
    #[must_use]
    pub fn sensitivity(&self, parameter: usize, sample: usize) -> Option<&[f64]> {
        if parameter >= self.parameter_dimension || sample >= self.primal.times().len() {
            return None;
        }
        let samples_per_parameter = self.primal.times().len() * self.primal.dimension();
        let start = parameter * samples_per_parameter + sample * self.primal.dimension();
        self.sensitivities
            .get(start..start + self.primal.dimension())
    }
}
