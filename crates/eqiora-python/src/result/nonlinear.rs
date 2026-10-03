//! Nonlinear acceptance summary, distinct from an individual linear solve.
use super::*;

/// Accepted nonlinear residual and iteration summary for a finite Result.
#[pyclass(
    name = "NonlinearSolveSummary",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyNonlinearSolveSummary {
    #[pyo3(get)]
    completed_iterations: usize,
    #[pyo3(get)]
    initial_residual_norm: f64,
    #[pyo3(get)]
    true_residual_norm: f64,
    #[pyo3(get)]
    residual_target: f64,
}
impl PyNonlinearSolveSummary {
    pub(crate) fn from_differentiation(
        value: &eqiora::api::DifferentiationEvidence,
    ) -> Option<Self> {
        Some(Self {
            completed_iterations: value.nonlinear_iterations()?,
            initial_residual_norm: value.nonlinear_initial_residual_norm()?,
            true_residual_norm: value.primal_residual_norm(),
            residual_target: value.residual_tolerance(),
        })
    }

    pub(super) fn from_result(result: &eqiora_numerics::CommonResult) -> PyResult<Self> {
        let missing = || PyRuntimeError::new_err("nonlinear Result omitted acceptance evidence");
        Ok(Self {
            completed_iterations: result.nonlinear_iterations().ok_or_else(missing)?,
            initial_residual_norm: result
                .nonlinear_initial_residual_norm()
                .ok_or_else(missing)?,
            true_residual_norm: result
                .finite_reference_residual_norm()
                .ok_or_else(missing)?,
            residual_target: result.nonlinear_residual_target().ok_or_else(missing)?,
        })
    }
}

#[pymethods]
impl PyNonlinearSolveSummary {
    fn __repr__(&self) -> String {
        format!(
            "NonlinearSolveSummary(completed_iterations={}, initial_residual_norm={}, true_residual_norm={}, residual_target={})",
            self.completed_iterations,
            self.initial_residual_norm,
            self.true_residual_norm,
            self.residual_target,
        )
    }
}
