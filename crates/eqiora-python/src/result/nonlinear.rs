//! Nonlinear acceptance summary, distinct from an individual linear solve.
use super::*;
use pyo3::types::PyTuple;

/// Accepted nonlinear residual and iteration summary for a static Result or retained spatial output.
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
    linear_solves: Option<Vec<PyLinearSolveSummary>>,
}
impl PyNonlinearSolveSummary {
    pub(crate) fn from_differentiation(
        value: &eqiora::api::DifferentiationStepEvidence,
    ) -> Option<Self> {
        Some(Self {
            completed_iterations: value.nonlinear_iterations()?,
            initial_residual_norm: value.nonlinear_initial_residual_norm()?,
            true_residual_norm: value.primal_residual_norm(),
            residual_target: value.residual_tolerance(),
            linear_solves: Some(
                value
                    .primal_solves()
                    .iter()
                    .map(PyLinearSolveSummary::from_evidence)
                    .collect(),
            ),
        })
    }

    pub(super) fn from_result(
        result: &eqiora_numerics::CommonResult,
        output_index: Option<usize>,
    ) -> PyResult<Self> {
        let missing = || PyRuntimeError::new_err("nonlinear Result omitted acceptance evidence");
        let completed_iterations = result
            .nonlinear_iterations(output_index)
            .ok_or_else(missing)?;
        let linear_solves = (0..completed_iterations)
            .map(|iteration| {
                PyLinearSolveSummary::from_common_result(result, output_index, Some(iteration))
                    .ok_or_else(missing)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            completed_iterations,
            initial_residual_norm: result
                .nonlinear_initial_residual_norm(output_index)
                .ok_or_else(missing)?,
            true_residual_norm: result
                .nonlinear_true_residual_norm(output_index)
                .ok_or_else(missing)?,
            residual_target: result
                .nonlinear_residual_target(output_index)
                .ok_or_else(missing)?,
            linear_solves: Some(linear_solves),
        })
    }
}

#[pymethods]
impl PyNonlinearSolveSummary {
    /// Actual correction solves, empty for zero updates; unavailable for differentiation-only summaries.
    #[getter]
    fn linear_solves(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        self.linear_solves
            .as_ref()
            .map(|solves| {
                let values = solves
                    .iter()
                    .map(|solve| Py::new(py, solve.clone()))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyTuple::new(py, values)?.unbind())
            })
            .transpose()
    }

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
