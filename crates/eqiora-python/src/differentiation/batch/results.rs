//! Complete results and indexed partial membership remain different Python types.

use super::products::{PyEvaluationMapJvp, PyEvaluationMapVjp};
use super::*;
use crate::error::PyDiagnostic;
use eqiora::api::EvaluationMapOccurrence;

/// Complete accepted batch with point-local primal and first-order products.
#[pyclass(name = "CompleteEvaluationMap", module = "eqiora._eqiora", frozen)]
pub(super) struct PyCompleteEvaluationMap {
    pub(super) native: Arc<CompleteEvaluationMap>,
    pub(super) plan: PyEvaluationMapPlan,
}

#[pymethods]
impl PyCompleteEvaluationMap {
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyDifferentiableEvaluation> {
        self.member(py, occurrence_index(index, self.plan.count())?)
    }
    fn __repr__(&self) -> String {
        format!(
            "CompleteEvaluationMap(point_shape={:?})",
            self.plan.point_shape
        )
    }
    fn __len__(&self) -> usize {
        self.native.len()
    }
    #[getter]
    fn plan(&self) -> PyEvaluationMapPlan {
        self.plan.clone()
    }
    #[getter]
    fn statuses(&self) -> Vec<&'static str> {
        vec!["accepted"; self.plan.count()]
    }
    /// Inspect one ordinary accepted evaluation without changing its lineage.
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<PyDifferentiableEvaluation> {
        let value = self
            .native
            .evaluation(index)
            .ok_or_else(|| PyIndexError::new_err("batch occurrence is out of range"))?;
        evaluation_result(py, value.clone())
    }
    /// Complete read-only output tensor, including statically typed empty axes.
    fn primal(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self
            .native
            .members()
            .ok_or_else(|| {
                PyRuntimeError::new_err("dense Python batch requires retained native members")
            })?
            .iter()
            .flat_map(|member| member.primal().output().coordinates().to_vec())
            .collect();
        readonly_array(
            py,
            values,
            &self
                .plan
                .with_components(self.plan.program.identity().output_dimension()),
        )
    }
    /// First-order JVP on a native nested point/seed grid.
    #[pyo3(signature = (mapped, *, shared=None, seed_shape=None, point_axes=None, numerical_bytes_limit=67_108_864))]
    fn jvp(
        &self,
        py: Python<'_>,
        mapped: &Bound<'_, PyAny>,
        shared: Option<&Bound<'_, PyAny>>,
        seed_shape: Option<Vec<usize>>,
        point_axes: Option<Vec<usize>>,
        numerical_bytes_limit: usize,
    ) -> PyResult<PyEvaluationMapJvp> {
        panic_boundary(py, || {
            let seeds = seed_shape.unwrap_or_default();
            let positions =
                point_axes.unwrap_or_else(|| (0..self.plan.point_shape.len()).collect());
            let view = EvaluationMapProducts::new(
                &self.native,
                &self.plan.point_shape,
                &seeds,
                &positions,
                numerical_bytes_limit,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
            let mut shared_shape = seeds.clone();
            shared_shape.push(self.plan.native.shared_inputs().len());
            let mut mapped_shape = view.shape().to_vec();
            mapped_shape.push(self.plan.mapped().len());
            let shared = match shared {
                Some(value) => stage_f64_shaped_input(py, value, &shared_shape, "shared tangents")?,
                None if self.plan.native.shared_inputs().is_empty() => Vec::new(),
                None => {
                    return Err(PyBufferError::new_err(
                        "shared tangents are required for shared inputs",
                    ));
                }
            };
            let mapped = stage_f64_shaped_input(py, mapped, &mapped_shape, "mapped tangents")?;
            let native = py
                .detach(move || view.jvp(&shared, &mapped))
                .map_err(|error| diagnostic_error(py, &[error]))?;
            Ok(PyEvaluationMapJvp {
                native,
                plan: self.plan.clone(),
                seeds,
                positions,
            })
        })
    }
    /// First-order VJP with summed shared covectors and separate mapped covectors.
    #[pyo3(signature = (cotangents, *, seed_shape=None, point_axes=None, numerical_bytes_limit=67_108_864))]
    fn vjp(
        &self,
        py: Python<'_>,
        cotangents: &Bound<'_, PyAny>,
        seed_shape: Option<Vec<usize>>,
        point_axes: Option<Vec<usize>>,
        numerical_bytes_limit: usize,
    ) -> PyResult<PyEvaluationMapVjp> {
        panic_boundary(py, || {
            let seeds = seed_shape.unwrap_or_default();
            let positions =
                point_axes.unwrap_or_else(|| (0..self.plan.point_shape.len()).collect());
            let view = EvaluationMapProducts::new(
                &self.native,
                &self.plan.point_shape,
                &seeds,
                &positions,
                numerical_bytes_limit,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
            let mut shape = view.shape().to_vec();
            shape.push(self.plan.program.identity().output_dimension());
            let values = stage_f64_shaped_input(py, cotangents, &shape, "output cotangents")?;
            let native = py
                .detach(move || view.vjp(&values))
                .map_err(|error| diagnostic_error(py, &[error]))?;
            Ok(PyEvaluationMapVjp {
                native,
                plan: self.plan.clone(),
                seeds,
                positions,
            })
        })
    }
}

/// Failed or cancelled batch with inspectable indexed members, never a dense result.
#[pyclass(
    name = "EvaluationMapTerminalReport",
    module = "eqiora._eqiora",
    frozen
)]
pub(super) struct PyEvaluationMapTerminalReport {
    pub(super) native: EvaluationMapTerminalReport,
    pub(super) plan: PyEvaluationMapPlan,
}

#[pymethods]
impl PyEvaluationMapTerminalReport {
    fn __getitem__(
        &self,
        py: Python<'_>,
        index: isize,
    ) -> PyResult<Option<PyDifferentiableEvaluation>> {
        self.member(py, occurrence_index(index, self.plan.count())?)
    }
    fn __repr__(&self) -> String {
        format!(
            "EvaluationMapTerminalReport(stopped_index={}, cancelled={})",
            self.native.stopped_index(),
            self.native.is_cancelled()
        )
    }
    fn __len__(&self) -> usize {
        self.plan.count()
    }
    #[getter]
    fn plan(&self) -> PyEvaluationMapPlan {
        self.plan.clone()
    }
    #[getter]
    fn stopped_index(&self) -> usize {
        self.native.stopped_index()
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.native.is_cancelled()
    }
    #[getter]
    fn diagnostics(&self) -> Vec<PyDiagnostic> {
        self.native
            .diagnostics()
            .iter()
            .map(PyDiagnostic::from)
            .collect()
    }
    #[getter]
    fn statuses(&self) -> Vec<&'static str> {
        (0..self.plan.count())
            .map(|index| match self.native.occurrence(index) {
                Some(
                    EvaluationMapOccurrence::Accepted(_) | EvaluationMapOccurrence::Released(_),
                ) => "accepted",
                Some(EvaluationMapOccurrence::Failed(_)) => "failed",
                Some(EvaluationMapOccurrence::Cancelled) => "cancelled",
                _ => "not_started",
            })
            .collect()
    }
    /// Return only an accepted member; failed/unstarted positions return None.
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<Option<PyDifferentiableEvaluation>> {
        let occurrence = self
            .native
            .occurrence(index)
            .ok_or_else(|| PyIndexError::new_err("batch occurrence is out of range"))?;
        match occurrence {
            EvaluationMapOccurrence::Accepted(value) => {
                evaluation_result(py, value.clone()).map(Some)
            }
            _ => Ok(None),
        }
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompleteEvaluationMap>()?;
    module.add_class::<PyEvaluationMapTerminalReport>()
}
