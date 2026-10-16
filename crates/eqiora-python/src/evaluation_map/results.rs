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
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
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
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.native.time_result(index) {
            return time::result(py, &self.plan, value);
        }
        let value = self
            .native
            .evaluation(index)
            .ok_or_else(|| PyIndexError::new_err("batch occurrence is out of range"))?;
        evaluation_result(py, value.clone())
            .and_then(|value| Py::new(py, value))
            .map(Py::into_any)
    }
    /// Complete immutable output leaves, preserving declared point axes and intrinsic shapes.
    fn primal(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let members = self.native.members().ok_or_else(|| {
            PyRuntimeError::new_err("Python batch requires retained native members")
        })?;
        let outputs = members
            .iter()
            .map(|member| member.primal())
            .collect::<Vec<_>>();
        values::project_collection(
            py,
            self.plan.evaluation_program()?.identity().outputs(),
            &self.plan.point_shape,
            members.len(),
            |row, leaf| {
                outputs[row]
                    .output()
                    .leaf(leaf)
                    .expect("admitted leaf")
                    .components()
            },
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
            let (shared_signatures, mapped_signatures) = input_signatures(
                self.plan.evaluation_program()?,
                self.plan
                    .native
                    .shared_inputs()
                    .map_err(|error| diagnostic_error(py, &[error]))?,
            );
            let empty = PyTuple::empty(py);
            let shared = values::stage_collection(
                py,
                &shared_signatures,
                &seeds,
                shared.unwrap_or(empty.as_any()),
                "shared tangents",
                numerical_bytes_limit,
                DifferentiableTangent::from_components,
            )?;
            let mapped = values::stage_collection(
                py,
                &mapped_signatures,
                view.shape(),
                mapped,
                "mapped tangents",
                numerical_bytes_limit,
                DifferentiableTangent::from_components,
            )?;
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
            let values = values::stage_collection(
                py,
                self.plan.evaluation_program()?.identity().outputs(),
                view.shape(),
                cotangents,
                "output cotangents",
                numerical_bytes_limit,
                DifferentiableCotangent::from_components,
            )?;
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
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Option<Py<PyAny>>> {
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
                    EvaluationMapOccurrence::Accepted(_)
                    | EvaluationMapOccurrence::Released(_)
                    | EvaluationMapOccurrence::TimeAccepted(_)
                    | EvaluationMapOccurrence::TimeReleased(_),
                ) => "accepted",
                Some(EvaluationMapOccurrence::Failed(_)) => "failed",
                Some(
                    EvaluationMapOccurrence::Cancelled | EvaluationMapOccurrence::TimeCancelled(_),
                ) => "cancelled",
                _ => "not_started",
            })
            .collect()
    }
    /// Return only an accepted member; failed/unstarted positions return None.
    fn member(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        let occurrence = self
            .native
            .occurrence(index)
            .ok_or_else(|| PyIndexError::new_err("batch occurrence is out of range"))?;
        match occurrence {
            EvaluationMapOccurrence::Accepted(value) => evaluation_result(py, value.clone())
                .and_then(|value| Py::new(py, value))
                .map(|value| Some(value.into_any())),
            EvaluationMapOccurrence::TimeAccepted(value) => {
                time::result(py, &self.plan, value).map(Some)
            }
            _ => Ok(None),
        }
    }
    /// Atomic original time checkpoint for one interrupted member.
    fn checkpoint(
        &self,
        index: usize,
    ) -> PyResult<Option<crate::execution_checkpoint::PyExecutionCheckpoint>> {
        if index >= self.plan.count() {
            return Err(PyIndexError::new_err("batch occurrence is out of range"));
        }
        Ok(self.native.time_checkpoint(index).map(|value| {
            crate::execution_checkpoint::PyExecutionCheckpoint {
                value: crate::execution_checkpoint::CheckpointValue::Time(Box::new(value.clone())),
            }
        }))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompleteEvaluationMap>()?;
    module.add_class::<PyEvaluationMapTerminalReport>()
}
