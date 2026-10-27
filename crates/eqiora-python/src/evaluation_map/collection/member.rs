//! One immutable occurrence, attributed to its group's exact request.
use super::*;
use crate::error::PyDiagnostic;
use crate::execution_checkpoint::record::PyRecordedTimeCheckpoint;
use crate::sampling::PySamplingIdentity;
use eqiora::api::RecordedEvaluationOutcome;

/// An immutable attributed occurrence, including original failure or cancellation.
#[pyclass(name = "RecordedEvaluation", module = "eqiora._eqiora", frozen)]
pub(super) struct PyRecordedEvaluation {
    pub(super) collection: Arc<EvaluationCollection>,
    pub(super) group: usize,
    pub(super) index: usize,
}
impl PyRecordedEvaluation {
    fn group(&self) -> &EvaluationCollectionGroup {
        &self.collection.groups()[self.group]
    }
    fn outcome(&self) -> &RecordedEvaluationOutcome {
        self.group()
            .occurrence(self.index)
            .expect("admitted occurrence")
            .1
    }
}
#[pymethods]
impl PyRecordedEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "RecordedEvaluation(occurrence_id={:?}, status={:?})",
            self.occurrence_id(),
            self.status()
        )
    }
    #[getter]
    fn occurrence_id(&self) -> String {
        self.group()
            .occurrence(self.index)
            .expect("admitted occurrence")
            .0
            .to_string()
    }
    #[getter]
    fn status(&self) -> &'static str {
        match self.outcome() {
            RecordedEvaluationOutcome::Accepted(_) | RecordedEvaluationOutcome::RunAccepted(_) => {
                "accepted"
            }
            RecordedEvaluationOutcome::Failed(_) => "failed",
            RecordedEvaluationOutcome::Cancelled(_) => "cancelled",
            RecordedEvaluationOutcome::NotStarted => "not_started",
        }
    }
    #[getter]
    fn reference(&self) -> Option<&str> {
        match self.outcome() {
            RecordedEvaluationOutcome::Accepted(value) => Some(value.identity()),
            RecordedEvaluationOutcome::RunAccepted(value) => Some(value.identity()),
            _ => None,
        }
    }
    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self.outcome() {
            RecordedEvaluationOutcome::Accepted(value) => {
                crate::differentiation::primal_result(py, value.result().clone())
                    .and_then(|value| Py::new(py, value))
                    .map(|value| Some(value.into_any()))
            }
            RecordedEvaluationOutcome::RunAccepted(value) => {
                let plan = Py::new(py, PyPlan::from_native_artifact(py, value.plan().clone())?)?;
                time::recorded_result(py, plan.borrow(py), value).map(Some)
            }
            _ => Ok(None),
        }
    }
    /// The exact Plan of this accepted ordinary Run, independent of the group template.
    #[getter]
    fn plan(&self, py: Python<'_>) -> PyResult<Option<PyPlan>> {
        self.group()
            .member_plan(self.index)
            .cloned()
            .map(|plan| PyPlan::from_native_artifact(py, plan))
            .transpose()
    }
    #[getter]
    fn point(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        if let Some(points) = self.group().parameter_points() {
            return parameters::point(py, &points[self.index]).map(Some);
        }
        self.group()
            .points()
            .map(|points| values::project(py, &points[self.index]))
            .transpose()
    }
    #[getter]
    fn time_request_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.group()
            .time_requests()
            .map(|requests| {
                requests[self.index]
                    .to_bytes()
                    .map(|bytes| PyBytes::new(py, &bytes))
                    .map_err(|error| validation_error(py, &[error]))
            })
            .transpose()
    }
    #[getter]
    fn sampling(&self) -> Option<PySamplingIdentity> {
        self.group()
            .samples()
            .map(|samples| samples[self.index].identity())
            .or_else(|| {
                self.group()
                    .time_requests()?
                    .get(self.index)?
                    .sampling()
                    .map(|(_, sample)| sample)
            })
            .map(|native| PySamplingIdentity {
                native: native.clone(),
            })
    }
    /// Full Parameter generator provenance in its existing canonical codec.
    #[getter]
    fn parameter_sample_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.group()
            .samples()
            .map(|samples| {
                samples[self.index]
                    .to_bytes()
                    .map(|bytes| PyBytes::new(py, &bytes))
                    .map_err(|error| validation_error(py, &[error]))
            })
            .transpose()
    }
    #[getter]
    fn diagnostics(&self) -> Vec<PyDiagnostic> {
        match self.outcome() {
            RecordedEvaluationOutcome::Failed(values) => {
                values.iter().map(PyDiagnostic::from).collect()
            }
            _ => Vec::new(),
        }
    }
    /// Original input checkpoint; inspection does not admit continuation.
    #[getter]
    fn input_checkpoint(&self) -> Option<PyRecordedTimeCheckpoint> {
        self.group()
            .time_checkpoints()?
            .get(self.index)?
            .as_ref()
            .cloned()
            .map(PyRecordedTimeCheckpoint::from)
    }
    /// Checkpoint delivered by this cancelled occurrence, distinct from its input.
    #[getter]
    fn checkpoint(&self) -> Option<PyRecordedTimeCheckpoint> {
        match self.outcome() {
            RecordedEvaluationOutcome::Cancelled(Some(value)) => {
                Some(PyRecordedTimeCheckpoint::from(value.as_ref().clone()))
            }
            _ => None,
        }
    }
}
