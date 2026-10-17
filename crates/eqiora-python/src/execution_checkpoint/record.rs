//! Checkpoint content may be inspected before explicit numerical restart admission.
use super::*;
use eqiora_numerics::RecordedTimeCheckpoint;

/// Canonical checkpoint content; numerical continuation requires explicit restore().
#[pyclass(name = "RecordedTimeCheckpoint", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyRecordedTimeCheckpoint {
    native: RecordedTimeCheckpoint,
}
impl From<RecordedTimeCheckpoint> for PyRecordedTimeCheckpoint {
    fn from(native: RecordedTimeCheckpoint) -> Self {
        Self { native }
    }
}
#[pymethods]
impl PyRecordedTimeCheckpoint {
    fn __repr__(&self) -> String {
        format!(
            "RecordedTimeCheckpoint(identity={:?})",
            self.native.identity()
        )
    }
    #[getter]
    fn identity(&self) -> &str {
        self.native.identity()
    }
    #[getter]
    fn request_identity(&self) -> &str {
        self.native.request().identity()
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.native.as_bytes())
    }
    #[staticmethod]
    #[pyo3(signature = (plan, data, /))]
    fn from_bytes(py: Python<'_>, plan: &PyPlan, data: &[u8]) -> PyResult<Self> {
        let plan = plan.native().as_time().ok_or_else(|| {
            PyTypeError::new_err("checkpoint content requires its exact time Plan")
        })?;
        RecordedTimeCheckpoint::from_bytes(data, plan)
            .map(Self::from)
            .map_err(|error| validation_error(py, &[error]))
    }
    /// Validate physical/path consistency and restart compatibility before execution.
    fn restore(&self, py: Python<'_>) -> PyResult<PyExecutionCheckpoint> {
        py.detach(|| self.native.restore())
            .map(|value| PyExecutionCheckpoint {
                value: CheckpointValue::Time(Box::new(value)),
            })
            .map_err(|error| validation_error(py, &[error]))
    }
}
