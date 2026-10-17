//! One Python checkpoint projection for reference and time execution owners.
use eqiora::sem::ExecutionSession;
use eqiora_numerics::CommonTimeCheckpoint;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::common_plan::PyPlan;
use crate::error::{compatibility_error, validation_error};
use crate::model_io::{
    ArtifactFileSpec, read_artifact_bytes, unicode_artifact_path, write_artifact_bytes,
};

pub(crate) mod record;
use record::PyRecordedTimeCheckpoint;

const FILE_SPEC: ArtifactFileSpec = ArtifactFileSpec {
    artifact_name: "ExecutionCheckpoint",
    extension: "eqcheckpoint",
    staging_name: "checkpoint",
    max_bytes: 512 * 1024 * 1024,
};

#[derive(Debug)]
pub(crate) enum CheckpointValue {
    Reference(Box<ExecutionSession>),
    Time(Box<CommonTimeCheckpoint>),
}

/// Exact continuation data; reference checkpoints currently remain in memory.
#[pyclass(
    name = "ExecutionCheckpoint",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyExecutionCheckpoint {
    pub(crate) value: CheckpointValue,
}

impl PyExecutionCheckpoint {
    pub(crate) fn time(&self) -> PyResult<&CommonTimeCheckpoint> {
        match &self.value {
            CheckpointValue::Time(value) => Ok(value),
            CheckpointValue::Reference(_) => Err(PyTypeError::new_err(
                "reference checkpoints resume through Model.resume_execution and remain in memory",
            )),
        }
    }

    fn decode(py: Python<'_>, plan: &PyPlan, bytes: &[u8]) -> PyResult<Self> {
        let plan = plan.native().as_time().ok_or_else(|| {
            PyTypeError::new_err("durable ExecutionCheckpoint requires its exact finite-time Plan")
        })?;
        let value = CommonTimeCheckpoint::from_bytes(bytes, plan)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self {
            value: CheckpointValue::Time(Box::new(value)),
        })
    }
}

#[pymethods]
impl PyExecutionCheckpoint {
    fn record(&self, py: Python<'_>) -> PyResult<PyRecordedTimeCheckpoint> {
        self.time()?
            .record()
            .map(PyRecordedTimeCheckpoint::from)
            .map_err(|error| validation_error(py, &[error]))
    }

    fn __repr__(&self) -> String {
        match &self.value {
            CheckpointValue::Reference(value) => {
                crate::execution_session::session_repr("ExecutionCheckpoint", value)
            }
            CheckpointValue::Time(value) => format!(
                "ExecutionCheckpoint(identity={:?}, accepted_steps={})",
                value.identity(),
                value.accepted_steps()
            ),
        }
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .time()?
            .to_bytes()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    #[pyo3(signature = (plan, data, /))]
    fn from_bytes(py: Python<'_>, plan: &PyPlan, data: &[u8]) -> PyResult<Self> {
        Self::decode(py, plan, data)
    }

    /// Atomically save complete time execution; a State alone is a fresh restart.
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = self
            .time()?
            .to_bytes()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        py.detach(move || write_artifact_bytes(&path, &bytes, FILE_SPEC))
            .map_err(|diagnostic| compatibility_error(py, &[diagnostic]))
    }

    #[staticmethod]
    #[pyo3(signature = (plan, path, /))]
    fn read(py: Python<'_>, plan: &PyPlan, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = py
            .detach(move || read_artifact_bytes(&path, FILE_SPEC))
            .map_err(|diagnostic| compatibility_error(py, &[diagnostic]))?;
        Self::decode(py, plan, &bytes)
    }
}
