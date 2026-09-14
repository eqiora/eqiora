//! Thin Python adapter to reference execution sessions and in-memory checkpoints.

use eqiora::api::ModelDocument;
use eqiora::entity::{Entity, kinds};
use eqiora::kernel::{KernelNode, SignalDirection};
use eqiora::sem::{ExecutionSession, ReferenceConfig};
use eqiora::{Id, RawId};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::{diagnostic_error, modeling::value_literal};

mod physical;

/// Mutable reference execution session bound to one exact compiled Model.
#[pyclass(
    name = "ExecutionSession",
    module = "eqiora._eqiora",
    skip_from_py_object
)]
pub(crate) struct PyExecutionSession {
    document: ModelDocument,
    value: ExecutionSession,
}

/// Immutable in-memory snapshot resumable only against its exact compiled Model.
#[pyclass(
    name = "ExecutionCheckpoint",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyExecutionCheckpoint {
    value: ExecutionSession,
}

fn session_repr(name: &str, value: &ExecutionSession) -> String {
    let next = value.next_tick().map_or_else(
        || "None".to_owned(),
        |time| format!("Fraction({}, {})", time.numerator(), time.denominator()),
    );
    format!("{name}(next_tick={next})")
}

#[pymethods]
impl PyExecutionCheckpoint {
    fn __repr__(&self) -> String {
        session_repr("ExecutionCheckpoint", &self.value)
    }
}

fn resolve<E: Entity>(document: &ModelDocument, name: &str) -> PyResult<RawId> {
    document
        .aliases()
        .get(name)
        .copied()
        .filter(|id| id.kind() == E::KIND)
        .or_else(|| {
            name.parse::<ulid::Ulid>()
                .ok()
                .map(|id| Id::<E>::from_ulid(id).erase())
        })
        .filter(|id| document.program().node(*id).is_some())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name:?} is not an exact {:?} in this Model",
                E::KIND
            ))
        })
}

pub(crate) fn start(
    py: Python<'_>,
    document: &ModelDocument,
    end_time_s: f64,
    max_step_s: f64,
    inputs: &Bound<'_, PyDict>,
) -> PyResult<PyExecutionSession> {
    let config = ReferenceConfig::new(end_time_s, max_step_s)
        .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?;
    let mut tables = Vec::new();
    let mut total = 0usize;
    for (name, table) in inputs.iter() {
        let name: String = name.extract()?;
        let input = resolve::<kinds::Port>(document, &name)?;
        let table = table.cast::<PyTuple>()?;
        if table.len() != 2 {
            return Err(PyTypeError::new_err(
                "input tables must be (clock name, ordered samples)",
            ));
        }
        let clock_name: String = table.get_item(0)?.extract()?;
        let clock = resolve::<kinds::ClockDomain>(document, &clock_name)?;
        let samples = table.get_item(1)?;
        if !(samples.is_instance_of::<PyTuple>() || samples.is_instance_of::<PyList>()) {
            return Err(PyTypeError::new_err(
                "input samples must be an ordered tuple or list",
            ));
        }
        total = total
            .checked_add(samples.len()?)
            .ok_or_else(|| PyValueError::new_err("sample count overflow"))?;
        if total > 1_000_000 {
            return Err(PyValueError::new_err(
                "input samples exceed the million-value bound",
            ));
        }
        let Some(KernelNode::Port(port)) = document.program().node(input) else {
            unreachable!()
        };
        let Some((SignalDirection::Input, value_type)) = port.signal_contract() else {
            return Err(PyValueError::new_err(
                "sample target must be a causal input Port",
            ));
        };
        let values = samples
            .try_iter()?
            .map(|value| value_literal::from_python(&value?, value_type.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        tables.push((input, clock, values));
    }
    let value = py
        .detach(|| document.execution_session(config, tables))
        .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
    Ok(PyExecutionSession {
        document: document.clone(),
        value,
    })
}

pub(crate) fn resume(
    py: Python<'_>,
    document: &ModelDocument,
    checkpoint: &PyExecutionCheckpoint,
) -> PyResult<PyExecutionSession> {
    let value = py
        .detach(|| document.resume_execution(&checkpoint.value))
        .map_err(|diagnostics| diagnostic_error(py, &diagnostics))?;
    Ok(PyExecutionSession {
        document: document.clone(),
        value,
    })
}

#[pymethods]
impl PyExecutionSession {
    fn __repr__(&self) -> String {
        session_repr("ExecutionSession", &self.value)
    }

    /// Advance to the next fully stabilized boundary; false means execution is complete.
    fn advance(&mut self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.value.advance())
            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))
    }

    /// Fresh observation of the last accepted boundary, without mutable execution state.
    #[getter]
    fn progress(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let progress = self.value.progress();
        let result = PyDict::new(py);
        result.set_item("model_time", progress.model_time())?;
        result.set_item("end_time", progress.end_time())?;
        result.set_item("accepted_steps", progress.accepted_steps())?;
        result.set_item("maximum_steps", progress.maximum_steps())?;
        Ok(result.unbind())
    }

    /// Exact activation ULIDs grouped by microstep at the last stabilized boundary.
    #[getter]
    fn activation_sequence(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let steps = self
            .value
            .activation_sequence()
            .iter()
            .map(|step| PyTuple::new(py, step.iter().map(|id| id.ulid().to_string())))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, steps)?.unbind())
    }

    fn advance_ticks(&mut self, py: Python<'_>, count: usize) -> PyResult<usize> {
        py.detach(|| self.value.advance_ticks(count))
            .map_err(|diagnostics| diagnostic_error(py, &diagnostics))
    }

    #[getter]
    fn next_tick(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.value
            .next_tick()
            .map(|time| {
                Ok(py
                    .import("fractions")?
                    .getattr("Fraction")?
                    .call1((time.numerator(), time.denominator()))?
                    .unbind())
            })
            .transpose()
    }

    fn checkpoint(&self) -> PyExecutionCheckpoint {
        PyExecutionCheckpoint {
            value: self.value.checkpoint(),
        }
    }

    /// Read accepted coherent-SI across value for an exact scalar physical Port.
    fn across(&self, name: &str) -> PyResult<Option<f64>> {
        physical::read(self, name, true)
    }

    /// Read accepted coherent-SI through value for an exact scalar physical Port.
    fn through(&self, name: &str) -> PyResult<Option<f64>> {
        physical::read(self, name, false)
    }

    fn field(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let id = self
            .document
            .field_ref(name)
            .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
            .id()
            .erase();
        self.value
            .field(id)
            .map(|value| value_literal::to_python(py, &value))
            .transpose()
    }

    fn output(&self, py: Python<'_>, name: &str, tick_index: u64) -> PyResult<Option<Py<PyTuple>>> {
        let id = resolve::<kinds::Port>(&self.document, name)?;
        self.value
            .output(id, tick_index)
            .map(|(time, value)| {
                let exact = py
                    .import("fractions")?
                    .getattr("Fraction")?
                    .call1((time.numerator(), time.denominator()))?
                    .unbind();
                Ok(PyTuple::new(py, [exact, value_literal::to_python(py, value)?])?.unbind())
            })
            .transpose()
    }
}
