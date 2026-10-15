//! Python ownership projection; all numerical execution remains native.

use std::sync::atomic::{AtomicBool, Ordering};

use eqiora::api::{
    CompleteEvaluationMap, EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapProducts,
    EvaluationMapTerminalReport,
};
use eqiora::{Id, entity::kinds};
use pyo3::exceptions::{PyBufferError, PyIndexError, PyRuntimeError};
use pyo3::types::PyTuple;

use super::*;
use crate::model::PyModelParameterRef;
use pyo3::types::PySequence;

mod products;
mod results;
use results::{PyCompleteEvaluationMap, PyEvaluationMapTerminalReport};

type Parameter = Id<kinds::Parameter>;

/// Cooperative native cancellation, observed only between batch occurrences.
#[pyclass(
    name = "EvaluationMapCancellation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
struct PyEvaluationMapCancellation {
    requested: Arc<AtomicBool>,
}

#[pymethods]
impl PyEvaluationMapCancellation {
    fn __repr__(&self) -> String {
        format!("EvaluationMapCancellation(requested={})", self.requested())
    }
    #[new]
    fn new() -> Self {
        Self::default()
    }
    /// Request cancellation without a Python callback or solver interruption.
    fn cancel(&self) {
        self.requested.store(true, Ordering::Release);
    }
    #[getter]
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

/// Immutable planned batch, with input ownership fixed before any solve.
#[pyclass(
    name = "EvaluationMapPlan",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyEvaluationMapPlan {
    native: Arc<EvaluationMapPlan>,
    program: Arc<DifferentiableProgram>,
    point_shape: Vec<usize>,
}

impl PyEvaluationMapPlan {
    fn count(&self) -> usize {
        self.native.points().len()
    }
    fn coordinates(&self, index: usize) -> PyResult<Vec<usize>> {
        if index >= self.count() {
            return Err(PyIndexError::new_err("batch occurrence is out of range"));
        }
        let mut remainder = index;
        let mut coordinates = vec![0; self.point_shape.len()];
        for (coordinate, &extent) in coordinates.iter_mut().zip(&self.point_shape).rev() {
            *coordinate = remainder % extent;
            remainder /= extent;
        }
        Ok(coordinates)
    }
}

#[pymethods]
impl PyEvaluationMapPlan {
    fn __repr__(&self) -> String {
        format!(
            "EvaluationMapPlan(point_shape={:?}, occurrences={})",
            self.point_shape,
            self.count()
        )
    }
    fn __len__(&self) -> usize {
        self.count()
    }
    /// Read one flat request occurrence, independently of point-grid rank.
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyTuple>> {
        let index = occurrence_index(index, self.count())?;
        values::project(py, &self.native.points()[index])
    }
    #[getter]
    fn program(&self) -> PyDifferentiableProgram {
        PyDifferentiableProgram {
            value: self.program.clone(),
        }
    }
    #[getter]
    fn point_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.point_shape)
    }
    #[getter]
    fn shared_input_ids(&self) -> Vec<String> {
        ids(self.native.shared_inputs())
    }
    #[getter]
    fn mapped_input_ids(&self) -> Vec<String> {
        let (_, mapped) = input_signatures(&self.program, self.native.shared_inputs());
        mapped
            .iter()
            .map(|value| value.entity().ulid().to_string())
            .collect()
    }
    #[getter]
    fn estimated_storage_bytes(&self) -> usize {
        self.native.estimated_storage_bytes()
    }
    /// Exact frozen complete input leaves in Program order.
    #[getter]
    fn points(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        values::project_collection(
            py,
            self.program.identity().inputs(),
            &self.point_shape,
            self.count(),
            |row, leaf| {
                self.native.points()[row]
                    .leaf(leaf)
                    .expect("admitted leaf")
                    .components()
            },
        )
    }
    fn occurrence_coordinates(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.coordinates(index)?)
    }
    /// Execute once through the native ordered map; a terminal prefix is not complete.
    #[pyo3(signature = (*, cancellation=None))]
    fn execute(
        &self,
        py: Python<'_>,
        cancellation: Option<PyRef<'_, PyEvaluationMapCancellation>>,
    ) -> PyResult<Py<PyAny>> {
        panic_boundary(py, || {
            let requested = cancellation.map_or_else(
                || Arc::new(AtomicBool::new(false)),
                |token| token.requested.clone(),
            );
            let plan = self.native.clone();
            let outcome = py.detach(move || {
                plan.execute_with_cancellation(|| requested.load(Ordering::Acquire))
            });
            match outcome {
                Ok(value) => Ok(Py::new(
                    py,
                    PyCompleteEvaluationMap {
                        native: Arc::new(value),
                        plan: self.clone(),
                    },
                )?
                .into_any()),
                Err(value) => Ok(Py::new(
                    py,
                    PyEvaluationMapTerminalReport {
                        native: value,
                        plan: self.clone(),
                    },
                )?
                .into_any()),
            }
        })
    }
}

pub(super) fn plan(
    py: Python<'_>,
    program: &PyDifferentiableProgram,
    mapped: &Bound<'_, PyAny>,
    point_shape: &[usize],
    shared_inputs: Option<&Bound<'_, PyAny>>,
    shared: Option<&Bound<'_, PyAny>>,
    limit: usize,
) -> PyResult<PyEvaluationMapPlan> {
    let mut selected = Vec::new();
    if let Some(inputs) = shared_inputs {
        let sequence = inputs.cast::<PySequence>()?;
        if sequence.len()? > program.value.identity().inputs().len() {
            return Err(PyBufferError::new_err(
                "shared_inputs exceeds Program input count",
            ));
        }
        for (index, item) in sequence.try_iter()?.enumerate() {
            let item = item?;
            let input = item.extract::<PyRef<'_, PyModelParameterRef>>()?;
            if input.value.model() != program.value.identity().model()
                || !program
                    .value
                    .identity()
                    .inputs()
                    .iter()
                    .any(|signature| signature.entity() == input.value.id().erase())
                || selected.contains(&input.value.id())
            {
                return Err(PyBufferError::new_err(format!(
                    "shared input {index} must be a distinct ParameterRef of this exact Program/Model"
                )));
            }
            selected.push(input.value.id());
        }
    }
    let (shared_signatures, mapped_signatures) = input_signatures(&program.value, &selected);
    let empty = PyTuple::empty(py);
    let shared_values = values::stage(
        py,
        &shared_signatures,
        shared.unwrap_or(empty.as_any()),
        "shared parameters",
        DifferentiableValues::from_components,
    )?;
    let points = values::stage_collection(
        py,
        &mapped_signatures,
        point_shape,
        mapped,
        "mapped parameters",
        limit,
        DifferentiableValues::from_components,
    )?;
    let native = EvaluationMapPlan::from_partition(
        program.value.clone(),
        &shared_values,
        &points,
        EvaluationMapExecutionPolicy::retained(limit),
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    Ok(PyEvaluationMapPlan {
        native: Arc::new(native),
        program: program.value.clone(),
        point_shape: point_shape.to_vec(),
    })
}

// Project the native Plan's declaration; semantic partition admission remains native.
fn input_signatures(
    program: &DifferentiableProgram,
    shared: &[Parameter],
) -> (
    Vec<eqiora::api::DifferentiableValueSignature>,
    Vec<eqiora::api::DifferentiableValueSignature>,
) {
    let inputs = program.identity().inputs();
    let shared_values = shared
        .iter()
        .map(|id| {
            inputs
                .iter()
                .find(|value| value.entity() == id.erase())
                .expect("validated shared input")
                .clone()
        })
        .collect();
    let mapped_values = inputs
        .iter()
        .filter(|value| !shared.iter().any(|id| value.entity() == id.erase()))
        .cloned()
        .collect();
    (shared_values, mapped_values)
}

fn tuple(py: Python<'_>, shape: &[usize]) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, shape)?.unbind())
}

fn occurrence_index(index: isize, len: usize) -> PyResult<usize> {
    let index = if index < 0 {
        len.checked_add_signed(index)
    } else {
        Some(index as usize)
    };
    index
        .filter(|&index| index < len)
        .ok_or_else(|| PyIndexError::new_err("batch occurrence is out of range"))
}
fn ids(inputs: &[Parameter]) -> Vec<String> {
    inputs.iter().map(|id| id.ulid().to_string()).collect()
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEvaluationMapPlan>()?;
    module.add_class::<PyEvaluationMapCancellation>()?;
    results::register(module)?;
    products::register(module)
}
