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
use crate::array::{stage_f64_shaped_input, stage_f64_tensor_input};
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
    shared: Vec<Parameter>,
    point_shape: Vec<usize>,
}

impl PyEvaluationMapPlan {
    fn count(&self) -> usize {
        self.native.points().len()
    }
    fn mapped(&self) -> Vec<Parameter> {
        self.program
            .identity()
            .inputs()
            .iter()
            .copied()
            .filter(|id| !self.shared.contains(id))
            .collect()
    }
    fn with_components(&self, count: usize) -> Vec<usize> {
        let mut shape = self.point_shape.clone();
        shape.push(count);
        shape
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
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let index = occurrence_index(index, self.count())?;
        let point = &self.native.points()[index];
        readonly_array(py, point.values().to_vec(), &[point.values().len()])
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
    fn input_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(
            py,
            &self.with_components(self.program.identity().input_dimension()),
        )
    }
    #[getter]
    fn output_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(
            py,
            &self.with_components(self.program.identity().output_dimension()),
        )
    }
    #[getter]
    fn shared_input_ids(&self) -> Vec<String> {
        ids(&self.shared)
    }
    #[getter]
    fn mapped_input_ids(&self) -> Vec<String> {
        ids(&self.mapped())
    }
    #[getter]
    fn estimated_storage_bytes(&self) -> usize {
        self.native.estimated_storage_bytes()
    }
    /// Exact frozen complete points in Program coordinate order.
    #[getter]
    fn points(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        readonly_array(
            py,
            self.native
                .points()
                .iter()
                .flat_map(|p| p.values().iter().copied())
                .collect(),
            &self.with_components(self.program.identity().input_dimension()),
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
    shared_inputs: Option<&Bound<'_, PyAny>>,
    shared: Option<&Bound<'_, PyAny>>,
    limit: usize,
) -> PyResult<PyEvaluationMapPlan> {
    let mut selected = Vec::new();
    if let Some(inputs) = shared_inputs {
        let sequence = inputs.cast::<PySequence>()?;
        if sequence.len()? > program.value.identity().input_dimension() {
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
                    .contains(&input.value.id())
                || selected.contains(&input.value.id())
            {
                return Err(PyBufferError::new_err(format!(
                    "shared input {index} must be a distinct ParameterRef of this exact Program/Model"
                )));
            }
            selected.push(input.value.id());
        }
    }
    let shared_values = match shared {
        Some(value) => stage_f64_shaped_input(py, value, &[selected.len()], "shared parameters")?,
        None if selected.is_empty() => Vec::new(),
        None => {
            return Err(PyBufferError::new_err(
                "shared parameters are required for shared_inputs",
            ));
        }
    };
    let (mapped_values, shape) = stage_f64_tensor_input(py, mapped, limit, "mapped parameters")?;
    let Some((&components, point_shape)) = shape.split_last() else {
        return Err(PyBufferError::new_err(
            "mapped parameters require a trailing Program coordinate axis",
        ));
    };
    let expected = program.value.identity().input_dimension() - selected.len();
    if components != expected {
        return Err(PyBufferError::new_err(format!(
            "mapped parameters trailing coordinate axis requires {expected}, received {components}; partial axis sharing is unsupported"
        )));
    }
    let native = EvaluationMapPlan::from_partition(
        program.value.clone(),
        &selected,
        &shared_values,
        &mapped_values,
        point_shape,
        EvaluationMapExecutionPolicy::retained(limit),
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    Ok(PyEvaluationMapPlan {
        native: Arc::new(native),
        program: program.value.clone(),
        shared: selected,
        point_shape: point_shape.to_vec(),
    })
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
fn readonly_array(py: Python<'_>, values: Vec<f64>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    let buffer = PyArrayBuffer::from_owned_result(py, values)?;
    let array = buffer.bind(py).borrow().numpy_array(py)?;
    Ok(array
        .bind(py)
        .call_method1("reshape", (shape.to_vec(),))?
        .unbind())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEvaluationMapPlan>()?;
    module.add_class::<PyEvaluationMapCancellation>()?;
    results::register(module)?;
    products::register(module)
}
