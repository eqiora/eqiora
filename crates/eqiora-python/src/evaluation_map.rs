//! Python ownership projection; all numerical execution remains native.

use std::sync::atomic::{AtomicBool, Ordering};

use eqiora::api::{
    CompleteEvaluationMap, EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapProducts,
    EvaluationMapTerminalReport,
};
use eqiora::{Id, entity::kinds};
use pyo3::exceptions::{PyBufferError, PyIndexError, PyRuntimeError};
use pyo3::types::PyTuple;

use crate::differentiation::{
    PyDifferentiableJvp, PyDifferentiableProgram, PyDifferentiableVjp, evaluation_result,
    jvp_result, values, vjp_result,
};
use crate::error::{diagnostic_error, panic_boundary};
use crate::model::PyModelParameterRef;
use eqiora::api::{
    DifferentiableCotangent, DifferentiableProgram, DifferentiableTangent, DifferentiableValues,
};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use std::sync::Arc;

mod collection;
use collection::PyEvaluationCollection;
mod parameters;
mod products;
mod results;
mod time;
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
    context: MapContext,
    point_shape: Vec<usize>,
}

#[derive(Clone)]
enum MapContext {
    Evaluations(Arc<DifferentiableProgram>),
    Time(Arc<Py<crate::common_plan::PyPlan>>),
    Parameters(Arc<Py<crate::common_plan::PyPlan>>),
}

impl PyEvaluationMapPlan {
    fn evaluation_program(&self) -> PyResult<&Arc<DifferentiableProgram>> {
        match &self.context {
            MapContext::Evaluations(program) => Ok(program),
            MapContext::Time(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "time maps do not admit steady parameter or derivative operations",
            )),
            MapContext::Parameters(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "stationary Run maps do not admit derivative operations",
            )),
        }
    }
    fn count(&self) -> usize {
        self.native.len()
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
    /// Execute once and capture original deliveries, including incomplete membership.
    #[pyo3(signature = (*, cancellation=None))]
    fn execute_recorded(
        &self,
        py: Python<'_>,
        cancellation: Option<PyRef<'_, PyEvaluationMapCancellation>>,
    ) -> PyResult<PyEvaluationCollection> {
        panic_boundary(py, || {
            let requested = cancellation.map_or_else(
                || Arc::new(AtomicBool::new(false)),
                |token| token.requested.clone(),
            );
            py.detach(|| {
                self.native
                    .execute_recorded(|| requested.load(Ordering::Acquire))
            })
            .map(PyEvaluationCollection::from)
            .map_err(|error| crate::error::validation_error(py, &[error]))
        })
    }

    /// Ordered ordinary stationary Runs at complete typed Parameter points.
    /// Values use the selected Parameters' declared units and complete intrinsic shapes.
    /// Harmonic Parameters must be selected from the retained original Model.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (plan, parameters, points, /, *, chunk_size=1, workers=1, storage_bytes_limit=67_108_864, member_bytes_limit=8_388_608))]
    fn from_parameters(
        py: Python<'_>,
        plan: Py<crate::common_plan::PyPlan>,
        parameters: Vec<PyRef<'_, PyModelParameterRef>>,
        points: &Bound<'_, PyAny>,
        chunk_size: usize,
        workers: usize,
        storage_bytes_limit: usize,
        member_bytes_limit: usize,
    ) -> PyResult<Self> {
        panic_boundary(py, || {
            parameters::plan(
                py,
                plan,
                parameters,
                points,
                chunk_size,
                workers,
                storage_bytes_limit,
                member_bytes_limit,
            )
        })
    }
    /// Selected original Parameters in the exact point-column order.
    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        self.native
            .parameters()
            .map(|ids| {
                parameters::refs(
                    py,
                    self.native
                        .parameter_template()
                        .expect("parameter inventory owns a template"),
                    ids,
                )
            })
            .transpose()
    }
    #[getter]
    fn parameter_template(&self, py: Python<'_>) -> Option<Py<crate::common_plan::PyPlan>> {
        match &self.context {
            MapContext::Parameters(plan) => Some(plan.clone_ref(py)),
            _ => None,
        }
    }
    /// Independent ordinary time Runs at one fixed observation schedule.
    /// Checkpoints bind the supplied original requests; mapping adds no trajectory AD.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (plan, states, /, *, until_s, output_times_s, master_seed=None, sampling=None, checkpoints=None, chunk_size=1, workers=1, storage_bytes_limit=67_108_864, member_bytes_limit=8_388_608))]
    fn from_time(
        py: Python<'_>,
        plan: Py<crate::common_plan::PyPlan>,
        states: Vec<Py<crate::trajectory::PyState>>,
        until_s: f64,
        output_times_s: Vec<f64>,
        master_seed: Option<&[u8]>,
        sampling: Option<Vec<crate::sampling::PySamplingIdentity>>,
        checkpoints: Option<Vec<Option<Py<crate::execution_checkpoint::PyExecutionCheckpoint>>>>,
        chunk_size: usize,
        workers: usize,
        storage_bytes_limit: usize,
        member_bytes_limit: usize,
    ) -> PyResult<Self> {
        panic_boundary(py, || {
            time::plan(
                py,
                plan,
                states,
                until_s,
                output_times_s,
                master_seed,
                sampling,
                checkpoints,
                chunk_size,
                workers,
                storage_bytes_limit,
                member_bytes_limit,
            )
        })
    }
    #[getter]
    fn time_plan(&self, py: Python<'_>) -> Option<Py<crate::common_plan::PyPlan>> {
        match &self.context {
            MapContext::Time(plan) => Some(plan.clone_ref(py)),
            MapContext::Evaluations(_) | MapContext::Parameters(_) => None,
        }
    }
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
        if let MapContext::Time(plan) = &self.context {
            let request = &self.native.time_requests().expect("time context")[index];
            return Py::new(
                py,
                crate::trajectory::PyState::from_common_time(
                    py,
                    &plan.borrow(py),
                    request.state().clone(),
                    Some(request.identity()),
                ),
            )
            .map(Py::into_any);
        }
        if let Some(points) = self.native.parameter_points() {
            return parameters::point(py, &points[index]).map(Py::into_any);
        }
        values::project(
            py,
            &self
                .native
                .points()
                .map_err(|error| diagnostic_error(py, &[error]))?[index],
        )
        .map(Py::into_any)
    }
    #[getter]
    fn program(&self) -> PyResult<PyDifferentiableProgram> {
        Ok(PyDifferentiableProgram {
            value: self.evaluation_program()?.clone(),
        })
    }
    #[getter]
    fn point_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, &self.point_shape)
    }
    #[getter]
    fn shared_input_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(ids(self
            .native
            .shared_inputs()
            .map_err(|error| diagnostic_error(py, &[error]))?))
    }
    #[getter]
    fn mapped_input_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let (_, mapped) = input_signatures(
            self.evaluation_program()?,
            self.native
                .shared_inputs()
                .map_err(|error| diagnostic_error(py, &[error]))?,
        );
        Ok(mapped
            .iter()
            .map(|value| value.entity().ulid().to_string())
            .collect())
    }
    #[getter]
    fn estimated_storage_bytes(&self) -> usize {
        self.native.estimated_storage_bytes()
    }
    /// Exact frozen complete input leaves in Program order.
    #[getter]
    fn points(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        if let Some(points) = self.native.parameter_points() {
            return PyTuple::new(
                py,
                points
                    .iter()
                    .map(|point| parameters::point(py, point))
                    .collect::<PyResult<Vec<_>>>()?,
            )
            .map(Bound::unbind);
        }
        let points = self
            .native
            .points()
            .map_err(|error| diagnostic_error(py, &[error]))?;
        values::project_collection(
            py,
            self.evaluation_program()?.identity().inputs(),
            &self.point_shape,
            self.count(),
            |row, leaf| points[row].leaf(leaf).expect("admitted leaf").components(),
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
        context: MapContext::Evaluations(program.value.clone()),
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
    collection::register(module)?;
    products::register(module)
}
