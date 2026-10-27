//! Time input/result projection; scheduling and Run execution remain native.
use super::*;
use crate::common_plan::PyPlan;
use crate::execution_checkpoint::PyExecutionCheckpoint;
use crate::sampling::PySamplingIdentity;
use crate::trajectory::PyState;
use eqiora::api::EvaluationMapRetention;
use eqiora::realization::Target;
use eqiora_numerics::{CommonResult, CommonTimeRunRequest};
use pyo3::exceptions::{PyTypeError, PyValueError};
use std::num::NonZeroUsize;

#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    py: Python<'_>,
    plan: Py<PyPlan>,
    states: Vec<Py<PyState>>,
    until_s: f64,
    output_times_s: Vec<f64>,
    master_seed: Option<&[u8]>,
    sampling: Option<Vec<PySamplingIdentity>>,
    checkpoints: Option<Vec<Option<Py<PyExecutionCheckpoint>>>>,
    chunk_size: usize,
    workers: usize,
    storage_bytes_limit: usize,
    member_bytes_limit: usize,
) -> PyResult<PyEvaluationMapPlan> {
    let count = states.len();
    let policy = EvaluationMapExecutionPolicy::new(
        chunk_size,
        Target::HostCpu {
            threads: NonZeroUsize::new(workers)
                .ok_or_else(|| PyValueError::new_err("workers must be positive"))?,
        },
        EvaluationMapRetention::Retain,
        storage_bytes_limit,
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    let master = match (master_seed, &sampling) {
        (None, None) => None,
        (Some(bytes), Some(samples)) if samples.len() == count => Some(
            <[u8; 32]>::try_from(bytes)
                .map_err(|_| PyValueError::new_err("master_seed must contain exactly 32 bytes"))?,
        ),
        _ => {
            return Err(PyValueError::new_err(
                "master_seed and one sampling identity per occurrence must be supplied together",
            ));
        }
    };
    let native_plan = Arc::new(
        plan.borrow(py)
            .native()
            .as_time()
            .ok_or_else(|| PyTypeError::new_err("time mapping requires a time Plan"))?
            .clone(),
    );
    let requests = states
        .iter()
        .enumerate()
        .map(|(index, state)| {
            let state = state.borrow(py);
            let state = state
                .common_time_native()
                .ok_or_else(|| PyTypeError::new_err("time mapping requires time States"))?;
            match (&master, &sampling) {
                (Some(master), Some(samples)) => CommonTimeRunRequest::sampled(
                    (*native_plan).clone(),
                    state.clone(),
                    until_s,
                    output_times_s.clone(),
                    *master,
                    samples[index].native.clone(),
                ),
                _ => CommonTimeRunRequest::new(
                    (*native_plan).clone(),
                    state.clone(),
                    until_s,
                    output_times_s.clone(),
                ),
            }
            .map_err(|error| diagnostic_error(py, &[error]))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let checkpoints = checkpoints
        .map(|values| {
            values
                .into_iter()
                .map(|value| {
                    value
                        .map(|checkpoint| checkpoint.borrow(py).time().cloned())
                        .transpose()
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let native = EvaluationMapPlan::from_time_requests(
        native_plan,
        &requests,
        checkpoints.as_deref(),
        policy,
        member_bytes_limit,
        crate::execution::run_mapped_time,
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    Ok(PyEvaluationMapPlan {
        native: Arc::new(native),
        context: MapContext::Time(Arc::new(plan)),
        point_shape: vec![count],
    })
}

pub(super) fn result(
    py: Python<'_>,
    plan: &PyEvaluationMapPlan,
    value: &CommonResult,
) -> PyResult<Py<PyAny>> {
    let _ = plan;
    let member_plan = Py::new(py, PyPlan::from_native_artifact(py, value.plan().clone())?)?;
    recorded_result(py, member_plan.borrow(py), value)
}

pub(super) fn recorded_result(
    py: Python<'_>,
    plan: PyRef<'_, crate::common_plan::PyPlan>,
    value: &CommonResult,
) -> PyResult<Py<PyAny>> {
    let identity = crate::execution::RunIdentity::from_common_result(value)
        .ok_or_else(|| PyTypeError::new_err("accepted Result lost its original Run identity"))?;
    crate::result::materialize_common_result(py, plan, identity, value.clone(), None)
        .and_then(|result| Py::new(py, result))
        .map(Py::into_any)
}
