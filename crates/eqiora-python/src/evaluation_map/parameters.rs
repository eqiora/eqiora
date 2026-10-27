//! Typed Parameter transport; native map admission and ordinary Runs own semantics.
use super::*;
use crate::common_plan::PyPlan;
use eqiora::api::EvaluationMapRetention;
use eqiora::artifact::CanonicalModelArtifact;
use pyo3::exceptions::{PyTypeError, PyValueError};

#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    py: Python<'_>,
    plan: Py<PyPlan>,
    parameters: Vec<PyRef<'_, PyModelParameterRef>>,
    points: &Bound<'_, PyAny>,
    chunk_size: usize,
    workers: usize,
    storage_bytes_limit: usize,
    member_bytes_limit: usize,
) -> PyResult<PyEvaluationMapPlan> {
    let native = plan.borrow(py).native().clone();
    let original = crate::common_plan::harmonic::original_model(&native)
        .unwrap_or_else(|| native.model_artifact());
    let reference = original
        .artifact_reference()
        .map_err(|error| diagnostic_error(py, &[error]))?;
    let (transaction, _) = original
        .to_transaction()
        .map_err(|errors| diagnostic_error(py, &errors))?;
    let mut point_bytes = size_of::<Vec<eqiora::ValueLiteral>>();
    let mut selected = Vec::with_capacity(parameters.len());
    let mut types = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        if parameter.value.model() != &reference {
            return Err(PyTypeError::new_err(
                "stationary map Parameters must belong to the retained original Model",
            ));
        }
        let id = parameter.value.id();
        let value = transaction
            .ops()
            .iter()
            .find_map(|op| match op {
                eqiora::graph::Op::SetValue { target, value } if *target == id.erase() => {
                    Some(value)
                }
                _ => None,
            })
            .ok_or_else(|| {
                PyTypeError::new_err("selected Parameter has no complete typed value")
            })?;
        point_bytes = value
            .component_count()
            .checked_mul(2 * size_of::<f64>())
            .and_then(|bytes| bytes.checked_add(size_of::<eqiora::ValueLiteral>()))
            .and_then(|bytes| bytes.checked_add(point_bytes))
            .ok_or_else(|| PyBufferError::new_err("stationary map point size overflows"))?;
        selected.push(id);
        types.push(value.value_type().clone());
    }
    let points = points.cast::<PySequence>()?;
    if points
        .len()?
        .checked_mul(point_bytes)
        .is_none_or(|charge| charge > storage_bytes_limit)
    {
        return Err(PyBufferError::new_err(
            "stationary map point inventory exceeds its byte limit",
        ));
    }
    let mut staged = Vec::with_capacity(points.len()?);
    for point in points.try_iter()? {
        let point = point?;
        let point = point.cast::<PySequence>()?;
        if point.len()? != types.len() {
            return Err(PyValueError::new_err(
                "each stationary point must contain one complete value per selected Parameter",
            ));
        }
        staged.push(
            types
                .iter()
                .enumerate()
                .map(|(index, value_type)| {
                    crate::modeling::value_literal::from_python(
                        &point.get_item(index)?,
                        value_type.clone(),
                    )
                })
                .collect::<PyResult<Vec<_>>>()?,
        );
    }
    let count = staged.len();
    let policy = EvaluationMapExecutionPolicy::new(
        chunk_size,
        eqiora::realization::Target::HostCpu {
            threads: std::num::NonZeroUsize::new(workers)
                .ok_or_else(|| PyValueError::new_err("workers must be positive"))?,
        },
        EvaluationMapRetention::Retain,
        storage_bytes_limit,
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    let native = EvaluationMapPlan::from_parameters(
        Arc::new(native),
        &selected,
        &staged,
        policy,
        member_bytes_limit,
        Arc::new(eqiora::backends::faer::FaerLinearSolver),
    )
    .map_err(|error| diagnostic_error(py, &[error]))?;
    Ok(PyEvaluationMapPlan {
        native: Arc::new(native),
        context: MapContext::Parameters(Arc::new(plan)),
        point_shape: vec![count],
    })
}

pub(super) fn point(py: Python<'_>, values: &[eqiora::ValueLiteral]) -> PyResult<Py<PyTuple>> {
    PyTuple::new(
        py,
        values
            .iter()
            .map(|value| crate::modeling::value_literal::to_python(py, value))
            .collect::<PyResult<Vec<_>>>()?,
    )
    .map(Bound::unbind)
}

/// The original request basis, without source aliases or a second signature owner.
pub(super) fn refs(
    py: Python<'_>,
    plan: &eqiora_numerics::ResolvedCommonPlan,
    selected: &[Parameter],
) -> PyResult<Py<PyTuple>> {
    let original =
        crate::common_plan::harmonic::original_model(plan).unwrap_or_else(|| plan.model_artifact());
    let parameters = selected
        .iter()
        .map(|id| {
            PyModelParameterRef::from_artifact(original, *id)
                .map_err(|errors| diagnostic_error(py, &errors))
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, parameters).map(Bound::unbind)
}
