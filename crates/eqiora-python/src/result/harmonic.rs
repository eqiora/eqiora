//! Python values are projections of the shared native reconstruction.
use super::*;
use eqiora::artifact::CanonicalModelArtifact;
use pyo3::types::PyTuple;

pub(super) fn reconstruct_block(
    py: Python<'_>,
    result: &eqiora_numerics::CommonResult,
    field: &PyModelFieldRef,
    time_seconds: f64,
    block: usize,
) -> PyResult<Py<PyArrayBuffer>> {
    let plan = result.plan().as_scalar().ok_or_else(|| {
        capability_error(
            py,
            "reconstruct_harmonic_field_block requires a spatial harmonic Result",
        )
    })?;
    let original = plan
        .harmonic_original_model()
        .ok_or_else(|| capability_error(py, "Result has no harmonic reconstruction"))?;
    let digest = original
        .artifact_reference()
        .map_err(|error| crate::error::validation_error(py, &[error]))?
        .artifact()
        .to_string();
    if field.exact_model_digest() != digest {
        return Err(PyValueError::new_err(
            "FieldRef belongs to a different exact original Model artifact",
        ));
    }
    let field = ulid::Ulid::from_string(field.exact_id())
        .map(eqiora::Id::from_ulid)
        .map_err(|_| PyValueError::new_err("invalid original Field identity"))?;
    let time = eqiora::DynQuantity::new(
        time_seconds,
        DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("seconds"),
    );
    let values = plan
        .reconstruct_harmonic_field_block(result, field, block, time)
        .map_err(|error| crate::error::validation_error(py, &[error]))?;
    PyArrayBuffer::from_owned_result(py, values)
}

pub(super) fn reconstruct_fields(
    py: Python<'_>,
    result: &eqiora_numerics::CommonResult,
    time_seconds: f64,
) -> PyResult<Py<PyTuple>> {
    let plan = result.plan().as_algebraic().ok_or_else(|| {
        capability_error(
            py,
            "reconstruct_harmonic_fields requires a finite harmonic Result",
        )
    })?;
    let original = plan
        .harmonic_original_model()
        .ok_or_else(|| capability_error(py, "Result has no harmonic reconstruction"))?;
    let digest = original
        .artifact_reference()
        .map_err(|error| crate::error::validation_error(py, &[error]))?
        .artifact()
        .to_string();
    let time = eqiora::DynQuantity::new(
        time_seconds,
        DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("seconds"),
    );
    let values = plan
        .reconstruct_harmonic_fields(result, time)
        .map_err(|error| crate::error::validation_error(py, &[error]))?;
    let values = values
        .into_iter()
        .map(|(field, value)| {
            Ok((
                PyModelFieldRef::from_exact(digest.clone(), field.ulid().to_string()),
                crate::modeling::value_literal::to_python(py, &value)?,
                crate::modeling::PyValueType {
                    value: value.value_type().clone(),
                },
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, values).map(Bound::unbind)
}
