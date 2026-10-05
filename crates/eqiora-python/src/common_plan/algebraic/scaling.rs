//! Existing resolve scaling input bound to exact finite equality conditions.
use super::*;
use crate::model::constraint::PyConstraintRef;
use crate::modeling::PyDimension;
use eqiora::realization::PositivePhysicalScale;
use eqiora_numerics::finite_constraints::ConstraintRef;
use pyo3::types::{PyBool, PyDict, PyTuple};

pub(super) fn extract(
    py: Python<'_>,
    model: &PyModel,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(ConstraintRef, PositivePhysicalScale)>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(Vec::new());
    };
    let entries = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("finite scaling requires {equality_ref: (positive_value, Dimension)}")
    })?;
    let digest = model
        .artifact()
        .artifact_reference()
        .map_err(|error| validation_error(py, &[error]))?
        .artifact()
        .to_string();
    entries
        .iter()
        .map(|(reference, scale)| {
            let reference = reference.extract::<PyRef<'_, PyConstraintRef>>()?;
            if reference.kind != "equality" || reference.model_digest != digest {
                return Err(PyTypeError::new_err(
                    "residual scale requires an equality from this exact Model",
                ));
            }
            let scale = scale.cast::<PyTuple>()?;
            if scale.len() != 2 {
                return Err(PyTypeError::new_err(
                    "residual scale requires (positive_value, Dimension)",
                ));
            }
            let value = scale.get_item(0)?;
            if value.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err("residual scale rejects bool"));
            }
            let dimension = scale
                .get_item(1)?
                .extract::<PyRef<'_, PyDimension>>()?
                .native();
            let scale =
                PositivePhysicalScale::new(eqiora::DynQuantity::new(value.extract()?, dimension))
                    .map_err(|error| validation_error(py, &[error]))?;
            Ok((reference.native, scale))
        })
        .collect()
}
