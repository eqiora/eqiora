//! Inspect the native Plan's exact coefficient entities and compatible maps.
use super::*;
use eqiora::meshing::MeshEntity;
use std::collections::BTreeMap;

fn field_id(plan: &PyPlan, field: &PyModelFieldRef) -> PyResult<eqiora::Id<eqiora::kinds::Field>> {
    if field.exact_model_digest() != plan.native.model_digest() {
        return Err(PyTypeError::new_err(
            "FieldRef belongs to a different exact Model",
        ));
    }
    ulid::Ulid::from_string(field.exact_id())
        .map(eqiora::Id::from_ulid)
        .map_err(|_| PyTypeError::new_err("FieldRef has an invalid exact identity"))
}

fn linear(plan: &PyPlan) -> PyResult<&CommonLinearPlan> {
    plan.linear_native().ok_or_else(|| {
        PyTypeError::new_err("coefficient inspection requires a linear spatial Plan")
    })
}

pub(super) fn entities(
    py: Python<'_>,
    plan: &PyPlan,
    field: &PyModelFieldRef,
) -> PyResult<Py<PyTuple>> {
    let entities = linear(plan)?
        .field_coefficient_entities(field_id(plan, field)?)
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    Ok(PyTuple::new(py, entities.into_iter().map(entity))?.unbind())
}

pub(super) fn gradient_modes(
    py: Python<'_>,
    plan: &PyPlan,
    field: &PyModelFieldRef,
) -> PyResult<Py<PyDict>> {
    let rows = linear(plan)?
        .field_gradient_modes(field_id(plan, field)?)
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    project(py, rows)
}

pub(super) fn exterior_derivative(
    py: Python<'_>,
    plan: &PyPlan,
    field: &PyModelFieldRef,
) -> PyResult<Py<PyDict>> {
    let rows = linear(plan)?
        .field_exterior_derivative(field_id(plan, field)?)
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    project(py, rows)
}

fn entity(value: MeshEntity) -> (usize, usize) {
    (value.dimension(), value.index())
}

fn project(
    py: Python<'_>,
    rows: BTreeMap<MeshEntity, Vec<(MeshEntity, i8)>>,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    for (row, entries) in rows {
        result.set_item(
            entity(row),
            PyTuple::new(
                py,
                entries
                    .into_iter()
                    .map(|(column, sign)| (entity(column), sign)),
            )?,
        )?;
    }
    Ok(result.unbind())
}
