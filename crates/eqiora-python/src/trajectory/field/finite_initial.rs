//! Shape-preserving numerical coefficients for one exact no-Mesh Field.
use super::*;
use pyo3::types::{PyList, PyTuple};

enum Visit<'py> {
    Value(Bound<'py, PyAny>, usize),
    Leave(usize),
}

pub(super) fn extract(value: &Bound<'_, PyAny>) -> PyResult<(eqiora::ValueShape, Vec<(f64, f64)>)> {
    let value = if value.hasattr("tolist")? {
        value.call_method0("tolist")?
    } else {
        value.clone()
    };
    let mut visits = vec![Visit::Value(value, 0)];
    let mut active = BTreeSet::new();
    let mut extents = Vec::new();
    let mut leaf_depth = None;
    let mut components = Vec::new();
    while let Some(visit) = visits.pop() {
        let (value, depth) = match visit {
            Visit::Leave(identity) => {
                active.remove(&identity);
                continue;
            }
            Visit::Value(value, depth) => (value, depth),
        };
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let identity = value.as_ptr() as usize;
            if !active.insert(identity) {
                return Err(PyValueError::new_err("InitialField value contains a cycle"));
            }
            let length = u32::try_from(value.len()?)
                .map_err(|_| PyValueError::new_err("InitialField axis exceeds portable extent"))?;
            if length == 0 || leaf_depth.is_some_and(|leaf| depth >= leaf) {
                return Err(PyValueError::new_err(
                    "InitialField value has an empty or ragged shape",
                ));
            }
            if depth == extents.len() {
                extents.push(length);
            } else if extents[depth] != length {
                return Err(PyValueError::new_err(
                    "InitialField value has a ragged shape",
                ));
            }
            visits.push(Visit::Leave(identity));
            for index in (0..length).rev() {
                visits.push(Visit::Value(value.get_item(index as usize)?, depth + 1));
            }
        } else {
            if depth != extents.len() || leaf_depth.is_some_and(|leaf| leaf != depth) {
                return Err(PyValueError::new_err(
                    "InitialField value has a ragged shape",
                ));
            }
            leaf_depth = Some(depth);
            if value.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err("InitialField value rejects booleans"));
            }
            components.push(crate::modeling::value_literal::scalar(&value)?);
        }
    }
    let shape = eqiora::ValueShape::new(extents)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((shape, components))
}
