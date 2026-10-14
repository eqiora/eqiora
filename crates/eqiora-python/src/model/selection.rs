//! Typed alias and canonical-ID selection bound to one exact Model artifact.

use super::{PyModel, validation_error};
use eqiora::{Id, artifact::CanonicalModelArtifact, entity::Entity};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

pub(super) fn select<K: Entity>(
    model: &PyModel,
    py: Python<'_>,
    selection: &str,
) -> PyResult<(String, Id<K>)> {
    let raw = model
        .document()
        .ok()
        .and_then(|document| document.aliases().get(selection).copied());
    let id = match raw {
        Some(raw) => raw.downcast::<K>().ok_or_else(|| {
            PyValueError::new_err(format!("selection does not identify an {:?}", K::KIND))
        })?,
        None => {
            let ulid = selection
                .parse::<ulid::Ulid>()
                .map_err(|_| PyKeyError::new_err(selection.to_owned()))?;
            Id::from_ulid(ulid)
        }
    };
    let canonical_id = id.ulid().to_string();
    if !model
        .artifact_ids(K::KIND)
        .map_err(|errors| validation_error(py, &errors))?
        .contains(&canonical_id)
    {
        return Err(PyKeyError::new_err(format!(
            "{:?} is outside this exact Model",
            K::KIND
        )));
    }
    let reference = model
        .artifact()
        .artifact_reference()
        .map_err(|error| validation_error(py, &[error]))?;
    Ok((reference.artifact().to_string(), id))
}

#[cfg(test)]
mod tests;
