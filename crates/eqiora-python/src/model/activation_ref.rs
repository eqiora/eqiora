//! Exact Activation selection happens once at the immutable Model boundary.
use super::*;
use eqiora::{Id, kinds};
use pyo3::exceptions::{PyKeyError, PyValueError};

/// Exact event guard Activation selected from one immutable Model.
#[pyclass(
    name = "ActivationRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyActivationRef {
    pub(crate) model_digest: String,
    pub(crate) id: Id<kinds::Activation>,
}

#[pymethods]
impl PyActivationRef {
    /// Exact canonical Model artifact digest.
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    /// Stable canonical Activation ULID.
    #[getter]
    fn id(&self) -> String {
        self.id.ulid().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ActivationRef(id={:?}, model_digest={:?})",
            self.id.ulid().to_string(),
            self.model_digest
        )
    }
}

pub(super) fn select(
    model: &PyModel,
    py: Python<'_>,
    selection: &str,
) -> PyResult<PyActivationRef> {
    let raw = model
        .document()
        .ok()
        .and_then(|document| document.aliases().get(selection).copied());
    let id = match raw {
        Some(raw) => raw
            .downcast::<kinds::Activation>()
            .ok_or_else(|| PyValueError::new_err("selection does not identify an Activation"))?,
        None => {
            let ulid = selection
                .parse::<ulid::Ulid>()
                .map_err(|_| PyKeyError::new_err(selection.to_owned()))?;
            Id::from_ulid(ulid)
        }
    };
    if !model
        .artifact_ids(EntityKind::Activation)
        .map_err(|errors| validation_error(py, &errors))?
        .iter()
        .any(|candidate| candidate == &id.ulid().to_string())
    {
        return Err(PyKeyError::new_err(
            "Activation is outside this exact Model",
        ));
    }
    let reference = model
        .artifact()
        .artifact_reference()
        .map_err(|error| validation_error(py, &[error]))?;
    Ok(PyActivationRef {
        model_digest: reference.artifact().to_string(),
        id,
    })
}
