//! Exact Activation selection happens once at the immutable Model boundary.
use super::*;
use eqiora::{Id, kinds};

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
    let (model_digest, id) = selection::select(model, py, selection)?;
    Ok(PyActivationRef { model_digest, id })
}
