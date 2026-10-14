//! Exact Observable selection happens once at the immutable Model boundary.
use super::*;
use eqiora::{Id, kinds};

/// Exact derived output selected from one immutable Model.
#[pyclass(
    name = "ObservableRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyObservableRef {
    pub(crate) model_digest: String,
    pub(crate) id: Id<kinds::Observable>,
}

#[pymethods]
impl PyObservableRef {
    fn __repr__(&self) -> String {
        format!(
            "ObservableRef(model_digest={:?}, id={:?})",
            self.model_digest,
            self.id.ulid().to_string()
        )
    }

    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn id(&self) -> String {
        self.id.ulid().to_string()
    }
}

pub(super) fn select(
    model: &PyModel,
    py: Python<'_>,
    selection: &str,
) -> PyResult<PyObservableRef> {
    let (model_digest, id) = selection::select(model, py, selection)?;
    Ok(PyObservableRef { model_digest, id })
}
