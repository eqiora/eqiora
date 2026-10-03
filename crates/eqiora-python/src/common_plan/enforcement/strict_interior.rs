//! Explicit positive inequality margins for a local nonlinear branch.
use super::*;

#[pyclass(
    name = "StrictInterior",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyStrictInterior {
    pub(super) model_digest: String,
    pub(super) native: FiniteConstraintEnforcement,
}

#[pymethods]
impl PyStrictInterior {
    #[new]
    #[pyo3(signature = (*, margins))]
    fn new(py: Python<'_>, margins: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let (model_digest, entries) = bound_tolerances(margins)?;
        let native = FiniteConstraintEnforcement::strict_interior(entries)
            .map_err(|error| validation_error(py, &[error]))?;
        Ok(Self {
            model_digest,
            native,
        })
    }
    #[getter]
    fn margins(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        project_tolerances(py, &self.native, &self.model_digest)
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    fn __repr__(&self) -> String {
        format!(
            "StrictInterior(model_digest={:?}, conditions={})",
            self.model_digest,
            self.native.tolerances().len()
        )
    }
}
