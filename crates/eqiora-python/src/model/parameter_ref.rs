//! Immutable selected Parameter identity and complete value projection.

use eqiora::api::{ModelDocument, ModelParameterRef};
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

/// Exact canonical Parameter selected from one immutable Model.
#[pyclass(
    name = "ParameterRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyModelParameterRef {
    pub(crate) value: ModelParameterRef,
    model_digest: String,
    id: String,
    literal: eqiora::ValueLiteral,
}

impl PyModelParameterRef {
    pub(crate) fn from_document(
        document: &ModelDocument,
        selection: &str,
    ) -> Result<Self, eqiora::Diagnostic> {
        let value = document.parameter_ref(selection)?;
        let literal = document
            .program()
            .typed_value(value.id().into())
            .cloned()
            .expect("an admitted ParameterRef owns a complete Parameter value");
        Ok(Self {
            model_digest: value.model().artifact().to_string(),
            id: value.id().ulid().to_string(),
            value,
            literal,
        })
    }
}

impl PartialEq for PyModelParameterRef {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for PyModelParameterRef {}

impl Hash for PyModelParameterRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model_digest.hash(state);
        self.id.hash(state);
    }
}

#[pymethods]
impl PyModelParameterRef {
    /// Complete immutable value at this exact Model revision.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.literal)
    }

    /// Complete mathematical type of the selected Parameter.
    #[getter]
    fn value_type(&self) -> crate::modeling::PyValueType {
        crate::modeling::PyValueType {
            value: self.literal.value_type().clone(),
        }
    }

    /// Exact canonical Model artifact digest.
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }

    /// Stable canonical Parameter ULID.
    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    fn __repr__(&self) -> String {
        format!(
            "ParameterRef(id={:?}, model_digest={:?})",
            self.id, self.model_digest
        )
    }
}
