//! Immutable selected Parameter identity and complete value projection.

use eqiora::api::ModelEntityRef;
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
    pub(crate) value: ModelEntityRef<eqiora::Id<eqiora::kinds::Parameter>>,
    model_digest: String,
    id: String,
    literal: eqiora::ValueLiteral,
}

impl PyModelParameterRef {
    pub(crate) fn from_artifact(
        model: &eqiora::artifact::ModelEnvelope,
        id: eqiora::Id<eqiora::kinds::Parameter>,
    ) -> Result<Self, Vec<eqiora::Diagnostic>> {
        let value = ModelEntityRef::from_model_artifact(model, id)?;
        let (transaction, _) = model.to_transaction()?;
        let literal = transaction
            .ops()
            .iter()
            .find_map(|op| match op {
                eqiora::graph::Op::SetValue { target, value } if *target == id.erase() => {
                    Some(value.clone())
                }
                _ => None,
            })
            .ok_or_else(|| {
                vec![eqiora::Diagnostic::error(
                    eqiora::diagnostic::codes::NODE_NOT_FOUND,
                    "selected Parameter has no complete value in its exact Model artifact",
                )]
            })?;
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
