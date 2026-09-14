//! Python inspection of fresh-compile authored mathematics.

use eqiora::api::ModelDocument;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Immutable inspection of one fresh-compile authored mathematical form.
#[pyclass(
    name = "AuthoredFormulation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PyAuthoredFormulation {
    source_identity: String,
    name: String,
    interval: Option<(String, String, String)>,
    test_restrictions: Vec<(String, String, Vec<String>)>,
    implication: String,
    assumptions: Vec<String>,
    relation_ids: Vec<String>,
    domain_id: String,
    trial_field_ids: Vec<String>,
    filename: String,
    range: (u32, u32),
}

#[pymethods]
impl PyAuthoredFormulation {
    #[getter]
    fn implication(&self) -> &str {
        &self.implication
    }
    #[getter]
    fn assumptions(&self) -> Vec<String> {
        self.assumptions.clone()
    }
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn test_restrictions(&self) -> Vec<(String, String, Vec<String>)> {
        self.test_restrictions.clone()
    }
    #[getter]
    fn interval(&self) -> Option<(String, String, String)> {
        self.interval.clone()
    }
    #[getter]
    fn kind(&self) -> &'static str {
        if self.interval.is_some() {
            "integral-conservative"
        } else if self.trial_field_ids.len() > 1 {
            "mixed-galerkin"
        } else {
            "primal"
        }
    }

    #[getter]
    fn source_identity(&self) -> &str {
        &self.source_identity
    }

    #[getter]
    fn relation_ids(&self) -> Vec<String> {
        self.relation_ids.clone()
    }

    #[getter]
    fn domain_id(&self) -> &str {
        &self.domain_id
    }

    #[getter]
    fn trial_field_ids(&self) -> Vec<String> {
        self.trial_field_ids.clone()
    }

    #[getter]
    fn filename(&self) -> &str {
        &self.filename
    }

    #[getter]
    const fn source_range(&self) -> (u32, u32) {
        self.range
    }

    fn __repr__(&self) -> String {
        format!(
            "AuthoredFormulation(kind={:?}, source_identity={:?}, relation_ids={:?}, domain_id={:?}, trial_field_ids={:?}, filename={:?}, source_range={:?})",
            self.kind(),
            self.source_identity,
            self.relation_ids,
            self.domain_id,
            self.trial_field_ids,
            self.filename,
            self.range,
        )
    }
}

pub(super) fn project(py: Python<'_>, document: Option<&ModelDocument>) -> PyResult<Py<PyTuple>> {
    let formulations = document
        .into_iter()
        .flat_map(ModelDocument::authored_formulations)
        .map(|form| PyAuthoredFormulation {
            source_identity: form.source_identity().to_owned(),
            name: form.projection().name().to_owned(),
            test_restrictions: form.projection().test_restrictions().to_vec(),
            interval: form
                .projection()
                .interval()
                .map(|(name, lower, upper)| (name.into(), lower.into(), upper.into())),
            implication: form.projection().implication().into(),
            assumptions: form.projection().assumptions().to_vec(),
            relation_ids: form
                .relations()
                .iter()
                .map(|id| id.ulid().to_string())
                .collect(),
            domain_id: form.domain().ulid().to_string(),
            trial_field_ids: form
                .trials()
                .iter()
                .map(|id| id.ulid().to_string())
                .collect(),
            filename: form.file().to_owned(),
            range: (form.range().start(), form.range().end()),
        })
        .map(|form| Py::new(py, form))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, formulations)?.unbind())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAuthoredFormulation>()
}
