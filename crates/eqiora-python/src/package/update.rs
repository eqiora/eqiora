//! Python inspection and explicit publication of the shared native proposal.

use super::*;
use eqiora::package::{PackagePreparationError, ProjectUpdate};

/// A validated, frozen project update. Inspect its request/selection lock before
/// committing once; commit never selects or downloads a different candidate.
#[pyclass(module = "eqiora", name = "ProjectUpdate")]
pub(crate) struct PyProjectUpdate {
    pending: Option<ProjectUpdate>,
    resolution: Vec<u8>,
    lock: Vec<u8>,
    explanation: String,
    root: String,
}

#[pymethods]
impl PyProjectUpdate {
    #[getter]
    fn resolution(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.resolution).unbind()
    }
    #[getter]
    fn lock(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.lock).unbind()
    }
    #[getter]
    fn explanation(&self) -> &str {
        &self.explanation
    }
    fn __repr__(&self) -> String {
        format!(
            "ProjectUpdate(root={:?}, pending={})",
            self.root,
            self.pending.is_some()
        )
    }

    /// Commit once into the explicit store. A stale or failed proposal must be
    /// replaced with a fresh preview before trying another publication.
    fn commit(&mut self, py: Python<'_>, store_root: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
        panic_boundary(py, || {
            let store = unicode_path(py, store_root)?;
            let proposal = self.pending.take().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("project update has already been consumed")
            })?;
            py.detach(move || proposal.commit(store))
                .map_err(|error| failure(py, error))?;
            Ok(PyBytes::new(py, &self.resolution).unbind())
        })
    }
}

/// Freeze explicit candidates and preview the validated selection without
/// installing packages or publishing the project's manifest/lock.
#[pyfunction]
fn preview_local_project(
    py: Python<'_>,
    project_root: &Bound<'_, PyAny>,
) -> PyResult<PyProjectUpdate> {
    panic_boundary(py, || {
        let project = unicode_path(py, project_root)?;
        let proposal = py
            .detach(move || PackagedModelDocument::preview_local_package_project_v1(project))
            .map_err(|error| failure(py, error))?;
        let resolution = proposal
            .resolution()
            .canonical_json()
            .map_err(|error| failure(py, error.into()))?;
        let lock = proposal.lock_bytes().map_err(|error| failure(py, error))?;
        let explanation = proposal.explanation();
        let root = format!(
            "{}@{}",
            proposal.resolution().root().name,
            proposal.resolution().root().version
        );
        Ok(PyProjectUpdate {
            pending: Some(proposal),
            resolution,
            lock,
            explanation,
            root,
        })
    })
}

fn failure(py: Python<'_>, error: PackagePreparationError) -> PyErr {
    compatibility_error(
        py,
        &[Diagnostic::error(
            codes::INVALID_ARTIFACT,
            format!("project update rejected: {error}"),
        )],
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyProjectUpdate>()?;
    module.add_function(wrap_pyfunction!(preview_local_project, module)?)?;
    Ok(())
}
