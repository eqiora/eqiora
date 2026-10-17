//! Python inspection of native recorded collections; no live Program is reconstructed.
use super::*;
use crate::common_plan::PyPlan;
use crate::differentiation::signature;
use crate::error::{compatibility_error, validation_error};
use crate::model_io::{
    ArtifactFileSpec, read_artifact_bytes, unicode_artifact_path, write_artifact_bytes,
};
use eqiora::api::{EvaluationCollection, EvaluationCollectionGroup};
use pyo3::types::PyBytes;

mod member;
use member::PyRecordedEvaluation;

const FILE_SPEC: ArtifactFileSpec = ArtifactFileSpec {
    artifact_name: "EvaluationCollection",
    extension: "eqcollection",
    staging_name: "collection",
    max_bytes: eqiora::artifact::MAX_EXECUTION_ARTIFACT_BYTES,
};

/// Ordered execution groups with exact recorded requests and original outcomes.
#[pyclass(name = "EvaluationCollection", module = "eqiora._eqiora", frozen)]
pub(super) struct PyEvaluationCollection {
    native: Arc<EvaluationCollection>,
}

impl From<EvaluationCollection> for PyEvaluationCollection {
    fn from(native: EvaluationCollection) -> Self {
        Self {
            native: Arc::new(native),
        }
    }
}

#[pymethods]
impl PyEvaluationCollection {
    /// Associate a previously computed statistic with one exact complete group.
    fn with_estimator(
        &self,
        py: Python<'_>,
        group: usize,
        estimator: &crate::moment_estimator::PyMomentEstimator,
    ) -> PyResult<Self> {
        self.native
            .with_estimator(group, &estimator.native)
            .map(Self::from)
            .map_err(|error| validation_error(py, &[error]))
    }
    fn __repr__(&self) -> String {
        format!(
            "EvaluationCollection(groups={}, complete={})",
            self.native.groups().len(),
            self.native.is_complete()
        )
    }
    #[getter]
    fn identity(&self) -> &str {
        self.native.identity()
    }
    #[getter]
    fn complete(&self) -> bool {
        self.native.is_complete()
    }
    fn __len__(&self) -> usize {
        self.native.groups().len()
    }
    fn __getitem__(&self, index: isize) -> PyResult<PyEvaluationCollectionGroup> {
        Ok(PyEvaluationCollectionGroup {
            collection: self.native.clone(),
            index: occurrence_index(index, self.native.groups().len())?,
        })
    }
    /// Combine explicitly associated groups without changing their execution identities.
    #[staticmethod]
    fn combine(py: Python<'_>, collections: Vec<PyRef<'_, Self>>) -> PyResult<Self> {
        let collections = collections
            .iter()
            .map(|value| value.native.as_ref().clone())
            .collect();
        EvaluationCollection::combine(collections)
            .map(Self::from)
            .map_err(|error| validation_error(py, &[error]))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.native
            .to_bytes()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|error| validation_error(py, &[error]))
    }
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        panic_boundary(py, || {
            py.detach(|| {
                EvaluationCollection::from_bytes(
                    data,
                    &eqiora::backends::faer::FaerLinearSolver,
                    eqiora::backends::diffsol::DiffsolTimeBackend::CAPABILITIES,
                )
            })
            .map(Self::from)
            .map_err(|errors| validation_error(py, &errors))
        })
    }
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = self
            .native
            .to_bytes()
            .map_err(|error| validation_error(py, &[error]))?;
        py.detach(move || write_artifact_bytes(&path, &bytes, FILE_SPEC))
            .map_err(|error| compatibility_error(py, &[error]))
    }
    #[staticmethod]
    fn read(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = unicode_artifact_path(py, path)?;
        let bytes = py
            .detach(move || read_artifact_bytes(&path, FILE_SPEC))
            .map_err(|error| compatibility_error(py, &[error]))?;
        Self::from_bytes(py, &bytes)
    }
}

/// One exact requested inventory and its immutable execution outcomes.
#[pyclass(name = "EvaluationCollectionGroup", module = "eqiora._eqiora", frozen)]
pub(super) struct PyEvaluationCollectionGroup {
    collection: Arc<EvaluationCollection>,
    index: usize,
}
impl PyEvaluationCollectionGroup {
    fn native(&self) -> &EvaluationCollectionGroup {
        &self.collection.groups()[self.index]
    }
}
#[pymethods]
impl PyEvaluationCollectionGroup {
    #[getter]
    fn estimators(&self) -> Vec<crate::moment_estimator::record::PyRecordedMomentEstimator> {
        self.native()
            .estimators()
            .iter()
            .map(
                |native| crate::moment_estimator::record::PyRecordedMomentEstimator {
                    native: native.clone(),
                    model_digest: self.native().plan().model_digest().to_owned(),
                },
            )
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationCollectionGroup(execution_id={:?}, members={}, complete={})",
            self.native().execution_id(),
            self.native().len(),
            self.native().is_complete()
        )
    }
    #[getter]
    fn execution_id(&self) -> String {
        self.native().execution_id()
    }
    #[getter]
    fn request_identity(&self, py: Python<'_>) -> PyResult<String> {
        self.native()
            .request_identity()
            .map_err(|error| validation_error(py, &[error]))
    }
    #[getter]
    fn complete(&self) -> bool {
        self.native().is_complete()
    }
    #[getter]
    fn plan(&self, py: Python<'_>) -> PyResult<PyPlan> {
        PyPlan::from_native_artifact(py, self.native().plan().clone())
    }
    /// Exact static Program encoding, without a solver or derivative executor.
    #[getter]
    fn program_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.native()
            .program_identity()
            .map(|program| {
                program
                    .to_bytes()
                    .map(|bytes| PyBytes::new(py, &bytes))
                    .map_err(|error| validation_error(py, &[error]))
            })
            .transpose()
    }
    #[getter]
    fn inputs(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        self.native()
            .program_identity()
            .map(|program| signature::project(py, program.inputs()))
            .transpose()
    }
    #[getter]
    fn outputs(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        self.native()
            .program_identity()
            .map(|program| signature::project(py, program.outputs()))
            .transpose()
    }
    #[getter]
    fn shared_input_ids(&self) -> Option<Vec<String>> {
        self.native().shared_inputs().map(ids)
    }
    fn __len__(&self) -> usize {
        self.native().len()
    }
    fn __getitem__(&self, index: isize) -> PyResult<PyRecordedEvaluation> {
        Ok(PyRecordedEvaluation {
            collection: self.collection.clone(),
            group: self.index,
            index: occurrence_index(index, self.native().len())?,
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEvaluationCollection>()?;
    module.add_class::<PyEvaluationCollectionGroup>()?;
    module.add_class::<PyRecordedEvaluation>()
}
