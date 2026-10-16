//! Python projection of the shared sample/channel identity owner.
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Stable sample/channel identity with occurrence lineage and explicit stream coupling.
#[pyclass(
    name = "SamplingIdentity",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PySamplingIdentity {
    pub(crate) native: eqiora::api::SamplingIdentity,
}

#[pymethods]
impl PySamplingIdentity {
    fn __repr__(&self) -> String {
        format!(
            "SamplingIdentity(sample_id={:?}, channel={:?}, lineage={:?}, coupling_id={:?})",
            self.native.sample_id(),
            self.native.channel(),
            self.native.lineage(),
            self.native.coupling_id()
        )
    }

    #[new]
    #[pyo3(signature = (sample_id, channel, /, *, lineage=Vec::new(), coupling_id=None))]
    fn new(
        py: Python<'_>,
        sample_id: &[u8],
        channel: &[u8],
        lineage: Vec<u64>,
        coupling_id: Option<&[u8]>,
    ) -> PyResult<Self> {
        eqiora::api::SamplingIdentity::new(sample_id, channel, &lineage, coupling_id)
            .map(|native| Self { native })
            .map_err(|error| crate::error::validation_error(py, &[error]))
    }

    #[getter]
    fn sample_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.native.sample_id())
    }

    #[getter]
    fn channel<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.native.channel())
    }

    #[getter]
    fn lineage(&self) -> Vec<u64> {
        self.native.lineage().to_vec()
    }

    #[getter]
    fn coupling_id<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.native.coupling_id().map(|id| PyBytes::new(py, id))
    }
}
