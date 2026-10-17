//! Read-only computed statistic projection; no weighted arithmetic is invoked here.
use super::*;
use eqiora::api::{GradientEstimator, RecordedMomentEstimator};
use pyo3::types::PyDict;

/// A saved computed statistic and its exact group attribution, without a live reducer.
#[pyclass(name = "RecordedMomentEstimator", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyRecordedMomentEstimator {
    pub(crate) native: RecordedMomentEstimator,
    pub(crate) model_digest: String,
}
#[pymethods]
impl PyRecordedMomentEstimator {
    fn __repr__(&self) -> String {
        format!(
            "RecordedMomentEstimator(reduction={:?}, members={})",
            self.reduction(),
            self.native.occurrences().len()
        )
    }
    #[getter]
    fn identity(&self) -> &str {
        self.native.identity()
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn occurrence_ids(&self) -> Vec<String> {
        self.native
            .occurrences()
            .iter()
            .map(ToString::to_string)
            .collect()
    }
    #[getter]
    fn result_identities(&self) -> Vec<String> {
        self.native.references().to_vec()
    }
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.native.weights().to_vec()
    }
    #[getter]
    fn reduction(&self) -> &'static str {
        reduction_name(self.native.reduction())
    }
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, self.native.value())
    }
    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.native.value().value_type().clone(),
        }
    }
    #[getter]
    fn output_indices(&self) -> Option<Vec<usize>> {
        self.native.output_indices().map(<[usize]>::to_vec)
    }
    #[getter]
    fn time_s(&self) -> Option<f64> {
        self.native.time_observation().map(|(time, _)| time)
    }
    #[getter]
    fn observables(&self) -> Option<Vec<PyObservableRef>> {
        self.native.time_observation().map(|(_, outputs)| {
            outputs
                .iter()
                .map(|id| PyObservableRef {
                    model_digest: self.model_digest.clone(),
                    id: *id,
                })
                .collect()
        })
    }
    #[getter]
    fn gradient_estimator(&self) -> &'static str {
        match self.native.gradient_estimator() {
            GradientEstimator::FixedSamples => "fixed-samples",
            GradientEstimator::PathwiseLocationScale { .. } => "pathwise-location-scale",
            GradientEstimator::EnumeratedScore { .. } => "enumerated-score",
        }
    }
    /// Exact Model entity roles retained by the gradient interpretation.
    #[getter]
    fn gradient_entities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        match self.native.gradient_estimator() {
            GradientEstimator::FixedSamples => {}
            GradientEstimator::PathwiseLocationScale {
                distribution,
                variable,
                location,
                scale,
            } => {
                result.set_item("distribution", distribution.ulid().to_string())?;
                result.set_item("variable", variable.ulid().to_string())?;
                result.set_item("location", location.ulid().to_string())?;
                result.set_item("scale", scale.ulid().to_string())?;
            }
            GradientEstimator::EnumeratedScore { .. } => {
                unreachable!("collection statistics are not exhaustive laws")
            }
        }
        Ok(result)
    }
}
