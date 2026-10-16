//! Python projection of the shared native statistic, without a second reducer.
use crate::{
    model::PyObservableRef, modeling::PyValueType, result::PyRunResult,
    sampling::PySamplingIdentity,
};
use eqiora::{
    api::{ModelEntityRef, MomentEstimator, MomentReduction},
    artifact::ReplayableCanonicalModelArtifact,
    kernel::ProbabilityMoment,
};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Typed finite statistic retaining accepted membership and its explicit convention.
#[pyclass(
    name = "MomentEstimator",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyMomentEstimator {
    native: MomentEstimator<'static>,
}

#[pymethods]
impl PyMomentEstimator {
    fn __repr__(&self) -> String {
        format!(
            "MomentEstimator(reduction={:?}, members={}, time_s={})",
            self.reduction(),
            self.native.time_results().expect("time statistic").len(),
            self.time_s(),
        )
    }

    /// Reduce every supplied accepted Result at one exact retained physical time.
    /// Explicit assume_iid additionally requires one uncoupled sampling design;
    /// it is the caller's mathematical assumption, not an inference from seeds.
    #[staticmethod]
    #[pyo3(signature = (results, outputs, /, *, time_s, moment="mean", assume_iid=false, weights=None))]
    fn from_time_results(
        py: Python<'_>,
        results: Vec<Py<PyRunResult>>,
        outputs: Vec<Py<PyObservableRef>>,
        time_s: f64,
        moment: &str,
        assume_iid: bool,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let reduction = match (moment, assume_iid) {
            ("sum", false) => MomentReduction::Sum,
            ("mean" | "variance" | "covariance", _) => {
                let moment = match moment {
                    "mean" => ProbabilityMoment::Expectation,
                    "variance" => ProbabilityMoment::Variance,
                    _ => ProbabilityMoment::Covariance,
                };
                if assume_iid {
                    MomentReduction::UnbiasedIid(moment)
                } else {
                    MomentReduction::Empirical(moment)
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "moment must be mean, variance, covariance, or sum; sum has no IID interpretation",
                ));
            }
        };
        let native_results = results
            .iter()
            .map(|result| result.borrow(py).native().clone())
            .collect::<Vec<_>>();
        let first = native_results
            .first()
            .ok_or_else(|| PyValueError::new_err("results must not be empty"))?;
        let model = first
            .plan()
            .model_artifact()
            .replay_model()
            .map_err(|error| crate::error::validation_error(py, &[error]))?;
        let selected = outputs
            .iter()
            .map(|output| {
                let output = output.borrow(py);
                if output.model_digest != model.artifact_reference().artifact().to_string() {
                    return Err(PyValueError::new_err(
                        "ObservableRef belongs to a different exact Model",
                    ));
                }
                ModelEntityRef::from_replayed_model(&model, output.id)
                    .map_err(|error| crate::error::validation_error(py, &[error]))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let weights = weights.unwrap_or_else(|| vec![1.; native_results.len()]);
        MomentEstimator::from_time_results(
            native_results.into(),
            time_s,
            reduction,
            &selected,
            weights,
        )
        .map(|native| Self { native })
        .map_err(|error| crate::error::validation_error(py, &[error]))
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, self.native.value())
    }

    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.native.value_type().clone(),
        }
    }

    /// Estimated equal-weight IID mean uncertainty, in the mean's physical units.
    fn standard_error(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self
            .native
            .standard_error()
            .map_err(|error| crate::error::validation_error(py, &[error]))?;
        crate::modeling::value_literal::to_python(py, &value)
    }

    #[getter]
    fn time_s(&self) -> f64 {
        self.native.time_observation().expect("time statistic").0
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.native.weights().to_vec()
    }

    #[getter]
    fn model_digest(&self) -> String {
        self.native.model().artifact().to_string()
    }

    #[getter]
    fn observables(&self) -> Vec<PyObservableRef> {
        self.native
            .time_observation()
            .expect("time statistic")
            .1
            .iter()
            .map(|output| PyObservableRef {
                model_digest: output.model().artifact().to_string(),
                id: output.id(),
            })
            .collect()
    }

    #[getter]
    fn result_identities(&self) -> Vec<String> {
        self.native
            .time_results()
            .expect("time statistic")
            .iter()
            .map(|result| result.identity().to_owned())
            .collect()
    }

    #[getter]
    fn sampling_identities(&self) -> Vec<Option<PySamplingIdentity>> {
        self.native
            .time_results()
            .expect("time statistic")
            .iter()
            .map(|result| {
                result
                    .trajectory()
                    .and_then(|trajectory| trajectory.time_request())
                    .and_then(|request| request.sampling())
                    .map(|(_, identity)| PySamplingIdentity {
                        native: identity.clone(),
                    })
            })
            .collect()
    }

    #[getter]
    fn reduction(&self) -> &'static str {
        match self.native.reduction() {
            MomentReduction::Sum => "sum",
            MomentReduction::Empirical(ProbabilityMoment::Expectation) => "empirical-mean",
            MomentReduction::Empirical(ProbabilityMoment::Variance) => "empirical-variance",
            MomentReduction::Empirical(ProbabilityMoment::Covariance) => "empirical-covariance",
            MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation) => "iid-mean",
            MomentReduction::UnbiasedIid(ProbabilityMoment::Variance) => "iid-variance",
            MomentReduction::UnbiasedIid(ProbabilityMoment::Covariance) => "iid-covariance",
            MomentReduction::Population(_) => {
                unreachable!("time collection is not an exhaustive law")
            }
        }
    }
}
