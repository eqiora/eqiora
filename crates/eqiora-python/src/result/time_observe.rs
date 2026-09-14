//! Typed time functionals delegate to the accepted native integration history.
use super::*;
use crate::model::{PyModelParameterRef, PyObservableRef};
use crate::modeling::{PyDimension, PyValueType};
use eqiora::{DynQuantity, ValueLiteral};
use eqiora_numerics::TimeFunctionalQuadrature;
use pyo3::types::PyDict;

/// Numerical quadrature over retained native steps, independent of requested outputs.
#[pyclass(
    name = "TimeFunctionalQuadrature",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyTimeFunctionalQuadrature {
    AcceptedStepSimpson,
}

impl From<PyTimeFunctionalQuadrature> for TimeFunctionalQuadrature {
    fn from(value: PyTimeFunctionalQuadrature) -> Self {
        match value {
            PyTimeFunctionalQuadrature::AcceptedStepSimpson => Self::AcceptedStepSimpson,
        }
    }
}

struct TrajectoryObservationParts<'a> {
    literal: &'a ValueLiteral,
    trajectory_identity: &'a str,
    observable: eqiora::Id<eqiora::kinds::Observable>,
    quadrature: Option<TimeFunctionalQuadrature>,
    parameter_jvp: bool,
    interval_s: [f64; 2],
    endpoint_convention: &'static str,
}

/// Typed terminal or time-integrated value with exact trajectory lineage.
#[pyclass(name = "TrajectoryObservation", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyTrajectoryObservation {
    value: ValueLiteral,
    #[pyo3(get)]
    result_identity: String,
    #[pyo3(get)]
    trajectory_identity: String,
    #[pyo3(get)]
    observable_id: String,
    #[pyo3(get)]
    evaluation_kind: &'static str,
    #[pyo3(get)]
    quadrature: Option<PyTimeFunctionalQuadrature>,
    #[pyo3(get)]
    interval_s: (f64, f64),
    #[pyo3(get)]
    endpoint_convention: &'static str,
    #[pyo3(get)]
    parameter_jvp: bool,
}

#[pymethods]
impl PyTrajectoryObservation {
    fn __repr__(&self) -> String {
        format!(
            "TrajectoryObservation(observable_id={:?}, evaluation_kind={:?}, interval_s={:?}, trajectory_identity={:?})",
            self.observable_id, self.evaluation_kind, self.interval_s, self.trajectory_identity
        )
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.value)
    }
    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.value.value_type().clone(),
        }
    }
}

impl PyRunResult {
    fn trajectory_observation(
        &self,
        parts: TrajectoryObservationParts<'_>,
    ) -> PyTrajectoryObservation {
        let quadrature = parts.quadrature.map(|rule| match rule {
            TimeFunctionalQuadrature::AcceptedStepSimpson => {
                PyTimeFunctionalQuadrature::AcceptedStepSimpson
            }
        });
        PyTrajectoryObservation {
            value: parts.literal.clone(),
            result_identity: self.native.identity().to_owned(),
            trajectory_identity: parts.trajectory_identity.to_owned(),
            observable_id: parts.observable.ulid().to_string(),
            evaluation_kind: match (quadrature.is_some(), parts.parameter_jvp) {
                (false, false) => "terminal",
                (true, false) => "time-integral",
                (false, true) => "terminal-parameter-jvp",
                (true, true) => "time-integral-parameter-jvp",
            },
            quadrature,
            interval_s: (parts.interval_s[0], parts.interval_s[1]),
            endpoint_convention: parts.endpoint_convention,
            parameter_jvp: parts.parameter_jvp,
        }
    }

    fn parameter_direction(
        &self,
        directions: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(eqiora::Id<eqiora::kinds::Parameter>, DynQuantity)>> {
        let mut native = Vec::with_capacity(directions.len());
        for (parameter, direction) in directions.iter() {
            let parameter = parameter.extract::<PyRef<'_, PyModelParameterRef>>()?;
            if parameter.value.model().artifact().to_string() != self.identity.model_digest() {
                return Err(PyValueError::new_err(
                    "ParameterRef belongs to a different exact Model artifact",
                ));
            }
            let (dimension, value) = direction.extract::<(PyRef<'_, PyDimension>, f64)>()?;
            native.push((
                parameter.value.id(),
                DynQuantity::new(value, dimension.native()),
            ));
        }
        Ok(native)
    }

    pub(super) fn observe_trajectory(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        quadrature: Option<PyTimeFunctionalQuadrature>,
    ) -> PyResult<PyTrajectoryObservation> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        let trajectory = self.native.trajectory().ok_or_else(|| {
            PyValueError::new_err("time observation requires an accepted Trajectory")
        })?;
        let model = self.native.plan().model_artifact();
        let value = match quadrature {
            None => trajectory.observe_terminal(model, observable.id),
            Some(quadrature) => {
                trajectory.observe_time_integral(model, observable.id, quadrature.into())
            }
        }
        .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(self.trajectory_observation(TrajectoryObservationParts {
            literal: value.value(),
            trajectory_identity: value.trajectory_identity(),
            observable: value.observable(),
            quadrature: value.quadrature(),
            parameter_jvp: value.is_parameter_jvp(),
            interval_s: value.interval_s(),
            endpoint_convention: value.endpoint_convention(),
        }))
    }

    pub(super) fn observe_trajectory_parameter_jvp(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        quadrature: Option<PyTimeFunctionalQuadrature>,
        directions: &Bound<'_, PyDict>,
    ) -> PyResult<PyTrajectoryObservation> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        let trajectory = self.native.trajectory().ok_or_else(|| {
            PyValueError::new_err("trajectory Parameter JVP requires an accepted Trajectory")
        })?;
        let sensitivity = self.native.parameter_sensitivity().ok_or_else(|| {
            PyValueError::new_err("Result has no accepted forward Parameter products")
        })?;
        let model = self.native.plan().model_artifact();
        let direction = self.parameter_direction(directions)?;
        let value = match quadrature {
            None => trajectory.observe_terminal_parameter_jvp(
                model,
                observable.id,
                sensitivity,
                direction,
            ),
            Some(quadrature) => trajectory.observe_time_integral_parameter_jvp(
                model,
                observable.id,
                quadrature.into(),
                sensitivity,
                direction,
            ),
        }
        .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(self.trajectory_observation(TrajectoryObservationParts {
            literal: value.value(),
            trajectory_identity: value.trajectory_identity(),
            observable: value.observable(),
            quadrature: value.quadrature(),
            parameter_jvp: value.is_parameter_jvp(),
            interval_s: value.interval_s(),
            endpoint_convention: value.endpoint_convention(),
        }))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTimeFunctionalQuadrature>()?;
    module.add_class::<PyTrajectoryObservation>()?;
    Ok(())
}
