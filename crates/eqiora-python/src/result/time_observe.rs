//! Accepted steps and shared spatial reconstruction own typed time functionals.
use super::*;
use crate::model::{PyModelParameterRef, PyObservableRef};
use crate::modeling::{PyDimension, PyValueType};
use eqiora::{DynQuantity, ValueLiteral};
use eqiora_numerics::TimeObservationInterpolation;
use pyo3::types::PyDict;

/// Numerical interpolation over the available accepted-step samples.
#[pyclass(
    name = "TimeObservationInterpolation",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyTimeObservationInterpolation {
    NativeMidpointQuadratic,
    EndpointLinear,
}

impl From<PyTimeObservationInterpolation> for TimeObservationInterpolation {
    fn from(value: PyTimeObservationInterpolation) -> Self {
        match value {
            PyTimeObservationInterpolation::NativeMidpointQuadratic => {
                Self::NativeMidpointQuadratic
            }
            PyTimeObservationInterpolation::EndpointLinear => Self::EndpointLinear,
        }
    }
}

pub(super) enum TimeReduction {
    Terminal,
    Integral(PyTimeObservationInterpolation),
    Minimum(PyTimeObservationInterpolation),
    Maximum(PyTimeObservationInterpolation),
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
    extremum_time_s: Option<f64>,
    #[pyo3(get)]
    interpolation: Option<PyTimeObservationInterpolation>,
    #[pyo3(get)]
    interval_s: (f64, f64),
    #[pyo3(get)]
    endpoint_convention: &'static str,
    #[pyo3(get)]
    parameter_jvp: bool,
    #[pyo3(get)]
    spatial_quadratures: std::collections::BTreeMap<String, (&'static str, usize, usize)>,
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
        value: &eqiora_numerics::CommonTrajectoryObservation,
        quadrature_points: Option<usize>,
    ) -> PyTrajectoryObservation {
        let interpolation = value.interpolation().map(|rule| match rule {
            TimeObservationInterpolation::EndpointLinear => {
                PyTimeObservationInterpolation::EndpointLinear
            }
            TimeObservationInterpolation::NativeMidpointQuadratic => {
                PyTimeObservationInterpolation::NativeMidpointQuadratic
            }
        });
        PyTrajectoryObservation {
            value: value.value().clone(),
            result_identity: self.native.identity().to_owned(),
            trajectory_identity: value.trajectory_identity().to_owned(),
            observable_id: value.observable().ulid().to_string(),
            evaluation_kind: value.evaluation_kind(),
            extremum_time_s: value.extremum_time_s(),
            interpolation,
            interval_s: (value.interval_s()[0], value.interval_s()[1]),
            endpoint_convention: value.endpoint_convention(),
            parameter_jvp: value.is_parameter_jvp(),
            spatial_quadratures: super::observe::quadrature_metadata(
                value.spatial_quadratures(),
                quadrature_points.unwrap_or(0),
            ),
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
        reduction: TimeReduction,
        interval_s: Option<[f64; 2]>,
        quadrature_points: Option<usize>,
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
        let rules = self.observation_rules(py, observable, quadrature_points)?;
        let value = match reduction {
            TimeReduction::Terminal => trajectory.observe_terminal(model, observable.id),
            TimeReduction::Integral(rule) => trajectory.observe_time_integral(
                model,
                observable.id,
                rule.into(),
                interval_s,
                &rules,
            ),
            TimeReduction::Minimum(rule) => trajectory.observe_time_minimum(
                model,
                observable.id,
                rule.into(),
                interval_s,
                &rules,
            ),
            TimeReduction::Maximum(rule) => trajectory.observe_time_maximum(
                model,
                observable.id,
                rule.into(),
                interval_s,
                &rules,
            ),
        }
        .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(self.trajectory_observation(&value, quadrature_points))
    }

    pub(super) fn observe_trajectory_parameter_jvp(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        interpolation: Option<PyTimeObservationInterpolation>,
        directions: &Bound<'_, PyDict>,
        interval_s: Option<[f64; 2]>,
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
        let value = match interpolation {
            None => trajectory.observe_terminal_parameter_jvp(
                model,
                observable.id,
                sensitivity,
                direction,
            ),
            Some(interpolation) => trajectory.observe_time_integral_parameter_jvp(
                model,
                observable.id,
                interpolation.into(),
                sensitivity,
                direction,
                interval_s,
            ),
        }
        .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(self.trajectory_observation(&value, None))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTimeObservationInterpolation>()?;
    module.add_class::<PyTrajectoryObservation>()?;
    super::spectrum::register(module)?;
    Ok(())
}
