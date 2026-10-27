//! Typed Python projections delegate evaluation and tangent admission to CommonResult.
use super::*;
use crate::model::PyObservableRef;
use crate::modeling::{PyDimension, PyValueType};
use eqiora::meshing::QuadratureRule;
use eqiora::{DynQuantity, Id, kinds};
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashMap};

type ObservationPoint = (String, Vec<(f64, PyDimension)>);

/// Typed Result-owned value with its effective numerical quadrature.
#[pyclass(
    name = "Observation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyObservation {
    pub(crate) native: eqiora_numerics::CommonObservation,
    quadrature_points: usize,
}

impl PyObservation {
    fn from_native(native: eqiora_numerics::CommonObservation, points: usize) -> Self {
        Self {
            native,
            quadrature_points: points,
        }
    }
}

#[pymethods]
impl PyObservation {
    fn __repr__(&self) -> String {
        format!(
            "Observation(observable_id={:?}, evaluation_kind={:?}, result_identity={:?})",
            self.observable_id(),
            self.evaluation_kind(),
            self.result_identity()
        )
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, self.native.value())
    }
    #[getter]
    fn point(&self) -> Option<ObservationPoint> {
        self.native.point().map(|(domain, coordinates)| {
            (
                domain.ulid().to_string(),
                coordinates
                    .iter()
                    .map(|value| (value.value(), PyDimension { value: value.dim() }))
                    .collect(),
            )
        })
    }
    #[getter]
    fn output(&self) -> Option<(usize, f64, String)> {
        self.native
            .output()
            .map(|(index, time, state)| (index, time, state.to_owned()))
    }
    #[getter]
    fn result_identity(&self) -> &str {
        self.native.result_identity()
    }
    #[getter]
    fn observable_id(&self) -> String {
        self.native.observable().ulid().to_string()
    }
    #[getter]
    fn evaluation_kind(&self) -> &'static str {
        self.native.evaluation_kind()
    }
    #[getter]
    fn quadratures(&self) -> BTreeMap<String, (&'static str, usize, usize)> {
        quadrature_metadata(self.native.quadratures(), self.quadrature_points)
    }

    /// Project one component through the native mathematical value owner.
    #[pyo3(signature = (projection, index=0))]
    fn project_component(
        &self,
        projection: &str,
        index: usize,
    ) -> PyResult<Option<(f64, PyDimension)>> {
        self.native
            .project_component(projection, index)
            .map(|value| value.map(|value| (value.value(), PyDimension { value: value.dim() })))
            .map_err(|error| PyValueError::new_err(error.message().to_owned()))
    }

    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.native.value().value_type().clone(),
        }
    }
}

/// Dimensioned Field coefficient variation bound to one exact accepted Result.
#[pyclass(
    name = "ObservableStateTangent",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyObservableStateTangent {
    native: eqiora_numerics::CommonObservableStateTangent,
    #[pyo3(get)]
    result_identity: String,
}

#[pymethods]
impl PyObservableStateTangent {
    fn __repr__(&self) -> String {
        format!(
            "ObservableStateTangent(result_identity={:?})",
            self.result_identity
        )
    }
}

impl PyRunResult {
    pub(super) fn observation_rules(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        points: Option<usize>,
    ) -> PyResult<HashMap<Id<kinds::Domain>, QuadratureRule>> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        points.map_or_else(
            || Ok(HashMap::new()),
            |points| {
                self.native
                    .plan()
                    .observation_quadratures(&[observable.id], points)
                    .map_err(|error| diagnostic_error(py, &[error]))
            },
        )
    }

    pub(super) fn observe_value(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        points: Option<usize>,
        output_index: Option<usize>,
    ) -> PyResult<PyObservation> {
        let rules = self.observation_rules(py, observable, points)?;
        let value = self
            .native
            .observe(
                self.native.plan().model_artifact(),
                observable.id,
                &rules,
                output_index,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation::from_native(value, points.unwrap_or(0)))
    }

    pub(super) fn observe_point(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        coordinates: Vec<(Py<PyAny>, Py<PyDimension>)>,
        points: Option<usize>,
        output_index: Option<usize>,
    ) -> PyResult<PyObservation> {
        let coordinates = coordinates
            .into_iter()
            .map(|(value, dimension)| {
                Ok(DynQuantity::new(
                    crate::modeling::value_literal::real(value.bind(py))?,
                    dimension.borrow(py).value,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let rules = self.observation_rules(py, observable, points)?;
        let value = self
            .native
            .observe_at(
                self.native.plan().model_artifact(),
                observable.id,
                &coordinates,
                &rules,
                output_index,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation::from_native(value, points.unwrap_or(0)))
    }

    pub(super) fn observe_parameter_direction(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        directions: &Bound<'_, PyDict>,
        output_index: usize,
    ) -> PyResult<PyObservation> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        let direction = self.parameter_direction(directions)?;
        let value = self
            .native
            .observe_parameter_jvp(
                self.native.plan().model_artifact(),
                observable.id,
                output_index,
                direction,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation::from_native(value, 0))
    }

    pub(super) fn bind_observable_tangent(
        &self,
        py: Python<'_>,
        directions: &Bound<'_, PyDict>,
    ) -> PyResult<PyObservableStateTangent> {
        let mut fields = Vec::with_capacity(directions.len());
        for (field, direction) in directions.iter() {
            let field = field.extract::<PyRef<'_, PyModelFieldRef>>()?;
            if field.exact_model_digest() != self.identity.model_digest() {
                return Err(PyValueError::new_err(
                    "FieldRef belongs to a different exact Model artifact",
                ));
            }
            let (dimension, values) = direction.extract::<(PyRef<'_, PyDimension>, Vec<f64>)>()?;
            let id = Id::<kinds::Field>::from_ulid(
                field
                    .exact_id()
                    .parse()
                    .map_err(|_| PyValueError::new_err("FieldRef has an invalid exact ID"))?,
            );
            fields.push((
                id,
                values
                    .into_iter()
                    .map(|value| DynQuantity::new(value, dimension.native()))
                    .collect(),
            ));
        }
        let native = self
            .native
            .observable_state_tangent(fields)
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservableStateTangent {
            native,
            result_identity: self.native.identity().to_owned(),
        })
    }

    pub(super) fn observe_jvp(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        tangent: &PyObservableStateTangent,
        points: usize,
    ) -> PyResult<PyObservation> {
        let rules = self.observation_rules(py, observable, Some(points))?;
        let value = self
            .native
            .observe_state_jvp(
                self.native.plan().model_artifact(),
                observable.id,
                &rules,
                &tangent.native,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation::from_native(value, points))
    }
    pub(super) fn observe_second_variation(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        directions: [&PyObservableStateTangent; 2],
        wrt: &PyModelFieldRef,
        points: usize,
    ) -> PyResult<PyObservation> {
        if wrt.exact_model_digest() != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "FieldRef belongs to a different exact Model artifact",
            ));
        }
        let field = Id::<kinds::Field>::from_ulid(
            wrt.exact_id()
                .parse()
                .map_err(|_| PyValueError::new_err("FieldRef has an invalid exact ID"))?,
        );
        let rules = self.observation_rules(py, observable, Some(points))?;
        let value = self
            .native
            .observe_state_second_variation(
                self.native.plan().model_artifact(),
                observable.id,
                &rules,
                field,
                [&directions[0].native, &directions[1].native],
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation::from_native(value, points))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyObservation>()?;
    module.add_class::<PyObservableStateTangent>()?;
    Ok(())
}

pub(super) fn quadrature_metadata(
    rules: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    points: usize,
) -> BTreeMap<String, (&'static str, usize, usize)> {
    rules
        .iter()
        .map(|(domain, rule)| {
            let dimension = rule.reference_cell().dimension();
            let (kind, points) = if dimension == 0 {
                ("Point", 1)
            } else if rule.reference_cell().family()
                == eqiora::meshing::ReferenceCellFamily::Simplex
            {
                ("SimplexDuffyGaussLegendre", points)
            } else {
                ("GaussLegendre", points)
            };
            (domain.ulid().to_string(), (kind, dimension, points))
        })
        .collect()
}

impl PyRunResult {
    pub(super) fn boundary_force_observation(
        &self,
        py: Python<'_>,
        selection: Py<PyGeometrySelection>,
        output_index: Option<usize>,
    ) -> PyResult<Py<PyBoundaryForce>> {
        let selected = selection.borrow(py);
        let name = selected.canonical_name().to_owned();
        let geometry_digest = selected.bound_source_digest().to_owned();
        let (force, mesh_digest) = if let Some(trajectory) = self.native.trajectory() {
            let plan = self.native.plan().as_transient_flow().ok_or_else(|| {
                capability_error(py, "boundary force requires a transient flow Result")
            })?;
            Self::validate_observable_selection(&selected, plan.geometry_digest())?;
            let index = output_index.ok_or_else(|| {
                PyValueError::new_err("transient boundary force requires an explicit output_index")
            })?;
            if trajectory
                .spatial_steps()
                .and_then(|steps| steps.get(index))
                .is_none()
            {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "spatial output index is outside the retained schedule",
                ));
            }
            let force = py
                .detach(|| trajectory.spatial_boundary_force(index, &name))
                .map_err(|error| crate::error::validation_error(py, &[error]))?;
            (force, plan.mesh_digest().to_owned())
        } else {
            if output_index.is_some() {
                return Err(PyValueError::new_err(
                    "steady boundary force has no output_index",
                ));
            }
            let payload = self.steady_stokes_payload(py, "boundary-force")?;
            let mesh = payload.outputs[0].borrow(py).mesh_handle(py)?;
            Self::validate_observable_selection(&selected, mesh.borrow(py).source_digest_value())?;
            let force = self
                .native
                .steady_stokes_boundary_reaction(&name)
                .map_err(|error| crate::error::validation_error(py, &[error]))?
                .ok_or_else(|| {
                    PyKeyError::new_err(format!(
                        "this Result has no boundary-force observable for {name:?}"
                    ))
                })?;
            (force, mesh.borrow(py).exact_mesh_digest().to_owned())
        };
        drop(selected);
        Py::new(
            py,
            PyBoundaryForce::new(
                selection,
                name,
                geometry_digest,
                self.native.identity(),
                output_index,
                &mesh_digest,
                force,
            ),
        )
    }
}
