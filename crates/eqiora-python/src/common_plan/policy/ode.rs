//! Common Python ODE policy construction and exact Model-bound contracts.
use super::*;
impl OdePolicyData {
    pub(super) fn absolute_tolerances(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        for ((field, order, component, imaginary), tolerance) in self
            .coordinates
            .iter()
            .zip(self.native.absolute_tolerances())
        {
            result.set_item(
                (field.clone_ref(py), *order, *component, *imaginary),
                tolerance.value(),
            )?;
        }
        Ok(result.unbind())
    }
    pub(super) fn new(
        py: Python<'_>,
        method: eqiora::time::TimeMethod,
        initial_step_s: f64,
        relative_tolerance: f64,
        absolute_tolerances: &Bound<'_, PyDict>,
        events: Option<&crate::common_plan::event_policy::PyEventPolicy>,
        forward_sensitivities: Option<&crate::common_plan::forward_policy::PyForwardSensitivity>,
    ) -> PyResult<Self> {
        let mut native = Vec::with_capacity(absolute_tolerances.len());
        let mut coordinates = Vec::with_capacity(absolute_tolerances.len());
        for (field, value) in absolute_tolerances.iter() {
            let (field, order, component, imaginary) = field.extract::<(Py<PyModelFieldRef>, u32, usize, bool)>().map_err(|_| {
                PyTypeError::new_err(
                    "absolute_tolerances keys must be (exact eqiora.FieldRef, derivative order, component, imaginary) tuples",
                )
            })?;
            let value = exact_time_float(&value)?;
            let id = Ulid::from_string(field.borrow(py).exact_id()).map_err(|_| {
                PyTypeError::new_err("absolute_tolerances contains an invalid exact FieldRef")
            })?;
            native.push(
                CommonTimeTolerance::new(
                    eqiora::TimeStateCoordinate::new(
                        Id::<kinds::Field>::from_ulid(id),
                        order,
                        component,
                        imaginary,
                    ),
                    value,
                )
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?,
            );
            coordinates.push((field, order, component, imaginary));
        }
        coordinates.sort_by_key(|(field, order, component, imaginary)| {
            (
                field.borrow(py).exact_id().to_owned(),
                *order,
                *component,
                *imaginary,
            )
        });
        let mut native = CommonOdePolicy::new(method, initial_step_s, relative_tolerance, native)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        if let Some(events) = events {
            native = native
                .with_events(
                    events.max_events,
                    events
                        .entries
                        .iter()
                        .map(|entry| (entry.activation.id, entry.quantity))
                        .collect(),
                )
                .map_err(|error| validation_error(py, &[error]))?;
        }
        if let Some(policy) = forward_sensitivities {
            native = native
                .with_forward_sensitivities(
                    policy.relative_tolerance,
                    policy
                        .entries
                        .iter()
                        .map(|entry| {
                            let field = Ulid::from_string(entry.field.exact_id())
                                .map(Id::<kinds::Field>::from_ulid)
                                .expect("validated exact FieldRef");
                            (
                                eqiora::TimeStateCoordinate::new(
                                    field,
                                    entry.derivative_order,
                                    entry.component,
                                    entry.imaginary,
                                ),
                                entry.parameter.value.id(),
                                entry.quantity,
                            )
                        })
                        .collect(),
                )
                .map_err(|error| validation_error(py, &[error]))?;
        }
        Ok(Self {
            native,
            coordinates,
            contract_models: Vec::new(),
            events: events.cloned(),
            forward_sensitivities: forward_sensitivities.cloned(),
        })
    }
    pub(in crate::common_plan) fn from_native(
        py: Python<'_>,
        model_digest: &str,
        native: CommonOdePolicy,
        document: &eqiora::api::ModelDocument,
    ) -> PyResult<Self> {
        let coordinates = native
            .absolute_tolerances()
            .iter()
            .map(|entry| {
                Py::new(
                    py,
                    PyModelFieldRef::from_exact(
                        model_digest.to_owned(),
                        entry.coordinate().field().ulid().to_string(),
                    ),
                )
                .map(|field| {
                    (
                        field,
                        entry.coordinate().derivative_order(),
                        entry.coordinate().component(),
                        entry.coordinate().is_imaginary(),
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let events =
            native
                .events()
                .map(|policy| crate::common_plan::event_policy::PyEventPolicy {
                    model_digest: model_digest.to_owned(),
                    max_events: policy.max_events(),
                    entries: policy
                        .guard_tolerances()
                        .iter()
                        .map(|entry| crate::common_plan::event_policy::PyGuardTolerance {
                            activation: crate::model::PyActivationRef {
                                model_digest: model_digest.to_owned(),
                                id: entry.activation(),
                            },
                            quantity: entry.quantity(),
                        })
                        .collect(),
                });
        let forward_sensitivities = native
            .forward_sensitivities()
            .map(|policy| {
                crate::common_plan::forward_policy::PyForwardSensitivity::from_native(
                    py,
                    document,
                    policy.relative_tolerance(),
                    policy
                        .absolute_tolerances()
                        .iter()
                        .map(|entry| (entry.coordinate(), entry.parameter(), entry.quantity()))
                        .collect(),
                )
            })
            .transpose()?;
        Ok(Self {
            native,
            coordinates,
            contract_models: vec![model_digest.to_owned()],
            events,
            forward_sensitivities,
        })
    }

    pub(in crate::common_plan) fn belongs_to_model(
        &self,
        py: Python<'_>,
        model_digest: &str,
    ) -> bool {
        self.contract_models
            .iter()
            .all(|digest| digest == model_digest)
            && self
                .forward_sensitivities
                .as_ref()
                .is_none_or(|policy| policy.model_digest == model_digest)
            && self
                .events
                .as_ref()
                .is_none_or(|events| events.model_digest == model_digest)
            && self
                .coordinates
                .iter()
                .all(|(field, _, _, _)| field.borrow(py).exact_model_digest() == model_digest)
    }
}

impl OdePolicyData {
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            native: self.native.clone(),
            coordinates: self
                .coordinates
                .iter()
                .map(|(f, o, c, i)| (f.clone_ref(py), *o, *c, *i))
                .collect(),
            events: self.events.clone(),
            forward_sensitivities: self.forward_sensitivities.clone(),
            contract_models: self.contract_models.clone(),
        }
    }
    pub(super) fn with_norm(
        &self,
        py: Python<'_>,
        fields: Vec<Py<PyModelFieldRef>>,
        target: f64,
        tolerance: f64,
        dimension: &crate::modeling::PyDimension,
    ) -> PyResult<Self> {
        let mut result = self.copy(py);
        let mut ids = Vec::new();
        for field in fields {
            let field = field.borrow(py);
            result
                .contract_models
                .push(field.exact_model_digest().to_owned());
            let id = Ulid::from_string(field.exact_id())
                .map_err(|_| PyTypeError::new_err("invalid norm FieldRef"))?;
            ids.push(Id::<kinds::Field>::from_ulid(id));
        }
        result.native = result
            .native
            .with_conserved_norm(
                ids,
                eqiora::DynQuantity::new(target, dimension.native()),
                eqiora::DynQuantity::new(tolerance, dimension.native()),
            )
            .map_err(|e| validation_error(py, &[e]))?;
        Ok(result)
    }
    pub(super) fn with_hermitian(
        &self,
        py: Python<'_>,
        parameter: &crate::model::PyModelParameterRef,
    ) -> PyResult<Self> {
        let mut result = self.copy(py);
        result
            .contract_models
            .push(parameter.value.model().artifact().to_string());
        result.native = result
            .native
            .with_hermitian_parameter(parameter.value.id())
            .map_err(|e| validation_error(py, &[e]))?;
        Ok(result)
    }
}
