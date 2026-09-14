//! Exact Activation references and unit-bearing explicit event controls.
use super::*;
use crate::model::PyActivationRef;
use crate::modeling::PyDimension;
use eqiora::DynQuantity;
use eqiora_numerics::CommonTsitouras45;

#[pyclass(
    name = "GuardTolerance",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PyGuardTolerance {
    pub(super) activation: PyActivationRef,
    pub(super) quantity: DynQuantity,
}
#[pymethods]
impl PyGuardTolerance {
    #[new]
    fn new(
        _py: Python<'_>,
        activation: &PyActivationRef,
        value: f64,
        dimension: &PyDimension,
    ) -> PyResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyTypeError::new_err(
                "guard tolerance must be positive and finite",
            ));
        }
        Ok(Self {
            activation: activation.clone(),
            quantity: DynQuantity::new(value, dimension.native()),
        })
    }
    #[getter]
    fn activation(&self) -> PyActivationRef {
        self.activation.clone()
    }
    #[getter]
    fn value(&self) -> f64 {
        self.quantity.value()
    }
    #[getter]
    fn dimension(&self) -> PyDimension {
        PyDimension::from_native(self.quantity.dim())
    }
}

#[pyclass(
    name = "EventPolicy",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyEventPolicy {
    pub(super) model_digest: String,
    pub(super) max_events: usize,
    pub(super) entries: Vec<PyGuardTolerance>,
}
#[pymethods]
impl PyEventPolicy {
    #[new]
    #[pyo3(signature = (*, max_events, guard_tolerances))]
    fn new(
        _py: Python<'_>,
        max_events: usize,
        guard_tolerances: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let entries = guard_tolerances
            .iter()
            .map(|entry| {
                entry
                    .extract::<PyRef<'_, PyGuardTolerance>>()
                    .map(|entry| entry.clone())
                    .map_err(PyErr::from)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let model_digest = entries
            .first()
            .ok_or_else(|| PyTypeError::new_err("EventPolicy requires explicit guard tolerances"))?
            .activation
            .model_digest
            .clone();
        if entries
            .iter()
            .any(|entry| entry.activation.model_digest != model_digest)
        {
            return Err(PyTypeError::new_err(
                "EventPolicy guard tolerances belong to different exact Models",
            ));
        }
        CommonTsitouras45::validate_event_controls(
            max_events,
            entries
                .iter()
                .map(|entry| (entry.activation.id, entry.quantity))
                .collect(),
        )
        .map_err(|error| validation_error(_py, &[error]))?;
        Ok(Self {
            model_digest,
            max_events,
            entries,
        })
    }
    #[getter]
    fn max_events(&self) -> usize {
        self.max_events
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn guard_tolerances(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(py, self.entries.clone()).map(Bound::unbind)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGuardTolerance>()?;
    module.add_class::<PyEventPolicy>()?;
    Ok(())
}
