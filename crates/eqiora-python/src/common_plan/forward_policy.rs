//! Exact Model-bound continuous forward derivative controls.
use super::*;
use crate::model::{PyModelFieldRef, PyModelParameterRef};
use crate::modeling::PyDimension;
use eqiora::{DynQuantity, Id, kinds};
use eqiora_numerics::CommonTsitouras45;
use ulid::Ulid;

#[pyclass(
    name = "SensitivityTolerance",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PySensitivityTolerance {
    pub(super) field: PyModelFieldRef,
    pub(super) parameter: PyModelParameterRef,
    pub(super) quantity: DynQuantity,
}
#[pymethods]
impl PySensitivityTolerance {
    #[new]
    fn new(
        _py: Python<'_>,
        field: &PyModelFieldRef,
        parameter: &PyModelParameterRef,
        value: f64,
        dimension: &PyDimension,
    ) -> PyResult<Self> {
        if field.exact_model_digest() != parameter.value.model().artifact().to_string() {
            return Err(PyTypeError::new_err(
                "sensitivity Field and Parameter belong to different exact Models",
            ));
        }
        let id = Ulid::from_string(field.exact_id())
            .map_err(|_| PyTypeError::new_err("invalid exact FieldRef"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(PyTypeError::new_err(
                "sensitivity tolerance must be positive and finite",
            ));
        }
        let _field_id = Id::<kinds::Field>::from_ulid(id);
        Ok(Self {
            field: field.clone(),
            parameter: parameter.clone(),
            quantity: DynQuantity::new(value, dimension.native()),
        })
    }
    #[getter]
    fn field(&self) -> PyModelFieldRef {
        self.field.clone()
    }
    #[getter]
    fn parameter(&self) -> PyModelParameterRef {
        self.parameter.clone()
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
    name = "ForwardSensitivity",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyForwardSensitivity {
    pub(super) model_digest: String,
    pub(super) relative_tolerance: f64,
    pub(super) entries: Vec<PySensitivityTolerance>,
}
impl PyForwardSensitivity {
    pub(super) fn from_native(
        py: Python<'_>,
        document: &eqiora::api::ModelDocument,
        relative_tolerance: f64,
        controls: Vec<(Id<kinds::Field>, Id<kinds::Parameter>, DynQuantity)>,
    ) -> PyResult<Self> {
        let entries = controls
            .into_iter()
            .map(|(field_id, parameter_id, quantity)| {
                let parameter =
                    PyModelParameterRef::from_document(document, &parameter_id.ulid().to_string())
                        .map_err(|error| validation_error(py, &[error]))?;
                let field = PyModelFieldRef::from_exact(
                    parameter.value.model().artifact().to_string(),
                    field_id.ulid().to_string(),
                );
                Ok(PySensitivityTolerance {
                    field,
                    parameter,
                    quantity,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let model_digest = entries[0].field.exact_model_digest().to_owned();
        Ok(Self {
            model_digest,
            relative_tolerance,
            entries,
        })
    }
}
#[pymethods]
impl PyForwardSensitivity {
    #[new]
    #[pyo3(signature = (*, relative_tolerance, absolute_tolerances))]
    fn new(
        _py: Python<'_>,
        relative_tolerance: f64,
        absolute_tolerances: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let mut entries = absolute_tolerances
            .iter()
            .map(|entry| {
                entry
                    .extract::<PyRef<'_, PySensitivityTolerance>>()
                    .map(|entry| entry.clone())
                    .map_err(PyErr::from)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let model_digest = entries
            .first()
            .ok_or_else(|| {
                PyTypeError::new_err("ForwardSensitivity requires explicit absolute tolerances")
            })?
            .field
            .exact_model_digest()
            .to_owned();
        if entries
            .iter()
            .any(|entry| entry.field.exact_model_digest() != model_digest)
        {
            return Err(PyTypeError::new_err(
                "forward sensitivity tolerances belong to different exact Models",
            ));
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err(PyTypeError::new_err(
                "relative_tolerance must be positive and finite",
            ));
        }
        entries.sort_by_key(|entry| {
            (
                entry.parameter.value.id().ulid(),
                entry.field.exact_id().to_owned(),
            )
        });
        CommonTsitouras45::validate_forward_sensitivity_controls(
            relative_tolerance,
            entries
                .iter()
                .map(|entry| {
                    let field = Id::<kinds::Field>::from_ulid(
                        Ulid::from_string(entry.field.exact_id()).expect("validated FieldRef"),
                    );
                    (field, entry.parameter.value.id(), entry.quantity)
                })
                .collect(),
        )
        .map_err(|error| validation_error(_py, &[error]))?;
        Ok(Self {
            model_digest,
            relative_tolerance,
            entries,
        })
    }
    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn absolute_tolerances(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(py, self.entries.clone()).map(Bound::unbind)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySensitivityTolerance>()?;
    module.add_class::<PyForwardSensitivity>()?;
    Ok(())
}
