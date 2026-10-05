use super::*;
use crate::modeling::PyValueType;

/// One accepted typed eigenpair tied to the exact source Fields and Result.
#[pyclass(
    name = "Eigenpair",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(super) struct PyEigenpair {
    eigenvalue: eqiora::ValueLiteral,
    mode: eqiora::ValueLiteral,
    mode_fields: Vec<(PyModelFieldRef, eqiora::ValueLiteral)>,
    #[pyo3(get)]
    eigenvalue_field: PyModelFieldRef,
    #[pyo3(get)]
    mode_field: PyModelFieldRef,
    #[pyo3(get)]
    result_identity: String,
    #[pyo3(get)]
    relative_residual: f64,
    #[pyo3(get)]
    normalization_defect: f64,
}

#[pymethods]
impl PyEigenpair {
    /// Return a source Field's value and exact type for this selected mode.
    fn field(
        &self,
        py: Python<'_>,
        field: PyRef<'_, PyModelFieldRef>,
    ) -> PyResult<(Py<PyAny>, PyValueType)> {
        let value = if *field == self.eigenvalue_field {
            &self.eigenvalue
        } else {
            &self
                .mode_fields
                .iter()
                .find(|(candidate, _)| candidate == &*field)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "Field is not owned by this exact eigenpair Model and coordinate chain",
                    )
                })?
                .1
        };
        Ok((
            crate::modeling::value_literal::to_python(py, value)?,
            PyValueType {
                value: value.value_type().clone(),
            },
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "Eigenpair(eigenvalue={:?}, relative_residual={}, normalization_defect={}, result_identity={:?})",
            self.eigenvalue.component(0),
            self.relative_residual,
            self.normalization_defect,
            self.result_identity
        )
    }

    #[getter]
    fn eigenvalue(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.eigenvalue)
    }
    #[getter]
    fn mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.mode)
    }
    #[getter]
    fn eigenvalue_type(&self) -> PyValueType {
        PyValueType {
            value: self.eigenvalue.value_type().clone(),
        }
    }
    #[getter]
    fn mode_type(&self) -> PyValueType {
        PyValueType {
            value: self.mode.value_type().clone(),
        }
    }
}

impl PyRunResult {
    pub(super) fn eigenpair_count(&self) -> usize {
        self.native.eigenpair_count()
    }
    pub(super) fn eigen_convergence(&self) -> Option<&'static str> {
        self.native.eigen_convergence()
    }
    pub(super) fn eigen_candidate_counts(&self) -> Option<(usize, usize)> {
        self.native.eigen_candidate_counts()
    }
    pub(super) fn eigenpair_value(&self, index: usize) -> PyResult<PyEigenpair> {
        let plan = self
            .native
            .plan()
            .as_eigen()
            .ok_or_else(|| PyValueError::new_err("Result is not spectral"))?;
        let (eigenvalue, mode, relative_residual, normalization_defect) =
            self.native.eigenpair(index).ok_or_else(|| {
                PyValueError::new_err("eigenpair index is outside the accepted modes")
            })?;
        Ok(PyEigenpair {
            eigenvalue: eigenvalue.clone(),
            mode: mode.clone(),
            mode_fields: self
                .native
                .eigenmode_fields(index)
                .expect("selected eigenpair")
                .iter()
                .map(|(field, value)| {
                    (
                        PyModelFieldRef::from_exact(
                            plan.model_digest().to_owned(),
                            field.ulid().to_string(),
                        ),
                        value.clone(),
                    )
                })
                .collect(),
            eigenvalue_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.eigenvalue_field().ulid().to_string(),
            ),
            mode_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.mode_field().ulid().to_string(),
            ),
            result_identity: self.native.identity().to_owned(),
            relative_residual,
            normalization_defect,
        })
    }
    /// Return the metric projector and its exact finite-map type.
    pub(super) fn eigenprojector_value(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<(Py<PyAny>, PyValueType)> {
        let value = self
            .native
            .eigenprojector(&indices)
            .map_err(|d| diagnostic_error(py, &[d]))?;
        Ok((
            crate::modeling::value_literal::to_python(py, &value)?,
            PyValueType {
                value: value.value_type().clone(),
            },
        ))
    }
}
