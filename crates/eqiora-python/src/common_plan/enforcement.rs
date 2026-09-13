//! Thin unit-bearing numerical enforcement policy over exact Model conditions.

use crate::error::validation_error;
use crate::model::constraint::PyConstraintRef;
use crate::modeling::PyDimension;
use eqiora::DynQuantity;
use eqiora_numerics::finite_constraints::{ConstraintTolerance, FiniteConstraintEnforcement};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Positive physical tolerances for one exact inequality or complementarity condition.
#[pyclass(
    name = "ConstraintTolerance",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PyConstraintTolerance {
    reference: PyConstraintRef,
    native: ConstraintTolerance,
}

#[pymethods]
impl PyConstraintTolerance {
    fn __repr__(&self) -> String {
        format!(
            "ConstraintTolerance(kind={:?}, left_value={}, right_value={:?})",
            self.reference.kind,
            self.left_value(),
            self.right_value()
        )
    }
    #[staticmethod]
    fn inequality(
        py: Python<'_>,
        reference: &PyConstraintRef,
        value: f64,
        dimension: &PyDimension,
    ) -> PyResult<Self> {
        if reference.kind != "inequality" {
            return Err(PyTypeError::new_err(
                "inequality tolerance requires an exact inequality ConstraintRef",
            ));
        }
        let native = ConstraintTolerance::inequality(
            reference.native,
            DynQuantity::new(value, dimension.native()),
        )
        .map_err(|error| validation_error(py, &[error]))?;
        Ok(Self {
            reference: reference.clone(),
            native,
        })
    }
    #[staticmethod]
    fn complementarity(
        py: Python<'_>,
        reference: &PyConstraintRef,
        left_value: f64,
        left_dimension: &PyDimension,
        right_value: f64,
        right_dimension: &PyDimension,
    ) -> PyResult<Self> {
        if reference.kind != "complementarity" {
            return Err(PyTypeError::new_err(
                "complementarity tolerances require an exact complementarity ConstraintRef",
            ));
        }
        let native = ConstraintTolerance::complementarity(
            reference.native,
            DynQuantity::new(left_value, left_dimension.native()),
            DynQuantity::new(right_value, right_dimension.native()),
        )
        .map_err(|error| validation_error(py, &[error]))?;
        Ok(Self {
            reference: reference.clone(),
            native,
        })
    }
    #[getter]
    fn reference(&self) -> PyConstraintRef {
        self.reference.clone()
    }
    #[getter]
    fn left_value(&self) -> f64 {
        self.native.left().value()
    }
    #[getter]
    fn left_dimension(&self) -> PyDimension {
        PyDimension::from_native(self.native.left().dim())
    }
    #[getter]
    fn right_value(&self) -> Option<f64> {
        self.native.right().map(|quantity| quantity.value())
    }
    #[getter]
    fn right_dimension(&self) -> Option<PyDimension> {
        self.native
            .right()
            .map(|quantity| PyDimension::from_native(quantity.dim()))
    }
}

/// Explicit bounded active-set enumeration bound to one exact Model artifact.
#[pyclass(
    name = "ActiveSet",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(super) struct PyActiveSet {
    pub(super) model_digest: String,
    pub(super) native: FiniteConstraintEnforcement,
}

#[pymethods]
impl PyActiveSet {
    fn __repr__(&self) -> String {
        format!(
            "ActiveSet(model_digest={:?}, conditions={}, max_active_sets={})",
            self.model_digest,
            self.native.tolerances().len(),
            self.native.max_active_sets()
        )
    }
    #[new]
    #[pyo3(signature = (*, tolerances, max_active_sets))]
    fn new(
        py: Python<'_>,
        tolerances: &Bound<'_, PyTuple>,
        max_active_sets: u32,
    ) -> PyResult<Self> {
        let entries = tolerances
            .iter()
            .map(|value| {
                value
                    .extract::<PyRef<'_, PyConstraintTolerance>>()
                    .map(|value| value.clone())
                    .map_err(PyErr::from)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let first = entries.first().ok_or_else(|| {
            PyValueError::new_err("ActiveSet requires explicit condition tolerances")
        })?;
        let model_digest = first.reference.model_digest.clone();
        if entries
            .iter()
            .any(|entry| entry.reference.model_digest != model_digest)
        {
            return Err(PyValueError::new_err(
                "ActiveSet tolerances cross different exact Models",
            ));
        }
        let native = FiniteConstraintEnforcement::active_set(
            entries.into_iter().map(|entry| entry.native).collect(),
            max_active_sets,
        )
        .map_err(|error| validation_error(py, &[error]))?;
        Ok(Self {
            model_digest,
            native,
        })
    }
    #[getter]
    fn tolerances(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(
            py,
            self.native
                .tolerances()
                .iter()
                .map(|native| PyConstraintTolerance {
                    reference: PyConstraintRef {
                        model_digest: self.model_digest.clone(),
                        native: native.reference(),
                        kind: if native.right().is_some() {
                            "complementarity"
                        } else {
                            "inequality"
                        },
                    },
                    native: native.clone(),
                }),
        )
        .map(|tuple| tuple.unbind())
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn max_active_sets(&self) -> u32 {
        self.native.max_active_sets()
    }
}

pub(super) fn from_plan(plan: &super::PyPlan) -> Option<PyActiveSet> {
    let plan = plan.native.as_algebraic()?;
    Some(PyActiveSet {
        model_digest: plan.model_digest().to_owned(),
        native: plan.enforcement()?.clone(),
    })
}
