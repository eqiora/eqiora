//! Read-only original-condition receipts from the accepted native Result.
use super::*;
use crate::model::constraint::PyConstraintRef;
use crate::modeling::PyDimension;
use eqiora_numerics::finite_constraints::{
    ConstraintActivity, ConstraintMeasurement, ConstraintTolerance,
};
use pyo3::types::PyTuple;

/// Independently rederived original condition values, units and activity from an accepted Result.
#[pyclass(name = "ConstraintMeasurement", module = "eqiora._eqiora", frozen)]
pub(super) struct PyConstraintMeasurement {
    reference: PyConstraintRef,
    measurement: ConstraintMeasurement,
    tolerance: ConstraintTolerance,
}

#[pymethods]
impl PyConstraintMeasurement {
    fn __repr__(&self) -> String {
        format!(
            "ConstraintMeasurement(activity={:?}, left_value={}, right_value={})",
            self.activity(),
            self.left_value(),
            self.right_value()
        )
    }
    #[getter]
    fn reference(&self) -> PyConstraintRef {
        self.reference.clone()
    }
    #[getter]
    fn activity(&self) -> &'static str {
        match self.measurement.activity() {
            ConstraintActivity::Active => "active",
            ConstraintActivity::Inactive => "inactive",
            ConstraintActivity::Biactive => "biactive",
            ConstraintActivity::Inequality => "inequality",
        }
    }
    #[getter]
    fn left_value(&self) -> f64 {
        self.measurement.left().value()
    }
    #[getter]
    fn right_value(&self) -> f64 {
        self.measurement.right().value()
    }
    #[getter]
    fn left_dimension(&self) -> PyDimension {
        PyDimension::from_native(self.measurement.left().dim())
    }
    #[getter]
    fn right_dimension(&self) -> PyDimension {
        PyDimension::from_native(self.measurement.right().dim())
    }
    #[getter]
    fn left_tolerance(&self) -> f64 {
        self.tolerance.left().value()
    }
    #[getter]
    fn right_tolerance(&self) -> Option<f64> {
        self.tolerance.right().map(|value| value.value())
    }
}

pub(super) fn measurements(result: &PyRunResult, py: Python<'_>) -> PyResult<Py<PyTuple>> {
    let measurements = result.native.constraint_measurements();
    if measurements.is_empty() {
        return Ok(PyTuple::empty(py).unbind());
    }
    let plan = result
        .native
        .plan()
        .as_algebraic()
        .ok_or_else(|| PyRuntimeError::new_err("constraint receipt requires finite Plan"))?;
    let enforcement = plan.enforcement().ok_or_else(|| {
        PyRuntimeError::new_err("constraint receipt requires explicit enforcement")
    })?;
    let entries = measurements
        .iter()
        .map(|measurement| {
            let tolerance = enforcement
                .tolerances()
                .iter()
                .find(|entry| entry.reference() == measurement.reference())
                .ok_or_else(|| {
                    PyRuntimeError::new_err("constraint receipt missing exact operand tolerance")
                })?;
            Py::new(
                py,
                PyConstraintMeasurement {
                    reference: PyConstraintRef {
                        model_digest: plan.model_digest().to_owned(),
                        native: measurement.reference(),
                        kind: if tolerance.right().is_some() {
                            "complementarity"
                        } else {
                            "inequality"
                        },
                    },
                    measurement: measurement.clone(),
                    tolerance: tolerance.clone(),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, entries).map(Bound::unbind)
}
