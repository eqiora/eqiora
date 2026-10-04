//! Dimensioned interval inputs adapt directly to the shared support contract.
use crate::modeling::PyDimension;
use eqiora::{DynQuantity, kernel::AxisBounds};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

/// Finite coherent-SI endpoints for one mathematical coordinate factor.
#[pyclass(
    name = "CoordinateInterval",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyCoordinateInterval {
    pub(crate) bounds: AxisBounds,
}

#[pymethods]
impl PyCoordinateInterval {
    #[new]
    #[pyo3(signature = (lower, upper, *, dimension))]
    fn new(
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
        dimension: &PyDimension,
    ) -> PyResult<Self> {
        let endpoint = |value: &Bound<'_, PyAny>| {
            if value.is_instance_of::<pyo3::types::PyComplex>() {
                return Err(PyValueError::new_err(
                    "coordinate interval endpoints must be real",
                ));
            }
            let (real, imaginary) = crate::modeling::value_literal::scalar(value)?;
            if imaginary != 0.0 {
                return Err(PyValueError::new_err(
                    "coordinate interval endpoints must be real",
                ));
            }
            Ok(DynQuantity::new(real, dimension.value))
        };
        Ok(Self {
            bounds: AxisBounds::new(endpoint(lower)?, endpoint(upper)?)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CoordinateInterval({:?}, {:?}, dimension={})",
            self.lower(),
            self.upper(),
            self.dimension().__repr__()
        )
    }

    fn __hash__(&self) -> u64 {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        for value in [self.lower(), self.upper()] {
            (if value == 0.0 { 0.0 } else { value })
                .to_bits()
                .hash(&mut state);
        }
        self.bounds.lower().dim().hash(&mut state);
        state.finish()
    }

    #[getter]
    fn lower(&self) -> f64 {
        self.bounds.lower().value()
    }
    #[getter]
    fn upper(&self) -> f64 {
        self.bounds.upper().value()
    }
    #[getter]
    fn dimension(&self) -> PyDimension {
        PyDimension {
            value: self.bounds.lower().dim(),
        }
    }
}
