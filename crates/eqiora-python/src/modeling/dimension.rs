use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::DimExponents;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyTuple};

use super::PyDimension;

fn extract_exponent(value: &Bound<'_, PyAny>) -> PyResult<(i32, i32)> {
    let pair = if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        (value.extract::<i32>()?, 1)
    } else if value.is_instance(&value.py().import("fractions")?.getattr("Fraction")?)? {
        (
            value.getattr("numerator")?.extract::<i32>()?,
            value.getattr("denominator")?.extract::<i32>()?,
        )
    } else {
        return Err(PyTypeError::new_err(
            "dimension exponent must be an int or fractions.Fraction",
        ));
    };
    if pair.0 == i32::MIN || pair.1 <= 0 {
        return Err(PyValueError::new_err(
            "dimension exponents require magnitudes at most 2147483647 and positive denominators",
        ));
    }
    Ok(pair)
}

pub(crate) fn exponents(py: Python<'_>, value: DimExponents) -> PyResult<Py<PyTuple>> {
    let fraction = py.import("fractions")?.getattr("Fraction")?;
    let values = value
        .exponents()
        .into_iter()
        .map(|pair| fraction.call1(pair))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}

impl PyDimension {
    pub(crate) const fn native(&self) -> DimExponents {
        self.value
    }

    pub(crate) const fn from_native(value: DimExponents) -> Self {
        Self { value }
    }
}

#[pymethods]
impl PyDimension {
    #[new]
    #[pyo3(signature = (*, mass=(0,1), length=(0,1), time=(0,1), current=(0,1), temperature=(0,1), amount=(0,1), luminous_intensity=(0,1)))]
    #[pyo3(
        text_signature = "(*, mass=0, length=0, time=0, current=0, temperature=0, amount=0, luminous_intensity=0)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = extract_exponent)] mass: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] length: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] time: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] current: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] temperature: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] amount: (i32, i32),
        #[pyo3(from_py_with = extract_exponent)] luminous_intensity: (i32, i32),
    ) -> PyResult<Self> {
        let value = DimExponents::from_rationals([
            mass,
            length,
            time,
            current,
            temperature,
            amount,
            luminous_intensity,
        ])
        .ok_or_else(|| PyValueError::new_err("invalid rational dimension exponents"))?;
        Ok(Self::from_native(value))
    }

    #[getter]
    fn exponents(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        exponents(py, self.native())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.value == other.value)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.__eq__(other)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.value.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn __repr__(&self) -> String {
        let arguments = [
            "mass",
            "length",
            "time",
            "current",
            "temperature",
            "amount",
            "luminous_intensity",
        ]
        .into_iter()
        .zip(self.value.exponents())
        .map(|(name, (numerator, denominator))| {
            if denominator == 1 {
                format!("{name}={numerator}")
            } else {
                format!("{name}=Fraction({numerator}, {denominator})")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
        format!("Dimension({arguments})")
    }
}
