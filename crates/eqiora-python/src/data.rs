//! Python projection of the existing canonical resolved-array owner.
use eqiora::artifact::{ResolvedArrayDecoderLimits, ResolvedArrayScalarV1, ResolvedArrayV1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};

use crate::diagnostic_error;
use crate::modeling::value_literal;

fn unsigned(values: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<u64>> {
    values
        .iter()
        .map(|value| value_literal::integer(value)?.extract())
        .collect()
}

#[pyclass(name = "ResolvedArray", module = "eqiora.data", frozen)]
pub(crate) struct PyResolvedArray {
    value: ResolvedArrayV1,
}

#[pymethods]
impl PyResolvedArray {
    /// Construct finite row-major binary64 values through the canonical owner.
    #[staticmethod]
    fn from_f64(
        py: Python<'_>,
        shape: Vec<Bound<'_, PyAny>>,
        values: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let values = values
            .iter()
            .map(value_literal::real)
            .collect::<PyResult<Vec<_>>>()?;
        ResolvedArrayV1::from_f64(unsigned(shape)?, values)
            .map(|value| Self { value })
            .map_err(|error| diagnostic_error(py, &[error]))
    }

    /// Construct exact row-major unsigned integers without binary64 conversion.
    #[staticmethod]
    fn from_u64(
        py: Python<'_>,
        shape: Vec<Bound<'_, PyAny>>,
        values: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        ResolvedArrayV1::from_u64(unsigned(shape)?, unsigned(values)?)
            .map(|value| Self { value })
            .map_err(|error| diagnostic_error(py, &[error]))
    }

    /// Decode the current artifact within its ordinary byte/shape/value limits.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        ResolvedArrayV1::from_json(data, ResolvedArrayDecoderLimits::default())
            .map(|value| Self { value })
            .map_err(|error| diagnostic_error(py, &[error]))
    }

    /// Canonical bytes for package assets and external-array consumers.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.value
            .canonical_json()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|error| diagnostic_error(py, &[error]))
    }

    #[getter]
    fn digest(&self, py: Python<'_>) -> PyResult<String> {
        self.value
            .digest()
            .map(|digest| digest.to_string())
            .map_err(|error| diagnostic_error(py, &[error]))
    }

    #[getter]
    fn scalar(&self) -> &'static str {
        match self.value.scalar() {
            ResolvedArrayScalarV1::F64 => "f64",
            ResolvedArrayScalarV1::U64 => "u64",
        }
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.value.shape().iter().copied())
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match self.value.scalar() {
            ResolvedArrayScalarV1::F64 => PyTuple::new(
                py,
                self.value.f64_values().expect("f64 array").iter().copied(),
            ),
            ResolvedArrayScalarV1::U64 => PyTuple::new(
                py,
                self.value.u64_values().expect("u64 array").iter().copied(),
            ),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResolvedArray>()
}
