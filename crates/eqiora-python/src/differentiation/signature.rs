//! Read-only Python projection of the native semantic value signature.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::api::DifferentiableValueSignature;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Exact native Model, mathematical type and coefficient support of a selected value.
#[pyclass(
    name = "DifferentiableValueSignature",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyDifferentiableValueSignature {
    value: DifferentiableValueSignature,
}

#[pymethods]
impl PyDifferentiableValueSignature {
    #[getter]
    fn id(&self) -> String {
        self.value.entity().ulid().to_string()
    }

    #[getter]
    fn model_digest(&self) -> String {
        self.value.model().artifact().to_string()
    }

    /// Original mathematical type, excluding coefficient and mapping axes.
    #[getter]
    fn value_type(&self) -> crate::modeling::PyValueType {
        crate::modeling::PyValueType {
            value: self.value.mathematical_type().value_type.clone(),
        }
    }

    /// Complete pointwise value shape, including any spatial coefficient axis.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.value.shape())?.unbind())
    }

    #[getter]
    fn mesh_digest(&self) -> Option<&str> {
        self.value.mesh_digest()
    }

    #[getter]
    fn space(&self) -> Option<&'static str> {
        self.value.space().map(crate::common_plan::space_name)
    }

    /// Exact (topological dimension, entity index) associations in coefficient order.
    #[getter]
    fn coefficient_entities(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.value
                .coefficient_entities()
                .iter()
                .map(|entity| (entity.dimension(), entity.index())),
        )?
        .unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "DifferentiableValueSignature(id={:?}, shape={:?})",
            self.id(),
            self.value.shape()
        )
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.value.model().artifact().hash(&mut hasher);
        self.value.entity().hash(&mut hasher);
        self.value.mesh_digest().hash(&mut hasher);
        self.value.space().hash(&mut hasher);
        self.value.shape().hash(&mut hasher);
        self.value.coefficient_entities().hash(&mut hasher);
        hasher.finish()
    }
}

pub(super) fn project(
    py: Python<'_>,
    values: &[DifferentiableValueSignature],
) -> PyResult<Py<PyTuple>> {
    let leaves = values
        .iter()
        .map(|value| {
            Py::new(
                py,
                PyDifferentiableValueSignature {
                    value: value.clone(),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, leaves)?.unbind())
}
