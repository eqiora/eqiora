//! Lowest-order oriented tetrahedral moment policies.
use pyo3::prelude::*;

/// Tangential line integrals (lowest-order Nedelec first kind).
#[pyclass(
    name = "TetrahedralEdge",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyTetrahedralEdge;

#[pymethods]
impl PyTetrahedralEdge {
    #[new]
    const fn new() -> Self {
        Self
    }
    #[getter]
    const fn method(&self) -> &'static str {
        "tetrahedral-edge"
    }
    #[getter]
    const fn space(&self) -> &'static str {
        "tetrahedral-edge"
    }
    #[getter]
    const fn quadrature(&self) -> &'static str {
        "tetrahedron-duffy-gauss-legendre-3-per-axis"
    }
    fn __repr__(&self) -> &'static str {
        "TetrahedralEdge()"
    }
}

/// Normal flux integrals (lowest-order Raviart–Thomas).
#[pyclass(
    name = "TetrahedralFace",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyTetrahedralFace;

#[pymethods]
impl PyTetrahedralFace {
    #[new]
    const fn new() -> Self {
        Self
    }
    #[getter]
    const fn method(&self) -> &'static str {
        "tetrahedral-face"
    }
    #[getter]
    const fn space(&self) -> &'static str {
        "tetrahedral-face"
    }
    #[getter]
    const fn quadrature(&self) -> &'static str {
        "tetrahedron-duffy-gauss-legendre-3-per-axis"
    }
    fn __repr__(&self) -> &'static str {
        "TetrahedralFace()"
    }
}
