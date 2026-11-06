//! A closed request for the ordinary quadratic scalar finite-element path.
use super::*;

/// Full tensor-product Q2 scalar Galerkin policy on affine quads and hexes.
#[pyclass(
    name = "Q2",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyQ2;

#[pymethods]
impl PyQ2 {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "q2"
    }

    #[getter]
    const fn space(&self) -> &'static str {
        "continuous-lagrange-q2"
    }

    #[pyo3(signature = (domain, /))]
    fn at(&self, domain: &PyModelDomainRef) -> PyScopedSpatialBinding {
        PyScopedSpatialBinding {
            domain: domain.clone(),
            policy: ScopedSpatialKind::Q2,
        }
    }

    fn __repr__(&self) -> &'static str {
        "Q2()"
    }
}

/// Continuous P2 scalar Galerkin policy on affine triangles and tetrahedra.
#[pyclass(
    name = "P2",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyP2;

#[pymethods]
impl PyP2 {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "p2"
    }

    #[getter]
    const fn space(&self) -> &'static str {
        "continuous-lagrange-p2"
    }

    #[pyo3(signature = (domain, /))]
    fn at(&self, domain: &PyModelDomainRef) -> PyScopedSpatialBinding {
        PyScopedSpatialBinding {
            domain: domain.clone(),
            policy: ScopedSpatialKind::P2,
        }
    }

    fn __repr__(&self) -> &'static str {
        "P2()"
    }
}
