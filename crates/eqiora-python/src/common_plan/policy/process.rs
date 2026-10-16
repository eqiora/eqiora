//! Numerical requests for finite serial-host stochastic evolution.
use pyo3::prelude::*;

/// Ito Euler-Maruyama on one dyadic partition of the Run horizon.
#[pyclass(
    name = "EulerMaruyama",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyEulerMaruyama {
    #[pyo3(get)]
    pub(crate) depth: u8,
}

#[pymethods]
impl PyEulerMaruyama {
    #[new]
    #[pyo3(signature = (*, depth))]
    const fn new(depth: u8) -> Self {
        Self { depth }
    }

    fn __repr__(&self) -> String {
        format!("EulerMaruyama(depth={})", self.depth)
    }
}

/// Constant-rate independent Poisson arrivals with a total event budget.
#[pyclass(
    name = "PoissonEvents",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyPoissonEvents {
    #[pyo3(get)]
    pub(crate) max_events: usize,
}

#[pymethods]
impl PyPoissonEvents {
    #[new]
    #[pyo3(signature = (*, max_events))]
    const fn new(max_events: usize) -> Self {
        Self { max_events }
    }

    fn __repr__(&self) -> String {
        format!("PoissonEvents(max_events={})", self.max_events)
    }
}
