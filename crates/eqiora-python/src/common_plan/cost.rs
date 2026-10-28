//! Typed projections of the existing solver planner's forecast and decision evidence.
use eqiora::solver::{SolverCandidateCost, SolverCostRequest};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::num::NonZeroUsize;

#[pyclass(
    name = "SolverCost",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PySolverCost {
    pub(super) native: SolverCostRequest,
}

fn positive(value: usize) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(value)
        .ok_or_else(|| PyValueError::new_err("cost counts and limits must be positive"))
}

#[pymethods]
impl PySolverCost {
    #[new]
    #[pyo3(signature = (*, solves=None, structures=None, numeric_preparations=None,
        identity_iterations=None, jacobi_iterations=None, numeric_payload_limit=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one typed forecast owns its counts, iteration assumptions and payload limit"
    )]
    fn new(
        solves: Option<usize>,
        structures: Option<usize>,
        numeric_preparations: Option<usize>,
        identity_iterations: Option<usize>,
        jacobi_iterations: Option<usize>,
        numeric_payload_limit: Option<usize>,
    ) -> PyResult<Self> {
        let mut native = match (solves, structures, numeric_preparations) {
            (None, None, None) => SolverCostRequest::default(),
            (Some(h), Some(s), Some(f)) => {
                SolverCostRequest::new(positive(h)?, positive(s)?, positive(f)?)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?
            }
            _ => {
                return Err(PyValueError::new_err(
                    "declare solves, structures and numeric_preparations together",
                ));
            }
        };
        native = native.with_iteration_forecasts(
            identity_iterations.map(positive).transpose()?,
            jacobi_iterations.map(positive).transpose()?,
        );
        if let Some(limit) = numeric_payload_limit {
            native = native.with_numeric_payload_limit(positive(limit)?);
        }
        Ok(Self { native })
    }
    #[getter]
    fn solves(&self) -> usize {
        self.native.solves().get()
    }
    #[getter]
    fn structures(&self) -> usize {
        self.native.structures().get()
    }
    #[getter]
    fn numeric_preparations(&self) -> usize {
        self.native.numeric_preparations().get()
    }
    #[getter]
    fn identity_iterations(&self) -> Option<usize> {
        self.native.identity_iterations().map(NonZeroUsize::get)
    }
    #[getter]
    fn jacobi_iterations(&self) -> Option<usize> {
        self.native.jacobi_iterations().map(NonZeroUsize::get)
    }
    #[getter]
    fn numeric_payload_limit(&self) -> Option<usize> {
        self.native.numeric_payload_limit().map(NonZeroUsize::get)
    }
    #[getter]
    fn declared_horizon(&self) -> bool {
        self.native.declared_horizon()
    }
    fn __repr__(&self) -> String {
        format!(
            "SolverCost(solves={}, numeric_payload_limit={:?})",
            self.solves(),
            self.numeric_payload_limit()
        )
    }
}

#[pyclass(
    name = "SolverCandidateCost",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PySolverCandidateCost {
    pub(super) native: SolverCandidateCost,
}

#[pymethods]
impl PySolverCandidateCost {
    #[getter]
    fn run_work(&self) -> Option<u128> {
        self.native.run_work()
    }
    #[getter]
    fn retained_numeric_bytes(&self) -> Option<usize> {
        self.native.retained_numeric_bytes()
    }
    #[getter]
    fn candidate(&self) -> &'static str {
        self.native.candidate()
    }
    #[getter]
    fn setup_work(&self) -> Option<u128> {
        self.native.setup_work()
    }
    #[getter]
    fn solve_work(&self) -> Option<u128> {
        self.native.solve_work()
    }
    #[getter]
    fn acceptance_work(&self) -> Option<u128> {
        self.native.acceptance_work()
    }
    #[getter]
    fn total_work(&self) -> Option<u128> {
        self.native.total_work()
    }
    #[getter]
    fn numeric_payload_bytes(&self) -> Option<usize> {
        self.native.numeric_payload_bytes()
    }
    fn __repr__(&self) -> String {
        format!(
            "SolverCandidateCost(candidate={:?}, work={:?}, numeric_payload_bytes={:?})",
            self.candidate(),
            self.total_work(),
            self.numeric_payload_bytes()
        )
    }
}
