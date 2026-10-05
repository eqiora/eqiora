use super::*;
use crate::modeling::PyDimension;
use eqiora::{DynQuantity, solver::SolverProvider};
use eqiora_numerics::{CommonEigenPlan, CommonEigenRequest};
use std::num::NonZeroUsize;
type InputQuantity<'py> = (f64, PyRef<'py, PyDimension>);
type OutputQuantity = (f64, PyDimension);

/// Dense Hermitian mode selection with explicit provider and typed spectral controls.
#[pyclass(
    name = "HermitianEigen",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(super) struct PyHermitianEigen {
    pub(super) native: CommonEigenRequest,
    provider: SolverProvider,
}

impl PyHermitianEigen {
    pub(super) fn from_native(plan: &CommonEigenPlan) -> Self {
        Self {
            native: plan.request(),
            provider: plan.solver_provider(),
        }
    }
}

#[pymethods]
impl PyHermitianEigen {
    #[new]
    #[pyo3(signature = (*, count, provider, residual_tolerance, normalization_tolerance, target=None, interval=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        count: usize,
        provider: &solver_request::PySolverProvider,
        residual_tolerance: f64,
        normalization_tolerance: f64,
        target: Option<InputQuantity<'_>>,
        interval: Option<(InputQuantity<'_>, InputQuantity<'_>)>,
    ) -> PyResult<Self> {
        let count = NonZeroUsize::new(count)
            .ok_or_else(|| PyTypeError::new_err("mode count must be positive"))?;
        let build = || {
            let mut native =
                CommonEigenRequest::dense(count, residual_tolerance, normalization_tolerance)?;
            if let Some((value, dimension)) = target {
                native = native.with_target(DynQuantity::new(value, dimension.native()))?;
            }
            if let Some(((lo, ld), (hi, hd))) = interval {
                native = native.within_interval([
                    DynQuantity::new(lo, ld.native()),
                    DynQuantity::new(hi, hd.native()),
                ])?;
            }
            Ok::<_, eqiora::Diagnostic>(native)
        };
        Ok(Self {
            native: build().map_err(|d| validation_error(py, &[d]))?,
            provider: provider.native,
        })
    }
    #[getter]
    fn count(&self) -> usize {
        self.native.count().get()
    }
    #[getter]
    fn provider(&self) -> solver_request::PySolverProvider {
        solver_request::PySolverProvider {
            native: self.provider,
        }
    }
    #[getter]
    fn residual_tolerance(&self) -> f64 {
        self.native.residual_tolerance()
    }
    #[getter]
    fn normalization_tolerance(&self) -> f64 {
        self.native.normalization_tolerance()
    }
    #[getter]
    fn target(&self) -> Option<(f64, PyDimension)> {
        self.native
            .target()
            .map(|q| (q.value(), PyDimension { value: q.dim() }))
    }
    #[getter]
    fn interval(&self) -> Option<(OutputQuantity, OutputQuantity)> {
        self.native.interval().map(|[lo, hi]| {
            (
                (lo.value(), PyDimension { value: lo.dim() }),
                (hi.value(), PyDimension { value: hi.dim() }),
            )
        })
    }
}

pub(super) fn resolve(
    py: Python<'_>,
    model: Py<PyModel>,
    policy: &PyHermitianEigen,
) -> PyResult<PyPlan> {
    let backend = crate::execution::resolved_linear_backend(policy.provider)
        .map_err(|d| validation_error(py, &d))?;
    let native = CommonEigenPlan::resolve(model.borrow(py).artifact(), policy.native, backend)
        .map_err(|d| validation_error(py, &[d]))?;
    let native = ResolvedCommonPlan::Eigen(Box::new(native));
    let (requested_solve, solve) = solve_handles_from_native(py, &native)?;
    Ok(PyPlan {
        native,
        model,
        mesh: None,
        spatial: None,
        requested_solve,
        solve,
        temporal: None,
    })
}

/// Source roles of one complete finite Hermitian pencil.
#[pyclass(
    name = "EigenPlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(super) struct PyEigenPlanView {
    #[pyo3(get)]
    mode_field: PyModelFieldRef,
    #[pyo3(get)]
    eigenvalue_field: PyModelFieldRef,
}
pub(super) fn view(py: Python<'_>, plan: &CommonEigenPlan) -> PyResult<Py<PyAny>> {
    Py::new(
        py,
        PyEigenPlanView {
            mode_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.mode_field().ulid().to_string(),
            ),
            eigenvalue_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.eigenvalue_field().ulid().to_string(),
            ),
        },
    )
    .map(Py::into_any)
}
