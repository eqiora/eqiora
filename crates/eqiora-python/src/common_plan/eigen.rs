use super::*;
use crate::modeling::{PyDimension, PyValueType};
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
    fn __repr__(&self) -> String {
        format!(
            "HermitianEigen(count={}, provider={:?}, target={:?}, interval={:?}, residual_tolerance={}, normalization_tolerance={})",
            self.native.count(),
            self.provider.id().as_str(),
            self.native.target(),
            self.native.interval(),
            self.native.residual_tolerance(),
            self.native.normalization_tolerance()
        )
    }

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
    let model_ref = model.borrow(py);
    let native = CommonEigenPlan::resolve(
        model_ref.artifact(),
        policy.native,
        backend,
        model_ref
            .authored_formulation_projection()
            .map_err(|d| validation_error(py, &[d]))?,
    )
    .map_err(|d| validation_error(py, &[d]))?;
    drop(model_ref);
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

/// One exact source equality `target = mapping * coordinate`.
#[pyclass(
    name = "EigenCoordinateMap",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PyEigenCoordinateMap {
    #[pyo3(get)]
    relation_id: String,
    #[pyo3(get)]
    target_field: PyModelFieldRef,
    #[pyo3(get)]
    coordinate_field: PyModelFieldRef,
    mapping: eqiora::ValueLiteral,
}

#[pymethods]
impl PyEigenCoordinateMap {
    fn __repr__(&self) -> String {
        format!(
            "EigenCoordinateMap(relation_id={:?}, target_field={:?}, coordinate_field={:?})",
            self.relation_id,
            self.target_field.exact_id(),
            self.coordinate_field.exact_id()
        )
    }
    #[getter]
    fn mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.mapping)
    }
    #[getter]
    fn mapping_type(&self) -> PyValueType {
        PyValueType {
            value: self.mapping.value_type().clone(),
        }
    }
}

/// Source roles and coordinate equalities of one finite Hermitian pencil.
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
    embeddings: Vec<PyEigenCoordinateMap>,
    #[pyo3(get)]
    excluded_space: Option<PyEigenExclusion>,
}
pub(super) fn view(py: Python<'_>, plan: &CommonEigenPlan) -> PyResult<Py<PyAny>> {
    Py::new(
        py,
        PyEigenPlanView {
            excluded_space: plan.excluded_space().map(
                |(projector, dimension, operator_defect, metric_defect)| PyEigenExclusion {
                    projector: projector.clone(),
                    dimension,
                    operator_defect,
                    metric_defect,
                    tolerance: plan.request().residual_tolerance(),
                },
            ),
            mode_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.mode_field().ulid().to_string(),
            ),
            eigenvalue_field: PyModelFieldRef::from_exact(
                plan.model_digest().to_owned(),
                plan.eigenvalue_field().ulid().to_string(),
            ),
            embeddings: plan
                .coordinate_embeddings()
                .map(|(relation, target, coordinate, map)| PyEigenCoordinateMap {
                    relation_id: relation.ulid().to_string(),
                    target_field: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        target.ulid().to_string(),
                    ),
                    coordinate_field: PyModelFieldRef::from_exact(
                        plan.model_digest().to_owned(),
                        coordinate.ulid().to_string(),
                    ),
                    mapping: map.clone(),
                })
                .collect(),
        },
    )
    .map(Py::into_any)
}

#[pymethods]
impl PyEigenPlanView {
    #[getter]
    fn coordinate_embeddings(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(
            py,
            self.embeddings
                .iter()
                .map(|value| Py::new(py, value.clone()))
                .collect::<PyResult<Vec<_>>>()?,
        )
        .map(Bound::unbind)
    }
    fn __repr__(&self) -> String {
        format!(
            "EigenPlanView(mode_field={:?}, eigenvalue_field={:?})",
            self.mode_field.exact_id(),
            self.eigenvalue_field.exact_id()
        )
    }
}

/// Excluded source directions and separately verified numerical nullspaces.
#[pyclass(
    name = "EigenExclusion",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PyEigenExclusion {
    projector: eqiora::ValueLiteral,
    dimension: usize,
    operator_defect: f64,
    metric_defect: f64,
    tolerance: f64,
}

#[pymethods]
impl PyEigenExclusion {
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }
    #[getter]
    fn projector(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.projector)
    }
    #[getter]
    fn projector_type(&self) -> PyValueType {
        PyValueType {
            value: self.projector.value_type().clone(),
        }
    }
    #[getter]
    fn operator_defect(&self) -> f64 {
        self.operator_defect
    }
    #[getter]
    fn metric_defect(&self) -> f64 {
        self.metric_defect
    }
    #[getter]
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
    #[getter]
    fn is_operator_null(&self) -> bool {
        self.operator_defect <= self.tolerance
    }
    #[getter]
    fn is_metric_null(&self) -> bool {
        self.metric_defect <= self.tolerance
    }
    fn __repr__(&self) -> String {
        format!(
            "EigenExclusion(dimension={}, operator_defect={:?}, metric_defect={:?}, tolerance={:?})",
            self.dimension, self.operator_defect, self.metric_defect, self.tolerance
        )
    }
}
