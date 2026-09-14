//! Closed Python numerical-policy requests consumed by the root resolver.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use super::solver_request::{PyLinearSolver, PyPreconditioner, PyReduction, PySolverProvider};
use eqiora::realization::NonlinearSolvePlan;
use eqiora::solver::{SolverPlan, SolverPlanningObjective};
use eqiora::{Id, kinds};
use eqiora_numerics::{
    CommonBackwardEuler, CommonLinearRequest, CommonPressureGauge2d, CommonTsitouras45,
    CommonTsitourasTolerance,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt};
use ulid::Ulid;

use crate::error::validation_error;
use crate::model::{PyModelDomainRef, PyModelFieldRef};

/// Continuous tensor-product Q1 Galerkin spatial policy.
#[pyclass(
    name = "Q1",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyQ1;

#[pymethods]
impl PyQ1 {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "q1"
    }

    #[getter]
    const fn space(&self) -> &'static str {
        "continuous-lagrange-q1"
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "gauss-legendre-2-per-axis"
    }

    fn __repr__(&self) -> &'static str {
        "Q1()"
    }
}

/// Mixed MINI velocity and continuous P1 pressure spatial policy.
#[pyclass(
    name = "MiniP1",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyMiniP1;

#[pymethods]
impl PyMiniP1 {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "mini-p1"
    }

    #[getter]
    const fn velocity_space(&self) -> &'static str {
        "simplex-p1-bubble"
    }

    #[getter]
    const fn pressure_space(&self) -> &'static str {
        "continuous-lagrange-p1"
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "triangle-duffy-gauss-legendre-3-per-axis"
    }

    fn at(&self, domain: &PyModelDomainRef) -> PyScopedSpatialBinding {
        PyScopedSpatialBinding {
            domain: domain.clone(),
            policy: ScopedSpatialKind::MiniP1,
        }
    }

    fn __repr__(&self) -> &'static str {
        "MiniP1()"
    }
}

/// Continuous simplex P1 Galerkin spatial policy.
#[pyclass(
    name = "P1",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyP1;

#[pymethods]
impl PyP1 {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "p1"
    }

    #[getter]
    const fn space(&self) -> &'static str {
        "continuous-lagrange-p1"
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "triangle-duffy-gauss-legendre-4-per-axis"
    }

    fn at(&self, domain: &PyModelDomainRef) -> PyScopedSpatialBinding {
        PyScopedSpatialBinding {
            domain: domain.clone(),
            policy: ScopedSpatialKind::P1,
        }
    }

    fn __repr__(&self) -> &'static str {
        "P1()"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ScopedSpatialKind {
    MiniP1,
    P1,
}

/// One exact Model-bound spatial policy assignment.
#[pyclass(
    name = "ScopedSpatialPolicy",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyScopedSpatialBinding {
    pub(super) domain: PyModelDomainRef,
    pub(super) policy: ScopedSpatialKind,
}

#[pymethods]
impl PyScopedSpatialBinding {
    #[getter]
    fn domain(&self) -> PyModelDomainRef {
        self.domain.clone()
    }

    #[getter]
    fn method(&self) -> &'static str {
        match self.policy {
            ScopedSpatialKind::MiniP1 => "mini-p1",
            ScopedSpatialKind::P1 => "p1",
        }
    }

    #[getter]
    fn spaces(&self) -> (&'static str, Option<&'static str>) {
        match self.policy {
            ScopedSpatialKind::MiniP1 => ("simplex-p1-bubble", Some("continuous-lagrange-p1")),
            ScopedSpatialKind::P1 => ("continuous-lagrange-p1", None),
        }
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "triangle-duffy-gauss-legendre-4-per-axis"
    }

    fn __repr__(&self) -> String {
        format!(
            "ScopedSpatialPolicy(method={:?}, domain={:?})",
            self.method(),
            self.domain.exact_id()
        )
    }
}

/// Cell-centred orthogonal two-point-flux finite-volume spatial policy.
#[pyclass(
    name = "CellCenteredTpfa",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyCellCenteredTpfa;

#[pymethods]
impl PyCellCenteredTpfa {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "cell-centered-tpfa"
    }

    #[getter]
    const fn space(&self) -> &'static str {
        "cell-constant"
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "cell-centroid/facet-midpoint"
    }

    fn __repr__(&self) -> &'static str {
        "CellCenteredTpfa()"
    }
}

/// Collocated cell-centred incompressible-flow spatial policy.
#[pyclass(
    name = "CellCentered",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyCellCentered;

#[pymethods]
impl PyCellCentered {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[getter]
    const fn method(&self) -> &'static str {
        "cell-centered"
    }

    #[getter]
    const fn velocity_space(&self) -> &'static str {
        "cell-constant"
    }

    #[getter]
    const fn pressure_space(&self) -> &'static str {
        "cell-constant"
    }

    #[getter]
    const fn quadrature(&self) -> &'static str {
        "cell-centroid/facet-midpoint"
    }

    fn __repr__(&self) -> &'static str {
        "CellCentered()"
    }
}

/// Program-owned preference used by the versioned host-serial solver planner.
#[pyclass(
    name = "SolverPlanningObjective",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PySolverPlanningObjective {
    Robust,
    Fast,
    LowMemory,
}

impl From<PySolverPlanningObjective> for SolverPlanningObjective {
    fn from(value: PySolverPlanningObjective) -> Self {
        match value {
            PySolverPlanningObjective::Robust => Self::Robust,
            PySolverPlanningObjective::Fast => Self::Fast,
            PySolverPlanningObjective::LowMemory => Self::LowMemory,
        }
    }
}

impl From<SolverPlanningObjective> for PySolverPlanningObjective {
    fn from(value: SolverPlanningObjective) -> Self {
        match value {
            SolverPlanningObjective::Robust => Self::Robust,
            SolverPlanningObjective::Fast => Self::Fast,
            SolverPlanningObjective::LowMemory => Self::LowMemory,
        }
    }
}

/// Closed linear-solve controls resolved against Model-owned operator meaning.
#[pyclass(
    name = "Linear",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PyLinear {
    pub(super) native: CommonLinearRequest,
}

impl PyLinear {
    pub(super) const fn from_native(native: CommonLinearRequest) -> Self {
        Self { native }
    }
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (*, relative_tolerance, absolute_tolerance, maximum_iterations, objective=None, algorithm=None, preconditioner=None, reduction=None, provider=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one explicit exact tuple or ranked objective accompanies common controls"
    )]
    fn new(
        py: Python<'_>,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: usize,
        objective: Option<PySolverPlanningObjective>,
        algorithm: Option<PyLinearSolver>,
        preconditioner: Option<PyPreconditioner>,
        reduction: Option<PyReduction>,
        provider: Option<PySolverProvider>,
    ) -> PyResult<Self> {
        let maximum_iterations = NonZeroUsize::new(maximum_iterations)
            .ok_or_else(|| PyTypeError::new_err("maximum_iterations must be a positive integer"))?;
        let native = match (objective, algorithm, preconditioner, reduction, provider) {
            (Some(objective), None, None, None, None) => CommonLinearRequest::program_controlled(
                relative_tolerance, absolute_tolerance, maximum_iterations, objective.into()),
            (None, Some(algorithm), Some(preconditioner), Some(reduction), Some(provider)) => {
                SolverPlan::new(algorithm.into(), relative_tolerance, absolute_tolerance, maximum_iterations)
                    .and_then(|plan| CommonLinearRequest::exact(plan.with_preconditioner(preconditioner.into())
                        .with_reduction(reduction.into()), provider.native))
            }
            _ => return Err(PyTypeError::new_err("Linear requires either objective or the complete exact algorithm, preconditioner, reduction, provider tuple; mixed and incomplete intent is invalid")),
        }.map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self { native })
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.native.relative_tolerance()
    }
    #[getter]
    fn absolute_tolerance(&self) -> f64 {
        self.native.absolute_tolerance()
    }
    #[getter]
    fn maximum_iterations(&self) -> usize {
        self.native.maximum_iterations().get()
    }
    #[getter]
    fn objective(&self) -> Option<PySolverPlanningObjective> {
        self.native.objective().map(Into::into)
    }
    #[getter]
    fn algorithm(&self) -> Option<PyLinearSolver> {
        self.native
            .exact_request()
            .map(|(p, _)| p.algorithm().into())
    }
    #[getter]
    fn preconditioner(&self) -> Option<PyPreconditioner> {
        self.native
            .exact_request()
            .map(|(p, _)| p.preconditioner().into())
    }
    #[getter]
    fn reduction(&self) -> Option<PyReduction> {
        self.native
            .exact_request()
            .map(|(p, _)| p.reduction().into())
    }
    #[getter]
    fn provider(&self) -> Option<PySolverProvider> {
        self.native
            .exact_request()
            .map(|(_, native)| PySolverProvider { native })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self == &*other)
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.relative_tolerance().to_bits().hash(&mut hasher);
        self.absolute_tolerance().to_bits().hash(&mut hasher);
        self.maximum_iterations().hash(&mut hasher);
        self.objective().hash(&mut hasher);
        self.algorithm().hash(&mut hasher);
        self.preconditioner().hash(&mut hasher);
        self.reduction().hash(&mut hasher);
        if let Some((_, provider)) = self.native.exact_request() {
            provider.id().as_str().hash(&mut hasher);
            provider.implementation_version().hash(&mut hasher);
            provider.libraries().hash(&mut hasher);
        }
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!("Linear({:?})", self.native)
    }
}

/// Backward-Euler temporal operator policy.
#[pyclass(
    name = "BackwardEuler",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PyBackwardEuler {
    pub(super) native: CommonBackwardEuler,
}

impl PyBackwardEuler {
    pub(super) const fn from_native(native: CommonBackwardEuler) -> Self {
        Self { native }
    }
}

#[pymethods]
impl PyBackwardEuler {
    #[new]
    #[pyo3(signature = (step_s))]
    fn new(py: Python<'_>, step_s: &Bound<'_, PyAny>) -> PyResult<Self> {
        let step_s = step_s
            .cast::<PyFloat>()
            .map_err(|_| PyTypeError::new_err("step_s must be a float"))?
            .value();
        CommonBackwardEuler::from_seconds(step_s)
            .map(|native| Self { native })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[getter]
    fn step_s(&self) -> f64 {
        self.native.step().value()
    }

    fn __repr__(&self) -> String {
        format!("BackwardEuler(step_s={})", self.step_s())
    }
}

/// Adaptive Tsitouras 5(4) policy with exact Field-bound SI tolerances.
#[pyclass(
    name = "Tsitouras45",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyTsitouras45 {
    pub(super) native: CommonTsitouras45,
    fields: Vec<Py<PyModelFieldRef>>,
    events: Option<super::event_policy::PyEventPolicy>,
    forward_sensitivities: Option<super::forward_policy::PyForwardSensitivity>,
}

impl PyTsitouras45 {
    pub(super) fn from_native(
        py: Python<'_>,
        model_digest: &str,
        native: CommonTsitouras45,
        document: &eqiora::api::ModelDocument,
    ) -> PyResult<Self> {
        let fields = native
            .absolute_tolerances()
            .iter()
            .map(|entry| {
                Py::new(
                    py,
                    PyModelFieldRef::from_exact(
                        model_digest.to_owned(),
                        entry.field().ulid().to_string(),
                    ),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let events = native
            .events()
            .map(|policy| super::event_policy::PyEventPolicy {
                model_digest: model_digest.to_owned(),
                max_events: policy.max_events(),
                entries: policy
                    .guard_tolerances()
                    .iter()
                    .map(|entry| super::event_policy::PyGuardTolerance {
                        activation: crate::model::PyActivationRef {
                            model_digest: model_digest.to_owned(),
                            id: entry.activation(),
                        },
                        quantity: entry.quantity(),
                    })
                    .collect(),
            });
        let forward_sensitivities = native
            .forward_sensitivities()
            .map(|policy| {
                super::forward_policy::PyForwardSensitivity::from_native(
                    py,
                    document,
                    policy.relative_tolerance(),
                    policy
                        .absolute_tolerances()
                        .iter()
                        .map(|entry| (entry.field(), entry.parameter(), entry.quantity()))
                        .collect(),
                )
            })
            .transpose()?;
        Ok(Self {
            native,
            fields,
            events,
            forward_sensitivities,
        })
    }

    pub(super) fn belongs_to_model(&self, py: Python<'_>, model_digest: &str) -> bool {
        self.forward_sensitivities
            .as_ref()
            .is_none_or(|policy| policy.model_digest == model_digest)
            && self
                .events
                .as_ref()
                .is_none_or(|events| events.model_digest == model_digest)
            && self
                .fields
                .iter()
                .all(|field| field.borrow(py).exact_model_digest() == model_digest)
    }
}

#[pymethods]
impl PyTsitouras45 {
    #[new]
    #[pyo3(signature = (*, initial_step_s, relative_tolerance, absolute_tolerances, events=None, forward_sensitivities=None))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = exact_time_float)] initial_step_s: f64,
        #[pyo3(from_py_with = exact_time_float)] relative_tolerance: f64,
        absolute_tolerances: &Bound<'_, PyDict>,
        events: Option<&super::event_policy::PyEventPolicy>,
        forward_sensitivities: Option<&super::forward_policy::PyForwardSensitivity>,
    ) -> PyResult<Self> {
        let mut native = Vec::with_capacity(absolute_tolerances.len());
        let mut fields = Vec::with_capacity(absolute_tolerances.len());
        for (field, value) in absolute_tolerances.iter() {
            let field = field.extract::<Py<PyModelFieldRef>>().map_err(|_| {
                PyTypeError::new_err(
                    "absolute_tolerances keys must be exact eqiora.FieldRef values",
                )
            })?;
            let value = exact_time_float(&value)?;
            let id = Ulid::from_string(field.borrow(py).exact_id()).map_err(|_| {
                PyTypeError::new_err("absolute_tolerances contains an invalid exact FieldRef")
            })?;
            native.push(
                CommonTsitourasTolerance::new(Id::<kinds::Field>::from_ulid(id), value)
                    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?,
            );
            fields.push(field);
        }
        fields.sort_by_key(|field| field.borrow(py).exact_id().to_owned());
        let mut native = CommonTsitouras45::new(initial_step_s, relative_tolerance, native)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        if let Some(events) = events {
            native = native
                .with_events(
                    events.max_events,
                    events
                        .entries
                        .iter()
                        .map(|entry| (entry.activation.id, entry.quantity))
                        .collect(),
                )
                .map_err(|error| validation_error(py, &[error]))?;
        }
        if let Some(policy) = forward_sensitivities {
            native = native
                .with_forward_sensitivities(
                    policy.relative_tolerance,
                    policy
                        .entries
                        .iter()
                        .map(|entry| {
                            let field = Ulid::from_string(entry.field.exact_id())
                                .map(Id::<kinds::Field>::from_ulid)
                                .expect("validated exact FieldRef");
                            (field, entry.parameter.value.id(), entry.quantity)
                        })
                        .collect(),
                )
                .map_err(|error| validation_error(py, &[error]))?;
        }
        Ok(Self {
            native,
            fields,
            events: events.cloned(),
            forward_sensitivities: forward_sensitivities.cloned(),
        })
    }

    #[getter]
    fn forward_sensitivities(&self) -> Option<super::forward_policy::PyForwardSensitivity> {
        self.forward_sensitivities.clone()
    }

    #[getter]
    fn events(&self) -> Option<super::event_policy::PyEventPolicy> {
        self.events.clone()
    }

    #[getter]
    fn initial_step_s(&self) -> f64 {
        self.native.initial_step_s()
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.native.relative_tolerance()
    }

    #[getter]
    fn absolute_tolerances(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        for (field, tolerance) in self.fields.iter().zip(self.native.absolute_tolerances()) {
            result.set_item(field.clone_ref(py), tolerance.value())?;
        }
        Ok(result.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tsitouras45(initial_step_s={}, relative_tolerance={}, fields={})",
            self.initial_step_s(),
            self.relative_tolerance(),
            self.fields.len()
        )
    }
}

/// Bounded Newton policy owning exact nested linear controls.
#[pyclass(
    name = "Newton",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyNewton {
    pub(super) linear: Py<PyLinear>,
    pub(super) native: NonlinearSolvePlan,
}

impl PyNewton {
    pub(super) fn from_native(linear: Py<PyLinear>, native: NonlinearSolvePlan) -> Self {
        Self { linear, native }
    }
}

#[pymethods]
impl PyNewton {
    #[new]
    #[pyo3(signature = (*, linear, relative_tolerance=1.0e-9, absolute_tolerance=1.0e-11, maximum_iterations=16, maximum_line_search_steps=12))]
    fn new(
        py: Python<'_>,
        linear: Py<PyLinear>,
        #[pyo3(from_py_with = exact_float_extract)] relative_tolerance: f64,
        #[pyo3(from_py_with = exact_float_extract)] absolute_tolerance: f64,
        #[pyo3(from_py_with = exact_usize_extract)] maximum_iterations: usize,
        #[pyo3(from_py_with = exact_usize_extract)] maximum_line_search_steps: usize,
    ) -> PyResult<Self> {
        let maximum_iterations = NonZeroUsize::new(maximum_iterations)
            .ok_or_else(|| PyTypeError::new_err("maximum_iterations must be positive"))?;
        NonlinearSolvePlan::new(
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
            maximum_line_search_steps,
        )
        .map(|native| Self { linear, native })
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[getter]
    fn linear(&self, py: Python<'_>) -> Py<PyLinear> {
        self.linear.clone_ref(py)
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.native.relative_tolerance()
    }

    #[getter]
    fn absolute_tolerance(&self) -> f64 {
        self.native.absolute_tolerance()
    }

    #[getter]
    fn maximum_iterations(&self) -> usize {
        self.native.maximum_iterations().get()
    }

    #[getter]
    fn maximum_line_search_steps(&self) -> usize {
        self.native.maximum_line_search_steps()
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<PyRef<'_, Self>>().is_ok_and(|other| {
            self.native == other.native && *self.linear.borrow(py) == *other.linear.borrow(py)
        })
    }

    fn __hash__(&self, py: Python<'_>) -> isize {
        let mut hasher = DefaultHasher::new();
        self.relative_tolerance().to_bits().hash(&mut hasher);
        self.absolute_tolerance().to_bits().hash(&mut hasher);
        self.maximum_iterations().hash(&mut hasher);
        self.maximum_line_search_steps().hash(&mut hasher);
        self.linear.borrow(py).__hash__().hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!(
            "Newton(linear=<Linear>, relative_tolerance={}, absolute_tolerance={}, maximum_iterations={}, maximum_line_search_steps={})",
            self.relative_tolerance(),
            self.absolute_tolerance(),
            self.maximum_iterations(),
            self.maximum_line_search_steps()
        )
    }
}

/// Closed pressure representative selected by transient resolution.
#[pyclass(
    name = "PressureGauge2d",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyPressureGauge2d {
    ZeroIntegral,
    BoundaryTraction,
}

impl From<CommonPressureGauge2d> for PyPressureGauge2d {
    fn from(value: CommonPressureGauge2d) -> Self {
        match value {
            CommonPressureGauge2d::ZeroIntegral => Self::ZeroIntegral,
            CommonPressureGauge2d::BoundaryTraction => Self::BoundaryTraction,
        }
    }
}

fn exact_float_extract(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(value
        .cast::<PyFloat>()
        .map_err(|_| PyTypeError::new_err("nonlinear tolerance must be a float"))?
        .value())
}

fn exact_usize_extract(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "nonlinear iteration bound must be an integer",
        ));
    }
    value
        .cast::<PyInt>()
        .map_err(|_| PyTypeError::new_err("nonlinear iteration bound must be an integer"))?
        .extract::<usize>()
        .map_err(|_| {
            PyTypeError::new_err("nonlinear iteration bound must be a non-negative integer")
        })
}

fn exact_time_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value
        .cast::<PyFloat>()
        .map(|value| value.value())
        .map_err(|_| PyTypeError::new_err("time integration values must be floats"))
}
