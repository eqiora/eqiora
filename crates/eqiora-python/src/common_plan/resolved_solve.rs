//! Typed inspection of the effective solver selected by root resolution.

use eqiora::realization::NonlinearSolvePlan;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
    SolverProvider,
};
use pyo3::prelude::*;

use super::policy::PySolverPlanningObjective;
use super::solver_request::PySolverProvider;

#[derive(Debug, Clone)]
pub(super) struct SolverPlanningAudit {
    objective: PySolverPlanningObjective,
    policy_id: &'static str,
    candidate_id: &'static str,
    evidence_case: &'static str,
    reasons: Vec<(&'static str, &'static str)>,
}

impl SolverPlanningAudit {
    pub(super) fn new(
        objective: PySolverPlanningObjective,
        policy_id: &'static str,
        candidate_id: &'static str,
        evidence_case: &'static str,
        reasons: Vec<(&'static str, &'static str)>,
    ) -> Self {
        Self {
            objective,
            policy_id,
            candidate_id,
            evidence_case,
            reasons,
        }
    }
}

#[pyclass(
    name = "ResolvedLinear",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyResolvedLinear {
    plan: SolverPlan,
    operator: LinearOperatorProperties,
    provider: SolverProvider,
    audit: Option<SolverPlanningAudit>,
}

impl PyResolvedLinear {
    pub(super) fn new(
        plan: SolverPlan,
        operator: LinearOperatorProperties,
        provider: SolverProvider,
        audit: Option<SolverPlanningAudit>,
    ) -> Self {
        Self {
            plan,
            operator,
            provider,
            audit,
        }
    }
}

#[pymethods]
impl PyResolvedLinear {
    #[getter]
    fn algorithm(&self) -> &'static str {
        match self.plan.algorithm() {
            LinearSolver::ConjugateGradient => "conjugate-gradient",
            LinearSolver::MinimumResidual => "minimum-residual",
            LinearSolver::BiConjugateGradientStabilized => "bicgstab",
            LinearSolver::SparseLu => "sparse-lu",
        }
    }

    #[getter]
    fn preconditioner(&self) -> &'static str {
        match self.plan.preconditioner() {
            PreconditionerPolicy::Identity => "identity",
            PreconditionerPolicy::Jacobi => "jacobi",
        }
    }

    #[getter]
    fn reduction(&self) -> &'static str {
        match self.plan.reduction() {
            ReductionPolicy::Reproducible => "reproducible",
            ReductionPolicy::Fast => "fast",
        }
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.plan.relative_tolerance()
    }

    #[getter]
    fn absolute_tolerance(&self) -> f64 {
        self.plan.absolute_tolerance()
    }

    #[getter]
    fn maximum_iterations(&self) -> usize {
        self.plan.maximum_iterations().get()
    }

    #[getter]
    fn operator(&self) -> &'static str {
        match self.operator {
            LinearOperatorProperties::General => "general",
            LinearOperatorProperties::SymmetricPositiveDefinite => "symmetric-positive-definite",
            LinearOperatorProperties::SymmetricIndefinite => "symmetric-indefinite",
            LinearOperatorProperties::Symmetric => "symmetric",
            LinearOperatorProperties::ComplexSymmetric => "complex-symmetric",
            LinearOperatorProperties::Hermitian => "hermitian",
            LinearOperatorProperties::HermitianPositiveDefinite => "hermitian-positive-definite",
        }
    }

    #[getter]
    const fn backend(&self) -> &'static str {
        self.provider.id().as_str()
    }

    #[getter]
    const fn backend_version(&self) -> &'static str {
        self.provider.implementation_version()
    }

    #[getter]
    const fn provider(&self) -> PySolverProvider {
        PySolverProvider {
            native: self.provider,
        }
    }

    #[getter]
    const fn objective(&self) -> Option<PySolverPlanningObjective> {
        match &self.audit {
            Some(audit) => Some(audit.objective),
            None => None,
        }
    }

    #[getter]
    const fn planning_policy_id(&self) -> Option<&'static str> {
        match &self.audit {
            Some(audit) => Some(audit.policy_id),
            None => None,
        }
    }

    #[getter]
    const fn selected_candidate_id(&self) -> Option<&'static str> {
        match &self.audit {
            Some(audit) => Some(audit.candidate_id),
            None => None,
        }
    }

    #[getter]
    const fn selected_evidence_case(&self) -> Option<&'static str> {
        match &self.audit {
            Some(audit) => Some(audit.evidence_case),
            None => None,
        }
    }

    #[getter]
    fn planning_reasons(&self) -> Vec<(&'static str, &'static str)> {
        self.audit
            .as_ref()
            .map_or_else(Vec::new, |audit| audit.reasons.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "ResolvedLinear(algorithm={:?}, preconditioner={:?}, reduction={:?}, backend={:?})",
            self.algorithm(),
            self.preconditioner(),
            self.reduction(),
            self.backend(),
        )
    }
}

#[pyclass(
    name = "ResolvedNewton",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyResolvedNewton {
    linear: Py<PyResolvedLinear>,
    nonlinear: NonlinearSolvePlan,
}

impl PyResolvedNewton {
    pub(super) const fn new(linear: Py<PyResolvedLinear>, nonlinear: NonlinearSolvePlan) -> Self {
        Self { linear, nonlinear }
    }
}

#[pymethods]
impl PyResolvedNewton {
    #[getter]
    fn linear(&self, py: Python<'_>) -> Py<PyResolvedLinear> {
        self.linear.clone_ref(py)
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.nonlinear.relative_tolerance()
    }

    #[getter]
    fn absolute_tolerance(&self) -> f64 {
        self.nonlinear.absolute_tolerance()
    }

    #[getter]
    fn maximum_iterations(&self) -> usize {
        self.nonlinear.maximum_iterations().get()
    }

    #[getter]
    fn maximum_line_search_steps(&self) -> usize {
        self.nonlinear.maximum_line_search_steps()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResolvedNewton(linear=<ResolvedLinear>, relative_tolerance={}, absolute_tolerance={}, maximum_iterations={}, maximum_line_search_steps={})",
            self.relative_tolerance(),
            self.absolute_tolerance(),
            self.maximum_iterations(),
            self.maximum_line_search_steps(),
        )
    }
}

use super::{
    CommonSolvePolicy, PyLinear, PyNewton, RequestedSolveHandle, ResolvedCommonPlan,
    ResolvedSolveHandle, eigen,
};

pub(super) fn solve_handles_from_native(
    py: Python<'_>,
    native: &ResolvedCommonPlan,
) -> PyResult<(Option<RequestedSolveHandle>, Option<ResolvedSolveHandle>)> {
    if let Some(plan) = native.as_eigen() {
        let policy = Py::new(py, eigen::PyHermitianEigen::from_native(plan))?;
        return Ok((
            Some(RequestedSolveHandle::Eigen(policy.clone_ref(py))),
            Some(ResolvedSolveHandle::Eigen(policy)),
        ));
    }
    let Some(request) = native.canonical_solve_request() else {
        return Ok((None, None));
    };
    let requested = match request {
        CommonSolvePolicy::Linear(linear) => {
            RequestedSolveHandle::Linear(Py::new(py, PyLinear::from_native(linear))?)
        }
        CommonSolvePolicy::Newton { nonlinear, linear } => {
            let linear = Py::new(py, PyLinear::from_native(linear))?;
            RequestedSolveHandle::Newton(Py::new(py, PyNewton::from_native(linear, nonlinear))?)
        }
    };
    let solver_planning_audit = native.solver_planning_objective().map(|objective| {
        SolverPlanningAudit::new(
            objective.into(),
            native
                .solver_planning_policy_id()
                .expect("planned solver retains its policy identity"),
            native
                .selected_solver_candidate_id()
                .expect("planned solver retains its selected candidate"),
            native
                .selected_solver_evidence_case()
                .expect("planned solver retains its evidence identity"),
            native.solver_planning_reasons().to_vec(),
        )
    });
    let linear = Py::new(
        py,
        PyResolvedLinear::new(
            native
                .effective_solver()
                .expect("spatial common Plan owns an effective linear solver"),
            native
                .operator_properties()
                .expect("spatial common Plan owns operator properties"),
            native
                .linear_solver_provider()
                .expect("spatial Plan owns its exact provider"),
            solver_planning_audit,
        ),
    )?;
    let resolved = match native {
        ResolvedCommonPlan::TransientFlow(plan) => ResolvedSolveHandle::Newton(Py::new(
            py,
            PyResolvedNewton::new(linear, plan.nonlinear()),
        )?),
        ResolvedCommonPlan::Eigen(_) | ResolvedCommonPlan::Ode(_) => {
            unreachable!("ODE Plan has no common solve request")
        }
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Scalar(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_)
        | ResolvedCommonPlan::Fsi(_) => ResolvedSolveHandle::Linear(linear),
    };
    Ok((Some(requested), Some(resolved)))
}
