use eqiora_numerics::{
    CommonElasticityPlan, CommonFsiRunRequest, CommonOdeRunRequest, CommonScalarPlan,
    CommonSteadyStokesPlan, CommonTransientRunRequest,
};
use pyo3::prelude::*;

use crate::trajectory::PyState;

/// Monotone public state of one native execution occurrence.
#[pyclass(
    name = "RunStatus",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyRunStatus {
    Created,
    Validating,
    Queued,
    Running,
    Cancelling,
    Cancelled,
    Completed,
    Failed,
}

impl PyRunStatus {
    pub(super) const fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Completed | Self::Failed)
    }
}

/// Last fully accepted common transient step boundary.
#[pyclass(
    name = "TransientRunProgress",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyCommonTransientRunProgress {
    pub(crate) accepted_steps: usize,
    pub(crate) maximum_steps: usize,
    pub(crate) model_time_bits: u64,
}

#[pymethods]
impl PyCommonTransientRunProgress {
    #[getter]
    const fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }
    #[getter]
    const fn maximum_steps(&self) -> usize {
        self.maximum_steps
    }
    #[getter]
    fn model_time_s(&self) -> f64 {
        f64::from_bits(self.model_time_bits)
    }
    fn __repr__(&self) -> String {
        format!(
            "TransientRunProgress(accepted_steps={}, maximum_steps={}, model_time_s={:?})",
            self.accepted_steps,
            self.maximum_steps,
            self.model_time_s(),
        )
    }
}

/// Exact accepted common transient boundary where cancellation terminated.
#[pyclass(
    name = "TransientRunCancellation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyCommonTransientRunCancellation {
    pub(crate) progress: PyCommonTransientRunProgress,
    pub(crate) request_identity: String,
    pub(crate) state: Py<PyState>,
}

#[pymethods]
impl PyCommonTransientRunCancellation {
    #[getter]
    fn progress(&self) -> PyCommonTransientRunProgress {
        self.progress.clone()
    }
    #[getter]
    fn request_identity(&self) -> &str {
        &self.request_identity
    }
    #[getter]
    fn state(&self, py: Python<'_>) -> Py<PyState> {
        self.state.clone_ref(py)
    }
    fn __repr__(&self) -> String {
        format!(
            "TransientRunCancellation(progress=TransientRunProgress(accepted_steps={}, maximum_steps={}, model_time_s={:?}), request_identity={:?})",
            self.progress.accepted_steps,
            self.progress.maximum_steps,
            self.progress.model_time_s(),
            self.request_identity,
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RunIdentity {
    model_id: String,
    model_digest: String,
    model_revision: u64,
    plan_key: String,
    adapter: &'static str,
    adapter_version: &'static str,
}

impl RunIdentity {
    pub(crate) fn from_common_algebraic(plan: &eqiora_numerics::CommonAlgebraicPlan) -> Self {
        Self::from_static(
            plan.model_id(),
            plan.model_digest(),
            plan.model_revision(),
            plan.identity(),
        )
    }
    pub(crate) fn from_common_result(result: &eqiora_numerics::CommonResult) -> Option<Self> {
        if let Some(trajectory) = result.trajectory() {
            return Some(match trajectory {
                eqiora_numerics::CommonTrajectory::Ode { request, .. } => {
                    Self::from_common_ode(request)
                }
                eqiora_numerics::CommonTrajectory::TransientFlow { request, .. } => {
                    Self::from_common_transient(request)
                }
                eqiora_numerics::CommonTrajectory::Fsi { request, .. } => {
                    Self::from_common_fsi(request)
                }
            });
        }
        Some(match result.plan() {
            eqiora_numerics::ResolvedCommonPlan::Algebraic(plan) => {
                Self::from_common_algebraic(plan)
            }
            eqiora_numerics::ResolvedCommonPlan::Scalar(plan) => Self::from_common_plan(plan),
            eqiora_numerics::ResolvedCommonPlan::Elasticity(plan) => {
                Self::from_common_elasticity(plan)
            }
            eqiora_numerics::ResolvedCommonPlan::SteadyStokes(plan) => {
                Self::from_common_steady_stokes(plan)
            }
            eqiora_numerics::ResolvedCommonPlan::Ode(_)
            | eqiora_numerics::ResolvedCommonPlan::TransientFlow(_)
            | eqiora_numerics::ResolvedCommonPlan::Fsi(_) => return None,
        })
    }

    pub(crate) fn from_common_ode(request: &CommonOdeRunRequest) -> Self {
        let plan = request.plan();
        Self {
            model_id: plan.model_id().to_owned(),
            model_digest: plan.model_digest().to_owned(),
            model_revision: plan.model_revision(),
            plan_key: request.identity().to_owned(),
            adapter: plan.backend().id(),
            adapter_version: plan.backend().version(),
        }
    }

    pub(crate) fn from_common_plan(plan: &CommonScalarPlan) -> Self {
        Self::from_static(
            plan.model_id(),
            plan.model_digest(),
            plan.model_revision(),
            plan.identity(),
        )
    }

    pub(crate) fn from_common_elasticity(plan: &CommonElasticityPlan) -> Self {
        Self::from_static(
            plan.model_id(),
            plan.model_digest(),
            plan.model_revision(),
            plan.identity(),
        )
    }

    pub(crate) fn from_common_steady_stokes(plan: &CommonSteadyStokesPlan) -> Self {
        Self::from_static(
            plan.model_id(),
            plan.model_digest(),
            plan.model_revision(),
            plan.identity(),
        )
    }

    fn from_static(
        model_id: &str,
        model_digest: &str,
        model_revision: u64,
        plan_key: &str,
    ) -> Self {
        Self {
            model_id: model_id.to_owned(),
            model_digest: model_digest.to_owned(),
            model_revision,
            plan_key: plan_key.to_owned(),
            adapter: eqiora::solver::SERIAL_EXECUTION_PROVIDER.id().as_str(),
            adapter_version: eqiora::solver::SERIAL_EXECUTION_PROVIDER.implementation_version(),
        }
    }

    pub(crate) fn from_common_transient(request: &CommonTransientRunRequest) -> Self {
        let plan = request.plan();
        Self {
            model_id: plan.model_id().to_owned(),
            model_digest: plan.model_digest().to_owned(),
            model_revision: plan.model_revision(),
            plan_key: request.identity().to_owned(),
            adapter: eqiora::solver::SERIAL_EXECUTION_PROVIDER.id().as_str(),
            adapter_version: eqiora::solver::SERIAL_EXECUTION_PROVIDER.implementation_version(),
        }
    }

    pub(crate) fn from_common_fsi(request: &CommonFsiRunRequest) -> Self {
        let plan = request.plan();
        Self {
            model_id: plan.model_id().to_owned(),
            model_digest: plan.model_digest().to_owned(),
            model_revision: plan.model_revision(),
            plan_key: request.identity().to_owned(),
            adapter: plan.execution_provider().id().as_str(),
            adapter_version: plan.execution_provider().implementation_version(),
        }
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.model_id
    }
    pub(crate) fn model_digest(&self) -> &str {
        &self.model_digest
    }
    pub(crate) const fn model_revision(&self) -> u64 {
        self.model_revision
    }
    pub(crate) fn plan_key(&self) -> &str {
        &self.plan_key
    }
    pub(crate) fn adapter(&self) -> &'static str {
        self.adapter
    }
    pub(crate) fn adapter_version(&self) -> &'static str {
        self.adapter_version
    }
}
