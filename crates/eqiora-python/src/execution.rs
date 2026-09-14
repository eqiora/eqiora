//! Bounded Python lifecycle over shared native worker execution.

mod evidence;
mod worker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;

use eqiora::diagnostic::codes;
use eqiora::{Diagnostic, GraphPath};
use eqiora_numerics::{
    CommonFsiRunRequest, CommonOdeRunRequest, CommonTransientRunRequest, ResolvedCommonPlan,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::common_plan::PyPlan;
use crate::error::{
    cancellation_error, catch_native_panic, execution_error, internal_diagnostic_error,
    internal_error, panic_boundary, validation_error,
};
use crate::trajectory::PyState;

pub(crate) use evidence::RunIdentity;
use evidence::{PyCommonTransientRunCancellation, PyCommonTransientRunProgress, PyRunStatus};
use worker::{NativeRunJob, run_worker};

const RESULT_MATERIALIZATION_FAILURE: &str =
    "the completed native Result could not be materialized";

#[derive(Debug, Clone)]
enum RunFailure {
    Execution(Vec<Diagnostic>),
    Internal(Vec<Diagnostic>),
}

#[derive(Debug, Clone)]
enum NativeRunProgress {
    CommonTransient(PyCommonTransientRunProgress),
}

impl NativeRunProgress {
    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::CommonTransient(progress) => Py::new(py, progress).map(Py::into_any),
        }
    }
}

#[derive(Debug)]
enum NativeRunOutput {
    Result(
        Box<eqiora_numerics::CommonResult>,
        Option<crate::profile::ProfileData>,
    ),
}

impl NativeRunOutput {
    fn attach_profile(&mut self, profile: crate::profile::ProfileData) {
        let Self::Result(_, slot) = self;
        *slot = Some(profile);
    }
}

enum ResultMaterializationContext {
    CommonPlan { plan: Py<PyPlan> },
}

#[derive(Debug, Clone)]
enum NativeRunCancellation {
    CommonTransient {
        accepted_steps: usize,
        maximum_steps: usize,
        model_time_s: f64,
        request_identity: String,
        state: eqiora_numerics::CommonState,
    },
}

impl NativeRunCancellation {
    fn into_python(self, py: Python<'_>, plan: &PyPlan) -> PyResult<Py<PyAny>> {
        match self {
            Self::CommonTransient {
                accepted_steps,
                maximum_steps,
                model_time_s,
                request_identity,
                state,
            } => Py::new(
                py,
                PyCommonTransientRunCancellation {
                    progress: PyCommonTransientRunProgress {
                        accepted_steps,
                        maximum_steps,
                        model_time_bits: model_time_s.to_bits(),
                    },
                    state: Py::new(
                        py,
                        crate::trajectory::PyState::from_common_cancellation(
                            py,
                            plan,
                            state,
                            accepted_steps as u64,
                            &request_identity,
                        )?,
                    )?,
                    request_identity,
                },
            )
            .map(Py::into_any),
        }
    }

    fn diagnostic(&self) -> Diagnostic {
        let message = match self {
            Self::CommonTransient {
                accepted_steps,
                maximum_steps: _,
                model_time_s,
                request_identity,
                state: _,
            } => format!(
                "common transient execution {request_identity} was cancelled at accepted model time() {model_time_s} after {accepted_steps} accepted steps"
            ),
        };
        Diagnostic::error(codes::EXECUTION_CANCELLED, message).with_graph_path(GraphPath::new([
            "execution".to_owned(),
            "cancellation".to_owned(),
        ]))
    }
}

#[derive(Debug)]
enum RunTerminal {
    Completed(Option<NativeRunOutput>),
    Cancelled(NativeRunCancellation),
    Failed(RunFailure),
}

#[derive(Debug, Clone)]
enum RunTerminalKind {
    Completed,
    Cancelled(NativeRunCancellation),
    Failed(RunFailure),
}

#[derive(Debug)]
struct RunState {
    status: PyRunStatus,
    history: Vec<PyRunStatus>,
    progress: Option<NativeRunProgress>,
    terminal: Option<RunTerminal>,
    integrity_failed: bool,
}

impl RunState {
    fn accepted() -> Self {
        Self {
            status: PyRunStatus::Queued,
            history: vec![
                PyRunStatus::Created,
                PyRunStatus::Validating,
                PyRunStatus::Queued,
            ],
            progress: None,
            terminal: None,
            integrity_failed: false,
        }
    }

    fn transition(&mut self, status: PyRunStatus) {
        if self.status == status {
            return;
        }
        let allowed = matches!(
            (self.status, status),
            (
                PyRunStatus::Queued,
                PyRunStatus::Running | PyRunStatus::Cancelling | PyRunStatus::Failed
            ) | (
                PyRunStatus::Running,
                PyRunStatus::Cancelling | PyRunStatus::Completed | PyRunStatus::Failed
            ) | (
                PyRunStatus::Cancelling,
                PyRunStatus::Cancelled | PyRunStatus::Completed | PyRunStatus::Failed
            )
        );
        if allowed {
            self.status = status;
            self.history.push(status);
        } else {
            self.integrity_failed = true;
            self.terminal = Some(RunTerminal::Failed(RunFailure::Internal(vec![
                Diagnostic::error(
                    codes::INTERNAL_FAILURE,
                    format!(
                        "invalid native Run lifecycle transition from {:?} to {status:?}",
                        self.status
                    ),
                ),
            ])));
            self.status = PyRunStatus::Failed;
            self.history.push(PyRunStatus::Failed);
        }
    }
}

#[derive(Debug)]
struct RunShared {
    cancellation_requested: AtomicBool,
    state: Mutex<RunState>,
    changed: Condvar,
}

impl RunShared {
    fn new() -> Self {
        Self {
            cancellation_requested: AtomicBool::new(false),
            state: Mutex::new(RunState::accepted()),
            changed: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, RunState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.integrity_failed = true;
                state.terminal = Some(RunTerminal::Failed(RunFailure::Internal(vec![
                    Diagnostic::error(
                        codes::INTERNAL_FAILURE,
                        "the native Run lifecycle lock was poisoned",
                    ),
                ])));
                state.transition(PyRunStatus::Failed);
                self.changed.notify_all();
                state
            }
        }
    }

    fn mark_running(&self) {
        let mut state = self.state();
        if !state.integrity_failed && state.status == PyRunStatus::Queued {
            state.transition(PyRunStatus::Running);
            self.changed.notify_all();
        }
    }

    fn request_cancellation(&self) -> bool {
        let mut state = self.state();
        if state.integrity_failed
            || state.status.is_terminal()
            || state.status == PyRunStatus::Cancelling
        {
            return false;
        }
        self.cancellation_requested.store(true, Ordering::Release);
        state.transition(PyRunStatus::Cancelling);
        self.changed.notify_all();
        true
    }

    fn cancellation_requested(&self) -> bool {
        self.cancellation_requested.load(Ordering::Acquire)
    }

    fn publish_progress(&self, progress: NativeRunProgress) {
        let mut state = self.state();
        if !state.integrity_failed {
            state.progress = Some(progress);
        }
        self.changed.notify_all();
    }

    fn finish(&self, terminal: RunTerminal) {
        let status = match terminal {
            RunTerminal::Completed(_) => PyRunStatus::Completed,
            RunTerminal::Cancelled(_) => PyRunStatus::Cancelled,
            RunTerminal::Failed(_) => PyRunStatus::Failed,
        };
        let mut state = self.state();
        if state.integrity_failed {
            return;
        }
        state.transition(status);
        if state.integrity_failed {
            return;
        }
        state.terminal = Some(terminal);
        self.changed.notify_all();
    }

    fn wait_terminal_kind(&self) -> RunTerminalKind {
        let mut state = self.state();
        loop {
            if let Some(terminal) = state.terminal.as_ref() {
                return match terminal {
                    RunTerminal::Completed(_) => RunTerminalKind::Completed,
                    RunTerminal::Cancelled(cancellation) => {
                        RunTerminalKind::Cancelled(cancellation.clone())
                    }
                    RunTerminal::Failed(failure) => RunTerminalKind::Failed(failure.clone()),
                };
            }
            state = match self.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    state.integrity_failed = true;
                    let failure = RunFailure::Internal(vec![Diagnostic::error(
                        codes::INTERNAL_FAILURE,
                        "the native Run lifecycle wait was poisoned",
                    )]);
                    state.terminal = Some(RunTerminal::Failed(failure.clone()));
                    state.transition(PyRunStatus::Failed);
                    return RunTerminalKind::Failed(failure);
                }
            };
        }
    }

    fn take_completed(&self) -> Option<NativeRunOutput> {
        let mut state = self.state();
        match state.terminal.as_mut() {
            Some(RunTerminal::Completed(result)) => result.take(),
            _ => None,
        }
    }

    fn cancellation(&self, py: Python<'_>, plan: &PyPlan) -> PyResult<Option<Py<PyAny>>> {
        let cancellation = {
            let state = self.state();
            match state.terminal.as_ref() {
                Some(RunTerminal::Cancelled(cancellation)) => Some(cancellation.clone()),
                _ => None,
            }
        };
        cancellation
            .map(|cancellation| cancellation.into_python(py, plan))
            .transpose()
    }
}

enum MaterializedResult {
    Empty,
    Materializing,
    Ready(Py<PyAny>),
    Failed(Vec<Diagnostic>),
}

struct ResultCache {
    state: Mutex<MaterializedResult>,
    changed: Condvar,
}

struct MaterializationClaim {
    cache: Arc<ResultCache>,
    committed: bool,
}

impl MaterializationClaim {
    fn new(cache: Arc<ResultCache>) -> Self {
        Self {
            cache,
            committed: false,
        }
    }

    fn commit(mut self, result: Py<PyAny>) {
        self.cache.store_ready(result);
        self.committed = true;
    }

    fn fail(mut self, diagnostics: Vec<Diagnostic>) {
        self.cache.store_failed(diagnostics);
        self.committed = true;
    }
}

impl Drop for MaterializationClaim {
    fn drop(&mut self) {
        if !self.committed {
            self.cache
                .store_failed(vec![result_materialization_diagnostic()]);
        }
    }
}

enum CacheClaim {
    Materialize,
    Wait,
    Ready,
    Failed(Vec<Diagnostic>),
}

fn result_materialization_diagnostic() -> Diagnostic {
    Diagnostic::error(codes::INTERNAL_FAILURE, RESULT_MATERIALIZATION_FAILURE)
}

impl ResultCache {
    fn new() -> Self {
        Self {
            state: Mutex::new(MaterializedResult::Empty),
            changed: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, MaterializedResult> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = MaterializedResult::Failed(vec![Diagnostic::error(
                    codes::INTERNAL_FAILURE,
                    "the completed Result cache lock was poisoned",
                )]);
                self.changed.notify_all();
                state
            }
        }
    }

    fn claim(&self) -> CacheClaim {
        let mut state = self.state();
        match &*state {
            MaterializedResult::Empty => {
                *state = MaterializedResult::Materializing;
                CacheClaim::Materialize
            }
            MaterializedResult::Materializing => CacheClaim::Wait,
            MaterializedResult::Ready(_) => CacheClaim::Ready,
            MaterializedResult::Failed(diagnostics) => CacheClaim::Failed(diagnostics.clone()),
        }
    }

    fn ready(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let state = self.state();
        match &*state {
            MaterializedResult::Ready(result) => Some(result.clone_ref(py)),
            _ => None,
        }
    }

    fn store_ready(&self, result: Py<PyAny>) {
        let mut state = self.state();
        *state = MaterializedResult::Ready(result);
        self.changed.notify_all();
    }

    fn store_failed(&self, diagnostics: Vec<Diagnostic>) {
        let mut state = self.state();
        *state = MaterializedResult::Failed(diagnostics);
        self.changed.notify_all();
    }

    fn wait_for_materializer(&self) {
        let mut state = self.state();
        while matches!(*state, MaterializedResult::Materializing) {
            state = match self.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    *state = MaterializedResult::Failed(vec![Diagnostic::error(
                        codes::INTERNAL_FAILURE,
                        "the completed Result cache wait was poisoned",
                    )]);
                    self.changed.notify_all();
                    return;
                }
            };
        }
    }
}

/// One process-local handle for an accepted native execution occurrence.
#[pyclass(name = "Run", module = "eqiora._eqiora", frozen, skip_from_py_object)]
pub(crate) struct PyRun {
    identity: RunIdentity,
    materialization: ResultMaterializationContext,
    shared: Arc<RunShared>,
    result_cache: Arc<ResultCache>,
    cancellation_supported: bool,
}

impl PyRun {
    fn submit_common(
        py: Python<'_>,
        plan: Py<PyPlan>,
        request: Option<CommonRunRequest>,
        profile: bool,
    ) -> PyResult<Self> {
        let plan_ref = plan.borrow(py);
        if plan_ref.scalar_native().is_some()
            && plan_ref.native().backward_euler().is_some()
            && request.is_none()
        {
            return Err(PyTypeError::new_err(
                "transient scalar submit requires State and an explicit output schedule",
            ));
        }
        let (identity, job, thread_name, cancellation_supported) = match (
            plan_ref.native(),
            request,
        ) {
            (ResolvedCommonPlan::Algebraic(native), Some(CommonRunRequest::Algebraic(state))) => (
                RunIdentity::from_common_algebraic(native),
                NativeRunJob::Algebraic(native.clone(), state),
                "eqiora-common-algebraic-run",
                false,
            ),
            (ResolvedCommonPlan::Ode(_), Some(CommonRunRequest::Ode(request))) => (
                RunIdentity::from_common_ode(&request),
                NativeRunJob::Ode(request),
                "eqiora-common-ode-run",
                false,
            ),
            (ResolvedCommonPlan::Scalar(native), None) => (
                RunIdentity::from_common_plan(native),
                NativeRunJob::Scalar(native.clone()),
                "eqiora-common-scalar-run",
                true,
            ),
            (ResolvedCommonPlan::Elasticity(native), None) => (
                RunIdentity::from_common_elasticity(native),
                NativeRunJob::Elasticity(native.clone()),
                "eqiora-common-elasticity-run",
                true,
            ),
            (ResolvedCommonPlan::SteadyStokes(native), None) => (
                RunIdentity::from_common_steady_stokes(native),
                NativeRunJob::SteadyStokes(native.clone()),
                "eqiora-common-steady-stokes-run",
                true,
            ),
            (
                ResolvedCommonPlan::TransientFlow(_) | ResolvedCommonPlan::Scalar(_),
                Some(CommonRunRequest::Transient(request)),
            ) => (
                RunIdentity::from_common_transient(&request),
                NativeRunJob::Transient(request),
                "eqiora-common-transient-run",
                true,
            ),
            (ResolvedCommonPlan::Fsi(_), Some(CommonRunRequest::Fsi(request))) => (
                RunIdentity::from_common_fsi(&request),
                NativeRunJob::Fsi(request),
                "eqiora-common-fsi-run",
                true,
            ),
            (ResolvedCommonPlan::TransientFlow(_), None) => {
                return Err(PyTypeError::new_err(
                    "transient submit requires State and one explicit horizon/output schedule family",
                ));
            }
            (ResolvedCommonPlan::Fsi(_), None) => {
                return Err(PyTypeError::new_err(
                    "FSI submit requires State and one explicit horizon/output schedule family",
                ));
            }
            (ResolvedCommonPlan::Ode(_), None) => {
                return Err(PyTypeError::new_err(
                    "ODE submit requires State, until_s, and output_times_s",
                ));
            }
            (ResolvedCommonPlan::TransientFlow(_), Some(CommonRunRequest::Ode(_))) => {
                return Err(PyTypeError::new_err(
                    "ODE Run request crossed a spatial transient Plan",
                ));
            }
            (
                ResolvedCommonPlan::Fsi(_),
                Some(CommonRunRequest::Ode(_) | CommonRunRequest::Transient(_)),
            )
            | (ResolvedCommonPlan::TransientFlow(_), Some(CommonRunRequest::Fsi(_))) => {
                return Err(PyTypeError::new_err(
                    "common Run request crossed an incompatible transient Plan",
                ));
            }
            (
                ResolvedCommonPlan::Scalar(_)
                | ResolvedCommonPlan::Elasticity(_)
                | ResolvedCommonPlan::SteadyStokes(_)
                | ResolvedCommonPlan::Ode(_),
                Some(_),
            ) => {
                return Err(PyTypeError::new_err(
                    "steady submit accepts Plan alone and no transient Run controls",
                ));
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "finite Run requires its exact Plan-bound State",
                ));
            }
        };
        drop(plan_ref);
        Self::spawn(
            identity,
            job,
            ResultMaterializationContext::CommonPlan { plan },
            thread_name,
            cancellation_supported,
            profile,
        )
        .map_err(|diagnostics| internal_diagnostic_error(py, &diagnostics))
    }

    fn spawn(
        identity: RunIdentity,
        job: NativeRunJob,
        materialization: ResultMaterializationContext,
        thread_name: &str,
        cancellation_supported: bool,
        profile: bool,
    ) -> Result<Self, Vec<Diagnostic>> {
        let shared = Arc::new(RunShared::new());
        let worker_shared = Arc::clone(&shared);
        let spawn = thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || run_worker(job, worker_shared, profile));
        if let Err(error) = spawn {
            let diagnostics = vec![Diagnostic::error(
                codes::INTERNAL_FAILURE,
                format!("the native Run worker could not start: {error}"),
            )];
            shared.finish(RunTerminal::Failed(RunFailure::Internal(
                diagnostics.clone(),
            )));
            return Err(diagnostics);
        }
        Ok(Self {
            identity,
            materialization,
            shared,
            result_cache: Arc::new(ResultCache::new()),
            cancellation_supported,
        })
    }
}

enum CommonRunRequest {
    Algebraic(eqiora_numerics::CommonAlgebraicState),
    Ode(Box<CommonOdeRunRequest>),
    Transient(Box<CommonTransientRunRequest>),
    Fsi(Box<CommonFsiRunRequest>),
}

#[pymethods]
impl PyRun {
    #[getter]
    fn status(&self) -> PyRunStatus {
        self.shared.state().status
    }

    #[getter]
    fn history(&self) -> Vec<PyRunStatus> {
        self.shared.state().history.clone()
    }

    #[getter]
    fn progress(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let progress = { self.shared.state().progress.clone() };
        progress
            .map(|progress| progress.into_python(py))
            .transpose()
    }

    /// Exact terminal cancellation evidence, once cancellation is accepted.
    #[getter]
    fn cancellation(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let ResultMaterializationContext::CommonPlan { plan } = &self.materialization;
        self.shared.cancellation(py, &plan.borrow(py))
    }

    #[getter]
    fn done(&self) -> bool {
        self.shared.state().status.is_terminal()
    }

    #[getter]
    fn model_id(&self) -> &str {
        self.identity.model_id()
    }

    #[getter]
    fn model_digest(&self) -> &str {
        self.identity.model_digest()
    }

    #[getter]
    const fn model_revision(&self) -> u64 {
        self.identity.model_revision()
    }

    #[getter]
    fn package_compilation_digest(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let ResultMaterializationContext::CommonPlan { plan } = &self.materialization;
        plan.borrow(py).package_compilation_digest_value(py)
    }

    #[getter]
    fn plan_key(&self) -> &str {
        self.identity.plan_key()
    }

    #[getter]
    fn adapter(&self) -> &'static str {
        self.identity.adapter()
    }

    #[getter]
    fn adapter_version(&self) -> &'static str {
        self.identity.adapter_version()
    }

    /// Request cancellation at the next supported execution-family boundary.
    ///
    /// Returns false when this execution family exposes no accepted cancellation
    /// boundary. Otherwise, a run can still complete when its last accepted
    /// boundary won the race.
    fn cancel(&self) -> bool {
        if !self.cancellation_supported {
            return false;
        }
        self.shared.request_cancellation()
    }

    /// Block without the GIL until one terminal outcome is available.
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        panic_boundary(py, || {
            let shared = Arc::clone(&self.shared);
            let terminal = py.detach(move || shared.wait_terminal_kind());
            match terminal {
                RunTerminalKind::Completed => {}
                RunTerminalKind::Cancelled(cancellation) => {
                    return Err(cancellation_error(py, &[cancellation.diagnostic()]));
                }
                RunTerminalKind::Failed(RunFailure::Execution(diagnostics)) => {
                    return Err(execution_error(py, &diagnostics));
                }
                RunTerminalKind::Failed(RunFailure::Internal(diagnostics)) => {
                    return Err(internal_diagnostic_error(py, &diagnostics));
                }
            }

            loop {
                match self.result_cache.claim() {
                    CacheClaim::Ready => {
                        return self.result_cache.ready(py).ok_or_else(|| {
                            internal_error(py, "the completed Result cache changed state")
                        });
                    }
                    CacheClaim::Failed(diagnostics) => {
                        return Err(internal_diagnostic_error(py, &diagnostics));
                    }
                    CacheClaim::Wait => {
                        let cache = Arc::clone(&self.result_cache);
                        py.detach(move || cache.wait_for_materializer());
                    }
                    CacheClaim::Materialize => break,
                }
            }
            let materialization = MaterializationClaim::new(Arc::clone(&self.result_cache));

            let Some(result) = self.shared.take_completed() else {
                return Err(internal_error(
                    py,
                    "the completed native Result payload was unavailable",
                ));
            };
            let projected = catch_native_panic(|| {
                materialize_result(py, result, &self.identity, &self.materialization)
            });
            match projected {
                Ok(Ok(result)) => {
                    materialization.commit(result.clone_ref(py));
                    Ok(result)
                }
                Ok(Err(_)) | Err(_) => {
                    let diagnostics = vec![result_materialization_diagnostic()];
                    materialization.fail(diagnostics.clone());
                    Err(internal_diagnostic_error(py, &diagnostics))
                }
            }
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Run(status={:?}, model_digest={:?}, plan_key={:?})",
            self.status(),
            self.model_digest(),
            self.plan_key()
        )
    }
}

fn materialize_result(
    py: Python<'_>,
    result: NativeRunOutput,
    identity: &RunIdentity,
    context: &ResultMaterializationContext,
) -> PyResult<Py<PyAny>> {
    let ResultMaterializationContext::CommonPlan { plan } = context;
    match result {
        NativeRunOutput::Result(result, profile) => crate::result::materialize_common_result(
            py,
            plan.borrow(py),
            identity.clone(),
            *result,
            profile,
        )
        .and_then(|result| Py::new(py, result))
        .map(Py::into_any),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (plan, /, *, state=None, until_s=None, output_times_s=None, steps=None, output_steps=None, profile=false))]
pub(crate) fn submit_plan(
    py: Python<'_>,
    plan: Py<PyPlan>,
    state: Option<&Bound<'_, PyState>>,
    until_s: Option<f64>,
    output_times_s: Option<Vec<f64>>,
    steps: Option<usize>,
    output_steps: Option<Vec<usize>>,
    profile: bool,
) -> PyResult<PyRun> {
    panic_boundary(py, || {
        let plan_ref = plan.borrow(py);
        let request = if let Some(native_plan) = plan_ref.native().as_algebraic() {
            if until_s.is_some()
                || output_times_s.is_some()
                || steps.is_some()
                || output_steps.is_some()
            {
                return Err(PyTypeError::new_err(
                    "finite submit accepts only Plan and State",
                ));
            }
            let state = state
                .ok_or_else(|| {
                    PyTypeError::new_err("finite submit requires state=State.initial(plan)")
                })?
                .borrow();
            let native_state = state
                .algebraic_native
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("State is not finite algebraic"))?;
            if native_state
                != &native_plan
                    .initial_state()
                    .map_err(|d| validation_error(py, &[d]))?
            {
                return Err(PyValueError::new_err(
                    "State belongs to a different finite Plan",
                ));
            }
            Some(CommonRunRequest::Algebraic(native_state.clone()))
        } else if let Some(native_plan) = plan_ref.ode_native() {
            let state = state.ok_or_else(|| {
                PyTypeError::new_err("ODE submit requires state=State.initial(plan)")
            })?;
            let state = state.borrow();
            let native_state = state.common_ode_native().ok_or_else(|| {
                PyValueError::new_err("State is not a no-Mesh explicit ODE State")
            })?;
            if steps.is_some() || output_steps.is_some() {
                return Err(PyTypeError::new_err(
                    "ODE submit accepts only state, until_s, and output_times_s; steps/output_steps are unsupported",
                ));
            }
            let until =
                until_s.ok_or_else(|| PyTypeError::new_err("ODE submit requires until_s"))?;
            let outputs = output_times_s
                .ok_or_else(|| PyTypeError::new_err("ODE submit requires output_times_s"))?;
            Some(CommonRunRequest::Ode(Box::new(
                CommonOdeRunRequest::new(native_plan.clone(), native_state.clone(), until, outputs)
                    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?,
            )))
        } else if plan_ref.native().backward_euler().is_some() && plan_ref.fsi_native().is_none() {
            let state = state.ok_or_else(|| {
                PyTypeError::new_err("transient submit requires state=State(...)")
            })?;
            let state = state.borrow();
            let native_state = state.common_native().ok_or_else(|| {
                PyValueError::new_err("State is not a common transient restart State")
            })?;
            let request = match (until_s, output_times_s, steps, output_steps) {
                (Some(until), Some(outputs), None, None) => CommonTransientRunRequest::from_times(
                    plan_ref.native().clone(), native_state.clone(), until, outputs,
                ),
                (None, None, Some(steps), Some(outputs)) => CommonTransientRunRequest::from_steps(
                    plan_ref.native().clone(), native_state.clone(), steps, outputs,
                ),
                _ => return Err(PyTypeError::new_err(
                    "transient submit requires exactly one complete until_s/output_times_s or steps/output_steps family",
                )),
            }.map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            Some(CommonRunRequest::Transient(Box::new(request)))
        } else if let Some(native_plan) = plan_ref.fsi_native() {
            let state = state.ok_or_else(|| {
                PyTypeError::new_err("FSI submit requires state=State.initial(plan, ...)")
            })?;
            let state = state.borrow();
            let native_state = state
                .common_native()
                .ok_or_else(|| PyValueError::new_err("State is not a common FSI restart State"))?;
            if native_state.state_space_identity() != native_plan.state_space_identity() {
                return Err(PyValueError::new_err(
                    "State belongs to a different exact common FSI state space",
                ));
            }
            let request = match (until_s, output_times_s, steps, output_steps) {
                (Some(until), Some(outputs), None, None) => CommonFsiRunRequest::from_times(
                    native_plan.clone(), native_state.clone(), until, outputs,
                ),
                (None, None, Some(steps), Some(outputs)) => CommonFsiRunRequest::from_steps(
                    native_plan.clone(), native_state.clone(), steps, outputs,
                ),
                _ => return Err(PyTypeError::new_err(
                    "FSI submit requires exactly one complete until_s/output_times_s or steps/output_steps family",
                )),
            }.map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            Some(CommonRunRequest::Fsi(Box::new(request)))
        } else {
            if state.is_some()
                || until_s.is_some()
                || output_times_s.is_some()
                || steps.is_some()
                || output_steps.is_some()
            {
                return Err(PyTypeError::new_err(
                    "steady submit accepts Plan alone and no transient Run controls",
                ));
            }
            None
        };
        drop(plan_ref);
        PyRun::submit_common(py, plan, request, profile)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRunStatus>()?;
    module.add_class::<PyCommonTransientRunProgress>()?;
    module.add_class::<PyCommonTransientRunCancellation>()?;
    module.add_class::<PyRun>()?;
    module.add_function(wrap_pyfunction!(submit_plan, module)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "execution/tests.rs"]
mod tests;
