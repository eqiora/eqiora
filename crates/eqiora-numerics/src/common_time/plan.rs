//! One finite-time Plan owning its admitted deterministic or stochastic method.
use super::*;
use crate::stochastic_path::{program::ProcessProgram, run::ProcessMethod};
use crate::{CommonOdePlan, CommonOdePolicy};
use eqiora_sem::KernelProgram;
use eqiora_time::{TimeBackendCapabilities, TimeBackendIdentity};

#[derive(Debug, Clone, PartialEq)]
enum Dynamics {
    Ode(Box<CommonOdePlan>),
    Process {
        model: Arc<ModelEnvelope>,
        model_id: String,
        model_revision: u64,
        program: Box<ProcessProgram>,
        method: ProcessMethod,
        identity: String,
    },
}

/// A no-Mesh time Plan. Sample identity, horizon and outputs belong to its Run.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTimePlan {
    dynamics: Dynamics,
}

impl From<CommonOdePlan> for CommonTimePlan {
    fn from(plan: CommonOdePlan) -> Self {
        Self {
            dynamics: Dynamics::Ode(Box::new(plan)),
        }
    }
}

impl CommonTimePlan {
    /// Exact linear stage provider, absent for methods without an implicit solve.
    pub fn linear_solver_provider(&self) -> Option<eqiora_solver::SolverProvider> {
        self.as_ode()?.linear.as_ref().map(|linear| linear.provider)
    }

    pub fn resolve_ode(
        model: &ModelEnvelope,
        kernel: &KernelProgram,
        policy: CommonOdePolicy,
        backend: TimeBackendCapabilities,
        solve: Option<crate::CommonSolvePolicy>,
        linear_backend: &dyn eqiora_solver::LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        CommonOdePlan::resolve(model, kernel, policy, backend, solve, linear_backend)
            .map(Into::into)
    }

    /// Admit Ito diffusion with one fixed dyadic refinement depth per Run horizon.
    pub fn resolve_euler_maruyama(
        model: &ModelEnvelope,
        kernel: &KernelProgram,
        depth: u8,
    ) -> Result<Self, Diagnostic> {
        Self::resolve_process(model, kernel, ProcessMethod::EulerMaruyama { depth })
    }

    /// Admit constant-rate independent Poisson laws with an exact total event budget.
    pub fn resolve_poisson_events(
        model: &ModelEnvelope,
        kernel: &KernelProgram,
        max_events: usize,
    ) -> Result<Self, Diagnostic> {
        Self::resolve_process(model, kernel, ProcessMethod::PoissonEvents { max_events })
    }

    pub(crate) fn resolve_process(
        model: &ModelEnvelope,
        kernel: &KernelProgram,
        method: ProcessMethod,
    ) -> Result<Self, Diagnostic> {
        let program = ProcessProgram::new(kernel)?;
        method.admit(&program)?;
        let reference = model.artifact_reference()?;
        if reference.artifact().to_string() != program.state_space().model_digest {
            return Err(invalid(
                "time Plan Model differs from its canonical process program",
            ));
        }
        let mut identity = Vec::new();
        push(&mut identity, program.state_space().identity.as_bytes());
        push(
            &mut identity,
            &serde_json::to_vec(&method).map_err(|error| invalid(error.to_string()))?,
        );
        push(&mut identity, b"serial-host-binary64/v1");
        Ok(Self {
            dynamics: Dynamics::Process {
                model: Arc::new(model.clone()),
                model_id: reference.model().ulid().to_string(),
                model_revision: reference.semantic_revision().get(),
                program: Box::new(program),
                method,
                identity: digest(b"eqiora.common-time-process-plan/v1\0", &identity),
            },
        })
    }

    /// Borrow the deterministic ODE capability, when that method was admitted.
    pub fn as_ode(&self) -> Option<&CommonOdePlan> {
        match &self.dynamics {
            Dynamics::Ode(plan) => Some(plan),
            _ => None,
        }
    }

    /// Fixed dyadic depth of the admitted Euler-Maruyama method, if selected.
    pub fn euler_maruyama_depth(&self) -> Option<u8> {
        match self.process()?.1 {
            ProcessMethod::EulerMaruyama { depth } => Some(depth),
            _ => None,
        }
    }

    /// Total arrival budget of the admitted Poisson event method, if selected.
    pub fn poisson_event_budget(&self) -> Option<usize> {
        match self.process()?.1 {
            ProcessMethod::PoissonEvents { max_events } => Some(max_events),
            _ => None,
        }
    }

    pub(crate) fn process(&self) -> Option<(&ProcessProgram, ProcessMethod)> {
        match &self.dynamics {
            Dynamics::Process {
                program, method, ..
            } => Some((program, *method)),
            _ => None,
        }
    }

    pub fn identity(&self) -> &str {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.identity(),
            Dynamics::Process { identity, .. } => identity,
        }
    }
    pub fn model_id(&self) -> &str {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.model_id(),
            Dynamics::Process { model_id, .. } => model_id,
        }
    }
    pub const fn model_revision(&self) -> u64 {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.model_revision(),
            Dynamics::Process { model_revision, .. } => *model_revision,
        }
    }
    pub fn model_digest(&self) -> &str {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.model_digest(),
            Dynamics::Process { program, .. } => &program.state_space().model_digest,
        }
    }
    pub fn model_artifact(&self) -> &ModelEnvelope {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.model_artifact(),
            Dynamics::Process { model, .. } => model,
        }
    }
    fn state_space(&self) -> &TimeStateSpace {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.state_space(),
            Dynamics::Process { program, .. } => program.state_space(),
        }
    }
    pub fn state_space_identity(&self) -> &str {
        &self.state_space().identity
    }
    pub fn state_coordinates(&self) -> impl ExactSizeIterator<Item = TimeStateCoordinate> + '_ {
        self.state_space().coordinates.iter().copied()
    }
    pub fn state_dimensions(&self) -> &[DimExponents] {
        &self.state_space().dimensions
    }
    pub fn initial_state(&self, time_s: f64) -> Result<CommonTimeState, Diagnostic> {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.initial_state(time_s),
            Dynamics::Process { program, .. } => program.initialize(time_s),
        }
    }
    pub fn state_from_bytes(&self, bytes: &[u8]) -> Result<CommonTimeState, Diagnostic> {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.state_from_bytes(bytes),
            Dynamics::Process { program, .. } => program.state_from_bytes(bytes),
        }
    }
    pub fn backend(&self) -> TimeBackendIdentity {
        match &self.dynamics {
            Dynamics::Ode(plan) => plan.backend(),
            Dynamics::Process { .. } => {
                TimeBackendIdentity::new("eqiora.time.serial-stochastic", "1")
            }
        }
    }
}
