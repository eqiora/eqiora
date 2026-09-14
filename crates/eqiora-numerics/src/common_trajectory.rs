//! One native authority for accepted common ODE and spatial trajectories.

use eqiora_core::Diagnostic;
use eqiora_time::{
    AcceptedTimeHistory, InitialConditionPolicy, TimeEquationClass, TimeMethod, TimeSolution,
};
use sha2::{Digest, Sha256};

use crate::{
    CommonFsiRunRequest, CommonOdeRunRequest, CommonOdeState, CommonState,
    CommonTransientRunRequest,
};

mod artifact;
mod functional;
mod history_boundary;
mod sensitivity;
pub use functional::TimeFunctionalQuadrature;
pub(crate) use sensitivity::CommonTrajectoryParameterSensitivity;

/// Accepted output States bound to the complete immutable Run request.
#[derive(Debug, Clone, PartialEq)]
pub enum CommonTrajectory {
    Ode {
        request: Box<CommonOdeRunRequest>,
        states: Vec<CommonOdeState>,
        history: AcceptedTimeHistory,
        identity: String,
    },
    SpatialTransient {
        request: Box<CommonTransientRunRequest>,
        states: Vec<(usize, CommonState)>,
        identity: String,
    },
    Fsi {
        request: Box<CommonFsiRunRequest>,
        states: Vec<(usize, CommonState)>,
        identity: String,
    },
}

impl CommonTrajectory {
    /// Reaccept one adaptive ODE backend solution against its exact request.
    pub fn accept_ode(
        request: CommonOdeRunRequest,
        solution: TimeSolution,
    ) -> Result<Self, Diagnostic> {
        if solution.report().method() != TimeMethod::Tsitouras45
            || solution.report().backend_identity() != request.plan().backend()
            || solution.report().equation_class() != TimeEquationClass::ExplicitOde
            || solution.report().initial_condition() != InitialConditionPolicy::Provided
            || solution.dimension() != request.plan().field_dimensions().len()
            || solution.times() != request.time_plan().output_times()
        {
            return Err(invalid(
                "adaptive backend result differs from the exact no-Mesh ODE request",
            ));
        }
        let history = solution.history().cloned().ok_or_else(|| {
            invalid("common ODE Trajectory requires accepted native integration history")
        })?;
        for (sample, &time) in solution.times().iter().enumerate() {
            history_boundary::validate(
                &history,
                time,
                solution.state(sample).expect("accepted sample shape"),
            )?;
        }
        let states = request
            .output_times_s()
            .iter()
            .enumerate()
            .map(|(sample, &time)| {
                let values = solution
                    .state(sample)
                    .ok_or_else(|| invalid("adaptive backend omitted one requested ODE State"))?;
                CommonOdeState::new(request.plan(), time, values.to_vec(), "result")
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::accept_ode_states(request, states, history)
    }

    pub(crate) fn accept_ode_states(
        request: CommonOdeRunRequest,
        states: Vec<CommonOdeState>,
        history: AcceptedTimeHistory,
    ) -> Result<Self, Diagnostic> {
        if states.len() != request.output_times_s().len()
            || states
                .iter()
                .zip(request.output_times_s())
                .any(|(state, time)| {
                    state.state_space_identity() != request.plan().state_space_identity()
                        || state.time_s().to_bits() != time.to_bits()
                })
        {
            return Err(invalid(
                "accepted ODE Trajectory differs from its exact Run request",
            ));
        }
        for state in &states {
            history_boundary::validate(&history, state.time_s(), state.values())?;
        }
        let first = history.steps().first().expect("history is nonempty");
        let last = history.steps().last().expect("history is nonempty");
        if history.dimension() != request.plan().field_dimensions().len()
            || first.start_time().to_bits() != request.state().time_s().to_bits()
            || first.start_state() != request.state().values()
            || last.end_time().to_bits() != request.until_s().to_bits()
        {
            return Err(invalid(
                "accepted ODE history differs from its exact Run interval, initial State, or activation profile",
            ));
        }
        if !history.events().is_empty() {
            let roots = request.plan().root_set()?.ok_or_else(|| {
                invalid("Trajectory events require an admitted registered root set")
            })?;
            let policy = request
                .plan()
                .event_policy()
                .expect("admitted root set has policy");
            if history.events().len() > policy.max_events() {
                return Err(invalid("Trajectory exceeds its exact Plan event budget"));
            }
            for event in history.events() {
                let proposal = event.proposal();
                if proposal.report().backend_identity() != request.plan().backend()
                    || proposal.report().method() != TimeMethod::Tsitouras45
                    || proposal.report().initial_condition() != InitialConditionPolicy::Provided
                {
                    return Err(invalid(
                        "Trajectory event report differs from its exact Plan",
                    ));
                }
                let tolerance = request.plan().guard_tolerance(proposal.root_index())?;
                let accepted = roots.linearize_proposal(proposal, tolerance.value())?;
                if accepted.post_state() != event.after_state() {
                    return Err(invalid(
                        "Trajectory event post-State differs from the canonical registered reset",
                    ));
                }
            }
        }
        let identity = ode_identity(request.identity(), &states, &history);
        Ok(Self::Ode {
            request: Box::new(request),
            states,
            history,
            identity,
        })
    }

    /// Accept exact requested scalar or flow output steps and grid times.
    pub fn accept_spatial_transient(
        request: CommonTransientRunRequest,
        states: Vec<(usize, CommonState)>,
    ) -> Result<Self, Diagnostic> {
        validate_spatial(
            request.plan().spatial_state_space_identity()?,
            request.output_steps(),
            &states,
        )?;
        let step_s = request
            .plan()
            .backward_euler()
            .ok_or_else(|| invalid("spatial trajectory requires BackwardEuler"))?
            .step()
            .value();
        if states.iter().any(|(step, state)| {
            state.time_s().to_bits() != (request.state().time_s() + *step as f64 * step_s).to_bits()
        }) {
            return Err(invalid(
                "spatial Trajectory time differs from its exact accepted-step grid",
            ));
        }
        let identity = spatial_identity(b"spatial-transient", request.identity(), &states)?;
        Ok(Self::SpatialTransient {
            request: Box::new(request),
            states,
            identity,
        })
    }

    /// Accept exact requested fixed-reference-FSI output steps.
    pub fn accept_fsi(
        request: CommonFsiRunRequest,
        states: Vec<(usize, CommonState)>,
    ) -> Result<Self, Diagnostic> {
        validate_spatial(
            request.plan().state_space_identity(),
            request.output_steps(),
            &states,
        )?;
        let identity = spatial_identity(b"fixed-reference-fsi", request.identity(), &states)?;
        Ok(Self::Fsi {
            request: Box::new(request),
            states,
            identity,
        })
    }

    /// Domain-separated identity of the request and all requested States.
    #[must_use]
    pub fn identity(&self) -> &str {
        match self {
            Self::Ode { identity, .. }
            | Self::SpatialTransient { identity, .. }
            | Self::Fsi { identity, .. } => identity,
        }
    }

    /// Exact immutable Run-request identity.
    #[must_use]
    pub fn request_identity(&self) -> &str {
        match self {
            Self::Ode { request, .. } => request.identity(),
            Self::SpatialTransient { request, .. } => request.identity(),
            Self::Fsi { request, .. } => request.identity(),
        }
    }

    /// Exact owning Plan identity.
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        match self {
            Self::Ode { request, .. } => request.plan().identity(),
            Self::SpatialTransient { request, .. } => request.plan().identity(),
            Self::Fsi { request, .. } => request.plan().identity(),
        }
    }

    /// Requested ODE States, when this is a no-Mesh trajectory.
    #[must_use]
    pub fn ode_states(&self) -> Option<&[CommonOdeState]> {
        match self {
            Self::Ode { states, .. } => Some(states),
            Self::SpatialTransient { .. } | Self::Fsi { .. } => None,
        }
    }

    /// Native accepted ODE step stencils, independent of requested output States.
    #[must_use]
    pub fn ode_history(&self) -> Option<&AcceptedTimeHistory> {
        match self {
            Self::Ode { history, .. } => Some(history),
            Self::SpatialTransient { .. } | Self::Fsi { .. } => None,
        }
    }

    /// Requested step-indexed States, when this is a spatial trajectory.
    #[must_use]
    pub fn spatial_states(&self) -> Option<&[(usize, CommonState)]> {
        match self {
            Self::Ode { .. } => None,
            Self::SpatialTransient { states, .. } | Self::Fsi { states, .. } => Some(states),
        }
    }
}

fn validate_spatial(
    state_space_identity: String,
    output_steps: &[usize],
    states: &[(usize, CommonState)],
) -> Result<(), Diagnostic> {
    if states.len() != output_steps.len()
        || states
            .iter()
            .zip(output_steps)
            .any(|((step, state), expected)| {
                step != expected || state.state_space_identity() != state_space_identity
            })
    {
        return Err(invalid(
            "accepted spatial Trajectory differs from its exact Run request",
        ));
    }
    Ok(())
}

fn ode_identity(
    request_identity: &str,
    states: &[CommonOdeState],
    history: &AcceptedTimeHistory,
) -> String {
    let mut bytes = Vec::new();
    push(&mut bytes, b"ode");
    push(&mut bytes, request_identity.as_bytes());
    for state in states {
        bytes.extend_from_slice(&state.time_s().to_bits().to_be_bytes());
        push(&mut bytes, state.identity().as_bytes());
    }
    for step in history.steps() {
        bytes.extend_from_slice(&step.start_time().to_bits().to_be_bytes());
        bytes.extend_from_slice(&step.end_time().to_bits().to_be_bytes());
        for value in step
            .start_state()
            .iter()
            .chain(step.midpoint_state())
            .chain(step.end_state())
        {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
    }
    bytes.extend_from_slice(&(history.events().len() as u64).to_be_bytes());
    for event in history.events() {
        let proposal = event.proposal();
        bytes.extend_from_slice(&proposal.registration().as_sha256());
        bytes.extend_from_slice(&(proposal.root_index() as u64).to_be_bytes());
        bytes.extend_from_slice(&proposal.time().to_bits().to_be_bytes());
        for value in proposal.state().iter().chain(event.after_state()) {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
    }
    digest(&bytes)
}

fn spatial_identity(
    family: &[u8],
    request_identity: &str,
    states: &[(usize, CommonState)],
) -> Result<String, Diagnostic> {
    let mut bytes = Vec::new();
    push(&mut bytes, family);
    push(&mut bytes, request_identity.as_bytes());
    for (step, state) in states {
        bytes.extend_from_slice(
            &u64::try_from(*step)
                .map_err(|_| invalid("Trajectory output step exceeds canonical u64 range"))?
                .to_be_bytes(),
        );
        push(&mut bytes, state.identity().as_bytes());
    }
    Ok(digest(&bytes))
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest([b"eqiora.common-trajectory/v3\0".as_slice(), bytes].concat())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn push(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value);
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
