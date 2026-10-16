//! One Run binding for finite-time Plans, physical State and output controls.
use super::*;
use crate::{CommonOdePlan, SamplingIdentity};
use eqiora_time::{TimePlan, TimeProblem};

#[derive(Debug, Clone, PartialEq)]
enum Execution {
    Ode(TimePlan),
    Process {
        sampling: ([u8; 32], SamplingIdentity),
    },
}

/// Exact finite-time request. Random sample identity belongs to this occurrence,
/// independently of the immutable Plan and reusable physical State.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTimeRunRequest {
    plan: CommonTimePlan,
    pub(crate) state: CommonTimeState,
    pub(crate) until_s: f64,
    output_times_s: Vec<f64>,
    pub(crate) execution_times_s: Vec<f64>,
    execution: Execution,
    identity: String,
}

impl CommonTimeRunRequest {
    /// Bind deterministic execution without a random sample identity.
    pub fn new(
        plan: impl Into<CommonTimePlan>,
        state: CommonTimeState,
        until_s: f64,
        output_times_s: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::admit(plan.into(), state, until_s, output_times_s, None)
    }

    /// Bind a stochastic occurrence to its complete stable sample identity.
    pub fn sampled(
        plan: CommonTimePlan,
        state: CommonTimeState,
        until_s: f64,
        output_times_s: Vec<f64>,
        master: [u8; 32],
        identity: SamplingIdentity,
    ) -> Result<Self, Diagnostic> {
        Self::admit(
            plan,
            state,
            until_s,
            output_times_s,
            Some((master, identity)),
        )
    }

    pub(super) fn admit(
        plan: CommonTimePlan,
        state: CommonTimeState,
        until_s: f64,
        output_times_s: Vec<f64>,
        sampling: Option<([u8; 32], SamplingIdentity)>,
    ) -> Result<Self, Diagnostic> {
        if state.state_space_identity() != plan.state_space_identity() {
            return Err(invalid(
                "State belongs to a different exact finite-time state space",
            ));
        }
        if !until_s.is_finite() || until_s <= state.time_s() {
            return Err(invalid(
                "time Run until_s must be finite and later than State.time_s",
            ));
        }
        if output_times_s.is_empty()
            || output_times_s
                .iter()
                .any(|time| !time.is_finite() || *time <= state.time_s() || *time > until_s)
            || output_times_s.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "time output_times_s must be finite, nonempty, strictly increasing, later than State.time_s, and within until_s",
            ));
        }
        let mut execution_times_s = output_times_s.clone();
        if execution_times_s
            .last()
            .is_none_or(|time| time.to_bits() != until_s.to_bits())
        {
            execution_times_s.push(until_s);
        }
        let execution = match (plan.as_ode(), sampling) {
            (Some(ode), None) => {
                Execution::Ode(ode.run_time_plan(&state, execution_times_s.clone())?)
            }
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "deterministic time Run does not consume a sample identity",
                ));
            }
            (None, Some(sampling)) => {
                let (_, method) = plan.process().expect("admitted process Plan");
                method.validate_times(state.time_s(), until_s, &execution_times_s)?;
                Execution::Process { sampling }
            }
            (None, None) => {
                return Err(invalid(
                    "stochastic time Run requires an explicit sample identity",
                ));
            }
        };
        let mut bytes = Vec::new();
        push(&mut bytes, plan.identity().as_bytes());
        push(&mut bytes, state.identity().as_bytes());
        bytes.extend_from_slice(&until_s.to_bits().to_be_bytes());
        bytes.extend_from_slice(&(output_times_s.len() as u64).to_be_bytes());
        for time in &output_times_s {
            bytes.extend_from_slice(&time.to_bits().to_be_bytes());
        }
        if let Execution::Process { sampling } = &execution {
            push(
                &mut bytes,
                &serde_json::to_vec(sampling).map_err(|error| invalid(error.to_string()))?,
            );
        }
        let identity = digest(b"eqiora.common-time-run-request/v1\0", &bytes);
        Ok(Self {
            plan,
            state,
            until_s,
            output_times_s,
            execution_times_s,
            execution,
            identity,
        })
    }

    pub fn plan(&self) -> &CommonTimePlan {
        &self.plan
    }
    pub fn state(&self) -> &CommonTimeState {
        &self.state
    }
    pub fn until_s(&self) -> f64 {
        self.until_s
    }
    pub fn output_times_s(&self) -> &[f64] {
        &self.output_times_s
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Master seed and sample binding, absent for deterministic execution.
    pub fn sampling(&self) -> Option<&([u8; 32], SamplingIdentity)> {
        match &self.execution {
            Execution::Process { sampling } => Some(sampling),
            Execution::Ode(_) => None,
        }
    }

    /// Borrow the ODE capability required by deterministic backend adapters.
    pub fn ode_plan(&self) -> Result<&CommonOdePlan, Diagnostic> {
        self.plan
            .as_ode()
            .ok_or_else(|| invalid("this time Run has no deterministic ODE capability"))
    }

    /// Complete deterministic controls, including an unrequested horizon sample.
    pub fn time_plan(&self) -> Result<&TimePlan, Diagnostic> {
        match &self.execution {
            Execution::Ode(plan) => Ok(plan),
            Execution::Process { .. } => Err(invalid(
                "stochastic time Run has no deterministic backend controls",
            )),
        }
    }

    pub fn problem(&self) -> Result<TimeProblem<'_>, Diagnostic> {
        let plan = self.ode_plan()?;
        if plan.event_policy().is_some() {
            return Err(invalid(
                "registered-event ODE requires the canonical event driver",
            ));
        }
        plan.problem(&self.state)
    }
}
