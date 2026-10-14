//! Shared leaves for current time-run artifacts. Method admission stays with each family.
use eqiora_core::Diagnostic;
use eqiora_time::{InitialConditionPolicy, TimeMethod, TimePlan};
use serde::{Deserialize, Serialize};

use crate::invalid_artifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTimePlan<M> {
    method: M,
    start_time: f64,
    initial_step: f64,
    relative_tolerance: f64,
    absolute_tolerances: Vec<f64>,
    output_times: Vec<f64>,
}

impl<M: Copy + TryFrom<TimeMethod, Error = Diagnostic> + Into<TimeMethod>> WireTimePlan<M> {
    pub(crate) fn state_dimension(&self) -> usize {
        self.absolute_tolerances.len()
    }

    pub(crate) fn encode(value: &TimePlan) -> Result<Self, Diagnostic> {
        let wire = Self {
            method: M::try_from(value.method())?,
            start_time: value.start_time(),
            initial_step: value.initial_step(),
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerances: value.absolute_tolerances().to_vec(),
            output_times: value.output_times().to_vec(),
        };
        wire.decode()?;
        Ok(wire)
    }

    pub(crate) fn decode(&self) -> Result<TimePlan, Diagnostic> {
        TimePlan::new(
            self.method.into(),
            self.start_time,
            self.initial_step,
            self.relative_tolerance,
            self.absolute_tolerances.clone(),
            self.output_times.clone(),
        )
        .map_err(|error| invalid_artifact(error.message()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireInitialCondition {
    Provided,
    SolveConsistent,
}

impl WireInitialCondition {
    pub(crate) const fn encode(value: InitialConditionPolicy) -> Self {
        match value {
            InitialConditionPolicy::Provided => Self::Provided,
            InitialConditionPolicy::SolveConsistent => Self::SolveConsistent,
        }
    }

    pub(crate) const fn decode(self) -> InitialConditionPolicy {
        match self {
            Self::Provided => InitialConditionPolicy::Provided,
            Self::SolveConsistent => InitialConditionPolicy::SolveConsistent,
        }
    }
}
