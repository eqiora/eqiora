//! Typed wire projection of accepted stencils; no live process reconstruction.
use super::super::super::retained;
use super::super::provenance::RunBinding;
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireProcessHistory {
    request: RunBinding,
    steps: Vec<WireStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStep {
    #[serde(with = "retained::value")]
    elapsed: BigRational,
    state_base64: String,
    inputs: WireInputs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "process",
    content = "stencils",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
enum WireInputs {
    Wiener(Vec<WireIncrement>),
    Poisson(Vec<WireJump>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireIncrement {
    group: String,
    #[serde(with = "retained::values")]
    values: Vec<BigRational>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireJump {
    #[serde(with = "retained::value")]
    elapsed: BigRational,
    channels: Vec<(String, usize)>,
    state_base64: String,
}

impl ProcessHistory {
    pub(crate) fn identity(&self) -> Result<String, Diagnostic> {
        let bytes = eqiora_artifact::JsonLimits::EXECUTION
            .encode(&WireProcessHistory::capture(self)?)
            .map_err(|error| invalid(&format!("cannot encode process history: {error}")))?;
        Ok(crate::common_time::digest(
            b"eqiora.accepted-process-history/v1\0",
            &bytes,
        ))
    }
}

impl WireProcessHistory {
    pub(crate) fn capture(history: &ProcessHistory) -> Result<Self, Diagnostic> {
        let steps = history
            .steps
            .iter()
            .map(|step| {
                Ok(WireStep {
                    elapsed: step.elapsed.clone(),
                    state_base64: STANDARD.encode(step.state.to_bytes()?),
                    inputs: match &step.inputs {
                        StepInputs::Wiener(increments) => WireInputs::Wiener(
                            increments
                                .iter()
                                .map(|input| WireIncrement {
                                    group: input.group.ulid().to_string(),
                                    values: input.values.clone(),
                                })
                                .collect(),
                        ),
                        StepInputs::Poisson(stencils) => WireInputs::Poisson(
                            stencils
                                .iter()
                                .map(|stencil| {
                                    Ok(WireJump {
                                        elapsed: stencil.events[0].elapsed.clone(),
                                        channels: stencil
                                            .events
                                            .iter()
                                            .map(|event| {
                                                (event.group.ulid().to_string(), event.component)
                                            })
                                            .collect(),
                                        state_base64: STANDARD.encode(stencil.state.to_bytes()?),
                                    })
                                })
                                .collect::<Result<_, Diagnostic>>()?,
                        ),
                    },
                })
            })
            .collect::<Result<_, Diagnostic>>()?;
        Ok(Self {
            request: history.provenance.clone(),
            steps,
        })
    }

    pub(crate) fn replay(
        &self,
        plan: &crate::CommonTimePlan,
    ) -> Result<ProcessHistory, Diagnostic> {
        let request = self.request.reopen_request(plan)?;
        let (program, _) = plan
            .process()
            .ok_or_else(|| invalid("process history requires its exact stochastic Plan"))?;
        let state = |encoded: &str| {
            let bytes = STANDARD
                .decode(encoded)
                .map_err(|_| invalid("invalid process history State encoding"))?;
            program.state_from_bytes(&bytes)
        };
        let group = |text: &str| {
            program
                .noises
                .iter()
                .find(|noise| noise.id.ulid().to_string() == text)
                .map(|noise| noise.id)
                .ok_or_else(|| invalid("process history references a foreign noise group"))
        };
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(AcceptedProcessStep {
                    elapsed: step.elapsed.clone(),
                    state: state(&step.state_base64)?,
                    inputs: match &step.inputs {
                        WireInputs::Wiener(increments) => StepInputs::Wiener(
                            increments
                                .iter()
                                .map(|input| {
                                    Ok(WienerIncrement {
                                        group: group(&input.group)?,
                                        values: input.values.clone(),
                                    })
                                })
                                .collect::<Result<_, Diagnostic>>()?,
                        ),
                        WireInputs::Poisson(stencils) => StepInputs::Poisson(
                            stencils
                                .iter()
                                .map(|stencil| {
                                    Ok(JumpStencil {
                                        state: state(&stencil.state_base64)?,
                                        events: stencil
                                            .channels
                                            .iter()
                                            .map(|(id, component)| {
                                                Ok(JumpEvent {
                                                    elapsed: stencil.elapsed.clone(),
                                                    group: group(id)?,
                                                    component: *component,
                                                })
                                            })
                                            .collect::<Result<_, Diagnostic>>()?,
                                    })
                                })
                                .collect::<Result<_, Diagnostic>>()?,
                        ),
                    },
                })
            })
            .collect::<Result<_, Diagnostic>>()?;
        ProcessHistory::accept(request, self.request.clone(), steps)
    }
}
