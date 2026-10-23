//! One canonical wire representation of native time steps, reports and resets.
use eqiora_core::Diagnostic;
use eqiora_time::{
    InitialConditionPolicy, TimeEquationClass, TimeExecutionReport, TimeHistoryStep, TimeMethod,
};
use serde::{Deserialize, Serialize};
pub(crate) mod events;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTimeStep {
    start_time: f64,
    end_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    end_state: Vec<f64>,
    dense_output: WireDenseOutput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDenseOutput {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
}

impl From<&TimeHistoryStep> for WireTimeStep {
    fn from(step: &TimeHistoryStep) -> Self {
        Self {
            start_time: step.start_time(),
            end_time: step.end_time(),
            start_state: step.start_state().to_vec(),
            midpoint_state: step.midpoint_state().to_vec(),
            end_state: step.end_state().to_vec(),
            dense_output: WireDenseOutput {
                times: step.dense_times().to_vec(),
                states: step.dense_states().to_vec(),
            },
        }
    }
}
impl WireTimeStep {
    pub(crate) fn replay(&self) -> Result<TimeHistoryStep, Diagnostic> {
        TimeHistoryStep::accepted(
            self.start_time,
            self.end_time,
            self.start_state.clone(),
            self.midpoint_state.clone(),
            self.end_state.clone(),
            self.dense_output.times.clone(),
            self.dense_output.states.clone(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireReport {
    backend: String,
    backend_version: String,
    method: WireMethod,
    equation_class: WireEquationClass,
    initial_condition: WireInitialCondition,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMethod {
    Tsitouras45,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireEquationClass {
    ExplicitOde,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireInitialCondition {
    Provided,
}

impl WireReport {
    pub(crate) fn from_native(report: TimeExecutionReport) -> Result<Self, Diagnostic> {
        if report.method() != TimeMethod::Tsitouras45
            || report.equation_class() != TimeEquationClass::ExplicitOde
            || report.initial_condition() != InitialConditionPolicy::Provided
        {
            return Err(invalid(
                "native time report requires admitted explicit Tsitouras45 execution with provided initial data",
            ));
        }
        Ok(Self {
            backend: report.backend().to_owned(),
            backend_version: report.backend_version().to_owned(),
            method: WireMethod::Tsitouras45,
            equation_class: WireEquationClass::ExplicitOde,
            initial_condition: WireInitialCondition::Provided,
        })
    }
    pub(crate) fn replay(
        &self,
        backend: eqiora_time::TimeBackendIdentity,
    ) -> Result<TimeExecutionReport, Diagnostic> {
        if self.backend != backend.id() || self.backend_version != backend.version() {
            return Err(invalid(
                "native time report backend differs from the exact ODE Plan",
            ));
        }
        Ok(TimeExecutionReport::new(
            backend,
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        ))
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_history_codec_preserves_samples_and_rejects_malformed_polynomials() {
        let step = TimeHistoryStep::accepted(
            0.,
            1.,
            vec![0.],
            vec![0.25],
            vec![1.],
            vec![0., 0.5, 1.],
            vec![vec![0.], vec![0.25], vec![1.]],
        )
        .unwrap();
        let encoded = serde_json::to_value(WireTimeStep::from(&step)).unwrap();
        let decoded: WireTimeStep = serde_json::from_value(encoded.clone()).unwrap();
        let replayed = decoded.replay().unwrap();
        assert_eq!(replayed, step);
        assert_eq!(replayed.sample(0.125).unwrap(), vec![1. / 64.]);
        let mut missing = encoded.clone();
        missing.as_object_mut().unwrap().remove("dense_output");
        assert!(serde_json::from_value::<WireTimeStep>(missing).is_err());
        for dense in [
            serde_json::Value::Null,
            serde_json::json!({"times": [0., 0.5, 0.5], "states": [[0.], [0.25], [1.]]}),
            serde_json::json!({"times": [0., 0.5, 2.], "states": [[0.], [0.25], [1.]]}),
            serde_json::json!({"times": [0., 1.], "states": [[0., 0.], [1., 1.]]}),
            serde_json::json!({"times": [0., 1.], "states": [[0.], [1.]], "unknown": 0}),
            serde_json::json!({"times": [0., 1.]}),
        ] {
            let mut forged = encoded.clone();
            forged["dense_output"] = dense;
            if let Ok(wire) = serde_json::from_value::<WireTimeStep>(forged) {
                assert!(wire.replay().is_err());
            }
        }
    }
}
