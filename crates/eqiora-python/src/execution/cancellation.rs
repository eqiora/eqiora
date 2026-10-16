//! Terminal cancellation projection for common execution families.
use super::evidence::{PyRunCancellation, PyRunProgress};
use crate::common_plan::PyPlan;
use eqiora::diagnostic::codes;
use eqiora::{Diagnostic, GraphPath};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub(super) enum NativeRunCancellation {
    Time(Box<eqiora_numerics::CommonTimeCheckpoint>),
    CommonTransient {
        accepted_steps: usize,
        maximum_steps: usize,
        model_time_s: f64,
        request_identity: String,
        state: eqiora_numerics::CommonState,
    },
}

impl NativeRunCancellation {
    pub(super) fn into_python(self, py: Python<'_>, plan: &PyPlan) -> PyResult<Py<PyAny>> {
        match self {
            Self::Time(checkpoint) => {
                let native_state = checkpoint
                    .state()
                    .map_err(|d| crate::error::validation_error(py, &[d]))?;
                let progress = PyRunProgress {
                    accepted_steps: checkpoint.accepted_steps(),
                    maximum_steps: None,
                    model_time_bits: native_state.time_s().to_bits(),
                };
                let request_identity = checkpoint.request().identity().to_owned();
                let state = Py::new(
                    py,
                    crate::trajectory::PyState::from_common_time(
                        py,
                        plan,
                        native_state,
                        Some(&request_identity),
                    ),
                )?;
                let checkpoint = Py::new(
                    py,
                    crate::execution_checkpoint::PyExecutionCheckpoint {
                        value: crate::execution_checkpoint::CheckpointValue::Time(checkpoint),
                    },
                )?;
                Py::new(
                    py,
                    PyRunCancellation {
                        progress,
                        request_identity,
                        state,
                        checkpoint: Some(checkpoint),
                    },
                )
                .map(Py::into_any)
            }
            Self::CommonTransient {
                accepted_steps,
                maximum_steps,
                model_time_s,
                request_identity,
                state,
            } => Py::new(
                py,
                PyRunCancellation {
                    progress: PyRunProgress {
                        accepted_steps,
                        maximum_steps: Some(maximum_steps),
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
                    checkpoint: None,
                },
            )
            .map(Py::into_any),
        }
    }

    pub(super) fn diagnostic(&self) -> Diagnostic {
        let message = match self {
            Self::Time(checkpoint) => format!(
                "time execution {} was cancelled after {} accepted steps; its exact checkpoint can resume",
                checkpoint.request().identity(),
                checkpoint.accepted_steps()
            ),
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
