//! Accepted continuous spatial history keeps output States separate from internal steps.
use super::*;
pub(super) mod artifact;
use crate::common_result::evidence::{CommonSolveEvidence, CommonSolveHistory};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpatialTimeStage {
    pub(crate) solve: CommonSolveHistory,
    pub(crate) convergence: crate::nonlinear_execution::NewtonConvergence,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SpatialTimeExecution {
    Midpoint(Vec<SpatialTimeStage>),
    Adaptive(eqiora_time::TimeExecutionReport),
}

impl CommonTrajectory {
    pub(crate) fn accept_spatial_time(
        request: CommonTransientRunRequest,
        states: Vec<CommonState>,
        history: AcceptedTimeHistory,
        execution: SpatialTimeExecution,
    ) -> Result<Self, Diagnostic> {
        request.validate_spatial_time(&states, &history, &execution)?;
        let mut bytes = Vec::new();
        push(&mut bytes, b"spatial-time");
        push(&mut bytes, request.identity().as_bytes());
        for state in &states {
            push(&mut bytes, state.identity().as_bytes());
        }
        let steps = history
            .steps()
            .iter()
            .map(crate::time_history_artifact::WireTimeStep::from)
            .collect::<Vec<_>>();
        let execution_wire = artifact::WireSpatialTimeExecution::capture(&execution)?;
        push(
            &mut bytes,
            &eqiora_artifact::JsonLimits::EXECUTION
                .encode(&(steps, execution_wire))
                .map_err(|error| invalid(error.to_string()))?,
        );
        Ok(Self {
            payload: TrajectoryPayload::SpatialTime {
                request: Box::new(request),
                states,
                history,
                execution,
                identity: digest(&bytes),
            },
        })
    }

    /// Physical spatial output by observation position, independent of internal steps.
    #[must_use]
    pub fn spatial_state(&self, output_index: usize) -> Option<&CommonState> {
        match &self.payload {
            TrajectoryPayload::SpatialTransient { states, .. } => {
                states.get(output_index).map(|(_, step)| step.state())
            }
            TrajectoryPayload::SpatialTime { states, .. } => states.get(output_index),
            _ => None,
        }
    }

    /// Continuous spatial stencils in free numerical coordinates.
    #[must_use]
    pub fn spatial_time_history(&self) -> Option<&AcceptedTimeHistory> {
        match &self.payload {
            TrajectoryPayload::SpatialTime { history, .. } => Some(history),
            _ => None,
        }
    }

    /// Actual linear solver receipts for one internal spatial time stage.
    /// The index denotes an internal stage, not a requested output.
    /// Adaptive provider stages do not expose retained linear receipts here.
    #[must_use]
    pub fn spatial_stage_solves(&self, stage: usize) -> Option<&[CommonSolveEvidence]> {
        match &self.payload {
            TrajectoryPayload::SpatialTime { execution, .. } => {
                let SpatialTimeExecution::Midpoint(stages) = execution else {
                    return None;
                };
                match &stages.get(stage)?.solve {
                    CommonSolveHistory::Newton { linear_solves, .. } => Some(linear_solves),
                    CommonSolveHistory::Linear(solve) => Some(std::slice::from_ref(solve.as_ref())),
                }
            }
            _ => None,
        }
    }
}
