//! Exact retained forward products inside the owning Result artifact.
use super::*;
use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use crate::time_history_artifact::WireTimeStep;
use eqiora_core::{Id, entity::kinds};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireParameterSensitivity {
    trajectory_identity: String,
    parameters: Vec<String>,
    steps: Vec<WireTimeStep>,
    outputs: Vec<Vec<f64>>,
    event_time_gradients: Vec<Vec<f64>>,
}
impl WireParameterSensitivity {
    pub(super) fn from_native(value: &CommonTrajectoryParameterSensitivity) -> Self {
        Self {
            trajectory_identity: value.trajectory_identity().to_owned(),
            parameters: value
                .parameters()
                .iter()
                .map(|id| id.ulid().to_string())
                .collect(),
            steps: value.steps().iter().map(WireTimeStep::from).collect(),
            outputs: value.outputs().to_vec(),
            event_time_gradients: value.event_time_gradients().to_vec(),
        }
    }
    pub(super) fn replay(
        &self,
        trajectory: &CommonTrajectory,
    ) -> Result<CommonTrajectoryParameterSensitivity, Diagnostic> {
        if self.trajectory_identity != trajectory.identity() {
            return Err(invalid(
                "persisted Parameter sensitivity belongs to a different exact Trajectory",
            ));
        }
        let parameters = self
            .parameters
            .iter()
            .map(|value| {
                ulid::Ulid::from_string(value)
                    .map(Id::<kinds::Parameter>::from_ulid)
                    .map_err(|_| invalid("invalid persisted sensitivity Parameter identity"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let steps = self
            .steps
            .iter()
            .map(WireTimeStep::replay)
            .collect::<Result<Vec<_>, _>>()?;
        let sensitivity = CommonTrajectoryParameterSensitivity::accept_event_history(
            trajectory,
            parameters,
            steps,
            self.outputs.clone(),
            self.event_time_gradients.clone(),
        )?;
        sensitivity.validate_for(trajectory)?;
        Ok(sensitivity)
    }
}
