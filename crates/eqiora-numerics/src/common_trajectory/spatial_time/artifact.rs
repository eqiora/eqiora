//! One representation of the execution witness selected by the temporal method.
use super::*;
use crate::common_result::artifact::WireSolveHistory;
use crate::time_history_artifact::WireReport;
use eqiora_time::TimeBackendIdentity;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(in crate::common_trajectory) enum WireSpatialTimeExecution {
    Midpoint { stages: Vec<WireSpatialTimeStage> },
    Adaptive { report: WireReport },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::common_trajectory) struct WireSpatialTimeStage {
    solve: WireSolveHistory,
    convergence: crate::nonlinear_execution::NewtonConvergence,
}

impl WireSpatialTimeExecution {
    pub(in crate::common_trajectory) fn capture(
        value: &SpatialTimeExecution,
    ) -> Result<Self, Diagnostic> {
        match value {
            SpatialTimeExecution::Midpoint(stages) => Ok(Self::Midpoint {
                stages: stages
                    .iter()
                    .map(|stage| {
                        Ok(WireSpatialTimeStage {
                            solve: WireSolveHistory::from_evidence(&stage.solve)?,
                            convergence: stage.convergence.clone(),
                        })
                    })
                    .collect::<Result<_, Diagnostic>>()?,
            }),
            SpatialTimeExecution::Adaptive(report) => Ok(Self::Adaptive {
                report: WireReport::from_native(*report)?,
            }),
        }
    }

    pub(in crate::common_trajectory) fn replay(
        &self,
        backend: TimeBackendIdentity,
    ) -> Result<SpatialTimeExecution, Diagnostic> {
        match self {
            Self::Midpoint { stages } => Ok(SpatialTimeExecution::Midpoint(
                stages
                    .iter()
                    .map(|stage| {
                        Ok(SpatialTimeStage {
                            solve: stage.solve.replay_metadata()?,
                            convergence: stage.convergence.clone(),
                        })
                    })
                    .collect::<Result<_, Diagnostic>>()?,
            )),
            Self::Adaptive { report } => {
                Ok(SpatialTimeExecution::Adaptive(report.replay(backend)?))
            }
        }
    }
}
