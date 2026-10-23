//! One representation of the execution witness selected by the temporal method.
use super::*;
use crate::common_result::artifact::WireSolveHistory;
use crate::time_history_artifact::WireReport;
use eqiora_time::TimeBackendIdentity;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(in crate::common_trajectory) enum WireSpatialTimeExecution {
    Midpoint {
        stages: Vec<WireSpatialTimeStage>,
        endpoints: Vec<WireSpatialTimeStage>,
        outputs: Vec<WireSpatialTimeStage>,
    },
    Adaptive {
        report: WireReport,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::common_trajectory) struct WireSpatialTimeStage {
    unknown: Vec<f64>,
    solve: WireSolveHistory,
    convergence: crate::nonlinear_execution::NewtonConvergence,
}

impl WireSpatialTimeExecution {
    pub(in crate::common_trajectory) fn capture(
        value: &SpatialTimeExecution,
    ) -> Result<Self, Diagnostic> {
        match value {
            SpatialTimeExecution::Midpoint {
                stages,
                endpoints,
                outputs,
            } => Ok(Self::Midpoint {
                stages: stages
                    .iter()
                    .map(WireSpatialTimeStage::capture)
                    .collect::<Result<_, _>>()?,
                endpoints: endpoints
                    .iter()
                    .map(WireSpatialTimeStage::capture)
                    .collect::<Result<_, _>>()?,
                outputs: outputs
                    .iter()
                    .map(WireSpatialTimeStage::capture)
                    .collect::<Result<_, _>>()?,
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
            Self::Midpoint {
                stages,
                endpoints,
                outputs,
            } => Ok(SpatialTimeExecution::Midpoint {
                stages: stages
                    .iter()
                    .map(WireSpatialTimeStage::replay)
                    .collect::<Result<_, _>>()?,
                endpoints: endpoints
                    .iter()
                    .map(WireSpatialTimeStage::replay)
                    .collect::<Result<_, _>>()?,
                outputs: outputs
                    .iter()
                    .map(WireSpatialTimeStage::replay)
                    .collect::<Result<_, _>>()?,
            }),
            Self::Adaptive { report } => {
                Ok(SpatialTimeExecution::Adaptive(report.replay(backend)?))
            }
        }
    }
}

impl WireSpatialTimeStage {
    fn capture(stage: &SpatialTimeStage) -> Result<Self, Diagnostic> {
        Ok(Self {
            unknown: stage.unknown.clone(),
            solve: WireSolveHistory::from_evidence(&stage.solve)?,
            convergence: stage.convergence.clone(),
        })
    }
    fn replay(&self) -> Result<SpatialTimeStage, Diagnostic> {
        Ok(SpatialTimeStage {
            unknown: self.unknown.clone(),
            solve: self.solve.replay_metadata()?,
            convergence: self.convergence.clone(),
        })
    }
}
