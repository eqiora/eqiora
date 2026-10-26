//! Untrusted retained primal inputs, reaccepted by the original numerical owners.
use crate::common_result::evidence::CommonSolveHistory;
use crate::common_trajectory::{SpatialTimeStage, spatial_time::artifact::WireSpatialTimeStage};
use crate::nonlinear_execution::NewtonConvergence;
use serde::{Deserialize, Serialize};

/// Retained coordinates and method-owned stopping witnesses for solve-free reopening.
/// Deserialization does not establish numerical acceptance or authenticate a producer.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationCandidate {
    pub(in crate::numerical_admission) kind: Candidate,
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) enum Candidate {
    Coordinates(Vec<f64>),
    Midpoint {
        stage: SpatialTimeStage,
        endpoint: Option<Box<SpatialTimeStage>>,
        observations: Vec<SpatialTimeStage>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireCandidate {
    Coordinates {
        unknown: Vec<f64>,
    },
    Midpoint {
        stage: WireSpatialTimeStage,
        endpoint: Option<Box<WireSpatialTimeStage>>,
        observations: Vec<WireSpatialTimeStage>,
    },
}

impl Serialize for CommonDifferentiationCandidate {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let wire = match &self.kind {
            Candidate::Coordinates(unknown) => WireCandidate::Coordinates {
                unknown: unknown.clone(),
            },
            Candidate::Midpoint {
                stage,
                endpoint,
                observations,
            } => WireCandidate::Midpoint {
                stage: WireSpatialTimeStage::capture(stage).map_err(S::Error::custom)?,
                endpoint: endpoint
                    .as_deref()
                    .map(WireSpatialTimeStage::capture)
                    .transpose()
                    .map_err(S::Error::custom)?
                    .map(Box::new),
                observations: observations
                    .iter()
                    .map(WireSpatialTimeStage::capture)
                    .collect::<Result<_, _>>()
                    .map_err(S::Error::custom)?,
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CommonDifferentiationCandidate {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let kind = match WireCandidate::deserialize(deserializer)? {
            WireCandidate::Coordinates { unknown } => Candidate::Coordinates(unknown),
            WireCandidate::Midpoint {
                stage,
                endpoint,
                observations,
            } => Candidate::Midpoint {
                stage: stage.replay().map_err(D::Error::custom)?,
                endpoint: endpoint
                    .as_deref()
                    .map(WireSpatialTimeStage::replay)
                    .transpose()
                    .map_err(D::Error::custom)?
                    .map(Box::new),
                observations: observations
                    .iter()
                    .map(WireSpatialTimeStage::replay)
                    .collect::<Result<_, _>>()
                    .map_err(D::Error::custom)?,
            },
        };
        Ok(Self { kind })
    }
}

impl CommonDifferentiationCandidate {
    /// Plain implicit coordinates have no retained multi-solve stopping witnesses.
    #[must_use]
    pub fn from_coordinates(unknown: Vec<f64>) -> Self {
        Self {
            kind: Candidate::Coordinates(unknown),
        }
    }

    /// Plain coordinates, absent for a method-owned multi-solve candidate.
    #[must_use]
    pub fn coordinates(&self) -> Option<&[f64]> {
        match &self.kind {
            Candidate::Coordinates(values) => Some(values),
            _ => None,
        }
    }

    fn stage(&self, index: usize) -> Option<&SpatialTimeStage> {
        match &self.kind {
            Candidate::Coordinates(_) => None,
            Candidate::Midpoint {
                stage,
                endpoint,
                observations,
            } => std::iter::once(stage)
                .chain(endpoint.as_deref())
                .chain(observations)
                .nth(index),
        }
    }

    /// Ordered original solves: midpoint stage, optional endpoint, then observations.
    #[must_use]
    pub fn solve_count(&self) -> usize {
        match &self.kind {
            Candidate::Coordinates(_) => 0,
            Candidate::Midpoint {
                endpoint,
                observations,
                ..
            } => 1 + usize::from(endpoint.is_some()) + observations.len(),
        }
    }

    /// Actual accepted update count for one retained solve, including zero.
    #[must_use]
    pub fn iterations(&self, solve: usize) -> Option<usize> {
        match &self.stage(solve)?.solve {
            CommonSolveHistory::Newton { iterations, .. } => Some(*iterations),
            _ => None,
        }
    }

    /// Original seed residual for one retained solve.
    #[must_use]
    pub fn initial_residual_norm(&self, solve: usize) -> Option<f64> {
        match &self.stage(solve)?.solve {
            CommonSolveHistory::Newton {
                initial_residual_norm,
                ..
            } => Some(*initial_residual_norm),
            _ => None,
        }
    }

    /// Reports remain grouped by the original solve, without reconstructing live providers.
    #[must_use]
    pub fn linear_solves(&self, solve: usize) -> Option<&[crate::CommonSolveEvidence]> {
        match &self.stage(solve)?.solve {
            CommonSolveHistory::Newton { linear_solves, .. } => Some(linear_solves),
            CommonSolveHistory::Linear(solve) => Some(std::slice::from_ref(solve)),
        }
    }

    /// A physical correction criterion has no residual threshold.
    #[must_use]
    pub fn residual_tolerance(&self, solve: usize) -> Option<f64> {
        match self.stage(solve)?.convergence {
            NewtonConvergence::Residual { target } => Some(target),
            NewtonConvergence::ExactResidualZero => Some(0.),
            NewtonConvergence::Correction { .. } => None,
        }
    }
}
