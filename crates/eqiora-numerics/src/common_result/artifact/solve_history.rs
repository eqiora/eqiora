//! Shared linear/Newton producer history; each consumer rechecks its original equations.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireSolveHistory {
    Linear {
        solve: Box<WireSolve>,
    },
    Newton {
        initial_residual_norm: f64,
        iterations: usize,
        linear_solves: Vec<WireSolve>,
    },
}

impl WireSolveHistory {
    pub(crate) fn from_evidence(value: &CommonSolveHistory) -> Result<Self, Diagnostic> {
        Ok(match value {
            CommonSolveHistory::Linear(solve) => Self::Linear {
                solve: Box::new(WireSolve::from_solve(solve)?),
            },
            CommonSolveHistory::Newton {
                initial_residual_norm,
                iterations,
                linear_solves,
            } => Self::Newton {
                initial_residual_norm: *initial_residual_norm,
                iterations: *iterations,
                linear_solves: linear_solves
                    .iter()
                    .map(WireSolve::from_solve)
                    .collect::<Result<_, _>>()?,
            },
        })
    }
    pub(crate) fn replay_metadata(&self) -> Result<CommonSolveHistory, Diagnostic> {
        Ok(match self {
            Self::Linear { solve } => CommonSolveHistory::Linear(Box::new(solve.replay()?)),
            Self::Newton {
                initial_residual_norm,
                iterations,
                linear_solves,
            } => {
                require_finite_nonnegative(&[*initial_residual_norm], "Newton initial residual")?;
                if *iterations != linear_solves.len() {
                    return Err(invalid(
                        "Newton iteration count differs from its linear solve history",
                    ));
                }
                CommonSolveHistory::Newton {
                    initial_residual_norm: *initial_residual_norm,
                    iterations: *iterations,
                    linear_solves: linear_solves
                        .iter()
                        .map(WireSolve::replay)
                        .collect::<Result<_, _>>()?,
                }
            }
        })
    }
}
