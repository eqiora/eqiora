//! Canonical finite solve records revalidated against the exact seed and Plan.
use super::*;
use crate::finite_constraints::ConstraintAssessment;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireAlgebraicSolve {
    Linear {
        solve: Box<WireSolve>,
    },
    Newton {
        initial_residual_norm: f64,
        iterations: usize,
        linear_solves: Vec<WireSolve>,
    },
}

impl WireAlgebraicSolve {
    pub(super) fn from_evidence(value: &AlgebraicSolveEvidence) -> Result<Self, Diagnostic> {
        Ok(match value {
            AlgebraicSolveEvidence::Linear(solve) => Self::Linear {
                solve: Box::new(WireSolve::from_solve(solve)?),
            },
            AlgebraicSolveEvidence::Newton {
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
    pub(super) fn replay(
        &self,
        plan: &ResolvedCommonPlan,
        state: &crate::CommonAlgebraicState,
        values: &[f64],
        mask: Option<u32>,
    ) -> Result<(AlgebraicSolveEvidence, f64, Option<ConstraintAssessment>), Diagnostic> {
        let native = plan
            .as_algebraic()
            .ok_or_else(|| invalid("finite solve requires finite Plan"))?;
        match self {
            Self::Linear { solve } => {
                if native.nonlinear().is_some() {
                    return Err(invalid("linear Result cannot belong to a Newton Plan"));
                }
                let solve = solve.replay()?;
                require_plan_solver(plan, &solve)?;
                let (norm, assessment) =
                    native.validate_values(values, solve.residual_target(), mask)?;
                Ok((
                    AlgebraicSolveEvidence::Linear(Box::new(solve)),
                    norm,
                    assessment,
                ))
            }
            Self::Newton {
                initial_residual_norm,
                iterations,
                linear_solves,
            } => {
                let nonlinear = native
                    .nonlinear()
                    .ok_or_else(|| invalid("Newton Result requires Newton Plan"))?;
                if mask.is_some()
                    || *iterations != linear_solves.len()
                    || *iterations > nonlinear.maximum_iterations().get()
                {
                    return Err(invalid(
                        "Newton Result has an active set or inconsistent iteration count",
                    ));
                }
                let (derived_initial, assessment) =
                    native.validate_nonlinear_values(state, values)?;
                if derived_initial.to_bits() != initial_residual_norm.to_bits() {
                    return Err(invalid(
                        "Newton Result initial residual differs from the original State",
                    ));
                }
                crate::common_result::algebraic::validate_iterations(
                    state,
                    values,
                    derived_initial,
                    assessment.residual_target(),
                    *iterations,
                )?;
                let reports = linear_solves
                    .iter()
                    .map(|report| {
                        let report = report.replay()?;
                        require_plan_solver(plan, &report)?;
                        Ok(report)
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                let norm = assessment.equality_residual_norm();
                Ok((
                    AlgebraicSolveEvidence::Newton {
                        initial_residual_norm: derived_initial,
                        iterations: *iterations,
                        linear_solves: reports,
                    },
                    norm,
                    Some(assessment),
                ))
            }
        }
    }
}
