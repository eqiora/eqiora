//! Canonical finite solve records revalidated against the exact seed and Plan.
use super::*;
use crate::finite_constraints::ConstraintAssessment;

impl WireSolveHistory {
    pub(super) fn replay(
        &self,
        plan: &ResolvedCommonPlan,
        state: &crate::CommonAlgebraicState,
        values: &[f64],
        mask: Option<u32>,
        nullspace: Option<&crate::nullspace::NullspaceEvidence>,
    ) -> Result<(CommonSolveHistory, f64, Option<ConstraintAssessment>), Diagnostic> {
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
                    native.validate_values(values, solve.residual_target(), mask, nullspace)?;
                Ok((
                    CommonSolveHistory::Linear(Box::new(solve)),
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
                if nullspace.is_some()
                    || mask.is_some()
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
                    CommonSolveHistory::Newton {
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
