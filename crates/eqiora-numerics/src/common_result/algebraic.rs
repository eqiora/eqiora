//! Nonlinear Result acceptance retains its seed and original condition assessment.
use super::*;
use crate::finite_constraints::FiniteNonlinearSolution;

impl CommonResult {
    pub(crate) fn from_nonlinear_algebraic(
        plan: &crate::CommonAlgebraicPlan,
        state: &crate::CommonAlgebraicState,
        solution: FiniteNonlinearSolution,
    ) -> Result<Self, Diagnostic> {
        let (initial_residual_norm, assessment) =
            plan.validate_nonlinear_values(state, &solution.values)?;
        if initial_residual_norm.to_bits() != solution.initial_residual_norm.to_bits()
            || assessment != solution.assessment
            || solution.iterations != solution.linear_solves.len()
            || solution.iterations
                > plan
                    .nonlinear()
                    .expect("validated Newton Plan")
                    .maximum_iterations()
                    .get()
        {
            return Err(invalid(
                "nonlinear finite Result has inconsistent acceptance records",
            ));
        }
        state.validate_nonlinear_update_count(
            &solution.values,
            initial_residual_norm,
            assessment.residual_target(),
            solution.iterations,
        )?;
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        let linear_solves = solution
            .linear_solves
            .iter()
            .map(CommonSolveEvidence::from_report)
            .collect::<Vec<_>>();
        for report in &linear_solves {
            report.validate_plan(&resolved)?;
        }
        Self {
            plan: resolved,
            family: CommonResultFamily::Algebraic,
            elapsed_seconds: 0.0,
            identity: String::new(),
            payload: CommonResultPayload::Algebraic {
                values: solution.values,
                solve: CommonSolveHistory::Newton {
                    initial_residual_norm,
                    iterations: solution.iterations,
                    linear_solves,
                },
                initial_state: state.clone(),
                reference_residual_norm: assessment.equality_residual_norm(),
                assessment: Some(assessment),
                nullspace: None,
            },
        }
        .refresh_identity()
    }
}
