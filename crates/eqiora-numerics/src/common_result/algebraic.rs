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
        validate_iterations(
            state,
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
            artifact::require_plan_solver(&resolved, report)?;
        }
        Self {
            plan: resolved,
            family: CommonResultFamily::Algebraic,
            elapsed_seconds: 0.0,
            identity: String::new(),
            payload: CommonResultPayload::Algebraic {
                values: solution.values,
                solve: AlgebraicSolveEvidence::Newton {
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

    /// Initial original-equation residual norm for Newton execution.
    #[must_use]
    pub fn nonlinear_initial_residual_norm(&self) -> Option<f64> {
        match &self.payload {
            CommonResultPayload::Algebraic {
                solve:
                    AlgebraicSolveEvidence::Newton {
                        initial_residual_norm,
                        ..
                    },
                ..
            } => Some(*initial_residual_norm),
            _ => None,
        }
    }
    /// Nonlinear target derived from the exact initial State and Newton controls.
    #[must_use]
    pub fn nonlinear_residual_target(&self) -> Option<f64> {
        self.nonlinear_iterations()?;
        self.constraint_assessment()
            .map(|assessment| assessment.residual_target())
    }

    /// Number of accepted nonlinear updates, absent for linear Results.
    #[must_use]
    pub fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.payload {
            CommonResultPayload::Algebraic {
                solve: AlgebraicSolveEvidence::Newton { iterations, .. },
                ..
            } => Some(*iterations),
            _ => None,
        }
    }
}

pub(super) fn validate_iterations(
    state: &crate::CommonAlgebraicState,
    values: &[f64],
    initial_norm: f64,
    target: f64,
    iterations: usize,
) -> Result<(), Diagnostic> {
    if (iterations == 0) != (initial_norm <= target)
        || (iterations == 0 && values != state.values())
    {
        return Err(invalid(
            "nonlinear zero-update record differs from initial State acceptance",
        ));
    }
    Ok(())
}
