//! Observations of actual linear solves and original nonlinear acceptance.
//!
//! `output_index` selects a retained trajectory output, or `None` for a static
//! Result. Linear-solve accessors take `iteration = None` for a primal linear
//! solve and `Some(i)` for the zero-based Newton correction. Missing records,
//! invalid indices and zero-update Newton solves never fabricate a linear receipt.
use super::*;

impl CommonResult {
    fn solve_history(&self, output_index: Option<usize>) -> Option<&CommonSolveHistory> {
        match (&self.payload, output_index) {
            (CommonResultPayload::Algebraic { solve, .. }, None) => Some(solve),
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => Some(
                &trajectory
                    .spatial_steps()?
                    .get(index)?
                    .1
                    .execution
                    .as_ref()?
                    .solve,
            ),
            _ => None,
        }
    }

    fn solve_evidence(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<&CommonSolveEvidence> {
        if let (CommonResultPayload::Static(payload), None, None) =
            (&self.payload, output_index, iteration)
        {
            return Some(&payload.solve);
        }
        match (self.solve_history(output_index)?, iteration) {
            (CommonSolveHistory::Linear(solve), None) => Some(solve),
            (CommonSolveHistory::Newton { linear_solves, .. }, Some(index)) => {
                linear_solves.get(index)
            }
            _ => None,
        }
    }

    #[must_use]
    pub fn solve_solver_id(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<&str> {
        self.solve_evidence(output_index, iteration)
            .map(|solve| solve.solver().id())
    }
    #[must_use]
    pub fn solve_execution_adapter(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<&str> {
        self.solve_evidence(output_index, iteration)
            .map(|solve| solve.execution().adapter())
    }
    #[must_use]
    pub fn solve_verification_adapter(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<&str> {
        self.solve_evidence(output_index, iteration)
            .map(|solve| solve.verification().adapter())
    }
    #[must_use]
    pub fn solve_orientation(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<LinearOperatorOrientation> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::orientation)
    }
    #[must_use]
    pub fn solve_algorithm(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<LinearSolver> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::algorithm)
    }
    #[must_use]
    pub fn solve_preconditioner(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<PreconditionerPolicy> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::preconditioner)
    }
    #[must_use]
    pub fn solve_reduction(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<ReductionPolicy> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::reduction)
    }
    #[must_use]
    pub fn solve_relative_tolerance(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::relative_tolerance)
    }
    #[must_use]
    pub fn solve_absolute_tolerance(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::absolute_tolerance)
    }
    #[must_use]
    pub fn solve_maximum_iterations(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<usize> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::maximum_iterations)
    }
    #[must_use]
    pub fn solve_reason(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<ConvergenceReason> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::reason)
    }
    #[must_use]
    pub fn solve_completed_iterations(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<usize> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::completed_iterations)
    }
    #[must_use]
    pub fn solve_initial_residual_norm(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::initial_residual_norm)
    }
    #[must_use]
    pub fn solve_reported_residual_norm(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::reported_residual_norm)
    }
    #[must_use]
    pub fn solve_true_residual_norm(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        let solve = self.solve_evidence(output_index, iteration)?;
        if iteration.is_none()
            && let (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) =
                (&self.payload, output_index)
            && let Some((norm, _)) = trajectory.spatial_steps()?.get(index)?.1.residual
        {
            return Some(norm);
        }
        Some(solve.true_residual_norm())
    }
    #[must_use]
    pub fn solve_residual_target(
        &self,
        output_index: Option<usize>,
        iteration: Option<usize>,
    ) -> Option<f64> {
        self.solve_evidence(output_index, iteration)
            .map(CommonSolveEvidence::residual_target)
    }

    /// Number of accepted Newton updates at a static Result or retained output position.
    #[must_use]
    pub fn nonlinear_iterations(&self, output_index: Option<usize>) -> Option<usize> {
        match self.solve_history(output_index)? {
            CommonSolveHistory::Newton { iterations, .. } => Some(*iterations),
            CommonSolveHistory::Linear(_) => None,
        }
    }

    /// Original-equation residual at the exact Newton predictor.
    #[must_use]
    pub fn nonlinear_initial_residual_norm(&self, output_index: Option<usize>) -> Option<f64> {
        match self.solve_history(output_index)? {
            CommonSolveHistory::Newton {
                initial_residual_norm,
                ..
            } => Some(*initial_residual_norm),
            CommonSolveHistory::Linear(_) => None,
        }
    }

    /// Original-equation residual independently checked at the accepted solution.
    #[must_use]
    pub fn nonlinear_true_residual_norm(&self, output_index: Option<usize>) -> Option<f64> {
        self.nonlinear_iterations(output_index)?;
        match (&self.payload, output_index) {
            (
                CommonResultPayload::Algebraic {
                    reference_residual_norm,
                    ..
                },
                None,
            ) => Some(*reference_residual_norm),
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => {
                Some(trajectory.spatial_steps()?.get(index)?.1.residual?.0)
            }
            _ => None,
        }
    }

    /// Newton target derived from the exact predictor and the Plan's controls.
    #[must_use]
    pub fn nonlinear_residual_target(&self, output_index: Option<usize>) -> Option<f64> {
        self.nonlinear_iterations(output_index)?;
        match (&self.payload, output_index) {
            (CommonResultPayload::Algebraic { assessment, .. }, None) => {
                Some(assessment.as_ref()?.residual_target())
            }
            (CommonResultPayload::Trajectory { trajectory, .. }, Some(index)) => {
                Some(trajectory.spatial_steps()?.get(index)?.1.residual?.1)
            }
            _ => None,
        }
    }
}
