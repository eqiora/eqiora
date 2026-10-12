use super::*;
use crate::common_result::artifact::{require_reference_assembly, require_solver_contract};

impl CommonSpatialExecution {
    pub(super) fn validate(
        &self,
        plan: &CommonSpatialPlan,
        residual_target: f64,
        packets: usize,
        targets: usize,
        derived_initial: Option<f64>,
    ) -> Result<(), Diagnostic> {
        let expected = plan.linear();
        use crate::common_result::evidence::CommonSolveHistory;
        let (reports, linear_target): (&[_], _) = match (
            &self.solve,
            plan.admission.nonlinear,
            derived_initial,
        ) {
            (CommonSolveHistory::Linear(solve), None, None) => {
                (std::slice::from_ref(solve.as_ref()), Some(residual_target))
            }
            (
                CommonSolveHistory::Newton {
                    initial_residual_norm,
                    iterations,
                    linear_solves,
                },
                Some(nonlinear),
                Some(initial),
            ) => {
                if initial_residual_norm.to_bits() != initial.to_bits()
                    || *iterations != linear_solves.len()
                    || *iterations > nonlinear.maximum_iterations().get()
                {
                    return Err(invalid(
                        "Newton step metadata differs from its original predictor or iteration bound",
                    ));
                }
                (linear_solves, None)
            }
            _ => {
                return Err(invalid(
                    "step solve metadata differs from its linear/Newton strategy",
                ));
            }
        };
        let workers = plan.admission.linear.workers.get();
        for solve in reports {
            require_solver_contract(
                expected,
                plan.admission.linear.provider,
                plan.admission.linear.execution,
                workers,
                solve,
            )?;
            if solve.orientation() != eqiora_solver::LinearOperatorOrientation::Normal
                || solve.completed_iterations() > expected.maximum_iterations().get()
                || linear_target
                    .is_some_and(|target| solve.residual_target().to_bits() != target.to_bits())
            {
                return Err(invalid("step solve metadata differs from its exact Plan"));
            }
        }
        require_reference_assembly(&self.assembly)?;
        if self.assembly.packet_count() != packets || self.assembly.target_count() != targets {
            return Err(invalid(
                "step assembly metadata differs from its exact reconstruction",
            ));
        }
        // Provider metadata must match the exact Plan. Iteration counts remain
        // producer metadata; physical coefficients were checked independently.
        Ok(())
    }
}
