use super::*;
use crate::common_result::evidence::CommonExecutionTopology;

impl CommonSpatialExecution {
    pub(super) fn validate(
        &self,
        plan: &CommonLinearPlan,
        residual_target: f64,
        packets: usize,
        targets: usize,
    ) -> Result<(), Diagnostic> {
        let expected = plan.linear();
        let solve = &self.solve;
        if solve.orientation() != eqiora_solver::LinearOperatorOrientation::Normal
            || solve.algorithm() != expected.algorithm()
            || solve.preconditioner() != expected.preconditioner()
            || solve.reduction() != expected.reduction()
            || solve.relative_tolerance().to_bits() != expected.relative_tolerance().to_bits()
            || solve.absolute_tolerance().to_bits() != expected.absolute_tolerance().to_bits()
            || solve.maximum_iterations() != expected.maximum_iterations().get()
            || solve.completed_iterations() > expected.maximum_iterations().get()
            || solve.residual_target().to_bits() != residual_target.to_bits()
        {
            return Err(invalid("step solve metadata differs from its exact Plan"));
        }
        let workers = plan.admission.linear.workers.get();
        for topology in [
            solve.execution().topology(),
            solve.verification().topology(),
            self.assembly.topology(),
        ] {
            if !matches!(topology, CommonExecutionTopology::Host { workers: actual } if actual > 0 && actual <= workers)
            {
                return Err(invalid("step execution exceeds its host worker contract"));
            }
        }
        if self.assembly.packet_count() != packets || self.assembly.target_count() != targets {
            return Err(invalid(
                "step assembly metadata differs from its exact reconstruction",
            ));
        }
        // Provider identity and iteration counts remain producer metadata. The
        // supplied physical coefficients were checked independently beforehand.
        Ok(())
    }
}
