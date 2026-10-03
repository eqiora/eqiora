//! Private shape and retained-numerical-storage projection for independent maps.

use super::*;
use eqiora_solver::LinearSolverBackend;
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

impl DifferentiableProgram {
    pub(crate) fn validate_map_provider(&self) -> Result<(), Diagnostic> {
        let receipt = self
            .default
            .native
            .receipt()
            .ok_or_else(|| invalid("bounded maps require an accepted linear point"))?;
        if receipt.solver_provider() != REFERENCE_LINEAR_SOLVER.provider()
            || receipt.report().execution() != eqiora_solver::ExecutionReport::host_serial()
            || receipt.report().verification() != eqiora_solver::ExecutionReport::host_serial()
            || receipt.acceptance_verification() != eqiora_solver::ExecutionReport::host_serial()
            || receipt.cuda_trace().is_some()
            || receipt.distributed_trace().is_some()
        {
            return Err(invalid(
                "bounded maps require the one-thread reference host provider",
            ));
        }
        Ok(())
    }

    pub(crate) fn map_binding_metadata_bytes(&self) -> Result<usize, Diagnostic> {
        Ok(self
            .default
            .native
            .receipt()
            .ok_or_else(|| invalid("bounded maps require an accepted linear point"))?
            .binding()
            .realization()
            .to_bytes()?
            .len())
    }

    pub(crate) fn share_map_receipt(
        &self,
        mut member: DifferentiableEvaluation,
    ) -> Result<DifferentiableEvaluation, Diagnostic> {
        member.native = member
            .native
            .with_shared_receipt_binding(&self.default.native)?;
        Ok(member)
    }
    pub(crate) fn validate_map_point(&self, values: &[f64]) -> Result<(), Diagnostic> {
        if values.len() != self.identity.input_dimension()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(format!(
                "evaluation map point must contain {} finite values in the program's Parameter order",
                self.identity.input_dimension()
            )));
        }
        Ok(())
    }

    /// Called only after all map points and the total resource estimate pass.
    pub(crate) fn map_point(&self, values: &[f64]) -> DifferentiableParameterPoint {
        DifferentiableParameterPoint {
            inputs: self.identity.inputs.clone(),
            values: values.to_vec(),
        }
    }

    pub(crate) fn map_occurrence_bytes(&self) -> Result<usize, Diagnostic> {
        let n = self.default.native.relation().unknown_dimension();
        let p = self.identity.input_dimension();
        let o = self.identity.output_dimension();
        // A fixed Program retains its state/output dimensions. n*n bounds CSR
        // nonzeros even when coefficients alter sparsity; no observed/default
        // nnz count or finite-difference output supplies this bound.
        let multiply = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
        let entries = multiply(n, n)?;
        let csr_entries = multiply(entries, size_of::<usize>() + size_of::<f64>())?;
        let offsets = multiply(n.checked_add(1).ok_or_else(overflow)?, size_of::<usize>())?;
        let state = multiply(n, 2 * size_of::<f64>())?; // accepted state and CSR RHS
        let design = multiply(multiply(n, p)?, size_of::<f64>())?;
        let output = multiply(o, size_of::<f64>() + size_of::<Option<usize>>())?;
        let direct = multiply(multiply(o, p)?, size_of::<f64>())?;
        // Planned point, accepted point and relation design values; two point
        // inventories plus the member identity's ordered Parameter inventory.
        let point_data = multiply(
            p,
            3 * (size_of::<f64>() + size_of::<Id<kinds::Parameter>>()),
        )?;
        [
            size_of::<DifferentiableEvaluation>(),
            size_of::<ExecutionReceipt>(), // the accepted linear point's one boxed receipt
            size_of::<DifferentiableParameterPoint>(),
            size_of::<eqiora_solver::CanonicalCsrSystemView>(),
            size_of_val(self.default.native.relation().design_coordinates()),
            self.identity.plan_identity.len(),
            64, // canonical Model digest
            csr_entries,
            offsets,
            state,
            design,
            output,
            direct,
            point_data,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(overflow)
        })
    }
}

impl DifferentiableEvaluation {
    pub(crate) fn map_receipt(&self) -> &ExecutionReceipt {
        self.native
            .receipt()
            .expect("bounded map admission requires a linear point")
    }
}

fn overflow() -> Diagnostic {
    invalid("evaluation map retained numerical storage estimate overflows usize")
}
