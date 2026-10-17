//! Private shape and retained-numerical-storage projection for independent maps.

use super::*;
use eqiora_solver::{ExecutionReport, SolveReport};

impl DifferentiableProgram {
    pub(crate) fn validate_map_execution(&self) -> Result<(), Diagnostic> {
        validate_host_execution(&self.default.native, self.backend.default_execution())
    }

    pub(crate) fn map_binding_metadata_bytes(&self) -> Result<usize, Diagnostic> {
        self.default.native.receipt().map_or(Ok(0), |receipt| {
            Ok(receipt.binding().realization().to_bytes()?.len())
        })
    }

    pub(crate) fn prepare_map_member(
        &self,
        mut member: DifferentiableEvaluation,
    ) -> Result<DifferentiableEvaluation, Diagnostic> {
        validate_host_execution(&member.native, self.backend.default_execution())?;
        if member.native.receipt().is_some() {
            member.native = member
                .native
                .with_shared_receipt_binding(&self.default.native)?;
        }
        Ok(member)
    }

    /// Fixed acceptance metadata survives release; no state/output vectors or
    /// Jacobians are retained here. Signature inventories remain shared.
    pub(crate) fn map_evidence_bytes(&self) -> Result<usize, Diagnostic> {
        let values = self
            .identity
            .input_dimension()
            .checked_mul(size_of::<f64>())
            .ok_or_else(overflow)?;
        [
            size_of::<DifferentiationEvidence>(),
            values,
            self.identity.plan_identity.len(),
            64, // canonical Model digest
            self.identity
                .initial_state_identity
                .as_ref()
                .map_or(0, String::len),
            // Provider strings and library inventory are Plan-owned policy,
            // independent of accepted values and completed iterations.
            self.default
                .native
                .linear_solves()
                .first()
                .map_or(Ok(0), |report| {
                    eqiora_numerics::CommonSolveEvidence::from_report(report)
                        .storage_bytes()
                        .ok_or_else(overflow)
                })?,
            self.initial
                .as_ref()
                .map_or(0, |initial| initial.identity().len()),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(overflow)
        })
    }

    pub(crate) fn map_occurrence_bytes(&self) -> Result<usize, Diagnostic> {
        let n = self
            .default
            .native
            .linearization()
            .relation()
            .unknown_dimension();
        let p = self.identity.input_dimension();
        let o = self.identity.output_dimension();
        // A fixed Program retains its state/output dimensions. n*n bounds CSR
        // nonzeros even when coefficients alter sparsity; no observed/default
        // nnz count or finite-difference output supplies this bound.
        let multiply = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
        let entries = multiply(n, n)?;
        let csr_entries = multiply(entries, size_of::<usize>() + size_of::<f64>())?;
        let offsets = multiply(n.checked_add(1).ok_or_else(overflow)?, size_of::<usize>())?;
        // Accepted state, original residual and CSR RHS, plus the nonlinear seed.
        let state = multiply(n, 4 * size_of::<f64>())?;
        let design = multiply(multiply(n, p)?, size_of::<f64>())?;
        // Dense finite outputs own O_w; indexed spatial outputs own DOF indices.
        // Charge both bounds so the estimate never depends on observed sparsity.
        let output = multiply(o, size_of::<f64>() + size_of::<Option<usize>>())?;
        let output_state = multiply(multiply(o, n)?, size_of::<f64>())?;
        let updates = match &self.plan {
            ResolvedCommonPlan::Algebraic(plan) => plan
                .nonlinear()
                .map_or(0, |nonlinear| nonlinear.maximum_iterations().get()),
            _ => 0,
        };
        let reports = multiply(updates, size_of::<SolveReport>())?;
        let direct = multiply(multiply(o, p)?, size_of::<f64>())?;
        // Planned point, accepted point and relation design values. Immutable
        // signatures are shared with the Program, not cloned per occurrence.
        let point_data = multiply(p, 3 * size_of::<f64>())?;
        [
            size_of::<DifferentiableEvaluation>(),
            size_of::<ExecutionReceipt>(), // at most one boxed deployment receipt
            self.map_evidence_bytes()?,    // temporary comparison/publication metadata
            self.initial
                .as_ref()
                .map_or(0, |initial| initial.identity().len()),
            self.initial
                .as_ref()
                .map_or(0, |initial| initial.plan_identity().len()),
            size_of::<DifferentiableValues>(),
            size_of::<eqiora_solver::CanonicalCsrSystemView>(),
            size_of_val(
                self.default
                    .native
                    .linearization()
                    .relation()
                    .design_coordinates(),
            ),
            self.identity.plan_identity.len(),
            64, // canonical Model digest
            self.identity
                .initial_state_identity
                .as_ref()
                .map_or(0, String::len),
            csr_entries,
            offsets,
            state,
            design,
            output,
            output_state,
            reports,
            direct,
            point_data,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(overflow)
        })
    }
}

fn validate_host_execution(
    native: &CommonDifferentiationPoint,
    placement: ExecutionReport,
) -> Result<(), Diagnostic> {
    // A Newton seed may accept without any linear solve. Check the backend's
    // declared placement before dispatch, and every actual report on acceptance.
    if placement != ExecutionReport::host_serial()
        || native.linear_solves().iter().any(|report| {
            report.execution() != ExecutionReport::host_serial()
                || report.verification() != ExecutionReport::host_serial()
        })
        || native.receipt().is_some_and(|receipt| {
            receipt.acceptance_verification() != ExecutionReport::host_serial()
                || receipt.cuda_trace().is_some()
                || receipt.distributed_trace().is_some()
        })
    {
        return Err(invalid(
            "bounded maps require serial host execution for each member",
        ));
    }
    Ok(())
}

fn overflow() -> Diagnostic {
    invalid("evaluation map retained numerical storage estimate overflows usize")
}
