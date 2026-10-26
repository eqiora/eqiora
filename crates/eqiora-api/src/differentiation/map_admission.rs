//! Private shape and retained-numerical-storage projection for independent maps.

use super::*;
use eqiora_solver::{ExecutionReport, SolveReport};

impl DifferentiableProgram {
    pub(crate) fn validate_map_execution(&self) -> Result<(), Diagnostic> {
        self.default
            .native
            .points()
            .iter()
            .try_for_each(|point| validate_host_execution(point, self.backend.default_execution()))
    }

    pub(crate) fn map_binding_metadata_bytes(&self) -> Result<usize, Diagnostic> {
        self.default
            .native
            .points()
            .iter()
            .try_fold(0usize, |total, point| {
                let bytes = point.receipt().map_or(Ok(0), |receipt| {
                    Ok(receipt.binding().realization().to_bytes()?.len())
                })?;
                total.checked_add(bytes).ok_or_else(overflow)
            })
    }

    pub(crate) fn prepare_map_member(
        &self,
        mut member: DifferentiableEvaluation,
    ) -> Result<DifferentiableEvaluation, Diagnostic> {
        for point in member.native.points() {
            validate_host_execution(point, self.backend.default_execution())?;
        }
        member.native = member
            .native
            .with_shared_receipt_binding(&self.default.native)?;
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
        let updates = match self.plan().canonical_solve_request() {
            Some(eqiora_numerics::CommonSolvePolicy::Newton { nonlinear, .. }) => {
                nonlinear.maximum_iterations().get()
            }
            _ => 1,
        };
        let mut bytes = size_of::<DifferentiationEvidence>()
            .checked_add(values)
            .ok_or_else(overflow)?;
        let report = eqiora_numerics::CommonSolveEvidence::planned_storage_bytes(self.plan())
            .ok_or_else(overflow)?;
        for point in self.default.native.points() {
            let groups = primal_group_count(point);
            let step = size_of::<DifferentiationStepEvidence>()
                .checked_add(product(
                    groups,
                    sum_bytes(&[
                        size_of::<super::evidence::DifferentiationPrimalEvidence>(),
                        product(report, updates)?,
                    ])?,
                )?)
                .ok_or_else(overflow)?;
            bytes = bytes.checked_add(step).ok_or_else(overflow)?;
        }
        for extra in [
            // The nonlinear primal owns its seed identity separately from the Program identity.
            self.context
                .initial()
                .as_ref()
                .map_or(0, |initial| initial.identity().len()),
            self.identity.plan_identity.len(),
            64,
            self.identity
                .run_request_identity
                .as_ref()
                .map_or(0, String::len),
            self.identity
                .initial_state_identity
                .as_ref()
                .map_or(0, String::len),
        ] {
            bytes = bytes.checked_add(extra).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }

    pub(crate) fn map_occurrence_bytes(&self) -> Result<usize, Diagnostic> {
        let p = self.identity.input_dimension();
        let updates = match self.plan().canonical_solve_request() {
            Some(eqiora_numerics::CommonSolvePolicy::Newton { nonlinear, .. }) => {
                nonlinear.maximum_iterations().get()
            }
            _ => 1,
        };
        let initial_payload =
            self.default
                .native
                .points()
                .iter()
                .try_fold(0usize, |total, point| {
                    let bytes = point
                        .nonlinear_initial_state()
                        .map_or(Ok(0), CommonInitialState::retained_payload_bytes)?;
                    total.checked_add(bytes).ok_or_else(overflow)
                })?;
        let request_payload = match &self.default.native {
            NativeEvaluation::Point(_) => 0,
            NativeEvaluation::Trajectory(trajectory) => sum_bytes(&[
                size_of_val(trajectory.as_ref()),
                CommonInitialState::from(trajectory.request().state()).retained_payload_bytes()?,
                size_of_val(trajectory.request().output_times_s()),
                trajectory.request().identity().len(),
            ])?,
        };
        let mut bytes = sum_bytes(&[
            size_of::<DifferentiableEvaluation>(),
            self.map_evidence_bytes()?,
            initial_payload,
            request_payload,
            self.default.projection.retained_payload_bytes(),
            product(self.default.unknowns.len(), size_of::<f64>())?,
            product(p, 2 * size_of::<f64>())?,
            self.identity.plan_identity.len(),
            64,
            self.identity
                .run_request_identity
                .as_ref()
                .map_or(0, String::len),
            self.identity
                .initial_state_identity
                .as_ref()
                .map_or(0, String::len),
        ])?;
        // Sum each independent step's numerical bound. Squaring the complete
        // trajectory dimension would charge nonexistent cross-step dense blocks.
        for point in self.default.native.points() {
            let linearization = point.linearization();
            let relation = linearization.relation();
            let n = relation.unknown_dimension();
            let groups = primal_group_count(point);
            let o = linearization.output_dimension();
            let history = linearization.discrete_step().map_or(0, |step| {
                eqiora_ir::DiscreteStepLinearization::state_coordinates(&step).len()
            });
            let step = sum_bytes(&[
                size_of::<CommonDifferentiationPoint>(),
                size_of::<ExecutionReceipt>(),
                size_of::<eqiora_solver::CanonicalCsrSystemView>(),
                size_of_val(relation.design_coordinates()),
                product(product(n, n)?, size_of::<usize>() + size_of::<f64>())?,
                product(n.checked_add(1).ok_or_else(overflow)?, size_of::<usize>())?,
                product(n, 3 * size_of::<f64>())?,
                product(product(n, p)?, size_of::<f64>())?,
                product(o, size_of::<f64>() + size_of::<Option<usize>>())?,
                product(product(o, n)?, size_of::<f64>())?,
                product(product(o, p)?, size_of::<f64>())?,
                product(product(updates, groups)?, size_of::<SolveReport>())?,
                // Retained multi-solve candidate and correction acceptance own
                // coordinate witnesses and sparse original correction systems.
                if groups > 1
                    || point.linearization().observation_times().is_some()
                        && matches!(
                            point.linearization().acceptance(),
                            eqiora_differentiation::PrimalAcceptance::Blocks(_)
                        )
                {
                    sum_bytes(&[
                        // Each retained stage and acceptance block owns metadata
                        // independently of its numerical coordinate buffers.
                        product(
                            groups,
                            size_of::<eqiora_numerics::CommonDifferentiationCandidate>(),
                        )?,
                        product(
                            groups,
                            size_of::<(usize, eqiora_differentiation::PrimalAcceptance)>(),
                        )?,
                        product(groups, size_of::<eqiora_solver::CanonicalCsrSystemView>())?,
                        product(n, 8 * size_of::<f64>())?,
                        product(product(n, n)?, size_of::<f64>() + size_of::<usize>())?,
                        product(
                            n.checked_add(groups).ok_or_else(overflow)?,
                            size_of::<usize>(),
                        )?,
                        product(
                            product(updates, groups)?,
                            eqiora_numerics::CommonSolveEvidence::planned_storage_bytes(
                                self.plan(),
                            )
                            .ok_or_else(overflow)?,
                        )?,
                    ])?
                } else {
                    0
                },
                product(p, size_of::<f64>())?,
                // The history CSR owns a separate row-offset array and shared coordinate identity.
                if history == 0 {
                    0
                } else {
                    sum_bytes(&[
                        product(n.checked_add(1).ok_or_else(overflow)?, size_of::<usize>())?,
                        self.identity.plan_identity.len(),
                    ])?
                },
                // Sparse predecessor action and complete-state output lift.
                product(
                    product(history, n)?,
                    2 * (size_of::<f64>() + size_of::<usize>()),
                )?,
                product(product(history, p)?, size_of::<f64>())?,
                product(
                    history,
                    4 * size_of::<f64>()
                        + size_of::<Option<usize>>()
                        + size_of::<eqiora_numerics::SpatialStepCoordinate>(),
                )?,
            ])?;
            bytes = bytes.checked_add(step).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
}

fn product(a: usize, b: usize) -> Result<usize, Diagnostic> {
    a.checked_mul(b).ok_or_else(overflow)
}

fn sum_bytes(values: &[usize]) -> Result<usize, Diagnostic> {
    values.iter().try_fold(0usize, |total, value| {
        total.checked_add(*value).ok_or_else(overflow)
    })
}

fn validate_host_execution(
    native: &CommonDifferentiationPoint,
    placement: ExecutionReport,
) -> Result<(), Diagnostic> {
    // A Newton seed may accept without any linear solve. Check the backend's
    // declared placement before dispatch, and every actual report on acceptance.
    if placement != ExecutionReport::host_serial()
        || native.linear_solves().iter().any(|report| {
            report.execution().topology()
                != eqiora_numerics::CommonExecutionTopology::Host { workers: 1 }
                || report.verification().topology()
                    != eqiora_numerics::CommonExecutionTopology::Host { workers: 1 }
                || report.execution().adapter() != ExecutionReport::host_serial().adapter().as_str()
                || report.verification().adapter()
                    != ExecutionReport::host_serial().adapter().as_str()
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

fn primal_group_count(point: &CommonDifferentiationPoint) -> usize {
    match point.linearization().acceptance() {
        eqiora_differentiation::PrimalAcceptance::Blocks(blocks) => blocks.len(),
        _ => 1,
    }
}
