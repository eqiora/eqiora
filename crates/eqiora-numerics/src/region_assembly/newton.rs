//! Region Newton execution uses the common globalization and linear solver owners.
use super::nonlinear::RegionAssemblyLinearization;
use super::*;
use crate::nonlinear_execution::{NewtonEvidence, NewtonTrial, solve_damped_newton};
use eqiora_assembly::{AssemblyTargetId, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_realization::{NonlinearSolvePlan, Target, VectorLayoutKind};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties, LinearSolveRequest};
use std::{num::NonZeroUsize, sync::Arc};

pub(crate) struct RegionNewtonSolution {
    pub(crate) values: Vec<f64>,
    pub(crate) residual_norm: f64,
    pub(crate) evidence: NewtonEvidence,
    pub(crate) assembly: RegionAssemblyLinearization,
}

struct RegionNewtonPoint {
    values: Vec<f64>,
    assembly: RegionAssemblyLinearization,
}

impl<S: Coefficient + Send + Sync, P: Projection<S, Scalar = f64>> PreparedRegionAssembly<S, P> {
    #[allow(
        clippy::too_many_arguments,
        reason = "exact target, numerical policies, and multiplier admission are independent inputs"
    )]
    pub(crate) fn solve_newton(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        initial: &[f64],
        nonlinear: NonlinearSolvePlan,
        linear: LinearSolveRequest<'_>,
        workers: NonZeroUsize,
        admit: impl Fn(&eqiora_assembly::CsrMatrix<f64>) -> Result<(), Diagnostic>,
    ) -> Result<RegionNewtonSolution, Diagnostic> {
        // Fixed packet maps belong to the Newton attempt, not each iterate.
        // Preserve an already sealed caller binding so drift still rejects.
        let plan = if plan.structure_identity().is_some() {
            plan.clone()
        } else {
            plan.clone().prepare(
                (0..self.packet_count())
                    .map(|index| Ok(self.packet_cell(index)?.mappings.clone()))
                    .collect::<Result<Vec<_>, Diagnostic>>()?,
            )?
        };
        let index = (0..plan.target_count())
            .find(|index| plan.target_id(*index) == Some(target))
            .ok_or_else(|| invalid("Newton target is outside its Region plan"))?;
        let (accepted, evidence) = solve_damped_newton(
            initial.to_vec(),
            nonlinear,
            |values| {
                let assembly = self.linearize(
                    &plan,
                    target,
                    RegionPoint::Bound(values),
                    &REFERENCE_ASSEMBLY_BACKEND,
                )?;
                crate::finalized_spatial::require_equation_rows(assembly.systems[index].matrix())?;
                // Convergence of a primal residual does not prove multiplier closure.
                admit(assembly.systems[index].matrix())?;
                Ok(RegionNewtonPoint {
                    values: values.to_vec(),
                    assembly,
                })
            },
            |point| residual_norm(&point.assembly.residuals[index]),
            |_, point| {
                let system = eqiora_assembly::LinearSystem::new(
                    point.assembly.systems[index].matrix().clone(),
                    point.assembly.residuals[index]
                        .iter()
                        .map(|value| -*value)
                        .collect(),
                )?;
                let canonical = Arc::new(CanonicalCsrSystemView::new(
                    &system,
                    system.rhs(),
                    LinearOperatorProperties::General,
                )?);
                let core = crate::finalized_spatial::FinalizedLinearCore::new(
                    linear.plan(),
                    VectorLayoutKind::Replicated,
                    Target::HostCpu { threads: workers },
                    canonical,
                );
                let solution = linear.solve(&core.linear_problem()?)?;
                core.validate_solution(&solution)?;
                Ok(solution.into_parts())
            },
            |candidate| {
                Ok(Some(NewtonTrial {
                    residual_norm: residual_norm(&self.residual(
                        &plan,
                        target,
                        RegionPoint::Bound(candidate),
                    )?)?,
                    linearization: None,
                }))
            },
        )?;
        // Check the retained accepted action too, not only the trial's rounded norm.
        let residual_norm = residual_norm(&accepted.assembly.residuals[index])?;
        if residual_norm > evidence.residual_target {
            return Err(invalid(
                "accepted Region Newton action exceeds its residual target",
            ));
        }
        Ok(RegionNewtonSolution {
            values: accepted.values,
            residual_norm,
            evidence,
            assembly: accepted.assembly,
        })
    }
}

pub(super) fn residual_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let norm = values.iter().fold(0_f64, |norm, value| norm.hypot(*value));
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(invalid("Region residual norm is non-finite"))
    }
}
