//! Nonlinear execution with the same physical coordinates and history admission.
use super::*;
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::{LinearOperatorProperties, LinearSolveRequest};
use std::num::NonZeroUsize;

pub(crate) struct RegionNonlinearOutput {
    pub(crate) fields: BTreeMap<RawId, RecoveredRegionField<f64>>,
    pub(crate) solve: crate::region_assembly::newton::RegionNewtonSolution,
}

impl RegionDofMap<f64> {
    /// The predictor is the prior physical state in the current free coordinates.
    /// Current essential traces are applied by lifting those coordinates, both
    /// during execution and during independent replay acceptance.
    pub(crate) fn physical_predictor(
        &self,
        initial: &BTreeMap<RawId, RecoveredRegionField<f64>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_physical(initial)?;
        let mut free = vec![0.; self.free_count()];
        for key in self.keys() {
            if let Some(dof) = self.free_dof(key) {
                free[dof.index()] =
                    initial[&key.field].coefficients[&key] / self.field_scale(key.field)?;
            }
        }
        Ok(free)
    }

    /// A Newton seed is independent of time history for stationary equations.
    /// A transient solve still starts from its exact previous physical Fields.
    pub(super) fn nonlinear_predictor(
        &self,
        input: &RegionSolveInput<f64>,
        initial: &BTreeMap<RawId, RecoveredRegionField<f64>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if input.operator_properties != LinearOperatorProperties::General {
            return Err(invalid(
                "nonlinear Region execution requires a general Jacobian",
            ));
        }
        if input
            .previous
            .as_ref()
            .is_some_and(|previous| previous != initial)
        {
            return Err(invalid(
                "transient Newton predictor must use its exact previous physical Fields",
            ));
        }
        self.physical_predictor(initial)
    }

    pub(crate) fn solve_nonlinear<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: RegionSolveInput<f64>,
        initial: &BTreeMap<RawId, RecoveredRegionField<f64>>,
        nonlinear: NonlinearSolvePlan,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_>,
    ) -> Result<RegionNonlinearOutput, Diagnostic> {
        let step = self.validate_solve_input(mesh, &input)?;
        let seed = self.nonlinear_predictor(&input, initial)?;
        let multipliers = input
            .forms
            .iter()
            .flat_map(|(form, _)| form.constraint_multipliers())
            .collect();
        let prepared = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            input.forms,
            input.natural,
            input.previous.as_ref(),
            input.geometry_action.as_deref(),
        )?;
        let solve = prepared.work.solve_newton(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            &seed,
            nonlinear,
            request,
            workers,
            |matrix| self.require_multiplier_closure(matrix, &multipliers),
        )?;
        let fields = if let Some(step) = &step {
            self.recover_step(
                &solve.values,
                input.previous.as_ref().expect("validated step history"),
                step,
                &input.prescribed_states,
            )?
        } else {
            self.recover(
                &solve.values,
                &self.fields.keys().copied().collect::<Vec<_>>(),
            )?
        };
        Ok(RegionNonlinearOutput { fields, solve })
    }
}
