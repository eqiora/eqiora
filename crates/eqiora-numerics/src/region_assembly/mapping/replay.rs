//! Reaccept physical step coefficients against the ordinary assembled equations.
use super::*;
use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

impl RegionDofMap<f64> {
    pub(crate) fn validate_step_coefficients<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<f64>,
        fields: &BTreeMap<RawId, RecoveredRegionField<f64>>,
        solver: eqiora_solver::SolverPlan,
        workers: std::num::NonZeroUsize,
    ) -> Result<(f64, f64, usize, usize), Diagnostic> {
        self.validate_physical(fields)?;
        let previous = input
            .previous
            .as_ref()
            .ok_or_else(|| invalid("step replay requires exact previous physical Fields"))?;
        if let Some(step) = super::solve::kinematic_step(&input.forms)? {
            self.validate_step_history(previous, &step)?;
            self.validate_step_history(fields, &step)?;
            self.validate_state_constraints(previous, &step, &input.prescribed_states)?;
            for state in step.eliminated_states() {
                let pair = state.pair();
                for (key, old) in &previous[&pair.state().erase()].coefficients {
                    let rate_key = FieldDof {
                        field: pair.rate().erase(),
                        ..*key
                    };
                    let new = fields[&pair.state().erase()].coefficients[key];
                    let increment = step.duration().value()
                        * fields[&pair.rate().erase()].coefficients[&rate_key];
                    // Same floating-operation bound used by physical volume sampling.
                    let roundoff = (8. * f64::EPSILON * old.abs())
                        + (8. * f64::EPSILON * new.abs())
                        + (8. * f64::EPSILON * increment.abs());
                    if !increment.is_finite() || (new - old - increment).abs() > roundoff {
                        return Err(invalid("accepted step violates backward-Euler kinematics"));
                    }
                    if input
                        .prescribed_states
                        .get(key)
                        .is_some_and(|expected| *expected != new)
                    {
                        return Err(invalid(
                            "accepted step contradicts its prescribed physical state",
                        ));
                    }
                }
            }
        }
        let mut full = vec![0.; self.full_count()];
        for key in self.keys() {
            full[self.global_dof(key).expect("mapped coordinate")] =
                fields[&key.field].coefficients[&key] / self.field_scale(key.field)?;
        }
        let reduced = self.restrict(&full)?;
        let lifted = self.lift(&reduced, false)?;
        for key in self.keys().filter(|key| self.free_dof(*key).is_none()) {
            let physical = lifted[self.global_dof(key).expect("mapped coordinate")]
                * self.field_scale(key.field)?;
            if physical != fields[&key.field].coefficients[&key] {
                return Err(invalid(
                    "accepted step contradicts its prescribed algebraic trace",
                ));
            }
        }
        let prepared = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            input.forms.clone(),
            input.natural.clone(),
            Some(previous),
            input.geometry_action.as_ref(),
        )?;
        let (systems, assembly) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&prepared.plan, &prepared.work)?
            .into_parts();
        let multipliers = input
            .forms
            .iter()
            .flat_map(|(form, _)| form.constraint_multipliers())
            .collect();
        self.require_multiplier_closure(systems[0].matrix(), &multipliers)?;
        let canonical = std::sync::Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &systems[0],
            input.operator_properties,
        )?);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            solver,
            eqiora_realization::VectorLayoutKind::Replicated,
            eqiora_realization::Target::HostCpu { threads: workers },
            canonical,
        );
        let (norm, target) = core.validate_coefficients(&reduced)?;
        Ok((
            norm,
            target,
            assembly.packet_count(),
            assembly.target_count(),
        ))
    }
}
