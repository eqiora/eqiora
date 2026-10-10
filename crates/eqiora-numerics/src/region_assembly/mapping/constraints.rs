//! Equation-local acceptance of algebraic rows in a complete physical State.
use super::*;
use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use eqiora_solver::SolverPlan;

impl RegionDofMap<f64> {
    pub(crate) fn validate_algebraic_state<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: RegionSolveInput<f64>,
        fields: &BTreeMap<RawId, RecoveredRegionField<f64>>,
        algebraic: &BTreeSet<RawId>,
        solver: SolverPlan,
    ) -> Result<(), Diagnostic> {
        self.validate_physical(fields)?;
        let mut values = vec![0.; self.free_count()];
        let mut rows = BTreeSet::new();
        let mut stored_rows = BTreeSet::new();
        for key in self.keys() {
            if let Some(dof) = self.free_dof(key) {
                let (_, layout) = self.field_layout(key.field).expect("mapped Field");
                values[dof.index()] = fields[&key.field].coefficients[&key] / layout.scale;
                if algebraic.contains(&key.field) {
                    rows.insert(dof.index());
                } else {
                    stored_rows.insert(dof.index());
                }
            }
        }
        // A trace quotient sums equation rows. Any stored contribution makes
        // the shared row differential, even when another Region is stationary.
        rows.retain(|row| !stored_rows.contains(row));
        if rows.is_empty() {
            return Ok(());
        }
        // This is a snapshot, not a new step. Invert exact kinematic recovery
        // so eliminated displacement terms evaluate at the supplied State.
        let mut previous = fields.clone();
        for (form, _) in &input.forms {
            if let Some(time) = form.time_binding() {
                for state in &time.states {
                    let pair = state.pair();
                    let displacement = previous
                        .get_mut(&pair.state().erase())
                        .ok_or_else(|| invalid("algebraic snapshot lacks its kinematic state"))?;
                    let rate = fields
                        .get(&pair.rate().erase())
                        .ok_or_else(|| invalid("algebraic snapshot lacks its kinematic rate"))?;
                    for (key, value) in &mut displacement.coefficients {
                        let rate_key = FieldDof {
                            field: pair.rate().erase(),
                            ..*key
                        };
                        *value -= time.step.value()
                            * rate.coefficients.get(&rate_key).ok_or_else(|| {
                                invalid("algebraic snapshot lacks its exact rate coordinate")
                            })?;
                    }
                }
            }
        }
        let prepared = self.prepare_assembly(
            mesh,
            input.forms,
            input.natural,
            Some(&previous),
            input.geometry_action.as_ref(),
        )?;
        let (systems, _) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&prepared.plan, &prepared.work)?
            .into_parts();
        let system = &systems[0];
        let matrix = system.matrix();
        let mut residual_norm: f64 = 0.;
        let mut scale_norm: f64 = 0.;
        for row in rows {
            let mut action = 0.;
            let mut scale = system.rhs()[row].abs();
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                let term = matrix.values()[entry] * values[matrix.column_indices()[entry]];
                action += term;
                scale += term.abs();
            }
            residual_norm = residual_norm.hypot(action - system.rhs()[row]);
            scale_norm = scale_norm.hypot(scale);
        }
        // Only algebraic equations contribute to this backward-error scale.
        // Stored rows and their 1/dt factors cannot hide a constraint violation.
        let target = solver.residual_target(scale_norm)?;
        if !residual_norm.is_finite() || !scale_norm.is_finite() || residual_norm > target {
            return Err(invalid(&format!(
                "linear State algebraic residual {residual_norm:e} exceeds equation-local target {target:e}"
            )));
        }
        Ok(())
    }
}
