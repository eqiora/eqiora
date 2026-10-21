//! Equation-local acceptance of algebraic rows in a complete physical State.
use super::*;
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use eqiora_solver::SolverPlan;

impl<S: Coefficient + Send + Sync> RegionDofMap<S>
where
    Self: RegionAlgebraicCoordinates<f64, S>,
{
    pub(crate) fn validate_algebraic_state<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        mut input: RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
        algebraic: &BTreeSet<RawId>,
        solver: SolverPlan,
    ) -> Result<(), Diagnostic> {
        self.validate_physical(fields)?;
        let values = self.validated_coordinates(fields)?;
        let coordinates = RegionRealCoordinates::new(self)?;
        let mut rows = BTreeSet::new();
        let mut stored_rows = BTreeSet::new();
        for key in self.keys() {
            for global in coordinates.global_coordinates(key) {
                if let Some(row) = coordinates.free_coordinate(global) {
                    if algebraic.contains(&key.field) {
                        rows.insert(row);
                    } else {
                        stored_rows.insert(row);
                    }
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
                        *value -= *rate.coefficients.get(&rate_key).ok_or_else(|| {
                            invalid("algebraic snapshot lacks its exact rate coordinate")
                        })? * time.step.value();
                    }
                }
            }
        }
        input.previous = Some(previous);
        let prepared = self.prepare_numeric_assembly(mesh, &input)?;
        // Snapshot constraint rows use the original State partial, without a
        // step-dependent mass contribution. Transformed kinematics retain their
        // explicit inverse recovery above; their coordinates are not independent.
        let rates = vec![0.; coordinates.full_count()];
        let point = if input.geometry_action.is_none()
            && input.forms.iter().all(|(form, _)| {
                form.time_binding()
                    .is_none_or(|time| time.states.is_empty())
            }) {
            crate::region_assembly::RegionPoint::Independent {
                state: &values,
                rate_target: prepared.plan.target_id(1).expect("full target"),
                rates: &rates,
                pullback: crate::form_compiler::region::FieldPullback {
                    current: Some(1.),
                    rate: None,
                },
            }
        } else {
            crate::region_assembly::RegionPoint::Bound(&values)
        };
        let action = prepared.work.linearize(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            point,
            &REFERENCE_ASSEMBLY_BACKEND,
        )?;
        let system = &action.systems[0];
        let matrix = system.matrix();
        let mut residual_norm: f64 = 0.;
        let mut scale_norm: f64 = 0.;
        for row in rows {
            let mut scale = system.rhs()[row].abs();
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                let term = matrix.values()[entry] * values[matrix.column_indices()[entry]];
                scale += term.abs();
            }
            residual_norm = residual_norm.hypot(action.residuals[0][row]);
            scale_norm = scale_norm.hypot(scale);
        }
        // Use the backward-error scale of the tangent equation J*x = J*x-R.
        // It reduces to |A|*|x|+|b| for affine equations. R itself comes from
        // the original local actions, never from subtracting the tangent RHS.
        // Only algebraic equations contribute to this scale.
        // Stored rows and their 1/dt factors cannot hide a constraint violation.
        let target = solver.residual_target(scale_norm)?;
        if !residual_norm.is_finite() || !scale_norm.is_finite() || residual_norm > target {
            return Err(invalid(&format!(
                "spatial State algebraic residual {residual_norm:e} exceeds equation-local target {target:e}"
            )));
        }
        Ok(())
    }
}
