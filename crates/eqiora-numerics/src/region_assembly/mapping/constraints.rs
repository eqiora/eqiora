//! Equation-local acceptance of algebraic rows in a complete physical State.
use super::*;
use crate::form_compiler::region::FieldPullback;
use crate::region_assembly::RegionPoint;
use eqiora_assembly::{CsrMatrix, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use eqiora_solver::SolverPlan;
use eqiora_time::ConstantDerivativeMatrixProof;

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
        let independent = input.geometry_action.is_none()
            && input.forms.iter().all(|(form, _)| {
                form.time_binding()
                    .is_none_or(|time| time.states.is_empty())
            });
        let mut weights = vec![[None, None]; values.len()];
        for &row in &rows {
            weights[row][0] = Some(1.);
        }
        let independent_action = |pullback| {
            prepared.work.linearize(
                &prepared.plan,
                prepared.plan.target_id(0).expect("reduced target"),
                RegionPoint::Independent {
                    state: &values,
                    rate_target: prepared.plan.target_id(1).expect("full target"),
                    rates: &rates,
                    pullback,
                },
                &REFERENCE_ASSEMBLY_BACKEND,
            )
        };
        let action = if independent {
            independent_action(FieldPullback::Coordinates(&weights))?
        } else {
            prepared.work.linearize(
                &prepared.plan,
                prepared.plan.target_id(0).expect("reduced target"),
                RegionPoint::Bound(&values),
                &REFERENCE_ASSEMBLY_BACKEND,
            )?
        };
        let system = &action.systems[0];
        let matrix = system.matrix();
        let mut residual_norm: f64 = 0.;
        let mut scale_norm: f64 = 0.;
        for &row in &rows {
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
        // Only algebraic equations and their unknown State columns contribute.
        // Held differential coordinates are absorbed into the tangent RHS.
        // Stored rows and their 1/dt factors cannot hide a constraint violation.
        let target = solver.residual_target(scale_norm)?;
        if !residual_norm.is_finite() || !scale_norm.is_finite() || residual_norm > target {
            return Err(invalid(&format!(
                "spatial State algebraic residual {residual_norm:e} exceeds equation-local target {target:e}"
            )));
        }
        if independent {
            let algebraic = rows.iter().copied().collect::<Vec<_>>();
            let block = square_projection(matrix, &algebraic)?;
            // Positive own-State capacities and a regular algebraic block certify
            // the unknown initial coordinates (differential rates, algebraic
            // values), without demanding the fixed differential-State partial.
            if ConstantDerivativeMatrixProof::new(algebraic.len(), block)?.exact_rank()
                != algebraic.len()
            {
                // Mixed constraints can determine multipliers through dynamics
                // even when A_A is singular. Check the full original pencil,
                // without imposing an index-one restriction on that profile.
                let state = independent_action(FieldPullback::Uniform {
                    current: Some(1.),
                    rate: None,
                })?;
                let mass = independent_action(FieldPullback::Uniform {
                    current: None,
                    rate: Some(1.),
                })?;
                let full = (0..matrix.rows()).collect::<Vec<_>>();
                ConstantDerivativeMatrixProof::new(
                    full.len(),
                    square_projection(mass.systems[0].matrix(), &full)?,
                )?
                .require_regular_pencil(&square_projection(state.systems[0].matrix(), &full)?)
                .map_err(|error| {
                    invalid(&format!(
                        "spatial State algebraic regularity: {}",
                        error.message()
                    ))
                })?;
            }
        }
        Ok(())
    }
}

fn square_projection(
    matrix: &CsrMatrix<f64>,
    coordinates: &[usize],
) -> Result<Vec<f64>, Diagnostic> {
    let dimension = coordinates.len();
    let entries = dimension
        .checked_mul(dimension)
        .ok_or_else(|| invalid("algebraic regularity matrix size overflows usize"))?;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(entries)
        .map_err(|_| invalid("cannot allocate algebraic regularity matrix"))?;
    coefficients.resize(entries, 0.);
    for (row, &global) in coordinates.iter().enumerate() {
        for entry in matrix.row_offsets()[global]..matrix.row_offsets()[global + 1] {
            if let Ok(column) = coordinates.binary_search(&matrix.column_indices()[entry]) {
                coefficients[row * dimension + column] = matrix.values()[entry];
            }
        }
    }
    Ok(coefficients)
}
