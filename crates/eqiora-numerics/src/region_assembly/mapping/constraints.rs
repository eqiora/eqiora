//! Equation-local acceptance of algebraic rows in a complete physical State.
use super::*;
mod rates;
use crate::form_compiler::region::FieldPullback;
use crate::form_compiler::spatial::InputPartialScope;
use crate::region_assembly::RegionPoint;
use eqiora_assembly::{CsrMatrix, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use eqiora_solver::SolverPlan;
use eqiora_time::ConstantDerivativeMatrixProof;

impl<S: Coefficient + Send + Sync> RegionDofMap<S>
where
    Self: RegionAlgebraicCoordinates<f64, S>,
{
    /// Partition free equations using exact Field/channel and quotient ownership.
    pub(crate) fn temporal_coordinate_partition(
        &self,
        temporal: &crate::form_compiler::region::TemporalStructure,
    ) -> Result<(BTreeSet<usize>, BTreeSet<usize>), Diagnostic> {
        let coordinates = RegionRealCoordinates::new(self)?;
        let mut rows = BTreeSet::new();
        let mut stored_rows = BTreeSet::new();
        for key in self.keys() {
            for (part, global) in coordinates.global_coordinates(key).enumerate() {
                if let Some(row) = coordinates.free_coordinate(global) {
                    if !temporal
                        .differential
                        .contains(&eqiora_core::TimeStateCoordinate::new(
                            key.field.downcast().expect("mapped Field"),
                            0,
                            key.component,
                            part != 0,
                        ))
                    {
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
        Ok((rows, stored_rows))
    }

    pub(crate) fn validate_temporal_state<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        mut input: RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
        temporal: &crate::form_compiler::region::TemporalStructure,
        solver: SolverPlan,
        initial_partial: impl FnOnce(
            InputPartialScope,
        ) -> Result<(Self, RegionSolveInput<S>), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        self.validate_prescribed_fields(fields)?;
        // A physical State preserves signed-zero words. Essential traces were
        // checked numerically above; reduction must not impose the distinct
        // exact-word contract of an already normalized algebraic point.
        let full_state = self.normalized_coefficients(fields)?;
        let values = self.encode_coordinates(&self.constraints.restrict_action(&full_state)?)?;
        let coordinates = RegionRealCoordinates::new(self)?;
        let (rows, stored_rows) = self.temporal_coordinate_partition(temporal)?;
        if values.is_empty() || (rows.is_empty() && temporal.positive_own_capacity) {
            return Ok(());
        }
        // This is a snapshot, not a new step. Invert exact kinematic recovery
        // so eliminated displacement terms evaluate at the supplied State.
        let accepted_step = input.previous.is_some();
        let discrete_rates = input
            .previous
            .as_ref()
            .map(|previous| {
                let step = input
                    .forms
                    .iter()
                    .find_map(|(form, _)| form.time_binding())
                    .ok_or_else(|| invalid("step State lacks its temporal binding"))?
                    .step
                    .value();
                let previous = self.normalized_coefficients(previous)?;
                let rates = full_state
                    .iter()
                    .zip(previous)
                    .map(|(current, old)| (*current - old) / step)
                    .collect::<Vec<_>>();
                coordinates.encode_full(&rates)
            })
            .transpose()?;
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
        input.previous = input
            .forms
            .iter()
            .any(|(form, _)| !form.previous_fields().is_empty())
            .then_some(previous);
        let prepared = self.prepare_numeric_assembly(mesh, &input)?;
        // Snapshot constraint rows use the original State partial, without a
        // step-dependent mass contribution. Transformed kinematics retain their
        // explicit inverse recovery above; their coordinates are not independent.
        let rates = discrete_rates
            .clone()
            .unwrap_or_else(|| vec![0.; coordinates.full_count()]);
        let independent = input.geometry_action.is_none()
            && input.forms.iter().all(|(form, _)| {
                form.time_binding()
                    .is_none_or(|time| time.states.is_empty())
            });
        let mut weights = vec![[None, None]; values.len()];
        for &row in &rows {
            weights[row][0] = Some(1.);
        }
        let independent_action = |rates: &[f64], pullback| {
            prepared.work.linearize(
                &prepared.plan,
                prepared.plan.target_id(0).expect("reduced target"),
                RegionPoint::Independent {
                    state: &values,
                    rate_target: prepared.plan.target_id(1).expect("full target"),
                    rates,
                    pullback,
                },
                &REFERENCE_ASSEMBLY_BACKEND,
            )
        };
        if !temporal.positive_own_capacity {
            if !independent {
                return Err(invalid(
                    "candidate-dependent storage requires independent State/rate coordinates",
                ));
            }
            // The authored rate inventory fixes the partition, even when a
            // coefficient vanishes at this particular State. Recover an actual
            // rate point before differentiating any State-dependent mass.
            let full = (0..values.len()).collect::<Vec<_>>();
            let rates = if let Some(rates) = discrete_rates {
                rates
            } else {
                let (partial_mapping, _) = initial_partial(InputPartialScope::EssentialTraces)?;
                let fixed = self.prescribed_rates(&partial_mapping)?;
                let mass = independent_action(
                    &fixed,
                    FieldPullback::Uniform {
                        current: None,
                        rate: Some(1.),
                    },
                )?;
                let mass_proof = ConstantDerivativeMatrixProof::new(
                    full.len(),
                    square_projection(mass.systems[0].matrix(), &full)?,
                )?;
                prepared.consistent_rates(
                    &coordinates,
                    (&values, &fixed),
                    &mass_proof,
                    &mass.residuals[0].iter().map(|v| -*v).collect::<Vec<_>>(),
                    &BTreeSet::new(),
                    solver,
                )?
            };
            for (column, weight) in weights.iter_mut().enumerate() {
                *weight = if stored_rows.contains(&column) {
                    [None, Some(1.)]
                } else {
                    [Some(1.), None]
                };
            }
            let selected = independent_action(&rates, FieldPullback::Coordinates(&weights))?;
            let matrix = selected.systems[0].matrix();
            let reduced_rates = coordinates.restrict_action(&rates)?;
            // The unknown coordinates are rates for D and values for A. Scale
            // each original equation in those coordinates: a large dynamic
            // forcing must not hide a small incompatible algebraic equation.
            let unknowns = (0..values.len())
                .map(|column| {
                    if stored_rows.contains(&column) {
                        reduced_rates[column]
                    } else {
                        values[column]
                    }
                })
                .collect::<Vec<_>>();
            for row in 0..matrix.rows() {
                let residual = selected.residuals[0][row];
                let scale = tangent_equation_scale(matrix, row, &unknowns, residual);
                let target = solver.residual_target(scale)?;
                if !residual.is_finite() || !scale.is_finite() || residual.abs() > target {
                    return Err(invalid(&format!(
                        "spatial State equation-local residual {:e} exceeds target {target:e}",
                        residual.abs(),
                    )));
                }
            }
            let proof =
                ConstantDerivativeMatrixProof::new(full.len(), square_projection(matrix, &full)?)?;
            if proof.exact_rank() != full.len() {
                return Err(invalid(
                    "spatial State differential-rate/algebraic-State Jacobian is singular at its consistent rate",
                ));
            }
            return Ok(());
        }
        let action = if independent {
            independent_action(&rates, FieldPullback::Coordinates(&weights))?
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
            let scale = tangent_equation_scale(matrix, row, &values, action.residuals[0][row]);
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
                let state_action = independent_action(
                    &rates,
                    FieldPullback::Uniform {
                        current: Some(1.),
                        rate: None,
                    },
                )?;
                let mass = independent_action(
                    &rates,
                    FieldPullback::Uniform {
                        current: None,
                        rate: Some(1.),
                    },
                )?;
                let full = (0..matrix.rows()).collect::<Vec<_>>();
                let mass = ConstantDerivativeMatrixProof::new(
                    full.len(),
                    square_projection(mass.systems[0].matrix(), &full)?,
                )?;
                let state = square_projection(state_action.systems[0].matrix(), &full)?;
                mass.require_regular_pencil(&state).map_err(|error| {
                    invalid(&format!(
                        "spatial State algebraic regularity: {}",
                        error.message()
                    ))
                })?;
                // A step checkpoint has already passed its original discrete
                // equations with the retained predecessor. Continuous tangent
                // compatibility is an initialization condition, not a BE equation.
                if accepted_step {
                    return Ok(());
                }
                // Differentiate the actual prescribed inputs: fixed trace
                // rates enter momentum, and R_A,t + J_A,fixed*g_dot enters
                // the first constraint tangent. Both use the original actions.
                let (fixed, mut partial) = self.initial_rate_data(
                    mesh,
                    &input,
                    &values,
                    initial_partial(InputPartialScope::Full)?,
                )?;
                let jacobian = state_action.systems[1].matrix();
                for global in 0..jacobian.rows() {
                    if let Some(row) = coordinates.free_coordinate(global) {
                        for entry in
                            jacobian.row_offsets()[global]..jacobian.row_offsets()[global + 1]
                        {
                            partial[row] +=
                                jacobian.values()[entry] * fixed[jacobian.column_indices()[entry]];
                        }
                    }
                }
                let source = prepared.work.residual(
                    &prepared.plan,
                    prepared.plan.target_id(0).expect("reduced target"),
                    RegionPoint::Independent {
                        state: &values,
                        rate_target: prepared.plan.target_id(1).expect("full target"),
                        rates: &fixed,
                        pullback: FieldPullback::Uniform {
                            current: None,
                            rate: None,
                        },
                    },
                )?;
                let rhs = source
                    .iter()
                    .enumerate()
                    .map(|(row, residual)| {
                        if rows.contains(&row) {
                            -partial[row]
                        } else {
                            -*residual
                        }
                    })
                    .collect::<Vec<_>>();
                let mut tangent = mass.coefficients().to_vec();
                for &row in &rows {
                    let range = row * full.len()..(row + 1) * full.len();
                    tangent[range.clone()].copy_from_slice(&state[range]);
                }
                let rate_proof = ConstantDerivativeMatrixProof::new(full.len(), tangent)?;
                prepared.consistent_rates(
                    &coordinates,
                    (&values, &fixed),
                    &rate_proof,
                    &rhs,
                    &rows,
                    solver,
                )?;
            }
        }
        Ok(())
    }
}

/// Backward-error scale |J|*|x| + |J*x-R| in the selected unknown coordinates.
fn tangent_equation_scale(
    matrix: &CsrMatrix<f64>,
    row: usize,
    point: &[f64],
    residual: f64,
) -> f64 {
    let mut rhs = -residual;
    let mut scale = 0.;
    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
        let term = matrix.values()[entry] * point[matrix.column_indices()[entry]];
        rhs += term;
        scale += term.abs();
    }
    scale + rhs.abs()
}

pub(crate) fn square_projection(
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

impl<S: Coefficient> RegionSolveInput<S> {
    /// Rebind the source rate inventory and sufficient mass certificate together.
    pub(crate) fn temporal_structure(
        &self,
    ) -> Result<crate::form_compiler::region::TemporalStructure, Diagnostic> {
        let mut temporal = crate::form_compiler::region::TemporalStructure {
            differential: Default::default(),
            positive_own_capacity: true,
        };
        for (form, _) in &self.forms {
            let region = form.temporal_structure()?;
            temporal.differential.extend(region.differential);
            temporal.positive_own_capacity &= region.positive_own_capacity;
        }
        Ok(temporal)
    }
}
