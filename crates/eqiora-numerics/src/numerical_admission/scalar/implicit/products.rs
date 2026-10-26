//! Original stage, endpoint and observation equations form a block-triangular relation.
use super::*;
use eqiora_assembly::CsrMatrix;
use eqiora_ir::ScalarSymbolCoordinate;

pub(in crate::numerical_admission) struct MidpointPartials {
    pub(in crate::numerical_admission) unknowns: Vec<f64>,
    pub(in crate::numerical_admission) residual: Vec<f64>,
    pub(in crate::numerical_admission) state: CsrMatrix<f64>,
    pub(in crate::numerical_admission) history: CsrMatrix<f64>,
    /// Row-major original-equation Parameter actions.
    pub(in crate::numerical_admission) parameters: Vec<f64>,
    /// Physical endpoint free coordinate -> joint unknown coordinate.
    pub(in crate::numerical_admission) physical: Vec<usize>,
    pub(in crate::numerical_admission) observations: Vec<MidpointObservationPartials>,
}

/// Physical observation coordinate is weight*joint[unknown] + history*predecessor.
pub(in crate::numerical_admission) struct MidpointObservationPartials {
    pub(in crate::numerical_admission) time: f64,
    pub(in crate::numerical_admission) physical: Vec<f64>,
    pub(in crate::numerical_admission) unknowns: Vec<usize>,
    pub(in crate::numerical_admission) weights: Vec<f64>,
    pub(in crate::numerical_admission) history: Vec<f64>,
}

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    /// Differentiate the equations, holding predecessor and stage/recovery
    /// unknowns independent. Newton iterations and recovery seeds are not differentiated.
    pub(in crate::numerical_admission) fn midpoint_partials(
        &self,
        time: f64,
        h: f64,
        initial: &[f64],
        step: &MidpointStep,
        parameters: &[ScalarSymbolCoordinate],
    ) -> Result<MidpointPartials, Diagnostic> {
        if step.endpoint.is_some() == self.algebraic.is_empty() {
            return Err(invalid(
                "midpoint partials require the exact endpoint recovery inventory",
            ));
        }
        let n = self.dimension;
        let block_count = 1_usize
            .checked_add(usize::from(step.endpoint.is_some()))
            .and_then(|count| {
                count.checked_add(if self.algebraic.is_empty() {
                    0
                } else {
                    step.observations.len()
                })
            })
            .ok_or_else(|| invalid("midpoint recovery count overflows"))?;
        let count = n
            .checked_mul(block_count)
            .ok_or_else(|| invalid("midpoint unknown count overflows"))?;
        let mut parameter_actions = vec![
            0.;
            count.checked_mul(parameters.len()).ok_or_else(|| {
                invalid("midpoint Parameter Jacobian shape overflows")
            })?
        ];
        let stage = self.midpoint_point(time, h, initial, &step.stage.unknown)?;
        let (stage_state, stage_rate) = self.stage_coordinates(initial, &step.stage.unknown, h);
        let history_weights = (0..n)
            .map(|i| {
                if self.algebraic.contains(&i) {
                    [None, None]
                } else {
                    [Some(0.5), Some(-1. / h)]
                }
            })
            .collect::<Vec<_>>();
        let history = self.linearize(
            time + 0.5 * h,
            &stage_state,
            &stage_rate,
            FieldPullback::Coordinates(&history_weights),
        )?;
        let mut unknowns = step.stage.unknown.clone();
        let mut residual = stage.assembly.residuals[0].clone();
        for (column, parameter) in parameters.iter().enumerate() {
            let (partial, _) =
                self.parameter_partial(time + 0.5 * h, &stage_state, &stage_rate, parameter)?;
            for (row, value) in partial.into_iter().enumerate() {
                parameter_actions[row * parameters.len() + column] = value;
            }
        }
        let mut physical = (0..n).collect::<Vec<_>>();
        let mut state_blocks = vec![(0, 0, stage.assembly.systems[0].matrix().clone())];
        let mut history_blocks = vec![(0, 0, history.systems[0].matrix().clone())];
        if let Some(endpoint) = &step.endpoint {
            let (fixed, _) = self.endpoint_recovery_seed(initial, &step.stage.unknown, h)?;
            let recovered = self.recovery_point(time + h, h, &fixed, &endpoint.unknown)?;
            let (state, rate) = self.recovery_coordinates(h, &fixed, &endpoint.unknown)?;
            let coupling_weights = (0..n)
                .map(|i| {
                    if self.algebraic.contains(&i) {
                        [None, None]
                    } else {
                        [Some(1.), None]
                    }
                })
                .collect::<Vec<_>>();
            let coupling = self.linearize(
                time + h,
                &state,
                &rate,
                FieldPullback::Coordinates(&coupling_weights),
            )?;
            for (column, parameter) in parameters.iter().enumerate() {
                let (partial, _) = self.parameter_partial(time + h, &state, &rate, parameter)?;
                for (row, value) in partial.into_iter().enumerate() {
                    parameter_actions[(n + row) * parameters.len() + column] = value;
                }
            }
            for &i in &self.algebraic {
                physical[i] = n + i;
            }
            unknowns.extend_from_slice(&endpoint.unknown);
            residual.extend_from_slice(&recovered.assembly.residuals[0]);
            state_blocks.push((n, 0, coupling.systems[0].matrix().clone()));
            state_blocks.push((n, n, recovered.assembly.systems[0].matrix().clone()));
        }
        let mut observations = Vec::new();
        for observation in &step.observations {
            self.validate(&observation.physical)?;
            if observation.recovery.is_some() != step.endpoint.is_some()
                || !observation.time.is_finite()
                || observation
                    .weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0. || *weight > 1.)
            {
                return Err(invalid(
                    "midpoint partials have a foreign observation inventory",
                ));
            }
            let [alpha, beta] = observation.weights;
            let mut projection = (0..n).collect::<Vec<_>>();
            let mut weights = vec![beta; n];
            let mut history_weights = vec![alpha; n];
            if let Some(recovery) = &observation.recovery {
                let offset = unknowns.len();
                let recovered = self.recovery_point(
                    observation.time,
                    h,
                    &observation.physical,
                    &recovery.unknown,
                )?;
                let (state, rate) =
                    self.recovery_coordinates(h, &observation.physical, &recovery.unknown)?;
                let coupling_weights = |weight| {
                    (0..n)
                        .map(|i| {
                            if self.algebraic.contains(&i) {
                                [None, None]
                            } else {
                                [Some(weight), None]
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let stage_weights = coupling_weights(beta);
                let previous_weights = coupling_weights(alpha);
                let coupling = self.linearize(
                    observation.time,
                    &state,
                    &rate,
                    FieldPullback::Coordinates(&stage_weights),
                )?;
                let previous = self.linearize(
                    observation.time,
                    &state,
                    &rate,
                    FieldPullback::Coordinates(&previous_weights),
                )?;
                state_blocks.push((offset, 0, coupling.systems[0].matrix().clone()));
                state_blocks.push((
                    offset,
                    offset,
                    recovered.assembly.systems[0].matrix().clone(),
                ));
                history_blocks.push((offset, 0, previous.systems[0].matrix().clone()));
                for (column, parameter) in parameters.iter().enumerate() {
                    let (partial, _) =
                        self.parameter_partial(observation.time, &state, &rate, parameter)?;
                    for (row, value) in partial.into_iter().enumerate() {
                        parameter_actions[(offset + row) * parameters.len() + column] = value;
                    }
                }
                for &i in &self.algebraic {
                    projection[i] = offset + i;
                    weights[i] = 1.;
                    history_weights[i] = 0.;
                }
                unknowns.extend_from_slice(&recovery.unknown);
                residual.extend_from_slice(&recovered.assembly.residuals[0]);
            }
            observations.push(MidpointObservationPartials {
                time: observation.time,
                physical: observation.physical.clone(),
                unknowns: projection,
                weights,
                history: history_weights,
            });
        }
        let state_refs = state_blocks
            .iter()
            .map(|(row, col, matrix)| (*row, *col, matrix))
            .collect::<Vec<_>>();
        let history_refs = history_blocks
            .iter()
            .map(|(row, col, matrix)| (*row, *col, matrix))
            .collect::<Vec<_>>();
        Ok(MidpointPartials {
            unknowns,
            residual,
            state: blocks(count, count, &state_refs)?,
            history: blocks(count, n, &history_refs)?,
            parameters: parameter_actions,
            physical,
            observations,
        })
    }
}

/// Copy already assembled sparse blocks; never reconstruct actions by probing basis vectors.
fn blocks(
    rows: usize,
    columns: usize,
    blocks: &[(usize, usize, &CsrMatrix<f64>)],
) -> Result<CsrMatrix<f64>, Diagnostic> {
    let mut entries = vec![BTreeMap::<usize, f64>::new(); rows];
    for &(row_offset, column_offset, matrix) in blocks {
        if row_offset
            .checked_add(matrix.rows())
            .is_none_or(|end| end > rows)
            || column_offset
                .checked_add(matrix.columns())
                .is_none_or(|end| end > columns)
        {
            return Err(invalid("midpoint sparse block has a foreign shape"));
        }
        for row in 0..matrix.rows() {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                *entries[row_offset + row]
                    .entry(column_offset + matrix.column_indices()[entry])
                    .or_default() += matrix.values()[entry];
            }
        }
    }
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in entries {
        for (column, value) in row {
            if value != 0. {
                indices.push(column);
                values.push(value);
            }
        }
        offsets.push(values.len());
    }
    CsrMatrix::from_sorted_csr(rows, columns, offsets, indices, values)
}
