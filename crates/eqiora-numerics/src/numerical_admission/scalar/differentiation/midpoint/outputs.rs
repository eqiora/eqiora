//! Compose original midpoint partials with complete history and requested outputs.
use super::*;
use crate::form_compiler::spatial::InputPartialScope;
use crate::numerical_admission::scalar::implicit::MidpointPartials;
use eqiora_assembly::CsrMatrix;

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn trace_parameter_lift(
        &self,
        time: f64,
        mapping: &RegionDofMap<S>,
        coordinate: &ScalarSymbolCoordinate,
    ) -> Result<Vec<f64>, Diagnostic> {
        let (partial, _) = self.plan.scalar_assembly_with_history(
            &self.equations,
            time,
            None,
            None,
            &[(coordinate, InputPartialScope::EssentialTraces)],
        )?;
        if !mapping.has_same_coordinate_layout(&partial) {
            return Err(invalid(
                "output Parameter partial changed the coordinate layout",
            ));
        }
        RegionRealCoordinates::new(mapping)?
            .encode_full(&partial.lift(&vec![S::zero(); mapping.free_count()], false)?)
    }

    pub(super) fn midpoint_linearization(
        &mut self,
        step: &eqiora_time::TimeHistoryStep,
        previous: &[f64],
        initial_mapping: Option<&RegionDofMap<S>>,
        physical: &[f64],
        partials: MidpointPartials,
        acceptance: PrimalAcceptance,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let time = step.end_time();
        let (mapping, _) = self.plan.scalar_assembly_at(&self.equations, time)?;
        let (start_mapping, _) = self
            .plan
            .scalar_assembly_at(&self.equations, step.start_time())?;
        let previous_mapping = initial_mapping.unwrap_or(&start_mapping);
        if !mapping.has_same_coordinate_layout(previous_mapping) {
            return Err(invalid(
                "midpoint history changed the exact coordinate layout",
            ));
        }
        let layout = RegionRealCoordinates::new(&mapping)?;
        let previous_layout = RegionRealCoordinates::new(previous_mapping)?;
        let full = layout.full_count();
        let unknowns = partials.unknowns.len();
        let parameters = self.coordinates.len();
        let mut globals = vec![0; layout.free_count()];
        for global in 0..full {
            if let Some(free) = layout.free_coordinate(global) {
                globals[free] = global;
            }
        }
        let jacobian = CsrMatrix::from_sorted_csr(
            partials.history.rows(),
            full,
            partials.history.row_offsets().to_vec(),
            partials
                .history
                .column_indices()
                .iter()
                .map(|&free| globals[free])
                .collect(),
            partials.history.values().to_vec(),
        )?;
        let preceding = previous_layout
            .encode_full(&previous_mapping.lift(&previous_layout.decode(previous)?, false)?)?;
        let physical_values =
            layout.encode_full(&mapping.lift(&layout.decode(physical)?, false)?)?;
        let mut direct = vec![
            0.;
            full.checked_mul(parameters).ok_or_else(|| invalid(
                "endpoint output Parameter shape overflows"
            ))?
        ];
        for (column, coordinate) in self.coordinates.iter().enumerate() {
            let lift = self.trace_parameter_lift(time, &mapping, coordinate)?;
            for (row, value) in lift.into_iter().enumerate() {
                direct[row * parameters + column] = value;
            }
        }
        let physical_projection = partials
            .physical
            .iter()
            .map(|&index| (index, 1.))
            .collect::<Vec<_>>();
        let complete_output = FieldCoefficientLinearization::new(
            physical_values,
            layout.free_count(),
            parameters,
            (0..full)
                .map(|global| layout.free_coordinate(global))
                .collect(),
            direct,
        )?
        .project_unknowns(unknowns, &physical_projection)?;
        let mut selected = Vec::new();
        let mut selected_history = Vec::new();
        let mut times = Vec::new();
        for observation in partials.observations {
            let (observed_mapping, _) = self
                .plan
                .scalar_assembly_at(&self.equations, observation.time)?;
            if !mapping.has_same_coordinate_layout(&observed_mapping) {
                return Err(invalid(
                    "midpoint observation changed the exact coordinate layout",
                ));
            }
            self.time_interval = Some([step.start_time(), observation.time]);
            let (_, output) =
                self.output_partials(&observed_mapping, &observation.physical, 0, |coordinate| {
                    Ok((
                        Vec::new(),
                        self.trace_parameter_lift(observation.time, &observed_mapping, coordinate)?,
                    ))
                })?;
            let projection = observation
                .unknowns
                .into_iter()
                .zip(observation.weights)
                .collect::<Vec<_>>();
            let history_projection = globals
                .iter()
                .copied()
                .zip(observation.history)
                .collect::<Vec<_>>();
            selected_history.push(output.clone().project_unknowns(full, &history_projection)?);
            selected.push(output.project_unknowns(unknowns, &projection)?);
            times.push(observation.time);
        }
        let output_history = if selected_history.is_empty() {
            None
        } else {
            Output::sequence(selected_history)?.state_jacobian()?
        };
        let output = if selected.is_empty() {
            Output::Empty {
                unknowns,
                parameters,
            }
        } else {
            Output::sequence(selected)?
        };
        let relation = AssembledLinearizedRelation::from_point(
            partials.state,
            partials.unknowns,
            partials.residual,
            self.design.clone(),
            self.values.clone(),
            partials.parameters,
            LinearOperatorProperties::General,
        )?;
        let identity: std::sync::Arc<str> = self.plan.identity().into();
        let coordinates = layout
            .full_keys()
            .into_iter()
            .map(|(key, imaginary)| SpatialStepCoordinate {
                plan: identity.clone(),
                key,
                imaginary,
            })
            .collect();
        self.time_interval = Some([step.start_time(), time]);
        CommonDifferentiationLinearization::from_spatial(
            relation,
            eqiora_core::ScalarDomain::Real,
            output,
            acceptance,
            None,
        )
        .with_step_history(
            jacobian,
            coordinates,
            preceding,
            [step.start_time(), time],
            complete_output,
        )?
        .with_step_observations(times, output_history)
    }
}
