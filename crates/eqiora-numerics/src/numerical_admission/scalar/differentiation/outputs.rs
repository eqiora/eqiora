//! Pull observation covectors through the exact constrained Field map.
use super::*;
use eqiora_core::{RawId, ScalarDomain};
use eqiora_ir::DenseOutputLinearization;

pub(super) struct ObservationPartials {
    id: RawId,
    values: Vec<f64>,
    state: Vec<f64>,
    parameters: Vec<f64>,
    lift: Vec<Vec<(usize, f64)>>,
}

impl ObservationPartials {
    pub(super) fn add_lift(
        &mut self,
        column: usize,
        columns: usize,
        lift: &[f64],
    ) -> Result<(), Diagnostic> {
        for (row, weights) in self.lift.iter().enumerate() {
            let value = &mut self.parameters[row * columns + column];
            *value += weights
                .iter()
                .map(|(index, weight)| weight * lift[*index])
                .sum::<f64>();
            if !value.is_finite() {
                return Err(invalid("observation boundary partial is non-finite"));
            }
        }
        Ok(())
    }
    fn output(self) -> Result<Output, Diagnostic> {
        DenseOutputLinearization::new(self.values, self.state, self.parameters).map(Output::Dense)
    }
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    pub(super) fn observation_partials(
        &self,
        mapping: &RegionDofMap<S>,
        full: &[S],
        states: usize,
    ) -> Result<Vec<ObservationPartials>, Diagnostic> {
        let selected = self
            .outputs
            .iter()
            .filter_map(|id| id.downcast::<kinds::Observable>())
            .collect::<Vec<_>>();
        if selected.is_empty() {
            if !self.quadratures.is_empty() {
                return Err(invalid("Field outputs have no observation quadrature"));
            }
            return Ok(Vec::new());
        }
        let layout = RegionRealCoordinates::new(mapping)?;
        let coordinates = layout.encode_full(full)?;
        let fields = self
            .plan
            .fields()
            .map(|(id, _)| {
                let values = mapping
                    .field_keys(id.erase())
                    .flat_map(|key| layout.global_coordinates(key))
                    .map(|index| coordinates[index])
                    .collect::<Vec<_>>();
                (id, values)
            })
            .collect::<Vec<_>>();
        let output = crate::common_result::observe::spatial_point_outputs(
            self.plan,
            &self.point,
            fields.iter().map(|(id, values)| (*id, values.as_slice())),
            &selected,
            &self.quadratures,
            Some(&self.coordinates),
        )?;
        selected
            .into_iter()
            .zip(output.values)
            .zip(output.partials.expect("requested partials"))
            .map(|((id, value), rows)| {
                let width = if value.value_type().scalar_domain() == ScalarDomain::Complex {
                    2
                } else {
                    1
                };
                let values = value
                    .components()
                    .ok_or_else(|| invalid("observation requires numeric components"))?
                    .flat_map(|(a, b)| [a, b].into_iter().take(width))
                    .collect::<Vec<_>>();
                if values.len() != rows.len() {
                    return Err(invalid(
                        "observation partials differ from output coordinates",
                    ));
                }
                let size = rows
                    .len()
                    .checked_mul(states)
                    .ok_or_else(|| invalid("observation state Jacobian overflows"))?;
                let mut state = vec![0.; size];
                let mut parameters = Vec::new();
                let mut lift = Vec::new();
                for (row, partial) in rows.into_iter().enumerate() {
                    parameters.extend(partial.parameters);
                    let mut weights = Vec::new();
                    for (field, values) in partial.fields {
                        let indices = mapping
                            .field_keys(field)
                            .flat_map(|key| layout.global_coordinates(key))
                            .collect::<Vec<_>>();
                        if values.len() != indices.len() {
                            return Err(invalid(
                                "observation covector differs from exact Field mapping",
                            ));
                        }
                        for (global, value) in indices.into_iter().zip(values) {
                            if let Some(free) = layout.free_coordinate(global) {
                                state[row * states + free] += value;
                            }
                            weights.push((global, value));
                        }
                    }
                    lift.push(weights);
                }
                Ok(ObservationPartials {
                    id: id.erase(),
                    values,
                    state,
                    parameters,
                    lift,
                })
            })
            .collect()
    }

    pub(super) fn selected_output(
        &self,
        mapping: &RegionDofMap<S>,
        keys: &[crate::region_assembly::mapping::FieldDof],
        full: &[S],
        states: usize,
        direct: Vec<f64>,
        observations: Vec<ObservationPartials>,
    ) -> Result<Output, Diagnostic> {
        let columns = self.coordinates.len();
        let layout = RegionRealCoordinates::new(mapping)?;
        let coordinates = layout.encode_full(full)?;
        let field_output = |keys: &[crate::region_assembly::mapping::FieldDof],
                            direct: Vec<f64>| {
            FieldCoefficientLinearization::new(
                keys.iter()
                    .flat_map(|key| layout.global_coordinates(*key))
                    .map(|index| coordinates[index])
                    .collect(),
                states,
                columns,
                keys.iter()
                    .flat_map(|key| layout.global_coordinates(*key))
                    .map(|index| layout.free_coordinate(index))
                    .collect(),
                direct,
            )
            .map(Output::Field)
        };
        if observations.is_empty() {
            return field_output(keys, direct);
        }
        let mut observations = observations
            .into_iter()
            .map(|value| (value.id, value))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut parts = Vec::new();
        let mut cursor = 0;
        let mut coordinate_cursor = 0;
        for id in &self.outputs {
            if let Some(observation) = observations.remove(id) {
                parts.push(observation.output()?);
            } else {
                let count = mapping.field_keys(*id).count();
                let end = cursor + count;
                let coordinates = keys[cursor..end]
                    .iter()
                    .map(|key| layout.global_coordinates(*key).len())
                    .sum::<usize>();
                let coordinate_end = coordinate_cursor + coordinates;
                parts.push(field_output(
                    &keys[cursor..end],
                    direct[coordinate_cursor * columns..coordinate_end * columns].to_vec(),
                )?);
                cursor = end;
                coordinate_cursor = coordinate_end;
            }
        }
        Output::sequence(parts)
    }
}
