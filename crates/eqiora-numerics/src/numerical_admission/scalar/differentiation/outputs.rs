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
        let fields = self
            .plan
            .fields()
            .map(|(id, _)| {
                let values = coordinates(
                    mapping
                        .field_keys(id.erase())
                        .map(|key| full[mapping.global_dof(key).expect("mapped Field DOF")]),
                )
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
                        let keys = mapping.field_keys(field).collect::<Vec<_>>();
                        if values.len() != keys.len() * S::WIDTH {
                            return Err(invalid(
                                "observation covector differs from exact Field mapping",
                            ));
                        }
                        for (index, key) in keys.into_iter().enumerate() {
                            let values = &values[index * S::WIDTH..(index + 1) * S::WIDTH];
                            let global =
                                mapping.global_dof(key).expect("mapped Field DOF") * S::WIDTH;
                            for (part, &value) in values.iter().enumerate() {
                                if let Some(free) = mapping.free_dof(key) {
                                    state[row * states + free.index() * S::WIDTH + part] += value;
                                }
                                weights.push((global + part, value));
                            }
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
        let field_output = |keys: &[crate::region_assembly::mapping::FieldDof],
                            direct: Vec<f64>| {
            FieldCoefficientLinearization::new(
                coordinates(
                    keys.iter()
                        .map(|key| full[mapping.global_dof(*key).expect("mapped Field DOF")]),
                )
                .collect(),
                states,
                columns,
                keys.iter()
                    .flat_map(|key| {
                        (0..S::WIDTH).map(move |part| {
                            mapping
                                .free_dof(*key)
                                .map(|dof| dof.index() * S::WIDTH + part)
                        })
                    })
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
        for id in &self.outputs {
            if let Some(observation) = observations.remove(id) {
                parts.push(observation.output()?);
            } else {
                let count = mapping.field_keys(*id).count();
                let end = cursor + count;
                parts.push(field_output(
                    &keys[cursor..end],
                    direct[cursor * S::WIDTH * columns..end * S::WIDTH * columns].to_vec(),
                )?);
                cursor = end;
            }
        }
        Output::sequence(parts)
    }
}
