//! Pull local Operator IR covectors through the primal reconstruction weights.
use std::collections::HashMap;

use super::*;
use crate::form_compiler::spatial::observation::PrescribedFieldPartials;
use eqiora_ir::RelationCotangent;

impl Projection {
    pub(in crate::common_result::observe::spatial) fn pullback(
        &self,
        occurrence: &super::super::Occurrence<'_>,
        coordinates: &[f64],
        fields: &BTreeMap<RawId, PointField>,
        prescribed: &PrescribedFieldPartials<'_>,
        weight: f64,
        partials: &mut super::super::IntegralPartials<'_>,
    ) -> Result<(), Diagnostic> {
        let mut bound = HashMap::new();
        for coordinate in self.0.rows().iter().flat_map(|row| row.symbols()) {
            if bound.contains_key(coordinate) {
                continue;
            }
            let (value, _, mut role) = resolve(occurrence, coordinates, fields, coordinate)?;
            if matches!(coordinate.symbol(), SymbolRef::Parameter(_))
                && partials.parameters.contains(coordinate)
            {
                role = DifferentiationRole::Parameter;
            }
            if let SymbolRef::Field(id) = coordinate.symbol()
                && fields[&id.erase()].reconstruction.is_none()
            {
                let field = &fields[&id.erase()];
                let component = super::super::super::projection::coordinate_index(
                    coordinate,
                    field.value.value_type(),
                )?;
                let axis = coordinate.spatial_derivative_axes().first().copied();
                if prescribed.active(id.erase(), component, axis)? {
                    role = DifferentiationRole::Unknown;
                }
            }
            bound.insert(coordinate.clone(), (value, role));
        }
        let action = self
            .0
            .linearize(|coordinate| bound.get(coordinate).copied())?;
        if action.residual_dimension() != partials.rows.len() {
            return Err(invalid(
                "Observable partials differ from its complete output",
            ));
        }
        let mut weights = vec![0.; action.residual_dimension()];
        let mut unknown = vec![0.; action.unknown_dimension()];
        let mut parameter = vec![0.; action.parameter_dimension()];
        for (row, output) in partials.rows.iter_mut().enumerate() {
            weights[row] = weight;
            action.vjp(
                &weights,
                RelationCotangent::Both {
                    unknown: &mut unknown,
                    parameter: &mut parameter,
                },
            )?;
            weights[row] = 0.;
            for (coordinate, &value) in action.parameter_coordinates().iter().zip(&parameter) {
                let index = partials
                    .parameters
                    .iter()
                    .position(|candidate| candidate == coordinate)
                    .expect("selected Parameter coordinate");
                output.parameters[index] += value;
            }
            for (coordinate, &value) in action.unknown_coordinates().iter().zip(&unknown) {
                let SymbolRef::Field(id) = coordinate.symbol() else {
                    return Err(invalid("spatial partial has an unadmitted unknown"));
                };
                let field = &fields[&id.erase()];
                let component = super::super::super::projection::coordinate_index(
                    coordinate,
                    field.value.value_type(),
                )?;
                let axis = match coordinate.spatial_derivative_axes() {
                    [] => None,
                    [axis] => Some(*axis),
                    _ => return Err(invalid("spatial partial requires a first Field gradient")),
                };
                if let Some(stencil) = &field.reconstruction {
                    let count = occurrence.field(id)?.values.len();
                    let target = output
                        .fields
                        .entry(id.erase())
                        .or_insert_with(|| vec![0.; count]);
                    stencil.transpose_add(component, axis, value, target)?;
                } else {
                    prescribed.pullback(
                        coordinates,
                        id.erase(),
                        component,
                        axis,
                        value,
                        &mut output.parameters,
                    )?;
                }
            }
            if output.parameters.iter().any(|value| !value.is_finite()) {
                return Err(invalid("Observable Parameter covector is non-finite"));
            }
        }
        Ok(())
    }
}
