//! Bind reconstructed spatial coordinates to the shared component arithmetic and JVP.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_ir::{
    ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationTangent,
    ScalarSymbolCoordinate,
};
use eqiora_schema::kernel::SymbolRef;
use eqiora_schema::kernel::typing::TypedResidual;

use super::{PointField, invalid};

mod partials;

pub(super) struct Projection(ComponentScalarization);

impl Projection {
    pub(super) fn new(
        typed: &TypedResidual<RawId>,
        normal: Option<&[f64]>,
    ) -> Result<Self, Diagnostic> {
        ComponentScalarization::lower_spatial_selected(typed, typed.expression().roots(), normal)
            .map(Self)
    }

    pub(super) fn evaluate(
        &self,
        occurrence: &super::Occurrence<'_>,
        coordinates: &[f64],
        fields: &BTreeMap<RawId, PointField>,
        derivative: bool,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.0
            .rows()
            .iter()
            .map(|row| {
                let bound = row
                    .symbols()
                    .iter()
                    .map(|coordinate| resolve(occurrence, coordinates, fields, coordinate))
                    .collect::<Result<Vec<_>, _>>()?;
                let values = bound.iter().map(|(value, _, _)| *value).collect::<Vec<_>>();
                if !derivative {
                    return row.evaluate(&values);
                }
                let roles = bound.iter().map(|(_, _, role)| *role).collect::<Vec<_>>();
                let tangents = bound
                    .iter()
                    .filter_map(|(_, tangent, role)| {
                        (*role == DifferentiationRole::Unknown).then_some(*tangent)
                    })
                    .collect::<Vec<_>>();
                let action = row.linearize(&values, &roles)?;
                let mut value = [0.];
                action.jvp(RelationTangent::Unknown(&tangents), &mut value)?;
                Ok(value[0])
            })
            .collect()
    }
}

fn resolve(
    occurrence: &super::Occurrence<'_>,
    coordinates: &[f64],
    fields: &BTreeMap<RawId, PointField>,
    coordinate: &ScalarSymbolCoordinate,
) -> Result<(f64, f64, DifferentiationRole), Diagnostic> {
    let program = occurrence
        .spatial_plan()
        .expect("spatial occurrence")
        .kernel();
    match coordinate.symbol() {
        SymbolRef::Field(field) => {
            let sample = fields
                .get(&field.erase())
                .ok_or_else(|| invalid("Observable Field sample is outside this Result"))?;
            let real_coordinate =
                super::super::projection::coordinate_index(coordinate, sample.value.value_type())?;
            let (value, tangent) = match coordinate.spatial_derivative_axes() {
                [] => (
                    coordinate.read_value(&sample.value)?,
                    sample.tangent[0].get(real_coordinate),
                ),
                [axis] if *axis < coordinates.len() => {
                    let index = real_coordinate * coordinates.len() + axis;
                    (
                        *sample.gradient.get(index).ok_or_else(|| {
                            invalid("Observable gradient coordinate is unavailable")
                        })?,
                        sample.gradient_tangent[0].get(index),
                    )
                }
                _ => {
                    return Err(invalid(
                        "Observable reconstruction does not supply the requested Field derivative",
                    ));
                }
            };
            Ok((
                value,
                *tangent.ok_or_else(|| invalid("Observable tangent coordinate is unavailable"))?,
                if sample.reconstruction.is_some() {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                },
            ))
        }
        SymbolRef::Parameter(id) => {
            let value = occurrence
                .parameter(program, id)
                .ok_or_else(|| invalid("Observable Parameter is unavailable"))?;
            Ok((
                coordinate.read_value(value)?,
                0.,
                DifferentiationRole::Frozen,
            ))
        }
        SymbolRef::Time => Ok((
            occurrence
                .time_s()
                .ok_or_else(|| invalid("Observable requires an explicit accepted time"))?,
            0.,
            DifferentiationRole::Frozen,
        )),
        SymbolRef::Coordinate {
            support,
            factor,
            axis,
        } => {
            if !crate::spatial_expression::physical_coordinate(program, support, factor) {
                return Err(invalid(
                    "Observable coordinate is not an ambient physical coordinate",
                ));
            }
            Ok((
                *coordinates
                    .get(axis)
                    .ok_or_else(|| invalid("Observable coordinate axis is unavailable"))?,
                0.,
                DifferentiationRole::Frozen,
            ))
        }
        _ => Err(invalid("spatial Observable contains an unadmitted symbol")),
    }
}
