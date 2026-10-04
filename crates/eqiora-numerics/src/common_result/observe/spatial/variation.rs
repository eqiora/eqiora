//! Bind the compiler's generated weak density to ordinary typed expression execution.
use std::collections::BTreeMap;

use eqiora_compiler::AuthoredFormExpressionV1 as E;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, RawId, ValueShape};
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::kernel::{ExprDagBuilder, ExprId};
use eqiora_sem::KernelProgram;

use super::{PointField, invalid};

pub(super) fn evaluate(
    variation: &E,
    program: &KernelProgram,
    coordinates: &[f64],
    fields: &BTreeMap<RawId, PointField>,
) -> Result<f64, Diagnostic> {
    let E::Integrate { integrand, .. } = variation else {
        return Err(invalid(
            "local variation must retain its exact spatial integral",
        ));
    };
    let mut projection = Projection {
        program,
        coordinates,
        fields,
        dag: ExprDagBuilder::new(),
        remaining: 65536,
    };
    let root = projection.scalar(integrand, &[], false, 0)?;
    let operator = ScalarOperatorIr::lower(&projection.dag.finish([root])?)?;
    let values = operator.evaluate_typed(&[root], &mut |_| None)?;
    values
        .first()
        .and_then(|value| value.real_scalar_value())
        .map(|value| value.value())
        .ok_or_else(|| invalid("variation local evaluation has no real scalar root"))
}

struct Projection<'a> {
    program: &'a KernelProgram,
    coordinates: &'a [f64],
    fields: &'a BTreeMap<RawId, PointField>,
    dag: ExprDagBuilder,
    remaining: usize,
}

impl Projection<'_> {
    fn scalar(
        &mut self,
        value: &E,
        indices: &[u32],
        gradient: bool,
        depth: usize,
    ) -> Result<ExprId, Diagnostic> {
        if depth > 96 || self.remaining == 0 {
            return Err(invalid(
                "variation sampling exceeds its bounded projection inventory",
            ));
        }
        self.remaining -= 1;
        let scalar = indices.is_empty() && !gradient;
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length dimension");
        let constant = match value {
            E::Component {
                value,
                indices: selected,
            } if scalar => return self.scalar(value, selected, false, depth + 1),
            E::Gradient { value } if !gradient => {
                return self.scalar(value, indices, true, depth + 1);
            }
            E::Trace { value } => return self.scalar(value, indices, gradient, depth + 1),
            E::Field { ulid }
            | E::Direction {
                field_ulid: ulid, ..
            } => {
                let id = Id::<kinds::Field>::from_ulid(
                    ulid.parse()
                        .map_err(|_| invalid("variation Field identity is invalid"))?,
                );
                let field = self
                    .fields
                    .get(&id.erase())
                    .ok_or_else(|| invalid("variation Field is outside the accepted Result"))?;
                let order = match value {
                    E::Direction { name, .. } => Some(match name.as_str() {
                        "first" => 0,
                        "second" => 1,
                        _ => return Err(invalid("variation has an unbound named direction")),
                    }),
                    _ => None,
                };
                let (index, dimension) = if gradient {
                    let Some((axis, indices)) = indices.split_last() else {
                        return Err(invalid("variation gradient has no spatial axis"));
                    };
                    if *axis as usize >= self.coordinates.len() {
                        return Err(invalid(
                            "variation gradient axis is outside the accepted mesh",
                        ));
                    }
                    (
                        component(field.value.value_type().shape(), indices)?
                            * self.coordinates.len()
                            + *axis as usize,
                        field
                            .value
                            .value_type()
                            .dimension()
                            .div(length)
                            .ok_or_else(|| {
                                invalid("variation gradient dimension is unrepresentable")
                            })?,
                    )
                } else {
                    (
                        component(field.value.value_type().shape(), indices)?,
                        field.value.value_type().dimension(),
                    )
                };
                let sample = match (gradient, order) {
                    (false, None) => field.value.component(index).map(|value| value.0),
                    (true, None) => field.gradient.get(index).copied(),
                    (false, Some(order)) => field.tangent[order].get(index).copied(),
                    (true, Some(order)) => field.gradient_tangent[order].get(index).copied(),
                }
                .ok_or_else(|| invalid("variation Field component is unavailable"))?;
                DynQuantity::new(sample, dimension)
            }
            E::Parameter { ulid } if !gradient => {
                let id = Id::<kinds::Parameter>::from_ulid(
                    ulid.parse()
                        .map_err(|_| invalid("variation Parameter identity is invalid"))?,
                );
                let value = self
                    .program
                    .typed_value(id.erase())
                    .ok_or_else(|| invalid("variation fixed Parameter is unavailable"))?;
                let sample = value
                    .component(component(value.value_type().shape(), indices)?)
                    .ok_or_else(|| invalid("variation Parameter component is unavailable"))?
                    .0;
                DynQuantity::new(sample, value.value_type().dimension())
            }
            E::Coordinate {
                support_ulid,
                factor_ulid,
                axis,
            } if scalar => {
                let support =
                    Id::<kinds::Domain>::from_ulid(support_ulid.parse().map_err(|_| {
                        invalid("variation coordinate support identity is invalid")
                    })?);
                let factor = Id::<kinds::Domain>::from_ulid(
                    factor_ulid
                        .parse()
                        .map_err(|_| invalid("variation coordinate factor identity is invalid"))?,
                );
                if !crate::spatial_expression::physical_coordinate(self.program, support, factor) {
                    return Err(invalid(
                        "variation coordinate is not an ambient physical coordinate",
                    ));
                }
                DynQuantity::new(
                    *self
                        .coordinates
                        .get(*axis)
                        .ok_or_else(|| invalid("variation coordinate is unavailable"))?,
                    length,
                )
            }
            E::Number { value } if scalar => DynQuantity::new(*value, DimExponents::DIMENSIONLESS),
            E::Rational {
                numerator,
                denominator,
                dimension,
            } if scalar => {
                if *denominator == 0 {
                    return Err(invalid("variation literal denominator is zero"));
                }
                DynQuantity::new(
                    *numerator as f64 / *denominator as f64,
                    DimExponents::from_rationals(*dimension)
                        .ok_or_else(|| invalid("variation literal dimension is invalid"))?,
                )
            }
            E::Neg { value } if scalar => {
                let value = self.scalar(value, &[], false, depth + 1)?;
                return self.dag.neg(value);
            }
            E::Add { left, right } | E::Mul { left, right } if scalar => {
                let left = self.scalar(left, &[], false, depth + 1)?;
                let right = self.scalar(right, &[], false, depth + 1)?;
                return if matches!(value, E::Add { .. }) {
                    self.dag.add(left, right)
                } else {
                    self.dag.mul(left, right)
                };
            }
            _ => {
                return Err(invalid(
                    "generated variation is outside the admitted scalar sampling profile",
                ));
            }
        };
        self.dag.constant(constant)
    }
}

fn component(shape: &ValueShape, indices: &[u32]) -> Result<usize, Diagnostic> {
    if shape.rank() != indices.len() {
        return Err(invalid("variation component rank differs from its Field"));
    }
    indices
        .iter()
        .zip(shape.extents())
        .try_fold(0usize, |offset, (index, extent)| {
            if *index >= extent.get() {
                return Err(invalid("variation component is outside its Field"));
            }
            Ok(offset * extent.get() as usize + *index as usize)
        })
}
