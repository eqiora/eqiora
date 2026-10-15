//! One physical-coordinate binding for Observable values and factor quadrature.
use super::*;
use crate::factor_measure::axes;
use eqiora_core::RawId;
use std::collections::BTreeMap;

pub(super) type Point = BTreeMap<(RawId, usize), f64>;

impl Context<'_> {
    pub(super) fn output_domain(&self, id: Id<kinds::Observable>) -> Option<Id<kinds::Domain>> {
        self.program
            .observable_output_support(id)
            .ok()
            .flatten()
            .and_then(|support| support.domain().downcast())
    }

    pub(super) fn output_point(
        &self,
        id: Id<kinds::Observable>,
        coordinates: Option<&[DynQuantity]>,
    ) -> Result<Point, Diagnostic> {
        match (self.output_domain(id), coordinates) {
            (None, None) => Ok(Point::new()),
            (Some(domain), Some(coordinates)) => {
                let axes = axes(self.program, domain)?;
                if coordinates.len() != axes.len() {
                    return Err(invalid(
                        "Observable point requires exactly its output support coordinates",
                    ));
                }
                axes.into_iter().zip(coordinates).map(|((axis, bounds), coordinate)| {
                    if coordinate.dim() != bounds.lower().dim()
                        || !coordinate.value().is_finite()
                        || coordinate.value() < bounds.lower().value()
                        || coordinate.value() > bounds.upper().value() {
                        return Err(invalid("Observable output coordinate has wrong units or lies outside its exact support"));
                    }
                    Ok((axis, coordinate.value()))
                }).collect()
            }
            (Some(_), None) => Err(invalid(
                "field-valued Observable requires an explicit output point",
            )),
            (None, Some(_)) => Err(invalid(
                "lumped Observable does not have output coordinates",
            )),
        }
    }

    pub(super) fn evaluation_point(
        &self,
        domain: Id<kinds::Domain>,
        point: &Point,
    ) -> Result<eqiora_sem::EvaluationPoint, Diagnostic> {
        let program = self.program;
        let coordinates = axes(program, domain)?
            .into_iter()
            .map(|(axis, bounds)| {
                let value = point
                    .get(&axis)
                    .ok_or_else(|| invalid("Observable point omits an input coordinate"))?;
                Ok((axis, DynQuantity::new(*value, bounds.lower().dim())))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        eqiora_sem::EvaluationPoint::new(program, domain, coordinates, None)
    }
}
