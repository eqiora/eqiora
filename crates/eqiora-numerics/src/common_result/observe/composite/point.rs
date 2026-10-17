//! One physical-coordinate binding for Observable values and factor quadrature.
use super::*;
use eqiora_core::RawId;
use eqiora_schema::kernel::typing::SpatialSupport;
use std::collections::BTreeMap;

pub(super) type Point = BTreeMap<(RawId, usize), DynQuantity>;

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
                let axes = coordinate_axes(self.program, domain)?;
                if coordinates.len() != axes.len() {
                    return Err(invalid(
                        "Observable point requires exactly its output support coordinates",
                    ));
                }
                let point = axes.into_iter().zip(coordinates.iter().copied()).collect();
                self.evaluation_point(domain, &point)?;
                Ok(point)
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
        let coordinates = coordinate_axes(program, domain)?
            .into_iter()
            .map(|axis| {
                let value = point
                    .get(&axis)
                    .ok_or_else(|| invalid("Observable point omits an input coordinate"))?;
                Ok((axis, *value))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        eqiora_sem::EvaluationPoint::new(program, domain, coordinates, None)
    }
}

// This projects ordering only; EvaluationPoint owns all admission and geometry checks.
fn coordinate_axes(
    program: &eqiora_sem::KernelProgram,
    domain: Id<kinds::Domain>,
) -> Result<Vec<(RawId, usize)>, Diagnostic> {
    match program.spatial_support(domain) {
        Some(SpatialSupport::Coordinates { factors, .. }) => Ok(factors
            .iter()
            .flat_map(|(factor, _, count)| (0..*count).map(move |axis| (*factor, axis)))
            .collect()),
        Some(SpatialSupport::Volume { domain, dimensions }) => {
            Ok((0..*dimensions).map(|axis| (*domain, axis)).collect())
        }
        Some(SpatialSupport::Boundary {
            parent, dimensions, ..
        }) => Ok((0..*dimensions).map(|axis| (*parent, axis)).collect()),
        _ => Err(invalid(
            "Observable point has no admitted coordinate support",
        )),
    }
}
