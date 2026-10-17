//! One mapping and cell traversal for coordinate integrals and probability reductions.
use super::*;
use crate::factor_measure::Axis;
use eqiora_schema::kernel::ObservableMeasure;

impl<'a> Context<'a> {
    pub(in crate::common_result::observe::composite) fn visit_factor_points(
        &mut self,
        domain: Id<kinds::Domain>,
        measure: ObservableMeasure,
        selected: &[Axis],
        point: &Point,
        mut visit: impl FnMut(&mut Self, &Point, eqiora_core::DynQuantity) -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        let rule = self.require_factor_rule(domain, selected.len())?;
        let cells = match self.occurrence.spatial_plan() {
            Some(plan) => plan.factor_quadrature_cells(selected, self.remaining.get())?,
            None => vec![selected.to_vec()],
        };
        self.remaining.set(
            self.remaining
                .get()
                .checked_sub(cells.len())
                .ok_or_else(|| invalid("factor quadrature exceeds its cell work bound"))?,
        );
        for cell in cells {
            for sample in rule.points() {
                let mut local = point.clone();
                let (coordinates, weight) = mapped_sample(&cell, sample, measure)?;
                for ((axis, _), coordinate) in selected.iter().zip(coordinates) {
                    local.insert(*axis, coordinate);
                }
                visit(self, &local, weight)?;
            }
        }
        Ok(())
    }
    pub(super) fn require_factor_rule(
        &mut self,
        domain: Id<kinds::Domain>,
        dimensions: usize,
    ) -> Result<&'a QuadratureRule, Diagnostic> {
        let rule = self.quadratures.get(&domain).ok_or_else(|| {
            invalid("factor integral requires explicit quadrature for its exact measure Domain")
        })?;
        if rule.reference_cell() != ReferenceCell::hypercube(dimensions)? {
            return Err(invalid(
                "factor quadrature dimension differs from the selected measure",
            ));
        }
        self.used_rules.insert(domain);
        Ok(rule)
    }
}
