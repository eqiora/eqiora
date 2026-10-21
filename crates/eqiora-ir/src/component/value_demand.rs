//! Explicit projection retains the complete source value's lazy demand graph.
use super::*;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn complete_value(
        &mut self,
        value: ExprId,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if let Some(&demand) = self.cache.complete_values.get(&value) {
            return Ok(demand);
        }
        let ty = self.node_types[value.index() as usize].value_type.clone();
        let count = ty
            .shape()
            .component_count()
            .ok_or_else(|| invalid_component_ir("complete value demand exceeds local usize"))?;
        let parts: &[ScalarPart] = if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
            &[ScalarPart::Real, ScalarPart::Imaginary]
        } else {
            &[ScalarPart::Real]
        };
        let mut demand = None;
        for index in 0..count {
            let coordinates = row_major_index(ty.shape(), index)?;
            for &part in parts {
                finite::charge_products(self.finite_products, 1)?;
                let component = self.lower_part(value, &coordinates, part)?;
                demand = Some(match demand {
                    Some(prior) => self.builder.demand(prior, component)?,
                    None => component,
                });
            }
        }
        let demand =
            demand.ok_or_else(|| invalid_component_ir("complete value has no components"))?;
        self.cache.complete_values.insert(value, demand);
        Ok(demand)
    }
}
