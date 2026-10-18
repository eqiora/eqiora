//! Coordinate Jacobians use the same scalar derivatives and finite-map outputs.
use super::*;
use crate::scalar::MapOutput;
use eqiora_core::DimExponents;
use eqiora_schema::kernel::CoordinateMapFactor;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_map_factor(
        &mut self,
        factor: CoordinateMapFactor,
        source: &[ExprId],
        at: &[(ExprId, ExprId)],
        dimension: DimExponents,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if !self.coordinate_bindings.is_empty() {
            return Err(invalid_component_ir(
                "a pulled-back coordinate Jacobian requires differentiation before substitution",
            ));
        }
        let extent = u32::try_from(source.len())
            .map_err(|_| invalid_component_ir("coordinate Jacobian extent exceeds u32"))?;
        let work = source
            .len()
            .checked_mul(at.len())
            .ok_or_else(|| invalid_component_ir("coordinate Jacobian work exceeds usize"))?;
        finite::charge_products(self.finite_products, work)?;
        let sources = source
            .iter()
            .map(|id| {
                let Some(ExprNode::Symbol(symbol @ SymbolRef::Coordinate { .. })) =
                    self.expression.node(*id)
                else {
                    return Err(invalid_component_ir(
                        "coordinate Jacobian needs exact source coordinates",
                    ));
                };
                Ok((*symbol, self.node_types[id.index() as usize].dimension()))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let one = self
            .builder
            .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))?;
        let mut entries = Vec::with_capacity(work);
        for (_, mapped) in at {
            let value = self.lower_part(*mapped, &[], ScalarPart::Real)?;
            let inputs = self.builder.directional_inputs(value)?;
            if inputs.iter().any(|input| {
                !matches!(
                    input.source().symbol(),
                    SymbolRef::Coordinate { .. } | SymbolRef::Parameter(_) | SymbolRef::Time
                )
            }) {
                return Err(invalid_component_ir(
                    "coordinate Jacobian needs explicit coordinate expressions; Field maps require reconstruction",
                ));
            }
            for &(symbol, source_dimension) in &sources {
                let mut seeds = vec![None; self.inputs.len()];
                for input in &inputs {
                    if input.source().symbol() == symbol {
                        seeds[input.ordinal() as usize] = Some(one);
                    }
                }
                let entry_dimension = self.node_types[mapped.index() as usize]
                    .dimension()
                    .div(source_dimension)
                    .ok_or_else(|| {
                        invalid_component_ir("coordinate Jacobian dimension overflow")
                    })?;
                let entry = match self.builder.directional(value, &seeds, |work| {
                    finite::charge_products(self.finite_products, work)
                })? {
                    Some(entry) => entry,
                    None => self
                        .builder
                        .constant(DynQuantity::new(0., entry_dimension))?,
                };
                // Axes may have different dimensions. Factor the coherent unit
                // of each entry out before using the common numeric matrix owner.
                let unit = self
                    .builder
                    .constant(DynQuantity::new(1., entry_dimension))?;
                entries.push(self.builder.div(entry, unit)?);
            }
        }
        let start = self.builder.map_entries(&entries)?;
        let value = self
            .builder
            .map_output(start, extent, MapOutput::coordinate_factor(factor))?;
        let unit = self.builder.constant(DynQuantity::new(1., dimension))?;
        self.builder.mul(value, unit)
    }
}

#[cfg(test)]
mod tests;
