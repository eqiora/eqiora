//! Physical derivative directions remain separate from Field value coordinates.
use super::*;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_field_gradient(
        &mut self,
        root: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if self.pullback_depth != 0 {
            return Err(invalid_component_ir(
                "spatial Field pullback requires an admitted point reconstruction",
            ));
        }
        let dimension = self.node_types[root.index() as usize].dimension();
        let mut operand = root;
        let mut component = component;
        let mut axes = Vec::new();
        while let Some(ExprNode::Gradient(value)) = self.expression.node(operand) {
            let (axis, prefix) = component.split_last().ok_or_else(|| {
                invalid_component_ir("Field gradient lacks its physical direction")
            })?;
            axes.push(*axis as usize);
            component = prefix;
            operand = *value;
        }
        let Some(ExprNode::Symbol(symbol @ SymbolRef::Field(_))) = self.expression.node(operand)
        else {
            return Err(invalid_component_ir(
                "retained spatial derivative requires an exact Field operand",
            ));
        };
        let ty = &self.node_types[operand.index() as usize];
        validate_component(ty.shape(), component)?;
        axes.reverse();
        self.input(*symbol, ty.shape(), dimension, component, part, &axes)
    }
}

#[cfg(test)]
mod tests;
