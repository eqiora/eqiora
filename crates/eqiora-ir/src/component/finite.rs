//! Explicit finite algebra lowers into the existing real-coordinate scalar SSA.
use super::*;
use eqiora_schema::kernel::{FiniteBinaryOperation, FiniteUnaryOperation};

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_finite_unary(
        &mut self,
        operation: FiniteUnaryOperation,
        operand: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let is_map = self.node_types[operand.index() as usize]
            .value_type
            .map_bases()
            .is_some();
        let coordinate = if is_map {
            vec![component[1], component[0]]
        } else {
            component.to_vec()
        };
        let value = self.lower_part(operand, &coordinate, part)?;
        if operation == FiniteUnaryOperation::Adjoint && part == ScalarPart::Imaginary {
            self.builder.neg(value)
        } else {
            Ok(value)
        }
    }

    pub(super) fn lower_finite_binary(
        &mut self,
        operation: FiniteBinaryOperation,
        left: ExprId,
        right: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let input = &self.node_types[left.index() as usize].value_type;
        let inner = if let Some((source, _)) = input.map_bases() {
            source.extent()
        } else {
            input
                .coordinate_basis()
                .expect("typed finite pairing")
                .extent()
        };
        *self.finite_products = self
            .finite_products
            .checked_add(inner as usize)
            .filter(|work| *work <= 1_000_000)
            .ok_or_else(|| {
                invalid_component_ir("finite scalarization exceeds one million component products")
            })?;
        let mut result = None;
        for k in 0..inner {
            let (a, b) = match operation {
                FiniteBinaryOperation::Apply => (vec![component[0], k], vec![k]),
                FiniteBinaryOperation::Compose => (vec![component[0], k], vec![k, component[1]]),
                FiniteBinaryOperation::Pair => (vec![k], vec![k]),
            };
            let product = self.lower_product_parts(left, &a, right, &b, part)?;
            result = Some(match result {
                None => product,
                Some(sum) => self.builder.add(sum, product)?,
            });
        }
        Ok(result.expect("finite bases are nonempty"))
    }
}
