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
        if matches!(
            operation,
            FiniteUnaryOperation::MatrixTrace
                | FiniteUnaryOperation::Determinant
                | FiniteUnaryOperation::Inverse
        ) {
            return self.lower_map_invariant(operation, operand, component, part);
        }
        let is_map = self.node_types[operand.index() as usize]
            .value_type
            .map_bases()
            .is_some();
        let coordinate = if let FiniteUnaryOperation::PermuteFactors(order) = operation {
            let ty = &self.node_types[operand.index() as usize].value_type;
            let reindex = |basis: eqiora_core::FiniteBasis, index: u32| {
                if order == [0, 1] {
                    index
                } else {
                    let [left, right] = basis.factors().expect("typed product");
                    (index % left.extent()) * right.extent() + index / left.extent()
                }
            };
            if let Some((source, target)) = ty.map_bases() {
                vec![reindex(target, component[0]), reindex(source, component[1])]
            } else {
                vec![reindex(
                    ty.coordinate_basis().expect("typed coordinates"),
                    component[0],
                )]
            }
        } else if is_map {
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
        if operation == FiniteBinaryOperation::TensorProduct {
            *self.finite_products = self
                .finite_products
                .checked_add(1)
                .filter(|work| *work <= 1_000_000)
                .ok_or_else(|| {
                    invalid_component_ir(
                        "finite scalarization exceeds one million component products",
                    )
                })?;
            let right_type = &self.node_types[right.index() as usize].value_type;
            let (a, b) = if let Some((source, target)) = right_type.map_bases() {
                (
                    vec![
                        component[0] / target.extent(),
                        component[1] / source.extent(),
                    ],
                    vec![
                        component[0] % target.extent(),
                        component[1] % source.extent(),
                    ],
                )
            } else {
                let extent = right_type
                    .coordinate_basis()
                    .expect("typed tensor factor")
                    .extent();
                (vec![component[0] / extent], vec![component[0] % extent])
            };
            return self.lower_product_parts(left, &a, right, &b, part);
        }
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
                FiniteBinaryOperation::TensorProduct => unreachable!("handled tensor product"),
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

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    fn lower_map_invariant(
        &mut self,
        operation: FiniteUnaryOperation,
        operand: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let ty = &self.node_types[operand.index() as usize].value_type;
        let (source, _) = ty.map_bases().expect("typed finite map");
        let n = source.extent();
        if operation == FiniteUnaryOperation::MatrixTrace {
            let mut sum = self.lower_part(operand, &[0, 0], part)?;
            for i in 1..n {
                let next = self.lower_part(operand, &[i, i], part)?;
                sum = self.builder.add(sum, next)?;
            }
            return Ok(sum);
        }
        // Scalarization gathers n² operands without expanding a factorization
        // circuit. The numerical owner separately bounds factorization work.
        let work = (n as usize)
            .checked_pow(2)
            .ok_or_else(|| invalid_component_ir("finite map operand work exceeds usize"))?;
        *self.finite_products = self
            .finite_products
            .checked_add(work)
            .filter(|work| *work <= 1_000_000)
            .ok_or_else(|| {
                invalid_component_ir("finite scalarization exceeds one million component products")
            })?;
        let start = if let Some(start) = self.map_entries.get(&operand) {
            *start
        } else {
            let mut entries = Vec::new();
            for row in 0..n {
                for column in 0..n {
                    entries.push(self.lower_part(operand, &[row, column], ScalarPart::Real)?);
                }
            }
            let start = self.builder.map_entries(&entries)?;
            self.map_entries.insert(operand, start);
            start
        };
        let output =
            (operation == FiniteUnaryOperation::Inverse).then(|| component[0] * n + component[1]);
        self.builder.map_invariant(start, n, output)
    }
}
