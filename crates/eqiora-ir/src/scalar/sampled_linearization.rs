//! Derivative slots for numerical samples of a retained expression.

use super::*;
use eqiora_schema::kernel::ExprId;

impl ScalarOperatorIr {
    /// Bind selected scalar constant leaves as numerical sample inputs.
    ///
    /// A discretization first projects sampled Field values or gradients to
    /// scalar constants. Their source expression IDs become dense tangent slots,
    /// without fabricating Semantic Parameters or changing Model meaning.
    /// Existing semantic symbols remain frozen. Unknown tangents follow `samples`
    /// order and use the ordinary scalar JVP/VJP executor.
    /// # Errors
    /// Rejects nonconstant, duplicate or missing samples, invalid symbols, and
    /// expression operations outside the ordinary scalar derivative profile.
    pub fn linearize_samples(
        &self,
        inputs: &[f64],
        samples: &[ExprId],
    ) -> Result<ScalarLinearization<'_>, Diagnostic> {
        validate_linearization_inputs(
            &self.instructions,
            self.symbols.len(),
            inputs,
            &vec![DifferentiationRole::Frozen; inputs.len()],
        )?;
        let mut ir = self.clone();
        let mut point = inputs.to_vec();
        let mut bindings = vec![InputBinding::Frozen; inputs.len()];
        let mut selected = std::collections::BTreeSet::new();
        for (ordinal, sample) in samples.iter().enumerate() {
            let id = self
                .source_values
                .get(sample.index() as usize)
                .ok_or_else(|| invalid_linearization("numerical sample expression is missing"))?;
            if !selected.insert(id.0) {
                return Err(invalid_linearization(
                    "numerical sample expression is repeated",
                ));
            }
            let Instruction::Constant(value) = ir.instructions[id.0 as usize] else {
                return Err(invalid_linearization(
                    "numerical sample must be a projected scalar constant leaf",
                ));
            };
            let slot = SymbolSlot(u32::try_from(point.len()).map_err(|_| ir_size_error())?);
            ir.instructions[id.0 as usize] = Instruction::Read(slot);
            point.push(value.value());
            bindings.push(InputBinding::Unknown(ordinal));
        }
        let bound = ScalarLinearization {
            instructions: std::borrow::Cow::Owned(ir.instructions),
            roots: std::borrow::Cow::Owned(ir.roots),
            inputs: point,
            bindings,
            unknown_dimension: samples.len(),
            parameter_dimension: 0,
        };
        bound.primal(&mut vec![0.0; bound.residual_dimension()])?;
        Ok(bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::ExprDagBuilder;

    #[test]
    fn numerical_samples_reuse_chain_rule_without_semantic_symbols() {
        let mut dag = ExprDagBuilder::new();
        let a = dag
            .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let b = dag
            .constant(DynQuantity::new(4.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let product = dag.mul(a, b).unwrap();
        let ir = ScalarOperatorIr::lower(&dag.finish([product]).unwrap()).unwrap();
        assert!(ir.symbols().is_empty());
        let bound = ir.linearize_samples(&[], &[a, b]).unwrap();
        let mut output = [0.0];
        bound
            .jvp(RelationTangent::Unknown(&[2.0, -1.0]), &mut output)
            .unwrap();
        assert_eq!(output, [5.0]);
        assert!(ir.linearize_samples(&[], &[a, a]).is_err());
        assert!(ir.linearize_samples(&[], &[product]).is_err());
    }
}
