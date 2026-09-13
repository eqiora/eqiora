//! Canonical evaluation of exact retained Relation roots in their owning Model.
use super::*;
use eqiora_core::ValueLiteral;

impl KernelProgram {
    /// Evaluate original ordered Relation operands using exact typed Field candidates.
    /// Parameters are resolved from this immutable program. This evaluates mathematical
    /// values; it does not enforce constraints or interpret solver success.
    ///
    /// # Errors
    /// Rejects foreign Relations or Fields, wrong candidate types and unsupported symbols.
    /// Canonical evaluation failures retain the original Relation/expression path.
    pub fn evaluate_relation_operands(
        &self,
        relation: Id<kinds::Relation>,
        fields: &[(Id<kinds::Field>, ValueLiteral)],
    ) -> Result<Vec<ValueLiteral>, Diagnostic> {
        let Some(KernelNode::Relation(definition)) = self.node(relation.erase()) else {
            return Err(kernel_error(
                relation.erase(),
                "original operands require an exact retained Relation",
            ));
        };
        let mut candidates = BTreeMap::new();
        for (id, value) in fields {
            let Some(KernelNode::Field(field)) = self.node(id.erase()) else {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate Field is outside this Model",
                ));
            };
            if candidates.insert(id.erase(), value).is_some() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidates repeat one exact Field",
                ));
            }
            if value.value_type() != field.value_type() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate differs from the exact Field type",
                ));
            }
        }
        crate::evaluate::evaluate_expression(
            relation.erase(),
            definition.expression(),
            &mut |symbol| match symbol {
                SymbolRef::Field(id) => candidates.get(&id.erase()).map(|value| (**value).clone()),
                SymbolRef::Parameter(id) => match self.node(id.erase()) {
                    Some(KernelNode::Parameter(parameter)) => Some(parameter.value().clone()),
                    _ => None,
                },
                _ => None,
            },
        )
    }
}
