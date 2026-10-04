//! Canonical evaluation of exact retained Relation roots in their owning Model.
use super::*;
use eqiora_core::ValueLiteral;

impl KernelProgram {
    /// Evaluate original ordered Relation operands at exact typed numerical candidates.
    /// Unselected Parameters retain their immutable Model values. This evaluates mathematical
    /// values; it does not enforce constraints or interpret solver success.
    ///
    /// # Errors
    /// Rejects foreign Relations, Fields, Parameters or Observables, duplicate coordinates, wrong
    /// candidate types, spatial Observable outputs and unsupported symbols.
    /// Observable values are supplied by the numerical realization; this performs no quadrature.
    /// Canonical evaluation failures retain the original Relation/expression path.
    pub fn evaluate_relation_operands(
        &self,
        relation: Id<kinds::Relation>,
        fields: &[(Id<kinds::Field>, ValueLiteral)],
        parameters: &[(Id<kinds::Parameter>, ValueLiteral)],
        observables: &[(Id<kinds::Observable>, ValueLiteral)],
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
        let mut parameter_candidates = BTreeMap::new();
        for (id, value) in parameters {
            let Some(KernelNode::Parameter(parameter)) = self.node(id.erase()) else {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate Parameter is outside this Model",
                ));
            };
            if parameter_candidates.insert(id.erase(), value).is_some() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidates repeat one exact Parameter",
                ));
            }
            if value.value_type() != parameter.value().value_type() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate differs from the exact Parameter type",
                ));
            }
        }
        let mut observable_candidates = BTreeMap::new();
        for (id, value) in observables {
            let Some(KernelNode::Observable(observable)) = self.node(id.erase()) else {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate Observable is outside this Model",
                ));
            };
            if observable_candidates.insert(id.erase(), value).is_some() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidates repeat one exact Observable",
                ));
            }
            if value.value_type() != observable.value_type() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate differs from the exact Observable type",
                ));
            }
            if self.observable_output_support(*id)?.is_some() {
                return Err(kernel_error(
                    id.erase(),
                    "operand candidate cannot erase an Observable output support",
                ));
            }
        }
        crate::evaluate::evaluate_expression(
            relation.erase(),
            definition.expression(),
            &mut |symbol| match symbol {
                SymbolRef::Field(id) => candidates.get(&id.erase()).map(|value| (**value).clone()),
                SymbolRef::Parameter(id) => match self.node(id.erase()) {
                    Some(KernelNode::Parameter(parameter)) => Some(
                        parameter_candidates
                            .get(&id.erase())
                            .copied()
                            .unwrap_or_else(|| parameter.value())
                            .clone(),
                    ),
                    _ => None,
                },
                SymbolRef::Observable(id) => observable_candidates
                    .get(&id.erase())
                    .map(|value| (**value).clone()),
                _ => None,
            },
        )
    }
}
