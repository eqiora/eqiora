//! Lower an event guard with the ordinary expression and type owners.
use super::*;

pub(in crate::lower) fn lower_event_guard(
    file: &str,
    guard: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
) -> Result<LoweredRelation, Diagnostic> {
    let guard = contextual::value(file, guard, bindings)?;
    let inferred = expression_type(file, &guard, bindings, None)?;
    if inferred.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        || !inferred.shape().is_scalar()
        || inferred.frame() != eqiora_core::ValueFrame::Invariant
        || inferred.support.is_some()
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            guard.range(),
            "event crossing guard requires one real invariant scalar quantity",
        ));
    }
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: None,
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: false,
        activation: &ActivationSyntax::Continuous,
        initial: false,
    };
    let root = lowerer.lower(&guard)?.id;
    let expression = lowerer.builder.finish([root]).map_err(|error| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            guard.range(),
            error.message(),
        )
    })?;
    Ok(LoweredRelation {
        expression,
        dependencies: lowerer.dependencies,
        ports: lowerer.ports,
    })
}

impl ExpressionLowerer<'_> {
    pub(super) fn eligible_evolution(&self, callee: &str, contract: &FieldContract) -> bool {
        let event = matches!(self.activation, ActivationSyntax::Named(name) if matches!(self.bindings.get(name), Some(Binding::Event(_))));
        let event_state = event && matches!(contract.activation, ActivationSyntax::Continuous);
        contract.role == eqiora_lang::FieldRoleSyntax::State
            && match callee {
                "derivative" => matches!(contract.activation, ActivationSyntax::Continuous),
                "pre" => {
                    event_state
                        || (matches!(contract.activation, ActivationSyntax::Named(_))
                            && (self.initial || contract.activation == *self.activation))
                }
                "next" => {
                    !self.initial
                        && (event_state
                            || (matches!(contract.activation, ActivationSyntax::Named(_))
                                && contract.activation == *self.activation))
                }
                _ => false,
            }
    }
}
