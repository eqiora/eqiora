//! Resolve exact value symbols with their activation and dependency contracts.
use super::*;

impl ExpressionLowerer<'_> {
    pub(super) fn lower_name(
        &mut self,
        expression: &LoweringExpression,
        name: &str,
    ) -> Result<TypedExpression, Diagnostic> {
        let Some(binding) = self.bindings.get(name).cloned() else {
            return Err(unresolved(
                self.file,
                expression.range(),
                name,
                "expression symbol",
            ));
        };
        let (symbol, id, dimension) = match binding {
            Binding::Field(id, contract) => {
                if self.sampling && !matches!(contract.activation, ActivationSyntax::Continuous) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "sample operand must be continuous",
                    ));
                }
                if contract.role == eqiora_lang::FieldRoleSyntax::Variable
                    && matches!(contract.activation, ActivationSyntax::Named(_))
                    && (self.initial || &contract.activation != self.activation)
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "clocked Variable read requires its exact declared activation",
                    ));
                }
                (SymbolRef::Field(id), id.erase(), contract.dimension)
            }
            Binding::Observable(id, value_type, domain, reduction) if self.allow_observables => {
                let ty = observable::reference_type(
                    self.file,
                    expression.range(),
                    &value_type,
                    domain.as_deref(),
                    reduction.as_ref(),
                    self.bindings,
                )?;
                (SymbolRef::Observable(id), id.erase(), ty.dimension())
            }
            Binding::Parameter(id, value_type) => {
                (SymbolRef::Parameter(id), id.erase(), value_type.dimension())
            }
            Binding::Port(id, contract) => match resolve_port_contract(
                self.file,
                expression.range(),
                &contract,
                self.bindings,
            )? {
                ResolvedPortContract::Signal {
                    value_type, clock, ..
                } => {
                    let expected = if self.sampling {
                        None
                    } else {
                        match self.activation {
                            ActivationSyntax::Named(name) => match self.bindings.get(name) {
                                Some(Binding::Clock(id, _)) => Some(*id),
                                _ => None,
                            },
                            _ => None,
                        }
                    };
                    if clock != expected {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.file,
                            expression.range(),
                            "signal Port read requires the exact declared activation; use an explicit transition",
                        ));
                    }
                    self.ports.insert(id.erase());
                    (SymbolRef::Port(id), id.erase(), value_type.dimension())
                }
                ResolvedPortContract::ScalarPhysical { .. } => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        format!(
                            "scalar physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    ));
                }
                ResolvedPortContract::BoundaryPhysical { .. } => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        format!(
                            "field-physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    ));
                }
            },
            Binding::Domain(_, _)
            | Binding::Representation(_)
            | Binding::Clock(_, _)
            | Binding::Observable(..)
            | Binding::Event(_)
            | Binding::Relation { .. } => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    format!("`{name}` is not a scalar Field, Parameter, or Port"),
                ));
            }
        };
        self.dependencies.insert(id);
        self.builder
            .symbol(symbol)
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }
}
