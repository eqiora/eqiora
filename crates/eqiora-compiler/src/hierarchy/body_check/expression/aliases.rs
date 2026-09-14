//! Intrinsic alias types and shared, use-context evolution obligations.

mod activation;
pub(in crate::hierarchy) use activation::DependencyActivation;

use super::*;
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(in crate::hierarchy::body_check) struct AliasContract {
    pub(super) inferred: ExpressionType<String>,
    pub(super) expression: Expr,
    pub(super) field_target: Option<String>,
    activation: DependencyActivation,
    event_context: Option<String>,
    range: eqiora_lang::TextRange,
    dependencies: Vec<Arc<AliasContract>>,
    evolution: Vec<EvolutionRequirement>,
    contextual: Vec<Expr>,
    endpoints: PhysicalEndpointSelections,
}

#[derive(Debug, Clone)]
pub(super) struct EvolutionRequirement {
    operator: String,
    target: String,
    range: eqiora_lang::TextRange,
}

pub(in crate::hierarchy::body_check) fn validate_aliases<'a>(
    scope: &mut DefinitionScope<'_, '_>,
    declarations: impl Iterator<Item = &'a eqiora_lang::NamedDefinitionDecl>,
    static_values: &crate::hierarchy::parameters::SymbolicParameterMap,
) -> Result<(), Vec<Diagnostic>> {
    scope.static_values = static_values.clone();
    let order = crate::hierarchy::parameters::alias_order(scope.file, declarations)?;
    let mut errors = Vec::new();
    for declaration in order {
        if static_values.contains_key(declaration.name()) {
            if declaration.domain().is_some() {
                errors.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "static let alias cannot assert spatial support",
                ));
            }
            if declaration.activation().is_some() {
                errors.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "static let alias cannot assert a clock activation",
                ));
            }
            continue;
        }
        let mut checker = ExpressionChecker {
            scope,
            relation_support: None,
            family_scope: None,
            allow_discrete_symbols: false,
            initial: false,
            activation: &ActivationSyntax::Continuous,
            physical_endpoints: PhysicalEndpointSelections::new(),
            intrinsic: true,
            alias_dependencies: Vec::new(),
            evolution: Vec::new(),
            contextual: Vec::new(),
            sampling: false,
        };
        let inferred = match checker.check(declaration.value()).and_then(|inferred| {
            if let Some(domain) = declaration.domain() {
                let expected = scope.spatial_support(domain).ok_or_else(|| {
                    scope.wrong_local_kind(declaration.range(), domain, "let alias spatial support")
                })?;
                if inferred.support.as_ref() != Some(&expected) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        scope.file,
                        declaration.range(),
                        "let alias support assertion does not match its inferred exact support",
                    ));
                }
            }
            if let Some(assertion) = declaration.value_type() {
                let assertion = crate::hierarchy::parameters::specialize_type(
                    scope.file,
                    assertion,
                    &scope.static_values,
                )?;
                let expected = crate::value_types::lower_value_type(
                    scope.file,
                    &assertion,
                    inferred.support.as_ref(),
                )?;
                if expected != inferred.value_type {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        scope.file,
                        declaration.range(),
                        "let alias type assertion does not match its inferred value type",
                    ));
                }
            }
            Ok(inferred)
        }) {
            Ok(value) => value,
            Err(error) => {
                errors.push(error);
                continue;
            }
        };
        let mut activation = DependencyActivation::infer(scope, declaration.value());
        if let Err(error) = activation.validate(scope, declaration) {
            errors.push(error);
            continue;
        }
        let event_context = declaration
            .activation()
            .filter(|name| matches!(scope.symbols.get(*name), Some(SymbolContract::Event)))
            .map(str::to_owned);
        if let Some(event) = &event_context {
            let eligible = |requirement: &EvolutionRequirement| {
                matches!(requirement.operator.as_str(), "pre" | "next")
                    && matches!(
                        scope.symbols.get(&requirement.target),
                        Some(SymbolContract::Field(
                            _,
                            eqiora_lang::FieldRoleSyntax::State,
                            ActivationSyntax::Continuous
                        ))
                    )
            };
            let mut has_reset = checker.evolution.iter().any(eligible);
            let mut pending = checker.alias_dependencies.clone();
            let mut seen = BTreeSet::new();
            while let Some(alias) = pending.pop() {
                if seen.insert(Arc::as_ptr(&alias) as usize) {
                    has_reset |= alias.evolution.iter().any(eligible);
                    pending.extend(alias.dependencies.iter().cloned());
                }
            }
            if !has_reset {
                errors.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "event-local let assertion requires a continuous-state reset use",
                ));
                continue;
            }
            activation = DependencyActivation::Event(event.clone());
        }
        let field_target = match declaration.value().kind() {
            ExprKind::Path(path) => match scope.symbols.get(path.as_str()) {
                Some(SymbolContract::Field(..)) => Some(path.as_str().to_owned()),
                Some(SymbolContract::Alias(alias)) => alias.field_target.clone(),
                _ => None,
            },
            ExprKind::Name(name) => match scope.symbols.get(name) {
                Some(SymbolContract::Field(..)) => Some(name.clone()),
                Some(SymbolContract::Alias(alias)) => alias.field_target.clone(),
                _ => None,
            },
            _ => None,
        };
        let alias = AliasContract {
            expression: declaration.value().clone(),
            inferred,
            activation,
            event_context,
            range: declaration.range(),
            field_target,
            dependencies: checker.alias_dependencies,
            evolution: checker.evolution,
            contextual: checker.contextual,
            endpoints: checker.physical_endpoints,
        };
        scope.symbols.insert(
            declaration.name().to_owned(),
            SymbolContract::Alias(Arc::new(alias)),
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn use_alias(
        &mut self,
        alias: Arc<AliasContract>,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let inferred = alias.inferred.clone();
        if self.intrinsic {
            self.alias_dependencies.push(alias);
            return Ok(inferred);
        }
        let mut pending = vec![alias];
        let mut seen = BTreeSet::new();
        while let Some(alias) = pending.pop() {
            if !seen.insert(Arc::as_ptr(&alias) as usize) {
                continue;
            }
            if let Some(event) = &alias.event_context
                && (self.activation != &ActivationSyntax::Named(event.clone())
                    || self.initial
                    || self.sampling)
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    alias.range,
                    "event-local let alias requires its exact event activation",
                ));
            }
            self.physical_endpoints
                .extend(alias.endpoints.iter().cloned());
            for requirement in &alias.evolution {
                let argument = eqiora_lang::SourceAstFactory::expression(
                    if requirement.target.contains('.') {
                        ExprKind::Path(
                            eqiora_lang::NamePath::from_segments(
                                requirement.target.split('.'),
                                requirement.range,
                            )
                            .expect("retained source path"),
                        )
                    } else {
                        ExprKind::Name(requirement.target.clone())
                    },
                    requirement.range,
                )
                .expect("retained source name and range");
                self.check_evolution(&requirement.operator, &argument, &argument)?;
            }
            for expression in &alias.contextual {
                self.check(expression)?;
            }
            pending.extend(alias.dependencies.iter().cloned());
        }
        Ok(inferred)
    }

    pub(super) fn check_evolution(
        &mut self,
        callee_name: &str,
        expression: &Expr,
        argument: &Expr,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        if self.sampling {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "sample operand cannot contain an evolution operator",
            ));
        }
        let name = match argument.kind() {
            ExprKind::Name(name) => name.as_str(),
            ExprKind::Path(path) => path.as_str(),
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    argument.range(),
                    format!("{callee_name}(...) requires one exact Field or record member"),
                ));
            }
        };
        if let Some(SymbolContract::Alias(alias)) = self.scope.symbols.get(name) {
            // Identity aliases retain their use obligations even inside pre/next/derivative.
            self.use_alias(alias.clone())?;
        }
        let target = match self.scope.symbols.get(name) {
            Some(SymbolContract::Alias(alias)) => {
                alias.field_target.as_deref().ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        argument.range(),
                        "evolution operator requires an identity alias of one eligible Field",
                    )
                })?
            }
            _ => name,
        };
        let inferred = match self.scope.symbols.get(target) {
            Some(SymbolContract::Field(inferred, role, activation)) => {
                if matches!(callee_name, "derivative" | "pre" | "next")
                    && (*role != eqiora_lang::FieldRoleSyntax::State
                        || (callee_name == "derivative"
                            && !matches!(activation, ActivationSyntax::Continuous)))
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        "evolution operator requires an eligible declared state",
                    ));
                }
                if matches!(callee_name, "pre" | "next") {
                    let continuous_state = matches!(activation, ActivationSyntax::Continuous);
                    let event_reset = matches!(self.activation, ActivationSyntax::Named(name)
                        if matches!(self.scope.symbols.get(name), Some(SymbolContract::Event)));
                    let clocked_state = matches!(activation, ActivationSyntax::Named(name)
                        if matches!(self.scope.symbols.get(name), Some(SymbolContract::Clock)));
                    let eligible = if self.intrinsic {
                        continuous_state || clocked_state
                    } else if self.initial {
                        clocked_state && callee_name == "pre"
                    } else {
                        (continuous_state && event_reset)
                            || (clocked_state
                                && self.scope.activation_matches(activation, self.activation))
                    };
                    if !eligible {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            expression.range(),
                            "state evolution requires its exact clock or an eligible continuous-state event reset",
                        ));
                    }
                }
                inferred.clone()
            }
            _ => {
                return Err(unresolved(
                    self.scope.file,
                    argument.range(),
                    name,
                    "Field operator argument",
                ));
            }
        };
        if matches!(callee_name, "pre" | "next") && !self.intrinsic && !self.allow_discrete_symbols
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                format!("continuous Relation cannot use `{callee_name}`"),
            ));
        }
        if callee_name == "derivative"
            && !self.intrinsic
            && self.allow_discrete_symbols
            && !self.initial
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "clocked Relation cannot use `derivative`",
            ));
        }
        if self.intrinsic {
            self.evolution.push(EvolutionRequirement {
                operator: callee_name.to_owned(),
                target: target.to_owned(),
                range: expression.range(),
            });
        }
        match callee_name {
            "derivative" => typing::time_derivative(&inferred)
                .map_err(|error| type_error(self.scope.file, expression, error)),
            "pre" | "next" => Ok(inferred),
            _ => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                format!("unknown scalar operator `{callee_name}`"),
            )),
        }
    }
}

#[cfg(test)]
mod event_alias_tests {
    fn model(alias: &str, relation: &str) -> String {
        format!(
            "model M() {{ state x:1; event impact=crossing(x,direction=falling); event other=crossing(x,direction=rising); clock tick=periodic(1[s]); initial {{x=1;}} {alias} {relation} }}"
        )
    }

    #[test]
    fn event_local_and_deferred_evolution_aliases_keep_exact_context() {
        for alias in [
            "let old=pre(x);",
            "let old at impact=pre(x);",
            "let local at impact=pre(x); let old=local;",
        ] {
            let source = model(alias, "relation reset at impact {next(x)=-old;}");
            crate::compile("event-alias.eqi", &source)
                .unwrap_or_else(|errors| panic!("{source}: {errors:?}"));
        }
        for (alias, relation) in [
            ("let old=pre(x);", "relation r {x=old;}"),
            ("let old=pre(x);", "relation r at tick {x=old;}"),
            (
                "let local at impact=pre(x); let old=local;",
                "relation r {x=old;}",
            ),
            (
                "let local at impact=pre(x); let old=local;",
                "relation r at other {next(x)=old;}",
            ),
            (
                "let local at impact=x;",
                "relation r at impact {next(x)=local;}",
            ),
            (
                "let local at impact=1;",
                "relation r at impact {next(x)=local;}",
            ),
        ] {
            let source = model(alias, relation);
            assert!(
                crate::compile("event-alias.eqi", &source).is_err(),
                "{source}"
            );
        }
    }
}
