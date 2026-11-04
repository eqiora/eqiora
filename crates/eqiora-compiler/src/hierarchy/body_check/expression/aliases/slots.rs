//! Expression requirements use the same typed alias and dependency owner as local values.
use super::*;

fn slot_type(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::FieldDecl,
    support: SpatialSupport<String>,
) -> Result<ExpressionType<String>, Diagnostic> {
    let invalid = |message| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            message,
        )
    };
    if declaration.role() != eqiora_lang::FieldRoleSyntax::Variable
        || declaration.spatial_regularity() != eqiora_schema::kernel::SpatialRegularity::Unspecified
        || declaration.activation() != &ActivationSyntax::Continuous
    {
        return Err(invalid(
            "expression slots require continuous values without Field roles or regularity",
        ));
    }
    let ty =
        crate::value_types::lower_value_type(scope.file, declaration.value_type(), Some(&support))?;
    Ok(ExpressionType::new(ty, Some(support)))
}

pub(in crate::hierarchy::body_check) fn bind_expression_slots(
    scope: &mut DefinitionScope<'_, '_>,
    signature: &[eqiora_lang::SignatureItem],
) -> Result<(), Vec<Diagnostic>> {
    let mut errors = Vec::new();
    for item in signature {
        let eqiora_lang::SignatureItem::Expression(declaration) = item else {
            continue;
        };
        let result = (|| {
            let support = declaration
                .domain()
                .and_then(|name| scope.spatial_support(name))
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        scope.file,
                        declaration.range(),
                        "expression slot requires an exact support",
                    )
                })?;
            let inferred = slot_type(scope, declaration, support)?;
            let expression = eqiora_lang::SourceAstFactory::expression(
                ExprKind::Name(declaration.name().into()),
                declaration.range(),
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    error.to_string(),
                )
            })?;
            let alias = AliasContract {
                inferred,
                expression,
                field_target: None,
                activation: DependencyActivation::Continuous,
                event_context: None,
                range: declaration.range(),
                dependencies: Vec::new(),
                evolution: Vec::new(),
                contextual: Vec::new(),
                endpoints: PhysicalEndpointSelections::new(),
            };
            if scope
                .symbols
                .insert(
                    declaration.name().into(),
                    SymbolContract::Alias(Arc::new(alias)),
                )
                .is_some()
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "expression slot collides with another declaration",
                ));
            }
            Ok(())
        })();
        if let Err(error) = result {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub(in crate::hierarchy::body_check) fn validate_expression_bindings(
    scope: &DefinitionScope<'_, '_>,
    instance: &eqiora_lang::InstanceDecl,
) -> Result<(), Diagnostic> {
    let Some(child) = scope.children.get(instance.name()) else {
        return Ok(());
    };
    for item in child.declaration.signature() {
        let eqiora_lang::SignatureItem::Expression(declaration) = item else {
            continue;
        };
        let invalid = |message| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                instance.range(),
                message,
            )
        };
        let binding = instance
            .bindings()
            .iter()
            .find(|binding| binding.name() == declaration.name())
            .ok_or_else(|| {
                invalid(format!(
                    "instance `{}` has no binding for required expression slot `{}`",
                    instance.name(),
                    declaration.name()
                ))
            })?;
        let support = declaration
            .domain()
            .and_then(|name| scope.child_bound_support(instance, name))
            .ok_or_else(|| invalid("expression slot requires an exact bound support".into()))?;
        let expected = slot_type(scope, declaration, support.clone())?;
        let mut checker = ExpressionChecker {
            scope,
            relation_support: Some(support.clone()),
            family_scope: None,
            allow_discrete_symbols: false,
            initial: false,
            activation: &ActivationSyntax::Continuous,
            physical_endpoints: PhysicalEndpointSelections::new(),
            intrinsic: false,
            alias_dependencies: Vec::new(),
            evolution: Vec::new(),
            contextual: Vec::new(),
            sampling: false,
            allow_observables: false,
        };
        let inferred = checker.check(binding.value())?;
        if inferred.value_type != expected.value_type
            || inferred
                .support
                .as_ref()
                .is_some_and(|actual| actual != &support)
        {
            return Err(invalid(format!(
                "expression slot `{}` requires its declared value type and exact bound support",
                declaration.name()
            )));
        }
        if !matches!(
            DependencyActivation::infer(scope, binding.value()),
            DependencyActivation::Static | DependencyActivation::Continuous
        ) {
            return Err(invalid(
                "expression slot requires continuous dependencies".into(),
            ));
        }
    }
    Ok(())
}
