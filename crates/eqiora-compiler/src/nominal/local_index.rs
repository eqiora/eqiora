//! Resolve closed root-local index types before their Parameter defaults are checked.
use super::*;
use eqiora_lang::Item;
use eqiora_schema::kernel::IndexSetDef;

pub(crate) fn bind_local_index_types(
    file: &str,
    document: &mut Document,
    namespace: &IdentityNamespace,
    mut native_identity: impl FnMut(&str) -> Option<RawId>,
) -> Result<(), Vec<Diagnostic>> {
    let mut sets = BTreeMap::new();
    for model in document.models() {
        let mut local = BTreeMap::new();
        for item in model.items() {
            let Item::IndexSet(declaration) = item else {
                continue;
            };
            let ExprKind::Call {
                callee,
                arguments: eqiora_lang::CallArguments::Positional(arguments),
            } = declaration.value().kind()
            else {
                continue;
            };
            let [extent] = arguments.as_slice() else {
                continue;
            };
            if callee.as_str() != "range" {
                continue;
            }
            // Unbound Parameter extents are resolved by occurrence allocation. They
            // cannot supply an exact typed index before that specialization.
            let Ok(value) = crate::hierarchy::closed_value(
                file,
                extent,
                ValueType::scalar(
                    eqiora_core::ScalarDomain::Integer,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .expect("admitted numeric scalar type"),
            ) else {
                continue;
            };
            let Some(extent) = value
                .integer_scalar_value()
                .and_then(|value| u32::try_from(value).ok())
            else {
                continue;
            };
            let key = ElaborationKey::entity(
                namespace.clone(),
                InstancePath::new([model.name()]).map_err(|e| vec![e])?,
                DeclarationPath::new(["indexset", model.name(), declaration.name()])
                    .map_err(|e| vec![e])?,
                EntityKind::IndexSet,
            )
            .map_err(|e| vec![e])?;
            let id = if let Some(id) = native_identity(declaration.name()) {
                id.downcast::<kinds::IndexSet>().ok_or_else(|| {
                    vec![source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        "native index set has a different entity kind",
                    )]
                })?
            } else {
                let mut staging = StagingIdAllocator::new();
                let full = staging.stage(&key).map_err(|e| vec![e])?;
                staging
                    .finish()
                    .resolve::<kinds::IndexSet>(full)
                    .map_err(|e| vec![e])?
                    .id()
            };
            let definition = IndexSetDef::new(id, extent).map_err(|error| {
                vec![source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    declaration.range(),
                    error.message(),
                )]
            })?;
            local.insert(declaration.name().to_owned(), definition);
        }
        sets.insert(model.name().to_owned(), local);
    }
    let mut errors = Vec::new();
    SourceAstFactory::visit_value_types(document, |scope, syntax| {
        let ValueTypeSyntaxKind::Index(name) = syntax.kind() else {
            return;
        };
        let definition = scope
            .and_then(|scope| sets.get(scope))
            .and_then(|sets| sets.get(name.as_str()));
        let Some(definition) = definition else {
            errors.push(source_error(codes::LANGUAGE_TYPE_ERROR,file,syntax.range(),format!("index type `{name}` requires a closed Model-local IndexSet before Parameter validation")));
            return;
        };
        if let Err(error) =
            SourceAstFactory::bind_nominal_value_type(syntax, definition.value_type())
        {
            errors.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                syntax.range(),
                error.message(),
            ));
        }
    });
    SourceAstFactory::visit_expressions(document, |scope, expression| {
        let ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } = expression.kind()
        else {
            return;
        };
        if callee.as_str() != "index" {
            return;
        }
        let Some(name) = arguments.first().and_then(|arg| match arg.kind() {
            ExprKind::Name(name) => Some(name.as_str()),
            ExprKind::Path(path) => Some(path.as_str()),
            _ => None,
        }) else {
            return;
        };
        let Some(definition) = scope
            .and_then(|scope| sets.get(scope))
            .and_then(|sets| sets.get(name))
        else {
            return;
        };
        let path = eqiora_lang::NamePath::from_segments(name.split('.'), expression.range())
            .expect("authored nominal name");
        if let Err(error) = SourceAstFactory::bind_nominal_expression(
            expression,
            std::slice::from_ref(&path),
            definition.value_type(),
        ) {
            errors.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                error.message(),
            ));
        }
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_index_parameter_type_and_default_share_the_allocated_identity() {
        let source = "model M() { indexset Rows=range(3); parameter chosen:index<Rows>=index(Rows,2); variable y:1; relation law {y=0;} }";
        let mut document = eqiora_lang::parse("index.eqi", source)
            .into_compilation_document()
            .unwrap();
        let identity =
            crate::source_identity::LocalSourceIdentity::from_document(&document).unwrap();
        let namespace = identity.namespace().unwrap();
        bind_local_index_types("index.eqi", &mut document, &namespace, |_| None).unwrap();
        let parameter = document.models()[0]
            .items()
            .iter()
            .find_map(|item| match item {
                Item::Parameter(value) => Some(value),
                _ => None,
            })
            .unwrap();
        let value_type = parameter.value_type().resolved_nominal().unwrap();
        let literal =
            crate::hierarchy::closed_value("index.eqi", parameter.value(), value_type.clone())
                .unwrap();
        assert_eq!(literal.ordinal().unwrap().integer_scalar_value(), Some(2));
        let key = ElaborationKey::entity(
            namespace,
            InstancePath::new(["M"]).unwrap(),
            DeclarationPath::new(["indexset", "M", "Rows"]).unwrap(),
            EntityKind::IndexSet,
        )
        .unwrap();
        let mut staging = StagingIdAllocator::new();
        let full = staging.stage(&key).unwrap();
        let id = staging
            .finish()
            .resolve::<kinds::IndexSet>(full)
            .unwrap()
            .id();
        assert_eq!(literal.value_type().index_set(), Some(id));
        assert_eq!(literal.value_type(), value_type);
    }
    #[test]
    fn closed_index_defaults_reject_foreign_sets_and_bounds() {
        for initializer in ["index(Rows,3)", "index(Rows,-1)", "index(Other,2)"] {
            let source = format!(
                "model M() {{ indexset Rows=range(3); indexset Other=range(3); parameter chosen:index<Rows>={initializer}; variable y:1; relation law {{y=0;}} }}"
            );
            let mut document = eqiora_lang::parse("index.eqi", &source)
                .into_compilation_document()
                .unwrap();
            let identity =
                crate::source_identity::LocalSourceIdentity::from_document(&document).unwrap();
            bind_local_index_types(
                "index.eqi",
                &mut document,
                &identity.namespace().unwrap(),
                |_| None,
            )
            .unwrap();
            let parameter = document.models()[0]
                .items()
                .iter()
                .find_map(|item| match item {
                    Item::Parameter(value) => Some(value),
                    _ => None,
                })
                .unwrap();
            assert!(
                crate::hierarchy::closed_value(
                    "index.eqi",
                    parameter.value(),
                    parameter.value_type().resolved_nominal().unwrap().clone()
                )
                .is_err(),
                "{initializer}"
            );
        }
    }
}
