//! Two-pass binding of atomic spaces and structural product aliases.
use super::*;

pub(crate) fn finite_spaces(
    file: &str,
    document: &Document,
    namespace: &IdentityNamespace,
    mut native_identity: impl FnMut(&str) -> Option<RawId>,
) -> Result<BTreeMap<String, BoundFiniteSpace>, Vec<Diagnostic>> {
    check_names(file, document)?;
    let mut atomic = declarations(file, document, namespace, None, &mut native_identity)?;
    let products = declarations(file, document, namespace, Some(&atomic), native_identity)?;
    atomic.extend(products);
    Ok(atomic)
}

pub(super) fn check_names(file: &str, document: &Document) -> Result<(), Vec<Diagnostic>> {
    let mut names = std::collections::BTreeSet::new();
    for declaration in document.finite_spaces() {
        if !names.insert(declaration.name()) {
            return Err(vec![source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "finite space name is declared more than once",
            )]);
        }
    }
    Ok(())
}

pub(super) fn declarations(
    file: &str,
    document: &Document,
    namespace: &IdentityNamespace,
    products: Option<&BTreeMap<String, BoundFiniteSpace>>,
    mut native_identity: impl FnMut(&str) -> Option<RawId>,
) -> Result<BTreeMap<String, BoundFiniteSpace>, Vec<Diagnostic>> {
    let mut values = BTreeMap::new();
    let mut errors = Vec::new();
    for declaration in document.finite_spaces() {
        let is_product = matches!(declaration.value().kind(), ExprKind::Call { callee, .. } if callee.as_str() == "product");
        if is_product != products.is_some() {
            continue;
        }
        let result = (|| {
            let invalid = |message: &str| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    declaration.range(),
                    message,
                )
            };
            let ExprKind::Call {
                callee,
                arguments: eqiora_lang::CallArguments::Positional(arguments),
            } = declaration.value().kind()
            else {
                return Err(invalid(
                    "finite space requires an orthonormal basis declaration",
                ));
            };
            if callee.as_str() != if is_product { "product" } else { "orthonormal" }
                || declaration.value_type().is_some()
                || declaration.domain().is_some()
                || declaration.activation().is_some()
            {
                return Err(invalid(
                    "finite space requires only its closed orthonormal basis declaration",
                ));
            }
            let key = ElaborationKey::entity(
                namespace.clone(),
                InstancePath::new(["$definitions"])?,
                DeclarationPath::new(["space", declaration.name()])?,
                EntityKind::FiniteSpace,
            )?;
            let id = if let Some(raw) = native_identity(declaration.name()) {
                raw.downcast::<kinds::FiniteSpace>()
                    .ok_or_else(|| invalid("native finite space has a different entity kind"))?
            } else {
                let mut identities = StagingIdAllocator::new();
                let full = identities.stage(&key)?;
                identities
                    .finish()
                    .resolve::<kinds::FiniteSpace>(full)?
                    .id()
            };
            let definition = if let Some(visible) = products {
                if arguments.len() != 2 {
                    return Err(invalid("product requires exactly two atomic factors"));
                }
                let factors = arguments
                    .iter()
                    .map(|argument| {
                        let syntax = eqiora_lang::FiniteBasisSyntax::from_expression(argument)
                            .ok_or_else(|| {
                                invalid("product factors require exact atomic space names")
                            })?;
                        let factor = visible.get(syntax.name.as_str()).ok_or_else(|| {
                            invalid("product factor is not a visible atomic space")
                        })?;
                        if syntax.dual || factor.definition.factors().is_some() {
                            return Err(invalid(
                                "product declarations require primal atomic factors",
                            ));
                        }
                        Ok(factor.definition.basis())
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                FiniteSpaceDef::product(id, factors[0], factors[1])
            } else {
                let labels = arguments
                    .iter()
                    .map(|argument| match argument.kind() {
                        ExprKind::Name(label) => Ok(label.clone()),
                        _ => Err(invalid(
                            "finite space basis entries must be distinct labels",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                FiniteSpaceDef::new(id, labels)
            }
            .map_err(|error| invalid(&error.to_string()))?;
            Ok(BoundFiniteSpace {
                definition,
                key,
                range: declaration.range(),
            })
        })();
        match result {
            Ok(value) if !values.contains_key(declaration.name()) => {
                values.insert(declaration.name().to_owned(), value);
            }
            Ok(_) => errors.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "finite space name is declared more than once",
            )),
            Err(error) => errors.push(error),
        }
    }
    if errors.is_empty() {
        Ok(values)
    } else {
        Err(errors)
    }
}
