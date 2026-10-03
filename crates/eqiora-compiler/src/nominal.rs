//! Lexical nominal declaration binding through existing elaboration identities.
use std::collections::BTreeMap;
mod local_index;
mod resolved;
pub(crate) use local_index::bind_local_index_types;
pub(crate) use resolved::bind_resolved;

use eqiora_core::{Diagnostic, EntityKind, RawId, ValueType, diagnostic::codes, entity::kinds};
use eqiora_lang::{Document, ExprKind, SourceAstFactory, ValueTypeSyntax, ValueTypeSyntaxKind};
use eqiora_schema::kernel::FiniteSpaceDef;

use crate::diagnostics::source_error;
use crate::identity::{
    DeclarationPath, ElaborationKey, IdentityNamespace, InstancePath, StagingIdAllocator,
};

#[derive(Debug, Clone)]
pub(crate) struct BoundFiniteSpace {
    pub(crate) definition: FiniteSpaceDef,
    pub(crate) key: ElaborationKey,
    pub(crate) range: eqiora_lang::TextRange,
}

pub(crate) fn finite_spaces(
    file: &str,
    document: &Document,
    namespace: &IdentityNamespace,
    mut native_identity: impl FnMut(&str) -> Option<RawId>,
) -> Result<BTreeMap<String, BoundFiniteSpace>, Vec<Diagnostic>> {
    let mut values = BTreeMap::new();
    let mut errors = Vec::new();
    for declaration in document.finite_spaces() {
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
            if callee.as_str() != "orthonormal"
                || declaration.value_type().is_some()
                || declaration.domain().is_some()
                || declaration.activation().is_some()
            {
                return Err(invalid(
                    "finite space requires only its closed orthonormal basis declaration",
                ));
            }
            let labels = arguments
                .iter()
                .map(|argument| match argument.kind() {
                    ExprKind::Name(label) => Ok(label.clone()),
                    _ => Err(invalid(
                        "finite space basis entries must be distinct labels",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
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
            let definition =
                FiniteSpaceDef::new(id, labels).map_err(|error| invalid(&error.to_string()))?;
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

pub(crate) fn bind_finite_types(
    file: &str,
    document: &mut Document,
    spaces: &BTreeMap<String, BoundFiniteSpace>,
) -> Result<(), Vec<Diagnostic>> {
    let mut errors = Vec::new();
    SourceAstFactory::visit_value_types(document, |_, syntax| {
        if let Err(error) = bind_type(file, syntax, spaces) {
            errors.push(error);
        }
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn bind_type(
    file: &str,
    syntax: &mut ValueTypeSyntax,
    spaces: &BTreeMap<String, BoundFiniteSpace>,
) -> Result<(), Diagnostic> {
    let value: Option<ValueType> = match syntax.kind() {
        ValueTypeSyntaxKind::Coordinates(name) | ValueTypeSyntaxKind::Counts(name) => {
            let definition = spaces.get(name.as_str()).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    syntax.range(),
                    format!("unresolved finite space `{name}`"),
                )
            })?;
            Some(if matches!(syntax.kind(), ValueTypeSyntaxKind::Counts(_)) {
                definition.definition.counts()
            } else {
                definition
                    .definition
                    .coordinates(
                        eqiora_core::ScalarDomain::Integer,
                        eqiora_core::DimExponents::DIMENSIONLESS,
                    )
                    .expect("integer coordinates")
            })
        }
        _ => None,
    };
    if let Some(value) = value {
        SourceAstFactory::bind_nominal_value_type(syntax, value).map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                syntax.range(),
                error.message(),
            )
        })?;
    }
    Ok(())
}

pub(crate) fn bind_finite_expressions(
    file: &str,
    document: &mut Document,
    spaces: &BTreeMap<String, BoundFiniteSpace>,
) -> Result<(), Vec<Diagnostic>> {
    let mut errors = Vec::new();
    SourceAstFactory::visit_expressions(document, |_, expression| {
        let ExprKind::Call {
            callee,
            arguments: eqiora_lang::CallArguments::Positional(arguments),
        } = expression.kind()
        else {
            return;
        };
        if !matches!(callee.as_str(), "counts" | "coordinates") {
            return;
        }
        let is_count = callee.as_str() == "counts";
        let name = arguments
            .first()
            .and_then(|argument| match argument.kind() {
                ExprKind::Name(name) => Some(name.clone()),
                ExprKind::Path(name) => Some(name.as_str().to_owned()),
                _ => None,
            });
        let result = (|| {
            let invalid = |message: &str| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    message,
                )
            };
            let name = name.as_deref().ok_or_else(|| {
                invalid("nominal constructor requires its exact declaration name")
            })?;
            let declaration = spaces
                .get(name)
                .ok_or_else(|| invalid("unresolved finite space in nominal constructor"))?;
            let value_type = if is_count {
                declaration.definition.counts()
            } else {
                declaration
                    .definition
                    .coordinates(
                        eqiora_core::ScalarDomain::Integer,
                        eqiora_core::DimExponents::DIMENSIONLESS,
                    )
                    .expect("integer coordinates")
            };
            let path = eqiora_lang::NamePath::from_segments(name.split('.'), expression.range())
                .map_err(|error| invalid(error.message()))?;
            SourceAstFactory::bind_nominal_expression(expression, &path, value_type).map_err(
                |error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        error.message(),
                    )
                },
            )?;
            literal(file, expression).map(|_| ())
        })();
        if let Err(error) = result {
            errors.push(error);
        }
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub(crate) fn literal(
    file: &str,
    expression: &eqiora_lang::Expr,
) -> Result<eqiora_core::ValueLiteral, Diagnostic> {
    if let Some(value) = expression.resolved_enum() {
        return Ok(value.clone());
    }

    let invalid = |message: &str| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let value_type = expression
        .resolved_nominal()
        .ok_or_else(|| invalid("nominal constructor requires exact lexical resolution"))?;
    let ExprKind::Call {
        arguments: eqiora_lang::CallArguments::Positional(arguments),
        ..
    } = expression.kind()
    else {
        return Err(invalid("nominal literal requires a constructor"));
    };
    let argument = arguments
        .get(1)
        .ok_or_else(|| invalid("nominal constructor requires one value argument"))?;
    let components = if value_type.index_set().is_some() {
        vec![integer_literal(file, argument)?]
    } else {
        let ExprKind::Array(elements) = argument.kind() else {
            return Err(invalid(
                "nominal coordinates require an explicit flat component list",
            ));
        };
        elements
            .iter()
            .map(|element| integer_literal(file, element))
            .collect::<Result<Vec<_>, _>>()?
    };
    eqiora_core::ValueLiteral::integer(value_type.clone(), components)
        .map_err(|error| invalid(&error.to_string()))
}
fn integer_literal(file: &str, expression: &eqiora_lang::Expr) -> Result<i64, Diagnostic> {
    crate::hierarchy::exact_signed_literal(expression)
        .ok_or_else(|| source_error(codes::LANGUAGE_TYPE_ERROR, file, expression.range(), "nominal constructor components must be closed integer literals; mutable Parameter-dependent construction is not admitted"))?
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, expression.range(), error.message()))
}
