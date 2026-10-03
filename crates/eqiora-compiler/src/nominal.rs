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
    let invalid =
        |message: String| source_error(codes::LANGUAGE_TYPE_ERROR, file, syntax.range(), message);
    let basis = |syntax: &eqiora_lang::FiniteBasisSyntax| {
        let declaration = spaces
            .get(syntax.name.as_str())
            .ok_or_else(|| invalid(format!("unresolved finite space `{}`", syntax.name)))?;
        let basis = declaration.definition.basis();
        Ok::<_, Diagnostic>(if syntax.dual { basis.dual() } else { basis })
    };
    let value: Option<ValueType> = match syntax.kind() {
        ValueTypeSyntaxKind::Counts(name) => Some(
            spaces
                .get(name.as_str())
                .ok_or_else(|| invalid(format!("unresolved finite space `{name}`")))?
                .definition
                .counts(),
        ),
        ValueTypeSyntaxKind::Coordinates {
            scalar,
            basis: coordinate,
        } => {
            let scalar = crate::value_types::lower_scalar_type(file, scalar)?;
            Some(
                ValueType::coordinates(
                    basis(coordinate)?,
                    scalar.scalar_domain(),
                    scalar.dimension(),
                )
                .map_err(|error| invalid(error.to_string()))?,
            )
        }
        ValueTypeSyntaxKind::LinearMap {
            scalar,
            source,
            target,
        } => {
            let scalar = crate::value_types::lower_scalar_type(file, scalar)?;
            Some(
                ValueType::linear_map(
                    basis(source)?,
                    basis(target)?,
                    scalar.scalar_domain(),
                    scalar.dimension(),
                )
                .map_err(|error| invalid(error.to_string()))?,
            )
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

mod values;
pub(crate) use values::{bind_finite_expressions, contextual_literal, literal};
