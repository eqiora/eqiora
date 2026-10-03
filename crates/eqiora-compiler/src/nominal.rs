//! Lexical nominal declaration binding through existing elaboration identities.
use std::collections::BTreeMap;
mod local_index;
mod resolved;
mod spaces;
pub(crate) use local_index::bind_local_index_types;
pub(crate) use resolved::bind_resolved;
pub(crate) use resolved::resolved_spaces;
pub(crate) use spaces::finite_spaces;

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
                .counts()
                .map_err(|error| invalid(error.to_string()))?,
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
