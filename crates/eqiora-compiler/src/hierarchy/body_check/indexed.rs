//! Definition checking of finite families with exact static ordinal substitution.
use super::{scope::DefinitionScope, *};
use eqiora_lang::{FamilyBinderSyntax, RelationDecl, RelationFamilyDecl, SourceAstFactory};

pub(super) fn extent(
    scope: &DefinitionScope<'_, '_>,
    binder: &FamilyBinderSyntax,
) -> Result<u32, Diagnostic> {
    if binder.member() == "time"
        || scope.symbols.contains_key(binder.member())
        || scope.static_values.contains_key(binder.member())
        || scope.children.contains_key(binder.member())
        || scope.index_sets.contains_key(binder.member())
    {
        return Err(crate::diagnostics::source_error(
            eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            binder.range(),
            "indexed family binder collides with an existing name",
        ));
    }
    scope
        .index_sets
        .get(binder.set().as_str())
        .copied()
        .flatten()
        .filter(|extent| {
            usize::try_from(*extent).is_ok_and(|n| n <= scope.elaborator.limits.max_parameter_terms)
        })
        .ok_or_else(|| {
            crate::diagnostics::source_error(
                eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                binder.range(),
                "indexed family requires an existing nonempty bounded IndexSet",
            )
        })
}

pub(super) fn relation(
    file: &str,
    family: &RelationFamilyDecl,
    ordinal: u32,
) -> Result<RelationDecl, Diagnostic> {
    let declaration = family.relation();
    let equations = declaration
        .conditions()
        .ok_or_else(|| {
            super::super::hierarchy_error("indexed Laws require retained term lowering")
        })?
        .iter()
        .map(|equation| {
            let left = super::super::reductions::instantiate(
                file,
                equation.left(),
                family.binder(),
                ordinal,
            )?;
            let right = super::super::reductions::instantiate(
                file,
                equation.right(),
                family.binder(),
                ordinal,
            )?;
            SourceAstFactory::condition(equation.kind(), left, right, equation.range())
                .map_err(|error| construction(file, family.range(), error))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    SourceAstFactory::relation(
        declaration.name(),
        declaration.activation().clone(),
        declaration.domain().map(str::to_owned),
        equations,
        declaration.range(),
    )
    .map_err(|error| construction(file, family.range(), error))
}

fn construction(
    file: &str,
    range: TextRange,
    error: eqiora_lang::AstConstructionError,
) -> Diagnostic {
    crate::diagnostics::source_error(
        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
        file,
        range,
        error.to_string(),
    )
}

pub(super) fn connections(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::ConnectionDecl,
    connected: &mut BTreeSet<Vec<String>>,
    limits: ConnectionSetLimits,
) -> Result<Vec<super::PhysicalConnectionFragment>, Diagnostic> {
    let Some(binder) = declaration.binder() else {
        return super::scope::validate_connection(scope, declaration, connected, limits)
            .map(|fragment| fragment.into_iter().collect());
    };
    let extent = extent(scope, binder)?;
    let mut fragments = Vec::new();
    for ordinal in 0..extent {
        let ports = declaration
            .port_expressions()
            .iter()
            .map(|value| super::super::reductions::instantiate(scope.file, value, binder, ordinal))
            .collect::<Result<Vec<_>, _>>()?;
        let member =
            SourceAstFactory::connection(declaration.syntax(), None, ports, declaration.range())
                .map_err(|error| construction(scope.file, declaration.range(), error))?;
        fragments.extend(super::scope::validate_connection(
            scope, &member, connected, limits,
        )?);
    }
    Ok(fragments)
}
