//! Exact declared coordinate selectors shared by definition and occurrence admission.
use crate::diagnostics::source_error;
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_lang::{ExprKind, NamedDefinitionDecl};

pub(super) fn selector<'a>(
    file: &str,
    declaration: &'a NamedDefinitionDecl,
) -> Result<(&'a str, usize), Diagnostic> {
    let invalid = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "coordinate requires an exact factor and a nonnegative integer axis",
        )
    };
    match declaration.value().kind() {
        ExprKind::Name(name) => Ok((name, 0)),
        ExprKind::Index { value, index } => {
            let (ExprKind::Name(name), ExprKind::Number(axis)) = (value.kind(), index.kind())
            else {
                return Err(invalid());
            };
            Ok((
                name,
                usize::try_from(axis.to_i64().map_err(|_| invalid())?).map_err(|_| invalid())?,
            ))
        }
        _ => Err(invalid()),
    }
}

pub(super) fn allocate<'a>(
    scope: &mut super::scope::Scope,
    file: &str,
    declarations: impl Iterator<Item = &'a NamedDefinitionDecl>,
) -> Result<(), Diagnostic> {
    for declaration in declarations {
        let (factor, axis) = selector(file, declaration)?;
        let support = declaration.domain().expect("validated coordinate support");
        let resolve = |name: &str| {
            scope
                .symbol(name)
                .filter(|symbol| matches!(symbol.kind, super::scope::SymbolKind::Domain))
                .map(|symbol| symbol.internal_name.clone())
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        format!("coordinate support `{name}` is unavailable"),
                    )
                })
        };
        let value = crate::lower::LoweringExpression::coordinate(
            resolve(support)?,
            resolve(factor)?,
            axis,
            declaration.range(),
        );
        scope
            .insert_coordinate(declaration.name().to_owned(), value)
            .map_err(super::hierarchy_error)?;
    }
    Ok(())
}
