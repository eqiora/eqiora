//! Exact authored endpoint projection before occurrence-specific family resolution.
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_lang::{Expr, ExprKind, NamePath};

pub(crate) fn path(file: &str, expression: &Expr) -> Result<NamePath, Diagnostic> {
    match expression.kind() {
        ExprKind::Path(path) => Ok(path.clone()),
        ExprKind::Name(name) => NamePath::from_segments([name.as_str()], expression.range())
            .map_err(|error| {
                crate::diagnostics::source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    error.message(),
                )
            }),
        _ => Err(crate::diagnostics::source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "Port endpoint requires an exact resolved occurrence selection",
        )),
    }
}
