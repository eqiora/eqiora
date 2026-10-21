//! Structural Jacobian action lists; they are not homogeneous value arrays.
use super::*;

pub(crate) fn arguments<'a>(
    file: &str,
    expression: &Expr,
    arguments: &'a CallArguments,
) -> Result<(&'a Expr, &'a [Expr], &'a [Expr]), Diagnostic> {
    let Some([value, bindings, directions]) = arguments.positional() else {
        return Err(pure_error(
            file,
            expression.range(),
            "jvp requires value, ordered binding list and ordered direction list",
        ));
    };
    let (ExprKind::Array(bindings), ExprKind::Array(directions)) =
        (bindings.kind(), directions.kind())
    else {
        return Err(pure_error(
            file,
            expression.range(),
            "jvp bindings and directions require explicit lists",
        ));
    };
    if bindings.is_empty()
        || bindings.len() != directions.len()
        || bindings.len() > eqiora_schema::kernel::pure_operator::MAX_FORMALS
    {
        return Err(pure_error(
            file,
            expression.range(),
            "jvp requires equally sized nonempty bounded lists",
        ));
    }
    Ok((value, bindings, directions))
}

pub(crate) fn binding(file: &str, expression: &Expr) -> Result<eqiora_lang::NamePath, Diagnostic> {
    match expression.kind() {
        ExprKind::Name(name) => {
            eqiora_lang::NamePath::from_segments([name.as_str()], expression.range())
                .map_err(|error| pure_error(file, expression.range(), error.to_string()))
        }
        ExprKind::Path(path) => Ok(path.clone()),
        _ => Err(pure_error(
            file,
            expression.range(),
            "derivative action selector must name an independent binding",
        )),
    }
}

pub(crate) fn vjp_arguments<'a>(
    file: &str,
    expression: &Expr,
    args: &'a CallArguments,
) -> Result<(&'a Expr, &'a Expr, &'a Expr), Diagnostic> {
    let Some([value, selector, cotangent]) = args.positional() else {
        return Err(pure_error(
            file,
            expression.range(),
            "vjp requires value, selected input and output cotangent",
        ));
    };
    binding(file, selector)?;
    Ok((value, selector, cotangent))
}
