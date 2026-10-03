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

/// Reuse ordinary partial binding resolution and ordered residual arithmetic in bodies.
/// Direction types are checked before this structural expansion.
pub(crate) fn expand(
    file: &str,
    expression: &Expr,
    args: &CallArguments,
) -> Result<Expr, Diagnostic> {
    let (value, selectors, directions) = arguments(file, expression, args)?;
    let bindings = selectors
        .iter()
        .map(|item| binding(file, item))
        .collect::<Result<Vec<_>, _>>()?;
    let make = |kind| {
        eqiora_lang::SourceAstFactory::expression(kind, expression.range())
            .map_err(|error| pure_error(file, expression.range(), error.to_string()))
    };
    let mut result = None;
    for (index, (wrt, direction)) in bindings.iter().zip(directions).enumerate() {
        let partial = make(ExprKind::Partial {
            value: Box::new(value.clone()),
            wrt: wrt.clone(),
            holding: bindings
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, binding)| binding.clone())
                .collect(),
        })?;
        let term = make(ExprKind::Binary {
            op: BinaryOp::Mul,
            left: Box::new(partial),
            right: Box::new(direction.clone()),
        })?;
        result = Some(match result {
            None => term,
            Some(prior) => make(ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(prior),
                right: Box::new(term),
            })?,
        });
    }
    Ok(result.expect("checked nonempty action"))
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

pub(crate) fn expand_vjp(
    file: &str,
    expression: &Expr,
    args: &CallArguments,
) -> Result<Expr, Diagnostic> {
    let (value, selector, cotangent) = vjp_arguments(file, expression, args)?;
    let make = |kind| {
        eqiora_lang::SourceAstFactory::expression(kind, expression.range())
            .map_err(|error| pure_error(file, expression.range(), error.to_string()))
    };
    let partial = make(ExprKind::Partial {
        value: Box::new(value.clone()),
        wrt: binding(file, selector)?,
        holding: Vec::new(),
    })?;
    make(ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(cotangent.clone()),
        right: Box::new(partial),
    })
}
