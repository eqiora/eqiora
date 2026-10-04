//! Spatial integrals are declaration reductions, not general expression operators.
use super::*;

type ObservableSource<'a> = (
    &'a eqiora_lang::Expr,
    Option<(&'a str, Option<eqiora_schema::kernel::ObservableMeasure>)>,
);

pub(in crate::hierarchy) fn split<'a>(
    file: &str,
    expression: &'a eqiora_lang::Expr,
) -> Result<ObservableSource<'a>, Diagnostic> {
    if let eqiora_lang::ExprKind::Call { callee, arguments } = expression.kind()
        && callee.as_str() == "integral"
    {
        let Some([integrand, measure]) = arguments.positional() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                "Observable integral requires an integrand and measure(domain)",
            ));
        };
        let eqiora_lang::ExprKind::Call { callee, arguments } = measure.kind() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                measure.range(),
                "Observable integral requires measure(domain)",
            ));
        };
        let Some([domain]) = arguments
            .positional()
            .filter(|_| matches!(callee.as_str(), "measure" | "spherical_measure"))
        else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                measure.range(),
                "Observable measure requires exactly one Domain",
            ));
        };
        let name = match domain.kind() {
            eqiora_lang::ExprKind::Name(name) => name.as_str(),
            eqiora_lang::ExprKind::Path(path) => path.as_str(),
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    domain.range(),
                    "Observable measure requires an exact Domain name",
                ));
            }
        };
        Ok((
            integrand,
            Some((
                name,
                (callee.as_str() == "spherical_measure")
                    .then_some(eqiora_schema::kernel::ObservableMeasure::SphericalVolume),
            )),
        ))
    } else {
        Ok((expression, None))
    }
}

pub(super) fn rewrite(
    file: &str,
    expression: &eqiora_lang::Expr,
    scope: &Scope,
) -> Result<
    (
        crate::lower::LoweringExpression,
        Option<crate::lower::LoweringIntegral>,
    ),
    Diagnostic,
> {
    let (value, reduction) = split(file, expression)?;
    let reduction = reduction
        .map(|(name, measure)| {
            let domain = resolve_local_kind(
                file,
                expression.range(),
                scope,
                name,
                |kind| matches!(kind, SymbolKind::Domain),
                "Observable integration Domain",
            )?;
            Ok::<_, Diagnostic>(crate::lower::LoweringIntegral {
                domain: domain.internal_name.clone(),
                measure,
            })
        })
        .transpose()?;
    Ok((
        crate::hierarchy::scope::rewrite_expression_with_boundary_member(file, value, scope, None)?,
        reduction,
    ))
}
