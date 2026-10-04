//! Spatial integrals are declaration reductions, not general expression operators.
use super::*;

#[derive(Clone, Copy)]
pub(in crate::hierarchy) struct IntegralSource<'a> {
    pub(in crate::hierarchy) domain: &'a str,
    pub(in crate::hierarchy) measure: Option<eqiora_schema::kernel::ObservableMeasure>,
    pub(in crate::hierarchy) limits: Option<[&'a eqiora_lang::Expr; 2]>,
}

type ObservableSource<'a> = (&'a eqiora_lang::Expr, Option<IntegralSource<'a>>);

pub(in crate::hierarchy) fn split<'a>(
    file: &str,
    expression: &'a eqiora_lang::Expr,
) -> Result<ObservableSource<'a>, Diagnostic> {
    if let eqiora_lang::ExprKind::Call { callee, arguments } = expression.kind()
        && callee.as_str() == "integral"
    {
        let (positional, named) = arguments.parts();
        let [integrand, measure] = positional else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                "Observable integral requires an integrand and measure(domain)",
            ));
        };
        let mut limits = [None, None];
        for binding in named {
            let index = match binding.name() {
                "lower" => 0,
                "upper" => 1,
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        binding.range(),
                        "integral options are exactly lower and upper",
                    ));
                }
            };
            if limits[index].replace(binding.value()).is_some() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    binding.range(),
                    "integral limit is bound more than once",
                ));
            }
        }
        let limits = match limits {
            [None, None] => None,
            [Some(lower), Some(upper)] => Some([lower, upper]),
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    expression.range(),
                    "explicit integral limits require both lower and upper",
                ));
            }
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
            Some(IntegralSource {
                domain: name,
                measure: (callee.as_str() == "spherical_measure")
                    .then_some(eqiora_schema::kernel::ObservableMeasure::SphericalVolume),
                limits,
            }),
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
        .map(|source| {
            let domain = resolve_local_kind(
                file,
                expression.range(),
                scope,
                source.domain,
                |kind| matches!(kind, SymbolKind::Domain),
                "Observable integration Domain",
            )?;
            Ok::<_, Diagnostic>(crate::lower::LoweringIntegral {
                domain: domain.internal_name.clone(),
                measure: source.measure,
                limits: source
                    .limits
                    .map(|limits| {
                        limits
                            .into_iter()
                            .map(|limit| {
                                crate::hierarchy::scope::rewrite_expression_with_boundary_member(
                                    file, limit, scope, None,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(|values| values.try_into().expect("two limits"))
                    })
                    .transpose()?,
            })
        })
        .transpose()?;
    Ok((
        crate::hierarchy::scope::rewrite_expression_with_boundary_member(file, value, scope, None)?,
        reduction,
    ))
}
