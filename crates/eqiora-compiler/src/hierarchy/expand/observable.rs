//! Declaration reductions bind measures and laws outside ordinary pure expressions.
use super::*;

use crate::observation::Reduction;
type ObservableSource<'a> = (
    Vec<&'a eqiora_lang::Expr>,
    Option<Reduction<&'a str, &'a eqiora_lang::Expr>>,
);

pub(in crate::hierarchy) fn name(expression: &eqiora_lang::Expr) -> Option<&str> {
    match expression.kind() {
        eqiora_lang::ExprKind::Name(name) => Some(name),
        eqiora_lang::ExprKind::Path(path) => Some(path.as_str()),
        _ => None,
    }
}

pub(in crate::hierarchy) fn split<'a>(
    file: &str,
    expression: &'a eqiora_lang::Expr,
) -> Result<ObservableSource<'a>, Diagnostic> {
    if let eqiora_lang::ExprKind::Call { callee, arguments } = expression.kind() {
        let invalid = |message| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                message,
            )
        };
        if callee.as_str() == "random_variable" {
            let Some([distribution, value]) = arguments.positional() else {
                return Err(invalid(
                    "random_variable requires a probability law and one expression",
                ));
            };
            let distribution = name(distribution)
                .ok_or_else(|| invalid("random_variable requires an exact law name"))?;
            return Ok((
                vec![value],
                Some(Reduction::RandomVariable { distribution }),
            ));
        }
        if let Some(moment) = crate::observation::moment(callee.as_str()) {
            let values = arguments
                .positional()
                .filter(|values| values.len() == moment.arity())
                .ok_or_else(|| {
                    invalid("probability moment requires its ordered random variables")
                })?;
            if values.iter().any(|value| name(value).is_none()) {
                return Err(invalid(
                    "probability moment requires named random variables",
                ));
            }
            return Ok((
                values.iter().collect(),
                Some(Reduction::ProbabilityMoment { moment }),
            ));
        }
    }
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
            vec![integrand],
            Some(Reduction::SpatialIntegral {
                domain: name,
                measure: (callee.as_str() == "spherical_measure")
                    .then_some(eqiora_schema::kernel::ObservableMeasure::SphericalVolume),
                limits,
            }),
        ))
    } else {
        Ok((vec![expression], None))
    }
}

pub(super) fn rewrite(
    file: &str,
    expression: &eqiora_lang::Expr,
    scope: &Scope,
) -> Result<
    (
        Vec<crate::lower::LoweringExpression>,
        Option<crate::lower::LoweringReduction>,
    ),
    Diagnostic,
> {
    let (values, reduction) = split(file, expression)?;
    let rewrite = |value| {
        crate::hierarchy::scope::rewrite_expression_with_boundary_member(file, value, scope, None)
    };
    let reduction = reduction
        .map(|source| {
            Ok::<_, Diagnostic>(match source {
                Reduction::SpatialIntegral {
                    domain,
                    measure,
                    limits,
                } => {
                    let domain = resolve_local_kind(
                        file,
                        expression.range(),
                        scope,
                        domain,
                        |kind| matches!(kind, SymbolKind::Domain),
                        "Observable integration Domain",
                    )?;
                    Reduction::SpatialIntegral {
                        domain: domain.internal_name.clone(),
                        measure,
                        limits: limits
                            .map(|[lower, upper]| {
                                Ok::<_, Diagnostic>([rewrite(lower)?, rewrite(upper)?])
                            })
                            .transpose()?,
                    }
                }
                Reduction::RandomVariable { distribution } => {
                    let law = resolve_local_kind(
                        file,
                        expression.range(),
                        scope,
                        distribution,
                        |kind| matches!(kind, SymbolKind::Relation),
                        "probability law",
                    )?;
                    Reduction::RandomVariable {
                        distribution: law.internal_name.clone(),
                    }
                }
                Reduction::ProbabilityMoment { moment } => Reduction::ProbabilityMoment { moment },
            })
        })
        .transpose()?;
    Ok((
        values.into_iter().map(rewrite).collect::<Result<_, _>>()?,
        reduction,
    ))
}
