//! Check declaration reductions, including unused Component definitions.
use super::*;
use crate::hierarchy::expand::observable::{name, split};
use crate::observation::Reduction;
use eqiora_lang::TextRange;
use eqiora_schema::kernel::ObservableMeasure;

type SourceReduction<'a> = Reduction<&'a str, &'a Expr>;

fn law_support(
    scope: &DefinitionScope<'_, '_>,
    law: &str,
    range: TextRange,
) -> Result<Option<SpatialSupport<String>>, Diagnostic> {
    match scope.symbols.get(law) {
        Some(SymbolContract::Relation(eqiora_lang::RelationBody::ProbabilityMass { .. })) => {
            Ok(None)
        }
        Some(SymbolContract::Relation(eqiora_lang::RelationBody::ProbabilityDensity {
            domain,
            ..
        })) => scope
            .spatial_support(domain)
            .map(Some)
            .ok_or_else(|| scope.wrong_local_kind(range, domain, "probability support Domain")),
        _ => Err(scope.wrong_local_kind(range, law, "probability law")),
    }
}

fn reduction_support(
    scope: &DefinitionScope<'_, '_>,
    reduction: Option<SourceReduction<'_>>,
    range: TextRange,
) -> Result<Option<SpatialSupport<String>>, Diagnostic> {
    match reduction {
        Some(Reduction::SpatialIntegral { domain, .. }) => scope
            .spatial_support(domain)
            .map(Some)
            .ok_or_else(|| scope.wrong_local_kind(range, domain, "Observable integration Domain")),
        Some(Reduction::RandomVariable { distribution }) => law_support(scope, distribution, range),
        _ => Ok(None),
    }
}

pub(super) fn outcome_type(
    scope: &DefinitionScope<'_, '_>,
    expression: &Expr,
    arguments: &eqiora_lang::CallArguments,
) -> Result<ExpressionType<String>, Diagnostic> {
    let invalid = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            expression.range(),
            "outcome requires one exact finite probability law",
        )
    };
    let Some([law]) = arguments.positional() else {
        return Err(invalid());
    };
    let Some(SymbolContract::Relation(eqiora_lang::RelationBody::ProbabilityMass {
        outcomes, ..
    })) = name(law).and_then(|name| scope.symbols.get(name))
    else {
        return Err(invalid());
    };
    let ty = outcomes
        .resolved_nominal()
        .filter(|ty| ty.enum_definition().is_some())
        .cloned()
        .ok_or_else(invalid)?;
    Ok(ExpressionType::new(ty, None))
}

fn moment_law(scope: &DefinitionScope<'_, '_>, values: &[&Expr]) -> Result<(), Diagnostic> {
    let mut law = None;
    for value in values {
        let invalid = || {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                value.range(),
                "probability moments require random variables bound to the same exact law",
            )
        };
        let Some(SymbolContract::Observable(variable)) =
            name(value).and_then(|name| scope.symbols.get(name))
        else {
            return Err(invalid());
        };
        let (_, Some(Reduction::RandomVariable { distribution })) =
            split(scope.file, variable.value())?
        else {
            return Err(invalid());
        };
        law_support(scope, distribution, value.range())?;
        if law
            .replace(distribution)
            .is_some_and(|previous| previous != distribution)
        {
            return Err(invalid());
        }
    }
    Ok(())
}

pub(in crate::hierarchy::body_check) fn validate_observable(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::ObservableDecl,
) -> Result<(), Diagnostic> {
    let (values, reduction) = split(scope.file, declaration.value())?;
    let support = reduction_support(scope, reduction, declaration.range())?;
    let output = declaration
        .domain()
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable output Domain")
            })
        })
        .transpose()?
        .or_else(|| {
            matches!(reduction, Some(Reduction::RandomVariable { .. }))
                .then(|| support.clone())
                .flatten()
        });
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let expected = crate::value_types::lower_signature_type(
        scope.file,
        &syntax,
        output.as_ref().or(support.as_ref()),
    )?;
    let mut checker = ExpressionChecker {
        scope,
        relation_support: support.clone().or(output.clone()),
        family_scope: None,
        allow_discrete_symbols: false,
        initial: false,
        activation: &ActivationSyntax::Continuous,
        physical_endpoints: PhysicalEndpointSelections::new(),
        intrinsic: false,
        alias_dependencies: Vec::new(),
        evolution: Vec::new(),
        contextual: Vec::new(),
        sampling: false,
        allow_observables: true,
    };
    let inferred = values
        .iter()
        .map(|value| checker.check_numeric_context(value, expected.scalar_domain()))
        .collect::<Result<Vec<_>, _>>()?;
    let inferred_type = match reduction {
        Some(Reduction::ProbabilityMoment { moment }) => {
            moment_law(scope, &values)?;
            if output.is_some() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "a population moment has no remaining probability support",
                ));
            }
            moment.output_type(
                &inferred
                    .iter()
                    .map(|value| &value.value_type)
                    .collect::<Vec<_>>(),
            )?
        }
        Some(Reduction::SpatialIntegral {
            measure, limits, ..
        }) => {
            let inferred = &inferred[0];
            let measure = measure.unwrap_or(
                if matches!(
                    support.as_ref(),
                    Some(
                        SpatialSupport::Boundary { .. } | SpatialSupport::PhysicalInterface { .. }
                    )
                ) {
                    ObservableMeasure::Boundary
                } else {
                    ObservableMeasure::Volume
                },
            );
            let input = inferred
                .support
                .as_ref()
                .or(support.as_ref())
                .expect("input support");
            let support = support.as_ref().expect("integration support");
            if let Some([lower, upper]) = limits {
                let lower =
                    checker.check_numeric_context(lower, eqiora_core::ScalarDomain::Real)?;
                let upper =
                    checker.check_numeric_context(upper, eqiora_core::ScalarDomain::Real)?;
                measure.validate_limits(input, support, [&lower, &upper])?;
            }
            measure
                .output_type(inferred, input, support, output.as_ref())?
                .value_type
        }
        None | Some(Reduction::RandomVariable { .. }) => {
            if inferred[0]
                .support
                .as_ref()
                .is_some_and(|actual| Some(actual) != output.as_ref())
                || matches!(reduction, Some(Reduction::RandomVariable { .. })) && output != support
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    "Observable value differs from its exact output support",
                ));
            }
            inferred[0].value_type.clone()
        }
    };
    if inferred_type != expected {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "Observable declared type differs from its expression and measure",
        ));
    }
    Ok(())
}

pub(super) fn reference_type(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::ObservableDecl,
) -> Result<ExpressionType<String>, Diagnostic> {
    let (_, reduction) = split(scope.file, declaration.value())?;
    let measure = reduction_support(scope, reduction, declaration.range())?;
    let support = declaration
        .domain()
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable output Domain")
            })
        })
        .transpose()?
        .or_else(|| {
            matches!(reduction, Some(Reduction::RandomVariable { .. }))
                .then(|| measure.clone())
                .flatten()
        });
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let value_type = crate::value_types::lower_signature_type(
        scope.file,
        &syntax,
        support.as_ref().or(measure.as_ref()),
    )?;
    Ok(ExpressionType::new(value_type, support))
}
