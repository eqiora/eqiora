//! Check every Observable body, including unused Component definitions.
use super::*;

pub(in crate::hierarchy::body_check) fn validate_observable(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::ObservableDecl,
) -> Result<(), Diagnostic> {
    let (value, reduction) =
        crate::hierarchy::expand::observable::split(scope.file, declaration.value())?;
    let support = reduction
        .map(|reduction| {
            let name = reduction.domain;
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable integration Domain")
            })
        })
        .transpose()?;
    let output = declaration
        .domain()
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable output Domain")
            })
        })
        .transpose()?;
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let expected = crate::value_types::lower_value_type(
        scope.file,
        &syntax,
        output.as_ref().or(support.as_ref()),
    )?;
    let mut checker = ExpressionChecker {
        scope,
        relation_support: support.clone(),
        family_scope: None,
        allow_discrete_symbols: true,
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
    let inferred = checker.check_numeric_context(value, expected.scalar_domain())?;
    let inferred_type = if reduction.is_some() {
        let measure = reduction
            .and_then(|reduction| reduction.measure)
            .unwrap_or({
                if matches!(
                    support.as_ref(),
                    Some(eqiora_schema::kernel::typing::SpatialSupport::Boundary { .. })
                ) {
                    eqiora_schema::kernel::ObservableMeasure::Boundary
                } else {
                    eqiora_schema::kernel::ObservableMeasure::Volume
                }
            });
        if let Some(limits) = reduction.and_then(|reduction| reduction.limits) {
            let lower =
                checker.check_numeric_context(limits[0], eqiora_core::ScalarDomain::Real)?;
            let upper =
                checker.check_numeric_context(limits[1], eqiora_core::ScalarDomain::Real)?;
            measure
                .validate_limits(
                    inferred
                        .support
                        .as_ref()
                        .or(support.as_ref())
                        .expect("input support"),
                    support.as_ref().expect("integration support"),
                    [&lower, &upper],
                )
                .map_err(|error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        scope.file,
                        declaration.range(),
                        error.message(),
                    )
                })?;
        }
        measure
            .output_type(
                &inferred,
                inferred
                    .support
                    .as_ref()
                    .or(support.as_ref())
                    .expect("resolved input support"),
                support.as_ref().expect("resolved integration support"),
                output.as_ref(),
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    error.message(),
                )
            })?
            .value_type
    } else {
        if inferred
            .support
            .as_ref()
            .is_some_and(|support| Some(support) != output.as_ref())
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "Observable value differs from its declared output support",
            ));
        }
        inferred.value_type
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
    let support = declaration
        .domain()
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable output Domain")
            })
        })
        .transpose()?;
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let (_, reduction) =
        crate::hierarchy::expand::observable::split(scope.file, declaration.value())?;
    let measure_support = reduction
        .map(|reduction| {
            let name = reduction.domain;
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable integration Domain")
            })
        })
        .transpose()?;
    let value_type = crate::value_types::lower_value_type(
        scope.file,
        &syntax,
        support.as_ref().or(measure_support.as_ref()),
    )?;
    Ok(ExpressionType::new(value_type, support))
}
