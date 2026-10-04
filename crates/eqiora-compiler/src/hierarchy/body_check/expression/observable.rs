//! Check every Observable body, including unused Component definitions.
use super::*;

pub(in crate::hierarchy::body_check) fn validate_observable(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::ObservableDecl,
) -> Result<(), Diagnostic> {
    let (value, reduction) =
        crate::hierarchy::expand::observable::split(scope.file, declaration.value())?;
    let support = reduction
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable integration Domain")
            })
        })
        .transpose()?;
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let expected = crate::value_types::lower_value_type(scope.file, &syntax, support.as_ref())?;
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
        allow_observables: reduction.is_none(),
    };
    let inferred = checker.check_numeric_context(value, expected.scalar_domain())?;
    let inferred_type = if reduction.is_some() {
        let measure = if matches!(
            support.as_ref(),
            Some(eqiora_schema::kernel::typing::SpatialSupport::Boundary { .. })
        ) {
            eqiora_schema::kernel::ObservableMeasure::Boundary
        } else {
            eqiora_schema::kernel::ObservableMeasure::Volume
        };
        measure
            .output_type(
                &inferred,
                support.as_ref().expect("resolved integration support"),
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    scope.file,
                    declaration.range(),
                    error.message(),
                )
            })?
    } else {
        if inferred.support.is_some() {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "Observable value requires no spatial support; use an explicit integral",
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
    let (_, reduction) =
        crate::hierarchy::expand::observable::split(scope.file, declaration.value())?;
    let support = reduction
        .map(|name| {
            scope.spatial_support(name).ok_or_else(|| {
                scope.wrong_local_kind(declaration.range(), name, "Observable integration Domain")
            })
        })
        .transpose()?;
    let syntax = crate::hierarchy::parameters::specialize_type(
        scope.file,
        declaration.value_type(),
        &scope.static_values,
    )?;
    let value_type = crate::value_types::lower_value_type(scope.file, &syntax, support.as_ref())?;
    Ok(ExpressionType::new(value_type, None))
}
