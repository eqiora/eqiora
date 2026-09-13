//! Admission of ordinary Relations and retained physical Laws.

use super::*;

pub(in crate::hierarchy::body_check) fn validate_relation_expression(
    scope: &DefinitionScope<'_, '_>,
    declaration: &RelationDecl,
    relation_support: Option<SpatialSupport<String>>,
) -> Result<PhysicalEndpointSelections, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    match declaration.activation() {
        ActivationSyntax::Continuous => {}
        ActivationSyntax::Named(clock) => match scope.symbols.get(clock) {
            Some(SymbolContract::Clock | SymbolContract::Event) => {}
            Some(_) => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                format!("`{clock}` is not a ClockDomain or Event"),
            )),
            None => diagnostics.push(unresolved(
                scope.file,
                declaration.range(),
                clock,
                "ClockDomain or Event",
            )),
        },
        _ => diagnostics.push(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            scope.file,
            declaration.range(),
            "Activation syntax is newer than definition-body validation",
        )),
    }
    let conditions = declaration.conditions();
    if conditions.is_some_and(|conditions| conditions.is_empty()) {
        diagnostics.push(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            scope.file,
            declaration.range(),
            "expression DAG requires at least one node and one residual root",
        ));
        return Err(diagnostics);
    }
    let discrete = matches!(declaration.activation(), ActivationSyntax::Named(_));
    let mut checker = ExpressionChecker {
        scope,
        relation_support,
        family_scope: None,
        allow_discrete_symbols: discrete,
        initial: false,
        activation: declaration.activation(),
        physical_endpoints: PhysicalEndpointSelections::new(),
        intrinsic: false,
        alias_dependencies: Vec::new(),
        evolution: Vec::new(),
        contextual: Vec::new(),
        sampling: false,
    };
    if let eqiora_lang::RelationBody::Conservation(terms) = declaration.body() {
        if let Err(error) = checker.check_law(terms) {
            diagnostics.push(error);
        }
        return if diagnostics.is_empty() {
            Ok(checker.physical_endpoints)
        } else {
            Err(diagnostics)
        };
    }
    for equation in conditions.expect("condition body was distinguished from Law") {
        let inferred = match checker.check_equation(equation) {
            Ok(inferred) => inferred,
            Err(error) => {
                diagnostics.push(error);
                continue;
            }
        };
        if let Err(error) = typing::residual(&inferred, checker.relation_support.as_ref()) {
            diagnostics.push(type_error(scope.file, equation.left(), error));
        }
    }
    if diagnostics.is_empty() {
        Ok(checker.physical_endpoints)
    } else {
        Err(diagnostics)
    }
}

pub(in crate::hierarchy::body_check) fn validate_relation_family_expression(
    scope: &DefinitionScope<'_, '_>,
    declaration: &RelationFamilyDecl,
    family_scope: &BoundaryFamilyScope,
) -> Result<PhysicalEndpointSelections, Vec<Diagnostic>> {
    let relation = declaration.relation();
    let mut diagnostics = Vec::new();
    if !matches!(relation.activation(), ActivationSyntax::Continuous) {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "boundary Relation family must be continuous",
        ));
    }
    if declaration.binder() != family_scope.binder() {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "boundary Relation family is not checked under its declared binder",
        ));
    }
    if relation.domain() != Some(declaration.binder().member()) {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            declaration.range(),
            "boundary Relation family support must name its binder member",
        ));
    }
    let conditions = relation.conditions().ok_or_else(|| {
        vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            relation.range(),
            "Law family requires retained term validation",
        )]
    })?;
    if conditions.is_empty() {
        diagnostics.push(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            scope.file,
            declaration.range(),
            "expression DAG requires at least one node and one residual root",
        ));
        return Err(diagnostics);
    }
    let mut checker = ExpressionChecker {
        scope,
        relation_support: Some(family_scope.support()),
        family_scope: Some(family_scope),
        allow_discrete_symbols: false,
        initial: false,
        activation: relation.activation(),
        physical_endpoints: PhysicalEndpointSelections::new(),
        intrinsic: false,
        alias_dependencies: Vec::new(),
        evolution: Vec::new(),
        contextual: Vec::new(),
        sampling: false,
    };
    for equation in conditions {
        let inferred = match checker.check_equation(equation) {
            Ok(inferred) => inferred,
            Err(error) => {
                diagnostics.push(error);
                continue;
            }
        };
        if let Err(error) = typing::residual(&inferred, checker.relation_support.as_ref()) {
            diagnostics.push(type_error(scope.file, equation.left(), error));
        }
    }
    if diagnostics.is_empty() {
        Ok(checker.physical_endpoints)
    } else {
        Err(diagnostics)
    }
}
