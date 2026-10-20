use super::*;

pub(super) fn collect_canonical_declarations(
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<CanonicalDeclarationIdentity> {
    let mut result = Vec::new();
    let mut paths = BTreeSet::new();
    for unit in units {
        let operator_formals = visible_operator_formals(unit, units, aliases);
        let resolved_aliases = aliases
            .iter()
            .filter(|alias| alias.declaring_module() == &unit.module)
            .map(|alias| (alias.alias().to_owned(), canonical_alias_target(alias)))
            .collect::<BTreeMap<_, _>>();
        for declaration in unit.document.dimensions() {
            let document = SourceAstFactory::document_with_dimensions(
                Vec::new(),
                vec![declaration.clone()],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .expect("parsed dimension document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, declaration.name()),
                CanonicalDeclarationKind::Dimension,
                declaration.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
        for (name, visibility, document) in unit.document.isolated_property_declarations() {
            let kind = if document.property_contract_syntax().next().is_some() {
                CanonicalDeclarationKind::PropertyContract
            } else if document.property_release_syntax().next().is_some() {
                CanonicalDeclarationKind::PropertyRelease
            } else {
                CanonicalDeclarationKind::MaterialComposition
            };
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, &name),
                kind,
                visibility,
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
            if let Some((
                _,
                _,
                _,
                eqiora_lang::PropertySourceSyntax::Table(table),
                _,
                _,
                citation,
                license,
                _,
            )) = document.property_release_syntax().next()
                && let Some(identity) = result.last_mut()
            {
                match crate::property::table::canonical_asset_identity(
                    unit,
                    table,
                    citation,
                    license,
                    &identity.canonical_form,
                ) {
                    Ok(form) => identity.canonical_form = form,
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
            }
        }
        for enumeration in unit.document.enumerations() {
            let document = SourceAstFactory::document(
                vec![enumeration.clone()],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .expect("parsed enum document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, enumeration.name()),
                CanonicalDeclarationKind::Enum,
                enumeration.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
        for connector in unit.document.connectors() {
            let document = SourceAstFactory::document(
                Vec::new(),
                vec![connector.clone()],
                Vec::new(),
                Vec::new(),
            )
            .expect("one parsed Connector is a valid document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, connector.name()),
                CanonicalDeclarationKind::Connector,
                connector.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
        for operator in unit.document.pure_operators() {
            let document = SourceAstFactory::document_with_pure_operators(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![operator.clone()],
                Vec::new(),
            )
            .expect("one parsed operator is a valid document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, operator.name()),
                CanonicalDeclarationKind::PureOperator,
                operator.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
        for component in unit.document.components() {
            let document = SourceAstFactory::document(
                Vec::new(),
                Vec::new(),
                vec![component.clone()],
                Vec::new(),
            )
            .expect("one parsed component is a valid document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, component.name()),
                CanonicalDeclarationKind::Component,
                component.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
        for model in unit.document.models() {
            let document =
                SourceAstFactory::document(Vec::new(), Vec::new(), Vec::new(), vec![model.clone()])
                    .expect("one parsed Model is a valid document");
            push_canonical(
                &mut result,
                &mut paths,
                unit.module.owner(),
                &canonical_declaration_path(&unit.module, model.name()),
                CanonicalDeclarationKind::Model,
                model.visibility(),
                &document,
                &resolved_aliases,
                &operator_formals,
                diagnostics,
            );
        }
    }
    result.sort_by(|left, right| {
        (
            &left.namespace,
            &left.path,
            left.kind,
            visibility_rank(left.visibility),
            &left.canonical_form,
        )
            .cmp(&(
                &right.namespace,
                &right.path,
                right.kind,
                visibility_rank(right.visibility),
                &right.canonical_form,
            ))
    });
    result
}

fn visible_operator_formals(
    unit: &AnalyzedSourceUnit,
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
) -> BTreeMap<String, Vec<String>> {
    let names = |operator: &eqiora_lang::PureOperatorDecl| {
        operator
            .formals()
            .iter()
            .map(|formal| formal.name().to_owned())
            .collect()
    };
    let mut visible = unit
        .document
        .pure_operators()
        .iter()
        .map(|operator| (operator.name().to_owned(), names(operator)))
        .collect::<BTreeMap<_, _>>();
    for alias in aliases
        .iter()
        .filter(|alias| alias.declaring_module() == &unit.module)
    {
        for target in units
            .iter()
            .filter(|target| &target.module == alias.target_module())
        {
            for operator in target
                .document
                .pure_operators()
                .iter()
                .filter(|operator| operator.visibility() == VisibilitySyntax::Public)
            {
                visible.insert(
                    format!("{}.{}", alias.alias(), operator.name()),
                    names(operator),
                );
            }
        }
    }
    visible
}

fn canonical_alias_target(alias: &ResolvedAlias) -> ResolvedAliasTarget {
    let target = alias.target_module();
    if alias.declaring_module().owner() == target.owner() {
        ResolvedAliasTarget::local_module(target.name().segments())
    } else {
        ResolvedAliasTarget::external_module(target.owner().segments(), target.name().segments())
    }
}

pub(super) fn canonical_declaration_path(
    module: &CompilationModuleId,
    declaration: &str,
) -> String {
    format!("{}.{}", module.name(), declaration)
}

#[allow(clippy::too_many_arguments)]
fn push_canonical(
    result: &mut Vec<CanonicalDeclarationIdentity>,
    paths: &mut BTreeSet<(CompilationNamespaceId, String)>,
    namespace: &CompilationNamespaceId,
    path: &str,
    kind: CanonicalDeclarationKind,
    visibility: VisibilitySyntax,
    document: &Document,
    resolved_aliases: &BTreeMap<String, ResolvedAliasTarget>,
    operator_formals: &BTreeMap<String, Vec<String>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if !paths.insert((namespace.clone(), path.to_owned())) {
        diagnostics.push(resolved_error(format!(
            "duplicate top-level declaration `{path}` in namespace `{namespace}`"
        )));
        return;
    }
    let identity = match LocalSourceIdentity::from_document_with_resolved_aliases(
        document,
        resolved_aliases,
        operator_formals,
    ) {
        Ok(identity) => identity,
        Err(error) => {
            diagnostics.push(error);
            return;
        }
    };
    result.push(CanonicalDeclarationIdentity {
        namespace: namespace.clone(),
        path: path.to_owned(),
        kind,
        visibility,
        canonical_form: declaration_identity_form(identity.digest()),
    });
}
