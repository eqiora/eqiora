//! Signature-directed external compilation reuses ordinary occurrence allocation.
use super::*;
use crate::StaticBindingValue;
use crate::resolved::{
    AnalyzedSourceUnit, CompilationModuleId, CompilationNamespaceId, ModuleName, ResolvedAlias,
    ValidatedResolvedHierarchy,
};

struct PropertyScope<'a> {
    units: &'a [AnalyzedSourceUnit],
    catalog: &'a crate::property::catalog::Catalog,
    binding_namespace: &'a CompilationModuleId,
    aliases: &'a [ResolvedAlias],
    local_namespace: Option<&'a CompilationModuleId>,
}
impl PropertyScope<'_> {
    fn namespace<'a>(
        &'a self,
        namespace: &'a preflight::DefinitionNamespace,
    ) -> Option<&'a CompilationModuleId> {
        match namespace {
            preflight::DefinitionNamespace::Resolved(value) => Some(value),
            preflight::DefinitionNamespace::Local => self.local_namespace,
        }
    }
}
use eqiora_lang::{ModelDecl, SignatureItem};

pub(crate) fn local(
    file: &str,
    source: &str,
    entry: &str,
    bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let mut models = local_entries(file, source, Some(entry), bindings)?;
    Ok(models.remove(0))
}

pub(crate) fn local_all(file: &str, source: &str) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    local_entries(file, source, None, &[])
}

fn local_entries(
    file: &str,
    source: &str,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    source_budget(file, source.len(), HierarchyLimits::default())?;
    let parsed = parse(file, source);
    let document = if entry.is_some() {
        parsed.into_document()?
    } else {
        parsed.into_compilation_document()?
    };
    local_document(
        file,
        source.len(),
        document,
        entry,
        bindings,
        HierarchyLimits::default(),
    )
}

pub(super) fn local_document(
    file: &str,
    source_bytes: usize,
    document: Document,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
    limits: HierarchyLimits,
) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    local_document_in(file, source_bytes, document, entry, bindings, limits, None)
}

pub(crate) fn native_document(
    native: &eqiora_lang::Module,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let bytes = crate::source_identity::module_input_bytes(native).map_err(|error| vec![error])?;
    let mut models = local_document_in(
        native.source_file().unwrap_or("<module>"),
        bytes,
        native.document().clone(),
        entry,
        bindings,
        HierarchyLimits::default(),
        native.has_native_metadata().then_some(native),
    )?;
    if models.len() != 1 {
        return Err(vec![Diagnostic::error(
            codes::LANGUAGE_LOWERING_ERROR,
            "Module compilation requires one selected Model",
        )]);
    }
    Ok(models.remove(0))
}

fn local_document_in(
    file: &str,
    source_bytes: usize,
    document: Document,
    entry: Option<&str>,
    bindings: &[(&str, StaticBindingValue<'_>)],
    limits: HierarchyLimits,
    native: Option<&eqiora_lang::Module>,
) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    source_budget(file, source_bytes, limits)?;
    let identity = LocalSourceIdentity::from_document(&document).map_err(|error| vec![error])?;
    let namespace = if native.is_some() {
        crate::identity::IdentityNamespace::new([
            "native".to_owned(),
            eqiora_core::OntologyId::<eqiora_schema::Model>::new().to_string(),
        ])
    } else {
        identity.namespace()
    }
    .map_err(|error| vec![error])?;
    let mut document = document;
    let enumerations = crate::enumeration::declarations(file, &document, &namespace, |name| {
        native.and_then(|native| native.nominal_identity(name))
    })?;
    crate::enumeration::bind_document(file, &mut document, &enumerations)?;
    let mut document =
        crate::dimensions::elaborate_dimension_aliases(file, &document)?.into_owned();
    let spaces = crate::nominal::finite_spaces(file, &document, &namespace, |name| {
        native.and_then(|native| native.nominal_identity(name))
    })?;
    crate::nominal::bind_finite_types(file, &mut document, &spaces)?;
    crate::nominal::bind_finite_expressions(file, &mut document, &spaces)?;
    crate::nominal::bind_local_index_types(file, &mut document, &namespace, |name| {
        native.and_then(|native| native.nominal_identity(name))
    })?;
    // This private lookup namespace never enters local source/occurrence identity.
    let module = CompilationModuleId::new(
        CompilationNamespaceId::new(["eqiora.local"]).map_err(|error| vec![error])?,
        ModuleName::new(["main"]).map_err(|error| vec![error])?,
    );
    let units = vec![AnalyzedSourceUnit {
        native: native.cloned().map(std::sync::Arc::new),
        resolved_arrays: Default::default(),
        resolved_documents: Default::default(),
        module: module.clone(),
        file: file.to_owned(),
        source_bytes,
        authored_document: std::sync::Arc::new(document.clone()),
        document,
    }];
    let property_catalog = std::sync::Arc::new(crate::property::catalog::build(&units, &[])?);
    crate::property::validate_bindings(&units, &[], &property_catalog)?;
    let context = PropertyScope {
        units: &units,
        catalog: &property_catalog,
        binding_namespace: &module,
        aliases: &[],
        local_namespace: Some(&module),
    };
    let selected_bound;
    let mut elaborator = Elaborator::with_identity(
        file,
        source_bytes,
        &units[0].document,
        namespace,
        native,
        limits,
    )?;
    elaborator.property_catalog = property_catalog.clone();
    elaborator.property_local_module = Some(module.clone());
    if let Some(entry) = entry
        && !bindings.is_empty()
        && let Some(model) = elaborator
            .find_entry_model(entry)
            .map_err(|message| vec![hierarchy_error(message)])?
    {
        let signature = authored_signature(&context, &model.namespace, model.name(), true)
            .unwrap_or_else(|| model.signature());
        let prepared = prepare(
            &parameters::RecordContext::model(&elaborator, &model),
            model.file,
            model.name(),
            signature,
            bindings,
            |requirement, value| {
                property(&context, &model.namespace, model.file, requirement, value)
            },
        )?;
        selected_bound = bind_model(
            &elaborator,
            model.declaration,
            &prepared,
            &parameters::RecordContext::model(&elaborator, &model),
        )?;
        elaborator.bind_selected_model(
            preflight::ModelDefinition {
                namespace: model.namespace,
                file: model.file,
                owned_interfaces: preflight::owned_model_items(&selected_bound),
                declaration: &selected_bound,
            },
            prepared.supports(),
        );
    }
    if let Some(entry) = entry
        && !bindings.is_empty()
        && elaborator
            .find_entry_model(entry)
            .map_err(|message| vec![hierarchy_error(message)])?
            .is_none()
    {
        let path = NamePath::from_segments(entry.split('.'), TextRange::default())
            .map_err(|error| vec![hierarchy_error(error.message())])?;
        let component = elaborator
            .resolve_component(
                &preflight::DefinitionNamespace::Local,
                &path,
                file,
                path.range(),
            )
            .map_err(|error| vec![error])?;
        let signature = authored_signature(&context, &component.namespace, component.name(), false)
            .unwrap_or_else(|| component.signature());
        let prepared = prepare(
            &parameters::RecordContext::component(&elaborator, &component),
            component.file,
            component.name(),
            signature,
            bindings,
            |requirement, value| {
                property(
                    &context,
                    &component.namespace,
                    component.file,
                    requirement,
                    value,
                )
            },
        )?;
        let values = parameters::resolve_external_parameters(
            component.file,
            component.declaration,
            prepared.parameters(),
            &parameters::RecordContext::component(&elaborator, &component),
        )?;
        elaborator.selected_component = Some((
            preflight::DefinitionKey {
                namespace: component.namespace.clone(),
                name: component.name().to_owned(),
            },
            values,
        ));
    }
    let checked = check::validate(&elaborator)?;
    let entries = entry.map_or_else(
        || {
            units[0]
                .document
                .models()
                .iter()
                .map(ModelDecl::name)
                .collect::<Vec<_>>()
        },
        |entry| vec![entry],
    );
    let mut models = Vec::with_capacity(entries.len());
    let mut diagnostics = Vec::new();
    for entry in entries {
        match compile(
            &elaborator,
            &checked,
            entry,
            bindings,
            &preflight::DefinitionNamespace::Local,
            &context,
        ) {
            Ok(model) => models.push(model),
            Err(mut errors) => diagnostics.append(&mut errors),
        }
    }
    if diagnostics.is_empty() {
        Ok(models)
    } else {
        Err(diagnostics)
    }
}

pub(crate) fn resolved(
    hierarchy: &ValidatedResolvedHierarchy,
    entry: &str,
    bindings: &[(&str, StaticBindingValue<'_>)],
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let mut elaborator = Elaborator::new_resolved(&hierarchy.analysis, HierarchyLimits::default())?;
    let context = PropertyScope {
        units: &hierarchy.analysis.units,
        catalog: &hierarchy.analysis.property_catalog,
        binding_namespace: &hierarchy.analysis.root,
        aliases: &hierarchy.analysis.aliases,
        local_namespace: None,
    };
    let selected_bound;
    let selected_checked;
    let checked = if let Some(model) = elaborator
        .find_entry_model(entry)
        .map_err(|message| vec![hierarchy_error(message)])?
    {
        let signature = authored_signature(&context, &model.namespace, model.name(), true)
            .unwrap_or_else(|| model.signature());
        let prepared = prepare(
            &parameters::RecordContext::model(&elaborator, &model),
            model.file,
            model.name(),
            signature,
            bindings,
            |requirement, value| {
                property(&context, &model.namespace, model.file, requirement, value)
            },
        )?;
        selected_bound = bind_model(
            &elaborator,
            model.declaration,
            &prepared,
            &parameters::RecordContext::model(&elaborator, &model),
        )?;
        elaborator.bind_selected_model(
            preflight::ModelDefinition {
                namespace: model.namespace,
                file: model.file,
                owned_interfaces: preflight::owned_model_items(&selected_bound),
                declaration: &selected_bound,
            },
            prepared.supports(),
        );
        // Resolve every deferred topology obligation of the selected definition
        // against the same validated support context used by occurrence allocation.
        selected_checked = check::validate(&elaborator)?;
        &selected_checked
    } else {
        &hierarchy.checked
    };
    compile(
        &elaborator,
        checked,
        entry,
        bindings,
        &preflight::DefinitionNamespace::Resolved(hierarchy.analysis.root.clone()),
        &context,
    )
}

fn compile(
    elaborator: &Elaborator<'_>,
    checked: &CheckedDefinitionGraph,
    entry: &str,
    bindings: &[(&str, StaticBindingValue<'_>)],
    root_namespace: &preflight::DefinitionNamespace,
    hierarchy: &PropertyScope<'_>,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let limits = elaborator.limits;
    if let Some(model) = elaborator
        .find_entry_model(entry)
        .map_err(|message| vec![hierarchy_error(message)])?
    {
        if let Some((_, component)) = elaborator
            .components()
            .find(|(_, component)| component.formulations().len() != 0)
        {
            let range = component
                .formulations()
                .next()
                .expect("nonempty authored forms")
                .4;
            return Err(vec![source_error(
                codes::LANGUAGE_TYPE_ERROR,
                component.file,
                range,
                "authored Component formulations require fresh external-Geometry component compilation",
            )]);
        }
        let signature = authored_signature(hierarchy, &model.namespace, model.name(), true)
            .unwrap_or_else(|| model.signature());
        let prepared = prepare(
            &parameters::RecordContext::model(elaborator, &model),
            model.file,
            model.name(),
            signature,
            bindings,
            |requirement, value| {
                property(hierarchy, &model.namespace, model.file, requirement, value)
            },
        )?;
        let bound = bind_model(
            elaborator,
            model.declaration,
            &prepared,
            &parameters::RecordContext::model(elaborator, &model),
        )?;
        let definition = preflight::ModelDefinition {
            namespace: model.namespace.clone(),
            file: model.file,
            owned_interfaces: preflight::owned_model_items(&bound),
            declaration: &bound,
        };
        let mut size =
            super::definition_graph::selected_expansion_size(elaborator, checked, &definition)?;
        size.declarations = checked_external_footprint(
            "declarations",
            size.declarations,
            prepared
                .supports()
                .iter()
                .map(ExternalGeometrySupportBinding::allocated_support_count)
                .sum::<usize>()
                + prepared.clocks.len()
                + prepared.parameters().len(),
            limits.max_declarations,
        )?;
        return RootExpansion::new(elaborator, definition, size)
            .map_err(|error| vec![error])?
            .expand_bound(prepared.supports(), &prepared.clocks, prepared.properties())?
            .compile(limits);
    }
    let path = NamePath::from_segments(entry.split('.'), TextRange::default())
        .map_err(|error| vec![hierarchy_error(error.message())])?;
    let component = elaborator
        .resolve_component(root_namespace, &path, "<selected-entry>", path.range())
        .map_err(|error| vec![error])?;
    let signature = authored_signature(hierarchy, &component.namespace, component.name(), false)
        .unwrap_or_else(|| component.signature());
    let prepared = prepare(
        &parameters::RecordContext::component(elaborator, &component),
        component.file,
        component.name(),
        signature,
        bindings,
        |requirement, value| {
            property(
                hierarchy,
                &component.namespace,
                component.file,
                requirement,
                value,
            )
        },
    )?;
    let compiled = compile_external_component_from_definition(
        elaborator,
        checked,
        component.clone(),
        &prepared,
        limits,
    )?;
    if component.formulations().len() != 0 {
        let geometry = bindings
            .iter()
            .find_map(|(_, value)| match value {
                StaticBindingValue::GeometrySupport { geometry, .. } => Some(*geometry),
                _ => None,
            })
            .ok_or_else(|| {
                vec![hierarchy_error(
                    "authored formulations require exact Geometry support bindings",
                )]
            })?;
        let formulations = crate::formulation::compile_component_formulations(
            component.file,
            component.declaration,
            compiled.symbols(),
            compiled.transaction(),
            (
                geometry.ambient_dimension(),
                geometry.topological_dimension(),
            ),
            prepared.supports(),
        )?;
        Ok(compiled.with_authored_formulations(formulations))
    } else {
        Ok(compiled)
    }
}

fn authored_signature<'a>(
    scope: &'a PropertyScope<'_>,
    namespace: &preflight::DefinitionNamespace,
    name: &str,
    model: bool,
) -> Option<&'a [SignatureItem]> {
    let namespace = scope.namespace(namespace)?;
    let unit = scope.units.iter().find(|unit| &unit.module == namespace)?;
    if model {
        unit.authored_document
            .models()
            .iter()
            .find(|value| value.name() == name)
            .map(ModelDecl::signature)
    } else {
        unit.authored_document
            .components()
            .iter()
            .find(|value| value.name() == name)
            .map(eqiora_lang::ComponentDecl::signature)
    }
}
fn property(
    scope: &PropertyScope<'_>,
    namespace: &preflight::DefinitionNamespace,
    file: &str,
    requirement: &eqiora_lang::ComponentPropertyDecl,
    value: &eqiora_lang::Expr,
) -> Result<eqiora_schema::kernel::PropertyRelease, Vec<Diagnostic>> {
    let namespace = scope.namespace(namespace).ok_or_else(|| {
        vec![hierarchy_error(
            "selected property scope is missing its source namespace",
        )]
    })?;
    scope.catalog.bind(
        namespace,
        requirement,
        scope.binding_namespace,
        value,
        scope.aliases,
        file,
    )
}

fn prepare(
    records: &parameters::RecordContext,
    file: &str,
    name: &str,
    signature: &[SignatureItem],
    bindings: &[(&str, StaticBindingValue<'_>)],
    mut property: impl FnMut(
        &eqiora_lang::ComponentPropertyDecl,
        &eqiora_lang::Expr,
    ) -> Result<eqiora_schema::kernel::PropertyRelease, Vec<Diagnostic>>,
) -> Result<ExternalComponentBinding, Vec<Diagnostic>> {
    use std::collections::{BTreeMap, BTreeSet};
    crate::external_compile::validate_selected_bindings(file, name, bindings)?;
    let fail = |range, message| {
        vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            message,
        )]
    };
    if bindings.len()
        > crate::source_identity::LocalSourceIdentityLimits::default().max_bindings_per_instance
    {
        return Err(fail(
            TextRange::default(),
            "selected binding count exceeds resource limit".to_owned(),
        ));
    }
    let frame_context = supports::signature_support_interface(file, signature)?
        .iter()
        .map(|(name, contract)| (name.to_owned(), contract.support().clone()))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut prepared = ExternalComponentBinding::new(name, name, Vec::new(), Vec::new());
    let mut parameters = Vec::new();
    let mut supports = Vec::new();
    let mut geometry_groups = BTreeMap::<
        [u8; 32],
        (
            &eqiora_geometry::CanonicalGeometryV1,
            Vec<(
                &str,
                &eqiora_geometry::NamedEntitySet,
                Option<(&str, &eqiora_geometry::NamedEntitySet)>,
            )>,
        ),
    >::new();
    for &(name, value) in bindings {
        if !seen.insert(name) {
            return Err(fail(
                TextRange::default(),
                format!("duplicate selected binding `{name}`"),
            ));
        }
        let target = signature
            .iter()
            .find(|item| item.name() == name)
            .ok_or_else(|| {
                fail(
                    TextRange::default(),
                    format!("selected signature has no binding target `{name}`"),
                )
            })?;
        match (target, value) {
            (
                SignatureItem::Parameter(_),
                StaticBindingValue::Value(_) | StaticBindingValue::Expression(_),
            ) => {}
            (SignatureItem::Property(requirement), StaticBindingValue::Expression(value)) => {
                prepared.properties.insert(
                    name.to_owned(),
                    std::sync::Arc::new(property(requirement, value)?),
                );
            }
            (SignatureItem::Clock(_), StaticBindingValue::Clock(clock))
                if matches!(
                    clock.kind(),
                    eqiora_schema::kernel::ClockKind::Periodic { .. }
                ) =>
            {
                prepared.clocks.push((name.to_owned(), clock.clone()))
            }
            (
                SignatureItem::Support(slot),
                StaticBindingValue::CompleteExterior {
                    geometry,
                    members,
                    parent,
                },
            ) => {
                let eqiora_lang::SupportSlotSyntax::CompleteExterior {
                    parent: parent_slot,
                } = slot.syntax()
                else {
                    return Err(fail(
                        slot.range(),
                        "CompleteExterior binding does not match the selected support contract"
                            .to_owned(),
                    ));
                };
                let group = geometry_groups
                    .entry(geometry.digest_bytes())
                    .or_insert_with(|| (geometry, Vec::new()));
                // The finite members remain individual exact selections; no union is constructed.
                for selection in members {
                    group
                        .1
                        .push((name, selection, Some((parent_slot.as_str(), parent))));
                }
                if members.is_empty() {
                    return Err(fail(
                        slot.range(),
                        "complete-exterior binding has no members".to_owned(),
                    ));
                }
                supports.push(ExternalGeometrySupportBinding::CompleteExterior {
                    slot: name.to_owned(),
                    geometry: eqiora_schema::kernel::GeometryDigest::new(geometry.digest_bytes()),
                    parent_slot: parent_slot.clone(),
                    members: members
                        .iter()
                        .map(
                            |selection| crate::external::ExternalGeometryBoundaryMember {
                                entity_set: selection.name().to_owned(),
                                embedding: geometry.cartesian_boundary_embedding(selection, parent),
                            },
                        )
                        .collect(),
                });
            }
            (
                SignatureItem::Support(slot),
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection,
                    parent,
                },
            ) => {
                let digest = eqiora_schema::kernel::GeometryDigest::new(geometry.digest_bytes());
                let parent_binding = match slot.syntax() {
                    eqiora_lang::SupportSlotSyntax::Volume { ambient_dimension }
                        if parent.is_none()
                            && *ambient_dimension == geometry.ambient_dimension() =>
                    {
                        None
                    }
                    eqiora_lang::SupportSlotSyntax::Boundary {
                        parent: parent_slot,
                    } => Some((
                        parent_slot.as_str(),
                        parent.ok_or_else(|| {
                            fail(
                                slot.range(),
                                "boundary support requires its exact parent selection".to_owned(),
                            )
                        })?,
                    )),
                    _ => {
                        return Err(fail(
                            slot.range(),
                            "Geometry binding does not match the selected support contract"
                                .to_owned(),
                        ));
                    }
                };
                geometry_groups
                    .entry(geometry.digest_bytes())
                    .or_insert_with(|| (geometry, Vec::new()))
                    .1
                    .push((name, selection, parent_binding));
                supports.push(match parent_binding {
                    Some((parent_slot, parent)) => ExternalGeometrySupportBinding::boundary(
                        name,
                        digest,
                        selection.name(),
                        parent_slot,
                        geometry.cartesian_boundary_embedding(selection, parent),
                    ),
                    None => ExternalGeometrySupportBinding::region(
                        name,
                        digest,
                        selection.name(),
                        geometry.ambient_dimension(),
                    ),
                });
            }
            _ => {
                return Err(fail(
                    target.range(),
                    format!("selected argument `{name}` has the wrong signature binding category"),
                ));
            }
        }
    }
    for item in signature {
        if seen.contains(item.name()) {
            continue;
        }
        let required = match item {
            SignatureItem::Parameter(value) => value.default().is_none(),
            SignatureItem::Clock(_)
            | SignatureItem::Support(_)
            | SignatureItem::Field(_)
            | SignatureItem::Property(_) => true,
            _ => false,
        };
        if required {
            return Err(fail(
                item.range(),
                format!("selected entry requires binding `{}`", item.name()),
            ));
        }
    }
    for (_, (geometry, bindings)) in geometry_groups {
        crate::external_compile::validate_geometry_bindings(file, geometry, &bindings)?;
    }
    parameters.extend(parameters::resolve_selected_parameters(
        file,
        signature,
        bindings,
        frame_context,
        records,
    )?);
    parameters.sort_by(|a, b| a.parameter().cmp(b.parameter()));
    supports.sort_by(|a, b| a.slot().cmp(b.slot()));
    let clocks = prepared.clocks;
    let properties = prepared.properties;
    prepared = ExternalComponentBinding::new(name, name, supports, parameters);
    prepared.clocks = clocks;
    prepared.properties = properties;
    prepared.clocks.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(prepared)
}

fn bind_model(
    elaborator: &Elaborator<'_>,
    model: &ModelDecl,
    bindings: &ExternalComponentBinding,
    records: &parameters::RecordContext,
) -> Result<ModelDecl, Vec<Diagnostic>> {
    let frame_context = supports::model_spatial_supports("<selected-entry>", model)?;
    let signature = model
        .signature()
        .iter()
        .map(|item| {
            let SignatureItem::Parameter(declaration) = item else {
                return Ok(item.clone());
            };
            if let Some(record) = records.record_for_type(declaration.value_type()) {
                let values = record
                    .definition
                    .members()
                    .iter()
                    .map(|(member, _)| {
                        let name = format!("{}.{member}", declaration.name());
                        bindings
                            .parameters()
                            .iter()
                            .find(|binding| binding.parameter() == name)
                    })
                    .collect::<Vec<_>>();
                if values.iter().all(Option::is_none) {
                    return Ok(item.clone());
                }
                let mut arguments = Vec::with_capacity(values.len());
                for (((member, _), syntax), binding) in record
                    .definition
                    .members()
                    .iter()
                    .zip(&record.member_syntax)
                    .zip(values)
                {
                    let binding = binding.ok_or_else(|| {
                        vec![hierarchy_error(
                            "selected record Parameter has incomplete leaf bindings",
                        )]
                    })?;
                    let value = SourceAstFactory::value_literal(
                        binding.value(),
                        None,
                        declaration.range(),
                        |id| record_nominal_path(syntax, id),
                        |id| elaborator.enum_definition(id),
                    )
                    .map_err(|error| vec![hierarchy_error(error.message())])?;
                    arguments.push(
                        SourceAstFactory::named_binding(member, value, declaration.range())
                            .map_err(|error| vec![hierarchy_error(error.message())])?,
                    );
                }
                let eqiora_lang::ValueTypeSyntaxKind::Named(callee) =
                    declaration.value_type().kind()
                else {
                    unreachable!("selected record type")
                };
                let value = SourceAstFactory::expression(
                    eqiora_lang::ExprKind::Call {
                        callee: callee.clone(),
                        arguments: eqiora_lang::CallArguments::Named(arguments),
                    },
                    declaration.range(),
                )
                .map_err(|error| vec![hierarchy_error(error.message())])?;
                return SourceAstFactory::component_parameter(
                    eqiora_lang::VisibilitySyntax::Public,
                    declaration.name(),
                    declaration.value_type().clone(),
                    Some(value),
                    declaration.range(),
                )
                .map(SignatureItem::Parameter)
                .map_err(|error| vec![hierarchy_error(error.message())]);
            }
            let Some(binding) = bindings
                .parameters()
                .iter()
                .find(|binding| binding.parameter() == declaration.name())
            else {
                return Ok(item.clone());
            };
            let value = SourceAstFactory::value_literal(
                binding.value(),
                projection_frame(
                    "<selected-entry>",
                    binding.value(),
                    declaration.default(),
                    &frame_context,
                    declaration.range(),
                )?,
                declaration.range(),
                |id| {
                    declaration
                        .value_type()
                        .resolved_nominal()
                        .filter(|ty| {
                            ty.enum_definition()
                                .is_some_and(|definition| definition.erase() == id)
                        })
                        .and_then(|_| match declaration.value_type().kind() {
                            eqiora_lang::ValueTypeSyntaxKind::Named(name) => Some(name.clone()),
                            _ => None,
                        })
                },
                |id| elaborator.enum_definition(id),
            )
            .map_err(|error| vec![hierarchy_error(error.message())])?;
            SourceAstFactory::component_parameter(
                eqiora_lang::VisibilitySyntax::Public,
                declaration.name(),
                declaration.value_type().clone(),
                Some(value),
                declaration.range(),
            )
            .map(SignatureItem::Parameter)
            .map_err(|error| vec![hierarchy_error(error.message())])
        })
        .collect::<Result<Vec<_>, Vec<Diagnostic>>>()?;
    SourceAstFactory::model(
        model.visibility(),
        model.name(),
        signature,
        model.items().to_vec(),
        model.range(),
    )
    .map_err(|error| vec![hierarchy_error(error.message())])
}

fn record_nominal_path(
    syntax: &eqiora_lang::ValueTypeSyntax,
    id: eqiora_core::RawId,
) -> Option<NamePath> {
    use eqiora_lang::ValueTypeSyntaxKind;
    if let ValueTypeSyntaxKind::Array { element, .. } = syntax.kind() {
        return record_nominal_path(element, id);
    }
    let value = syntax.resolved_nominal()?;
    if value
        .enum_definition()
        .is_some_and(|definition| definition.erase() == id)
        || value
            .finite_space()
            .is_some_and(|definition| definition.erase() == id)
        || value
            .index_set()
            .is_some_and(|definition| definition.erase() == id)
    {
        match syntax.kind() {
            ValueTypeSyntaxKind::Named(name)
            | ValueTypeSyntaxKind::Coordinates(name)
            | ValueTypeSyntaxKind::Counts(name)
            | ValueTypeSyntaxKind::Index(name) => Some(name.clone()),
            _ => None,
        }
    } else {
        None
    }
}

fn source_budget(
    file: &str,
    source_bytes: usize,
    limits: HierarchyLimits,
) -> Result<(), Vec<Diagnostic>> {
    if source_bytes > limits.max_source_bytes {
        Err(vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            TextRange::new(0, u32::try_from(source_bytes).unwrap_or(u32::MAX)),
            format!(
                "source requires {source_bytes} bytes, exceeding the {} byte hierarchy limit",
                limits.max_source_bytes
            ),
        )])
    } else {
        Ok(())
    }
}
