//! Bounded local declaration and complete-exterior footprint accounting.

use super::*;

pub(super) fn component_local_footprint(
    elaborator: &Elaborator<'_>,
    definition: &ComponentDefinition<'_>,
    diagnostics: &mut Vec<Diagnostic>,
    values: Option<&super::super::parameters::SymbolicParameterMap>,
) -> (LocalFootprint, BTreeSet<DefinitionKey>) {
    let generic_values;
    let require_exact = values.is_some();
    let values = match values {
        Some(values) => values,
        None => {
            generic_values = family_component_values(elaborator, definition, diagnostics);
            &generic_values
        }
    };
    let families = super::families::Families::new(
        definition.file,
        definition.owned_items().filter_map(|item| match item {
            ComponentItem::IndexSet(set) => Some(set),
            _ => None,
        }),
        values,
        require_exact,
        diagnostics,
    );
    let mut expression_nodes = 0usize;
    let mut declarations = 0_usize;
    let mut connections = 0_usize;
    let mut local_connectors = BTreeSet::new();
    for item in definition.owned_items() {
        match item {
            ComponentItem::Domain(_)
            | ComponentItem::IndexSet(_)
            | ComponentItem::Port(_)
            | ComponentItem::Initial(_)
            | ComponentItem::Clock(_)
            | ComponentItem::Observable(_)
            | ComponentItem::Event(_) => checked_local_add(
                &mut declarations,
                1,
                definition.file,
                definition.declaration.range(),
                "declaration",
                diagnostics,
            ),
            ComponentItem::Parameter(parameter) => checked_local_add(
                &mut declarations,
                elaborator
                    .record_for_type(&definition.namespace, parameter.value_type())
                    .map_or(1, |record| 1 + record.definition.members().len()),
                definition.file,
                parameter.range(),
                "Parameter record and members",
                diagnostics,
            ),
            ComponentItem::Field(field) => checked_local_add(
                &mut declarations,
                elaborator
                    .record_for_type(&definition.namespace, field.value_type())
                    .map_or_else(
                        || if field.domain().is_some() { 2 } else { 1 },
                        |record| {
                            1 + record.definition.members().len()
                                * if field.domain().is_some() { 2 } else { 1 }
                        },
                    ),
                definition.file,
                field.range(),
                "Field and continuum representation",
                diagnostics,
            ),
            ComponentItem::Relation(_) => checked_local_add(
                &mut declarations,
                2,
                definition.file,
                definition.declaration.range(),
                "declaration",
                diagnostics,
            ),
            ComponentItem::Connection(connection) => checked_local_add(
                &mut connections,
                families.members(connection.binder(), diagnostics),
                definition.file,
                definition.declaration.range(),
                "Connection",
                diagnostics,
            ),
            ComponentItem::PortFamily(family) => {
                if let Some(members) = complete_exterior_cardinality(
                    definition,
                    family.binder().set().as_str(),
                    family.range(),
                    diagnostics,
                ) {
                    checked_local_add(
                        &mut declarations,
                        members,
                        definition.file,
                        family.range(),
                        "complete-exterior Port-family declaration",
                        diagnostics,
                    );
                }
            }
            ComponentItem::RelationFamily(family) => {
                if let Some(members) =
                    families.extent(family.binder().set().as_str()).or_else(|| {
                        complete_exterior_cardinality(
                            definition,
                            family.binder().set().as_str(),
                            family.range(),
                            diagnostics,
                        )
                    })
                {
                    match members.checked_mul(2) {
                        Some(declarations_per_family) => checked_local_add(
                            &mut declarations,
                            declarations_per_family,
                            definition.file,
                            family.range(),
                            "complete-exterior Relation and Activation declarations",
                            diagnostics,
                        ),
                        None => diagnostics.push(source_error(
                            codes::LANGUAGE_LOWERING_ERROR,
                            definition.file,
                            family.range(),
                            "complete-exterior Relation-family declaration count overflows usize",
                        )),
                    }
                }
            }
            ComponentItem::BoundaryConnection(connection) => {
                let Some(binder) = connection.binder() else {
                    checked_local_add(
                        &mut connections,
                        1,
                        definition.file,
                        connection.range(),
                        "selected boundary Connection",
                        diagnostics,
                    );
                    continue;
                };
                if let Some(members) = complete_exterior_cardinality(
                    definition,
                    binder.set().as_str(),
                    connection.range(),
                    diagnostics,
                ) {
                    checked_local_add(
                        &mut connections,
                        members,
                        definition.file,
                        connection.range(),
                        "complete-exterior Connection family",
                        diagnostics,
                    );
                }
            }
            ComponentItem::Coordinate(_) | ComponentItem::Let(_) => {}
            ComponentItem::Instance(instance) => {
                let count = input_binding_count(
                    elaborator,
                    &definition.namespace,
                    definition.file,
                    instance,
                    diagnostics,
                );
                checked_local_add(
                    &mut connections,
                    count,
                    definition.file,
                    instance.range(),
                    "Input connections",
                    diagnostics,
                );
            }
            _ => {}
        }
        match item {
            ComponentItem::Relation(relation) => families.expressions(
                relation.expressions(),
                1,
                &mut expression_nodes,
                diagnostics,
            ),
            ComponentItem::Initial(initial) => {
                families.equations(initial.equations(), 1, &mut expression_nodes, diagnostics)
            }
            ComponentItem::Connection(connection) => families.expressions(
                connection.port_expressions(),
                families.members(connection.binder(), diagnostics),
                &mut expression_nodes,
                diagnostics,
            ),
            ComponentItem::Event(value) => {
                families.expressions([value.guard()], 1, &mut expression_nodes, diagnostics)
            }
            ComponentItem::Observable(value) => {
                families.expressions([value.value()], 1, &mut expression_nodes, diagnostics)
            }
            ComponentItem::Coordinate(value) | ComponentItem::Let(value) => {
                families.expressions([value.value()], 1, &mut expression_nodes, diagnostics)
            }
            ComponentItem::RelationFamily(family) => {
                if let Some(members) =
                    families.extent(family.binder().set().as_str()).or_else(|| {
                        complete_exterior_cardinality(
                            definition,
                            family.binder().set().as_str(),
                            family.range(),
                            diagnostics,
                        )
                    })
                {
                    families.expressions(
                        family.relation().expressions(),
                        members,
                        &mut expression_nodes,
                        diagnostics,
                    );
                }
            }
            _ => {}
        }
        let port = match item {
            ComponentItem::Port(port) => port,
            ComponentItem::PortFamily(family) => family.port(),
            _ => continue,
        };
        let connector = match port.syntax() {
            PortSyntax::ScalarPhysicalConnector { connector }
            | PortSyntax::FieldPhysical { connector, .. } => connector,
            _ => continue,
        };
        match elaborator.resolve_connector(
            &definition.namespace,
            connector,
            definition.file,
            port.range(),
        ) {
            Ok(connector) => {
                local_connectors.insert(DefinitionKey {
                    namespace: connector.namespace,
                    name: connector.declaration.name().to_owned(),
                });
            }
            Err(error) => diagnostics.push(error),
        }
    }
    (
        LocalFootprint {
            declarations,
            connections,
            expression_nodes,
        },
        local_connectors,
    )
}

fn complete_exterior_cardinality(
    definition: &ComponentDefinition<'_>,
    set: &str,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    signature_complete_exterior_cardinality(
        definition.file,
        definition.declaration.signature(),
        set,
        range,
        diagnostics,
    )
}

fn signature_complete_exterior_cardinality(
    file: &str,
    signature: &[eqiora_lang::SignatureItem],
    set: &str,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    let support = signature.iter().find_map(|item| match item {
        eqiora_lang::SignatureItem::Support(support) if support.name() == set => Some(support),
        _ => None,
    });
    let Some(support) = support else {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("boundary family refers to unknown complete-exterior support `{set}`"),
        ));
        return None;
    };
    let eqiora_lang::SupportSlotSyntax::CompleteExterior { parent } = support.syntax() else {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("boundary family support `{set}` is not a complete exterior"),
        ));
        return None;
    };
    let parent_name = parent;
    let parent = signature.iter().find_map(|item| match item {
        eqiora_lang::SignatureItem::Support(parent_support)
            if parent_support.name() == parent_name =>
        {
            Some(parent_support)
        }
        _ => None,
    });
    let Some(parent) = parent else {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("complete exterior `{set}` refers to unknown parent support `{parent_name}`"),
        ));
        return None;
    };
    let eqiora_lang::SupportSlotSyntax::Volume { ambient_dimension } = parent.syntax() else {
        diagnostics.push(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("complete exterior `{set}` requires a volume parent support"),
        ));
        return None;
    };
    match ambient_dimension.checked_mul(2) {
        Some(members) => Some(members),
        None => {
            diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                range,
                format!("complete exterior `{set}` member count overflows usize"),
            ));
            None
        }
    }
}

pub(super) fn model_local_footprint(
    elaborator: &Elaborator<'_>,
    definition: &ModelDefinition<'_>,
    diagnostics: &mut Vec<Diagnostic>,
    values: Option<&super::super::parameters::SymbolicParameterMap>,
) -> LocalFootprint {
    let generic_values;
    let require_exact = values.is_some();
    let values = match values {
        Some(values) => values,
        None => {
            generic_values = family_model_values(elaborator, definition, diagnostics);
            &generic_values
        }
    };
    let families = super::families::Families::new(
        definition.file,
        definition.owned_items().filter_map(|item| match item {
            Item::IndexSet(set) => Some(set),
            _ => None,
        }),
        values,
        require_exact,
        diagnostics,
    );
    let mut footprint = LocalFootprint::default();
    for item in definition.owned_items() {
        match item {
            Item::Parameter(parameter) => checked_local_add(
                &mut footprint.declarations,
                elaborator
                    .record_for_type(&definition.namespace, parameter.value_type())
                    .map_or(1, |record| 1 + record.definition.members().len()),
                definition.file,
                parameter.range(),
                "Parameter record and members",
                diagnostics,
            ),
            Item::Field(field) => checked_local_add(
                &mut footprint.declarations,
                elaborator
                    .record_for_type(&definition.namespace, field.value_type())
                    .map_or_else(
                        || if field.domain().is_some() { 2 } else { 1 },
                        |record| {
                            1 + record.definition.members().len()
                                * if field.domain().is_some() { 2 } else { 1 }
                        },
                    ),
                definition.file,
                field.range(),
                "Field and continuum representation",
                diagnostics,
            ),
            Item::Relation(_) => checked_local_add(
                &mut footprint.declarations,
                2,
                definition.file,
                definition.declaration.range(),
                "declaration",
                diagnostics,
            ),
            Item::Connection(connection) => checked_local_add(
                &mut footprint.connections,
                families.members(connection.binder(), diagnostics),
                definition.file,
                connection.range(),
                "indexed Connection",
                diagnostics,
            ),
            Item::RelationFamily(family) => checked_local_add(
                &mut footprint.declarations,
                families
                    .extent(family.binder().set().as_str())
                    .or_else(|| {
                        signature_complete_exterior_cardinality(
                            definition.file,
                            definition.declaration.signature(),
                            family.binder().set().as_str(),
                            family.range(),
                            diagnostics,
                        )
                    })
                    .unwrap_or(0)
                    .checked_mul(2)
                    .unwrap_or_else(|| {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_LOWERING_ERROR,
                            definition.file,
                            family.range(),
                            "complete-exterior Relation-family declaration count overflows usize",
                        ));
                        0
                    }),
                definition.file,
                family.range(),
                "indexed Relation and Activation declarations",
                diagnostics,
            ),
            Item::BoundaryConnection(_) => checked_local_add(
                &mut footprint.connections,
                1,
                definition.file,
                definition.declaration.range(),
                "Connection",
                diagnostics,
            ),
            Item::Coordinate(_) | Item::Let(_) => {}
            Item::Instance(instance) => {
                let count = input_binding_count(
                    elaborator,
                    &definition.namespace,
                    definition.file,
                    instance,
                    diagnostics,
                );
                checked_local_add(
                    &mut footprint.connections,
                    count,
                    definition.file,
                    instance.range(),
                    "Input connections",
                    diagnostics,
                );
            }
            _ => checked_local_add(
                &mut footprint.declarations,
                1,
                definition.file,
                definition.declaration.range(),
                "declaration",
                diagnostics,
            ),
        }
        match item {
            Item::Relation(relation) => families.expressions(
                relation.expressions(),
                1,
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::Initial(initial) => families.equations(
                initial.equations(),
                1,
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::RelationFamily(family) => families.expressions(
                family.relation().expressions(),
                families
                    .extent(family.binder().set().as_str())
                    .or_else(|| {
                        signature_complete_exterior_cardinality(
                            definition.file,
                            definition.declaration.signature(),
                            family.binder().set().as_str(),
                            family.range(),
                            diagnostics,
                        )
                    })
                    .unwrap_or(0),
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::Connection(connection) => families.expressions(
                connection.port_expressions(),
                families.members(connection.binder(), diagnostics),
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::Event(value) => families.expressions(
                [value.guard()],
                1,
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::Observable(value) => families.expressions(
                [value.value()],
                1,
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            Item::Coordinate(value) | Item::Let(value) => families.expressions(
                [value.value()],
                1,
                &mut footprint.expression_nodes,
                diagnostics,
            ),
            _ => {}
        }
    }
    footprint
}

fn checked_local_add(
    count: &mut usize,
    additional: usize,
    file: &str,
    range: TextRange,
    resource: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match count.checked_add(additional) {
        Some(next) => *count = next,
        None => diagnostics.push(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            format!("local {resource} count overflows usize"),
        )),
    }
}

pub(super) fn input_binding_count(
    elaborator: &Elaborator<'_>,
    namespace: &super::super::preflight::DefinitionNamespace,
    file: &str,
    instance: &eqiora_lang::InstanceDecl,
    diagnostics: &mut Vec<Diagnostic>,
) -> usize {
    match elaborator.resolve_component(namespace, instance.definition(), file, instance.range()) {
        Ok(child) => instance.bindings().iter().filter(|binding| child.signature().iter().any(|item| matches!(item, eqiora_lang::SignatureItem::Input(input) if input.name() == binding.name()))).count(),
        Err(error) => { diagnostics.push(error); 0 }
    }
}

fn family_component_values(
    elaborator: &Elaborator<'_>,
    definition: &ComponentDefinition<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> super::super::parameters::SymbolicParameterMap {
    use super::super::{clocks, parameters};
    let result: Result<_, Vec<Diagnostic>> = (|| {
        let mut values = parameters::resolve_component_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| clocks::component(definition.file, definition.declaration, name),
            &parameters::RecordContext::component(elaborator, definition),
        )?;
        elaborator.bind_symbolic_properties(
            &definition.namespace,
            definition.file,
            definition.declaration.signature(),
            &mut values,
        )?;
        parameters::resolve_component_lets(
            definition.file,
            definition.declaration,
            &mut values,
            |name| clocks::component(definition.file, definition.declaration, name),
        )?;
        Ok(values)
    })();
    match result {
        Ok(values) => values,
        Err(errors) => {
            diagnostics.extend(errors);
            Default::default()
        }
    }
}
fn family_model_values(
    elaborator: &Elaborator<'_>,
    definition: &ModelDefinition<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> super::super::parameters::SymbolicParameterMap {
    use super::super::{clocks, parameters};
    let result: Result<_, Vec<Diagnostic>> = (|| {
        let mut values = parameters::resolve_model_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| clocks::model(definition.file, definition.declaration, name),
            &parameters::RecordContext::model(elaborator, definition),
        )?;
        elaborator.bind_symbolic_properties(
            &definition.namespace,
            definition.file,
            definition.declaration.signature(),
            &mut values,
        )?;
        parameters::resolve_model_lets(
            definition.file,
            definition.declaration,
            &mut values,
            |name| clocks::model(definition.file, definition.declaration, name),
        )?;
        Ok(values)
    })();
    match result {
        Ok(values) => values,
        Err(errors) => {
            diagnostics.extend(errors);
            Default::default()
        }
    }
}
