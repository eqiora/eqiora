//! Materialize already-specialized Component occurrence declarations.
use super::*;

impl<'a, 'd> RootExpansion<'a, 'd> {
    pub(super) fn materialize_component_items(
        &mut self,
        occurrence: ComponentOccurrence<'_, 'd>,
        scope: &Scope,
        identities: &ScopeIdentities,
        support_bindings: &ResolvedSupportBindings<FullElaborationIdentity>,
    ) -> Result<(), Diagnostic> {
        let component = occurrence.definition;
        let instance = occurrence.instance;
        let mut initial_duplicates = BTreeMap::<String, usize>::new();
        for item in component.owned_items() {
            match item {
                ComponentItem::Initial(declaration) => {
                    let name = crate::source_identity::initial_declaration_name(declaration)?;
                    let duplicate = initial_duplicates.entry(name.clone()).or_default();
                    let name = format!("{name}-{duplicate}");
                    *duplicate += 1;
                    let identity = self.relation_identity(
                        occurrence.instance_path,
                        definition_path(&component.namespace, "component", component.name(), &name),
                        SourceLocation::new(component.file, declaration.range()),
                        SourceLocation::new(occurrence.instance_file, instance.range()),
                        Vec::new(),
                    )?;
                    let equations =
                        rewrite_equations(component.file, declaration.equations(), scope, None)?;
                    self.items.push(FlatItemBlueprint::Relation {
                        name: internal_name(identity.entity.full),
                        activation: eqiora_lang::ActivationSyntax::Continuous,
                        domain: None,
                        body: equations.into(),
                        initial: true,
                        range: declaration.range(),
                        identity,
                    });
                }

                ComponentItem::Parameter(_)
                | ComponentItem::IndexSet(_)
                | ComponentItem::Let(_) => {}
                ComponentItem::Port(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    if let PortSyntax::Signal { value_type, .. } = declaration.syntax() {
                        self.record_type_structure(
                            &internal_name(identity.full),
                            component.file,
                            value_type,
                            &scope.symbolic_parameters(),
                        )?;
                    }
                    let (contract, materialization) =
                        self.component_port_syntax(component, declaration, scope)?;
                    if let Some(materialization) = materialization {
                        let occurrence =
                            self.physical_ports.get_mut(&identity.full).ok_or_else(|| {
                                hierarchy_error("physical Port occurrence was not registered")
                            })?;
                        occurrence.contract = Some(materialization.contract);
                    }
                    self.items.push(FlatItemBlueprint::Port {
                        name: internal_name(identity.full),
                        contract,
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::PortFamily(family) => {
                    let declaration = family.port();
                    let set = support_bindings
                        .boundary_set(family.binder().set().as_str())
                        .ok_or_else(|| {
                            hierarchy_error(format!(
                                "Port family `{}` has no resolved complete-exterior binding `{}`",
                                declaration.name(),
                                family.binder().set().as_str()
                            ))
                        })?;
                    for side in set.witness().sides() {
                        let boundary = *side.boundary();
                        let member = set.member(&boundary).ok_or_else(|| {
                            hierarchy_error(
                                "complete-exterior witness has no identity-keyed member locator",
                            )
                        })?;
                        let identity = identities
                            .boundary_family_entities
                            .get(&(declaration.name().to_owned(), boundary))
                            .cloned()
                            .ok_or_else(|| {
                                hierarchy_error(format!(
                                    "Port family `{}` member identity was not allocated",
                                    declaration.name()
                                ))
                            })?;
                        let (contract, materialization) = self.component_port_family_syntax(
                            component,
                            family,
                            boundary,
                            member.target(),
                            set.witness().ambient_dimension(),
                        )?;
                        let occurrence =
                            self.physical_ports.get_mut(&identity.full).ok_or_else(|| {
                                hierarchy_error(
                                    "physical Port-family occurrence was not registered",
                                )
                            })?;
                        occurrence.contract = Some(materialization.contract);
                        self.items.push(FlatItemBlueprint::Port {
                            name: internal_name(identity.full),
                            contract,
                            range: family.range(),
                            identity,
                        });
                    }
                }
                ComponentItem::Observable(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    let (value, reduction) =
                        super::observable::rewrite(component.file, declaration.value(), scope)?;
                    self.record_type_structure(
                        &internal_name(identity.full),
                        component.file,
                        declaration.value_type(),
                        &scope.symbolic_parameters(),
                    )?;
                    self.record_structural(
                        &internal_name(identity.full),
                        value.structural_parameters(),
                    )?;
                    self.items.push(FlatItemBlueprint::Observable {
                        name: internal_name(identity.full),
                        value_type: super::super::parameters::specialize_type(
                            component.file,
                            declaration.value_type(),
                            &scope.symbolic_parameters(),
                        )?,
                        value,
                        reduction,
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::Field(declaration) => {
                    if let Some(record) = self
                        .elaborator
                        .record_for_type(&component.namespace, declaration.value_type())
                        .cloned()
                    {
                        self.emit_record_field(
                            scope,
                            declaration,
                            &record,
                            identities,
                            component.file,
                        )?;
                        continue;
                    }
                    let identity = identities.entities[declaration.name()].clone();
                    self.record_type_structure(
                        &internal_name(identity.full),
                        component.file,
                        declaration.value_type(),
                        &scope.symbolic_parameters(),
                    )?;
                    let (domain, activation) =
                        rewrite_field_scope(component.file, declaration, scope)?;
                    let representation = self.add_support_representation(domain.as_deref())?;
                    self.items.push(FlatItemBlueprint::Field {
                        name: internal_name(identity.full),
                        domain,
                        representation,
                        value_type: super::super::parameters::specialize_type(
                            component.file,
                            declaration.value_type(),
                            &scope.symbolic_parameters(),
                        )?,
                        role: declaration.role(),
                        activation,
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::Event(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    self.items.push(FlatItemBlueprint::Event {
                        name: internal_name(identity.full),
                        guard: crate::hierarchy::scope::rewrite_expression_with_boundary_member(
                            component.file,
                            declaration.guard(),
                            scope,
                            None,
                        )?,
                        direction: declaration.direction(),
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::Clock(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    self.items.push(FlatItemBlueprint::Clock {
                        supplied_id: None,
                        name: internal_name(identity.full),
                        period: declaration.period().clone(),
                        phase: declaration.phase().clone(),
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::Relation(declaration) => {
                    let identity = identities.relations[declaration.name()].clone();
                    let (activation, domain, equations) =
                        rewrite_relation(component.file, declaration, scope)?;
                    self.record_physical_relation_owners(
                        component.file,
                        declaration.range(),
                        identity.entity.full,
                        &equations,
                    )?;
                    self.items.push(FlatItemBlueprint::Relation {
                        initial: false,
                        name: internal_name(identity.entity.full),
                        activation,
                        domain,
                        body: equations,
                        range: declaration.range(),
                        identity,
                    });
                }
                ComponentItem::RelationFamily(family) => {
                    if scope.index_set(family.binder().set().as_str()).is_some() {
                        self.add_indexed_relations(
                            family,
                            scope,
                            (occurrence.instance_path, occurrence.display_prefix),
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                family.relation().name(),
                            ),
                            EntitySourceOrigin {
                                definition: SourceLocation::new(component.file, family.range()),
                                instance: SourceLocation::new(
                                    occurrence.instance_file,
                                    instance.range(),
                                ),
                                bindings: scope.occurrence_bindings().to_vec(),
                            },
                        )?;
                        continue;
                    }
                    let declaration = family.relation();
                    let set = support_bindings
                        .boundary_set(family.binder().set().as_str())
                        .ok_or_else(|| {
                            hierarchy_error(format!(
                                "Relation family `{}` has no resolved complete-exterior binding `{}`",
                                declaration.name(),
                                family.binder().set().as_str()
                            ))
                        })?;
                    for side in set.witness().sides() {
                        let boundary = *side.boundary();
                        let member = set.member(&boundary).ok_or_else(|| {
                            hierarchy_error(
                                "complete-exterior witness has no identity-keyed member locator",
                            )
                        })?;
                        let identity = identities
                            .boundary_family_relations
                            .get(&(declaration.name().to_owned(), boundary))
                            .cloned()
                            .ok_or_else(|| {
                                hierarchy_error(format!(
                                    "Relation family `{}` member identity was not allocated",
                                    declaration.name()
                                ))
                            })?;
                        let active = Some(ActiveBoundaryMember::new(
                            family.binder().member(),
                            boundary,
                        ));
                        let equations = rewrite_equations(
                            component.file,
                            declaration.conditions().ok_or_else(|| {
                                hierarchy_error(
                                    "Law families require explicit retained term lowering",
                                )
                            })?,
                            scope,
                            active,
                        )?;
                        let equations: crate::lower::LoweringRelationBody = equations.into();
                        self.record_physical_relation_owners(
                            component.file,
                            family.range(),
                            identity.entity.full,
                            &equations,
                        )?;
                        self.items.push(FlatItemBlueprint::Relation {
                            initial: false,
                            name: internal_name(identity.entity.full),
                            activation: eqiora_lang::ActivationSyntax::Continuous,
                            domain: Some(member.target().to_owned()),
                            body: equations,
                            range: family.range(),
                            identity,
                        });
                    }
                }
                ComponentItem::Connection(declaration) => {
                    self.add_connection(
                        declaration,
                        scope,
                        occurrence.instance_path,
                        definition_path(&component.namespace, "component", component.name(), "net"),
                        ConnectionOrigin {
                            instance: SourceLocation::new(
                                occurrence.instance_file,
                                instance.range(),
                            ),
                            bindings: scope.occurrence_bindings().to_vec(),
                            definition_file: component.file.to_owned(),
                        },
                    )?;
                }
                ComponentItem::BoundaryConnection(declaration) => {
                    if let Some(binder) = declaration.binder() {
                        let set = support_bindings.boundary_set(binder.set().as_str()).ok_or_else(|| {
                            hierarchy_error(format!(
                                "Connection family has no resolved complete-exterior binding `{}`",
                                binder.set().as_str()
                            ))
                        })?;
                        for side in set.witness().sides() {
                            let boundary = *side.boundary();
                            let member = set.member(&boundary).ok_or_else(|| {
                                hierarchy_error(
                                    "complete-exterior witness has no identity-keyed member locator",
                                )
                            })?;
                            self.add_boundary_connection(
                                declaration,
                                scope,
                                Some(ActiveBoundaryMember::new(binder.member(), boundary)),
                                occurrence.instance_path,
                                definition_path(
                                    &component.namespace,
                                    "component",
                                    component.name(),
                                    "net",
                                ),
                                ConnectionOrigin {
                                    instance: SourceLocation::new(
                                        occurrence.instance_file,
                                        instance.range(),
                                    ),
                                    bindings: boundary_family_bindings(
                                        scope.occurrence_bindings(),
                                        occurrence.instance_file,
                                        member.source_range(),
                                    ),
                                    definition_file: component.file.to_owned(),
                                },
                            )?;
                        }
                    } else {
                        self.add_boundary_connection(
                            declaration,
                            scope,
                            None,
                            occurrence.instance_path,
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                "net",
                            ),
                            ConnectionOrigin {
                                instance: SourceLocation::new(
                                    occurrence.instance_file,
                                    instance.range(),
                                ),
                                bindings: scope.occurrence_bindings().to_vec(),
                                definition_file: component.file.to_owned(),
                            },
                        )?;
                    }
                }
                ComponentItem::Instance(child) => {
                    self.add_input_bindings(
                        child,
                        scope,
                        occurrence.instance_path,
                        definition_path(
                            &component.namespace,
                            "component",
                            component.name(),
                            "input_binding",
                        ),
                        &component.namespace,
                        ConnectionOrigin {
                            instance: SourceLocation::new(
                                occurrence.instance_file,
                                instance.range(),
                            ),
                            bindings: scope.occurrence_bindings().to_vec(),
                            definition_file: component.file.to_owned(),
                        },
                    )?;
                }
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        component.file,
                        component.range(),
                        "component item is newer than hierarchy elaboration",
                    ));
                }
            }
        }
        Ok(())
    }
}
