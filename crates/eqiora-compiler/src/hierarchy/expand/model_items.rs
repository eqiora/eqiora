//! Materialize owned Model declarations after all occurrence symbols are bound.
use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn materialize_model_items(
        &mut self,
        scope: &Scope,
        identities: &ScopeIdentities,
    ) -> Result<(), Diagnostic> {
        let model = self.model.clone();
        let mut initial_duplicates = BTreeMap::<String, usize>::new();
        for item in model.owned_items() {
            match item {
                Item::Initial(declaration) => {
                    let name = crate::source_identity::initial_declaration_name(declaration)?;
                    let duplicate = initial_duplicates.entry(name.clone()).or_default();
                    let name = format!("{name}-{duplicate}");
                    *duplicate += 1;
                    let identity = self.relation_identity(
                        &self.root_path,
                        definition_path(&self.model.namespace, "model", self.model.name(), &name),
                        SourceLocation::new(self.model.file, declaration.range()),
                        SourceLocation::new(self.model.file, self.model.range()),
                        Vec::new(),
                    )?;
                    let equations =
                        rewrite_equations(self.model.file, declaration.equations(), scope, None)?;
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

                Item::Domain(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    let syntax = match declaration.syntax() {
                        DomainSyntax::Product { .. } => {
                            super::coordinate_products::rewrite_product(declaration, scope)?
                        }
                        DomainSyntax::CartesianBox(bounds) => DomainSyntax::CartesianBox(
                            super::cartesian::rewrite_coordinates(self.model.file, bounds, scope)?,
                        ),
                        DomainSyntax::Boundary { parent, axis, side } => {
                            let parent = resolve_local_kind(
                                self.model.file,
                                declaration.range(),
                                scope,
                                parent,
                                |kind| matches!(kind, SymbolKind::Domain),
                                "boundary parent Domain",
                            )?;
                            DomainSyntax::Boundary {
                                parent: parent.internal_name.clone(),
                                axis: *axis,
                                side: *side,
                            }
                        }
                        DomainSyntax::ScalarPhysical {
                            across_name,
                            across_type,
                            through_name,
                            through_type,
                        } => DomainSyntax::ScalarPhysical {
                            across_name: across_name.clone(),
                            through_name: through_name.clone(),
                            across_type: across_type.clone(),
                            through_type: through_type.clone(),
                        },
                        _ => {
                            return Err(source_error(
                                codes::LANGUAGE_LOWERING_ERROR,
                                self.model.file,
                                declaration.range(),
                                "Domain syntax is newer than hierarchy elaboration",
                            ));
                        }
                    };
                    self.items.push(FlatItemBlueprint::Domain {
                        name: internal_name(identity.full),
                        contract: LoweringDomainContract::Source(syntax),
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Observable(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    let (value, reduction) =
                        super::observable::rewrite(self.model.file, declaration.value(), scope)?;
                    self.record_type_structure(
                        &internal_name(identity.full),
                        self.model.file,
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
                            self.model.file,
                            declaration.value_type(),
                            &scope.symbolic_parameters(),
                        )?,
                        value,
                        reduction,
                        domain: declaration
                            .domain()
                            .map(|name| {
                                resolve_local_kind(
                                    self.model.file,
                                    declaration.range(),
                                    scope,
                                    name,
                                    |kind| matches!(kind, SymbolKind::Domain),
                                    "Observable output Domain",
                                )
                                .map(|symbol| symbol.internal_name.clone())
                            })
                            .transpose()?,
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Field(declaration) => {
                    if let Some(record) = self
                        .elaborator
                        .record_for_type(&self.model.namespace, declaration.value_type())
                        .cloned()
                    {
                        self.emit_record_field(
                            scope,
                            declaration,
                            &record,
                            identities,
                            self.model.file,
                        )?;
                        continue;
                    }
                    let identity = identities.entities[declaration.name()].clone();
                    self.record_type_structure(
                        &internal_name(identity.full),
                        self.model.file,
                        declaration.value_type(),
                        &scope.symbolic_parameters(),
                    )?;
                    let (domain, activation) =
                        rewrite_field_scope(self.model.file, declaration, scope)?;
                    let representation = self.add_support_representation(domain.as_deref())?;
                    self.items.push(FlatItemBlueprint::Field {
                        name: internal_name(identity.full),
                        domain,
                        representation,
                        value_type: super::super::parameters::specialize_type(
                            self.model.file,
                            declaration.value_type(),
                            &scope.symbolic_parameters(),
                        )?,
                        role: declaration.role(),
                        activation,
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Parameter(declaration) => {
                    if scope
                        .record_context
                        .record_for_type(declaration.value_type())
                        .is_some()
                    {
                        continue;
                    }
                    let identity = identities.entities[declaration.name()].clone();
                    self.record_type_structure(
                        &internal_name(identity.full),
                        self.model.file,
                        declaration.value_type(),
                        &scope.symbolic_parameters(),
                    )?;
                    let expression = super::super::scope::rewrite_expression_with_boundary_member(
                        self.model.file,
                        declaration.value(),
                        scope,
                        None,
                    )?;
                    self.record_structural(
                        &internal_name(identity.full),
                        expression.structural_parameters(),
                    )?;
                    self.items.push(FlatItemBlueprint::Parameter {
                        name: internal_name(identity.full),
                        value: scope
                            .parameter(declaration.name())
                            .expect("allocated model parameter")
                            .value
                            .clone(),
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Port(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    if let eqiora_lang::PortSyntax::Signal { value_type, .. } = declaration.syntax()
                    {
                        self.record_type_structure(
                            &internal_name(identity.full),
                            self.model.file,
                            value_type,
                            &scope.symbolic_parameters(),
                        )?;
                    }
                    let syntax = rewrite_model_port(
                        self.model.file,
                        declaration.syntax(),
                        declaration.range(),
                        scope,
                    )?;
                    self.items.push(FlatItemBlueprint::Port {
                        name: internal_name(identity.full),
                        contract: LoweringPortContract::Source(syntax),
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Event(declaration) => {
                    let identity = identities.entities[declaration.name()].clone();
                    self.items.push(FlatItemBlueprint::Event {
                        name: internal_name(identity.full),
                        guard: crate::hierarchy::scope::rewrite_expression_with_boundary_member(
                            self.model.file,
                            declaration.guard(),
                            scope,
                            None,
                        )?,
                        direction: declaration.direction(),
                        priority: declaration.priority(),
                        range: declaration.range(),
                        identity,
                    });
                }
                Item::Clock(declaration) => {
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
                Item::Relation(declaration) => {
                    let identity = identities.relations[declaration.name()].clone();
                    let (activation, domain, equations) =
                        rewrite_relation(self.model.file, declaration, scope)?;
                    self.record_physical_relation_owners(
                        self.model.file,
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
                Item::RelationFamily(family) => {
                    if scope.boundary_set(family.binder().set().as_str()).is_some()
                        && scope.index_set(family.binder().set().as_str()).is_none()
                    {
                        self.add_model_boundary_relations(family, scope)?;
                        continue;
                    }
                    self.add_indexed_relations(
                        family,
                        scope,
                        (&self.root_path.clone(), ""),
                        definition_path(
                            &model.namespace,
                            "model",
                            model.name(),
                            family.relation().name(),
                        ),
                        EntitySourceOrigin {
                            definition: SourceLocation::new(model.file, family.range()),
                            instance: SourceLocation::new(model.file, model.range()),
                            bindings: Vec::new(),
                        },
                    )?;
                }
                Item::Connection(declaration) => {
                    self.add_connection(
                        declaration,
                        scope,
                        &self.root_path.clone(),
                        definition_path(&self.model.namespace, "model", self.model.name(), "net"),
                        ConnectionOrigin {
                            instance: SourceLocation::new(self.model.file, self.model.range()),
                            bindings: Vec::new(),
                            definition_file: self.model.file.to_owned(),
                        },
                    )?;
                }
                Item::BoundaryConnection(declaration) => {
                    self.add_boundary_connection(
                        declaration,
                        scope,
                        None,
                        &self.root_path.clone(),
                        definition_path(&self.model.namespace, "model", self.model.name(), "net"),
                        ConnectionOrigin {
                            instance: SourceLocation::new(self.model.file, self.model.range()),
                            bindings: Vec::new(),
                            definition_file: self.model.file.to_owned(),
                        },
                    )?;
                }
                Item::Coordinate(_) | Item::Let(_) | Item::IndexSet(_) => {}
                Item::Instance(instance) => {
                    self.add_input_bindings(
                        instance,
                        scope,
                        &self.root_path.clone(),
                        definition_path(
                            &self.model.namespace,
                            "model",
                            self.model.name(),
                            "input_binding",
                        ),
                        &self.model.namespace.clone(),
                        ConnectionOrigin {
                            instance: SourceLocation::new(self.model.file, self.model.range()),
                            bindings: Vec::new(),
                            definition_file: self.model.file.to_owned(),
                        },
                    )?;
                }
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        self.model.file,
                        self.model.range(),
                        "model item is newer than hierarchy elaboration",
                    ));
                }
            }
        }
        let ports = self
            .model
            .signature()
            .iter()
            .filter_map(|item| match item {
                eqiora_lang::SignatureItem::Input(value)
                | eqiora_lang::SignatureItem::Output(value) => Some(value.name()),
                eqiora_lang::SignatureItem::Port(value) => Some(value.name()),
                _ => None,
            })
            .map(|name| {
                scope
                    .symbol(name)
                    .expect("allocated signature endpoint")
                    .internal_name
                    .clone()
            })
            .collect();
        self.items.push(FlatItemBlueprint::Boundary {
            ports,
            range: self.model.range(),
        });
        Ok(())
    }
}

impl RootExpansion<'_, '_> {
    fn add_model_boundary_relations(
        &mut self,
        family: &eqiora_lang::RelationFamilyDecl,
        scope: &Scope,
    ) -> Result<(), Diagnostic> {
        let set = scope
            .boundary_set(family.binder().set().as_str())
            .ok_or_else(|| {
                hierarchy_error("Model Relation family has no resolved complete-exterior binding")
            })?;
        let declaration = family.relation();
        for side in set.witness().sides() {
            let boundary = *side.boundary();
            let member = set.member(&boundary).ok_or_else(|| {
                hierarchy_error("complete-exterior witness has no identity-keyed member locator")
            })?;
            let identity = self.boundary_family_relation_identity(
                &self.root_path,
                definition_path(
                    &self.model.namespace,
                    "model",
                    self.model.name(),
                    declaration.name(),
                ),
                boundary,
                EntitySourceOrigin {
                    definition: SourceLocation::new(self.model.file, family.range()),
                    instance: SourceLocation::new(self.model.file, self.model.range()),
                    bindings: boundary_family_bindings(
                        scope.occurrence_bindings(),
                        self.model.file,
                        member.source_range(),
                    ),
                },
            )?;
            self.register_family_relation_display(
                boundary_family_display("", declaration.name(), side.axis(), side.side()),
                &identity,
            )?;
            let equations = super::super::scope::rewrite_relation_body(
                self.model.file,
                declaration,
                scope,
                Some(ActiveBoundaryMember::new(
                    family.binder().member(),
                    boundary,
                )),
            )?;
            self.record_physical_relation_owners(
                self.model.file,
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
        Ok(())
    }
}
