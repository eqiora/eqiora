//! Allocate one root Model and its exact owned declarations.
use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_model_scope(
        &mut self,
        scope: &mut Scope,
    ) -> Result<ScopeIdentities, Diagnostic> {
        let model = self.model.clone();
        let mut identities = ScopeIdentities::default();
        let parameters = super::super::parameters::resolve_model_parameters(
            model.file,
            model.declaration,
            |name| {
                super::super::clocks::occurrence(scope, name)
                    .or_else(|| super::super::clocks::model(model.file, model.declaration, name))
            },
            scope.frame_supports(),
            &scope.record_context,
        )
        .map_err(|mut errors| errors.remove(0))?;
        let mut owned_items = model.owned_items().collect::<Vec<_>>();
        owned_items.sort_by_key(|item| !matches!(item, Item::Clock(_)));
        for item in owned_items {
            if let Item::Parameter(parameter) = item
                && self.allocate_model_record_parameter(scope, parameter, &parameters)?
            {
                continue;
            }
            if let Item::Field(field) = item
                && self.allocate_record_field(
                    scope,
                    field,
                    records::RecordFieldOccurrence {
                        namespace: &model.namespace,
                        definition_name: model.name(),
                        instance_path: &self.root_path.clone(),
                        display_prefix: "",
                        file: model.file,
                        instance: SourceLocation::new(model.file, field.range()),
                        bindings: Vec::new(),
                    },
                    &mut identities,
                )?
            {
                continue;
            }
            let (name, kind, symbol_kind, parameter_value, range) = match item {
                Item::Domain(value) => (
                    value.name(),
                    EntityKind::Domain,
                    SymbolKind::Domain,
                    None,
                    value.range(),
                ),
                Item::Observable(value) => (
                    value.name(),
                    EntityKind::Observable,
                    SymbolKind::Observable,
                    None,
                    value.range(),
                ),
                Item::Field(value) => (
                    value.name(),
                    EntityKind::Field,
                    SymbolKind::Field,
                    None,
                    value.range(),
                ),
                Item::Parameter(declaration) => {
                    let value = parameters[declaration.name()].value.clone();
                    (
                        declaration.name(),
                        EntityKind::Parameter,
                        SymbolKind::Parameter,
                        Some(value),
                        declaration.range(),
                    )
                }
                Item::Port(value) => (
                    value.name(),
                    EntityKind::Port,
                    SymbolKind::Port {
                        activation: super::super::scope::port_activation(
                            model.file,
                            value.syntax(),
                            value.range(),
                            scope,
                        )?,
                        quantities: self.port_quantities(
                            value.syntax(),
                            &model.namespace,
                            model.file,
                            value.range(),
                        )?,
                    },
                    None,
                    value.range(),
                ),
                Item::Event(value) => (
                    value.name(),
                    EntityKind::Activation,
                    SymbolKind::Event,
                    None,
                    value.range(),
                ),
                Item::Clock(value) => (
                    value.name(),
                    EntityKind::ClockDomain,
                    SymbolKind::Clock(
                        crate::units::lower_clock(self.model.file, value.period(), value.phase())?
                            .0,
                    ),
                    None,
                    value.range(),
                ),
                Item::Relation(value) => {
                    let path = definition_path(
                        &self.model.namespace,
                        "model",
                        self.model.name(),
                        value.name(),
                    );
                    let identity = self.relation_identity(
                        &self.root_path,
                        path,
                        SourceLocation::new(self.model.file, value.range()),
                        SourceLocation::new(self.model.file, self.model.range()),
                        Vec::new(),
                    )?;
                    self.register_symbol(
                        value.name().to_owned(),
                        value.name(),
                        &identity.entity,
                        SymbolKind::Relation,
                        scope,
                    )?;
                    identities
                        .relations
                        .insert(value.name().to_owned(), identity);
                    continue;
                }
                Item::IndexSet(_)
                | Item::RelationFamily(_)
                | Item::Initial(_)
                | Item::Connection(_)
                | Item::BoundaryConnection(_)
                | Item::Let(_)
                | Item::Instance(_) => continue,
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        self.model.file,
                        self.model.range(),
                        "model item is newer than hierarchy elaboration",
                    ));
                }
            };
            let identity = self.entity_identity(
                &self.root_path,
                definition_path(&self.model.namespace, "model", self.model.name(), name),
                kind,
                SourceLocation::new(self.model.file, range),
                SourceLocation::new(self.model.file, self.model.range()),
                Vec::new(),
            )?;
            self.register_symbol(name.to_owned(), name, &identity, symbol_kind, scope)?;
            if let Some(value) = parameter_value
                && scope
                    .insert_parameter(
                        name.to_owned(),
                        ResolvedParameter::model_parameter(
                            value,
                            identity.full,
                            internal_name(identity.full),
                            range,
                        ),
                    )
                    .is_some()
            {
                return Err(hierarchy_error(format!(
                    "duplicate flattened Parameter term `{name}`"
                )));
            }
            identities.entities.insert(name.to_owned(), identity);
        }
        self.allocate_model_lets(scope, &model)?;
        self.allocate_nominals(
            scope,
            &model.namespace,
            model.name(),
            &self.root_path.clone(),
            &model
                .owned_items()
                .filter_map(|item| match item {
                    Item::IndexSet(value) => Some(value),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            model.file,
        )?;
        for item in model.owned_items() {
            let Item::Domain(declaration) = item else {
                continue;
            };
            match declaration.syntax() {
                DomainSyntax::CartesianBox(bounds) => {
                    scope.insert_spatial_support(
                        declaration.name().to_owned(),
                        SpatialSupport::Volume {
                            domain: identities.entities[declaration.name()].full,
                            dimensions: bounds.len(),
                        },
                    );
                }
                DomainSyntax::Product { .. }
                | DomainSyntax::Boundary { .. }
                | DomainSyntax::ScalarPhysical { .. } => {}
                _ => {
                    return Err(source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        self.model.file,
                        declaration.range(),
                        "Domain syntax is newer than spatial-support allocation",
                    ));
                }
            }
        }
        let products =
            super::super::supports::model_spatial_supports(model.file, model.declaration)
                .map_err(|errors| errors.into_iter().next().expect("product diagnostic"))?;
        super::coordinate_products::project_products(scope, &products)?;
        self.allocate_cartesian_boundaries(scope, &identities)?;
        for item in model.owned_items() {
            let Item::Field(declaration) = item else {
                continue;
            };
            let support = declaration
                .domain()
                .and_then(|domain| scope.spatial_support(domain).cloned());
            if let Some(record) = self
                .elaborator
                .record_for_type(&self.model.namespace, declaration.value_type())
            {
                let activation = super::super::scope::rewrite_activation(
                    self.model.file,
                    declaration.activation(),
                    declaration.range(),
                    scope,
                )?;
                for (name, value_type) in record.definition.members() {
                    let local = format!("{}.{name}", declaration.name());
                    scope
                        .field_evolution
                        .insert(local.clone(), (declaration.role(), activation.clone()));
                    scope.insert_field_type(
                        local,
                        eqiora_schema::kernel::typing::ExpressionType::new(
                            value_type.clone(),
                            support.clone(),
                        ),
                    );
                }
                continue;
            }
            let field_type = field_expression_type(
                self.model.file,
                declaration,
                support,
                &scope.symbolic_parameters(),
            )?;
            scope.field_evolution.insert(
                declaration.name().to_owned(),
                (
                    declaration.role(),
                    super::super::scope::rewrite_activation(
                        self.model.file,
                        declaration.activation(),
                        declaration.range(),
                        scope,
                    )?,
                ),
            );
            if scope
                .insert_field_type(declaration.name().to_owned(), field_type)
                .is_some()
            {
                return Err(hierarchy_error(format!(
                    "duplicate flattened Field type `{}`",
                    declaration.name()
                )));
            }
        }
        for item in model.owned_items() {
            let Item::Port(declaration) = item else {
                continue;
            };
            if !matches!(declaration.syntax(), PortSyntax::ScalarPhysical { .. }) {
                continue;
            }
            let PortSyntax::ScalarPhysical { domain } = declaration.syntax() else {
                unreachable!("scalar-physical Port syntax was selected above");
            };
            let connector = scope.symbol(domain).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    self.model.file,
                    declaration.range(),
                    "resolved scalar-physical Port Domain has no flattened symbol",
                )
            })?;
            let identity = identities.entities[declaration.name()].clone();
            self.register_physical_port_occurrence(
                identity,
                declaration.name().to_owned(),
                self.root_path.clone(),
                false,
                Some(PhysicalExposureContractIdentity::ScalarPhysical {
                    connector: connector.full_identity,
                }),
            )?;
        }
        Ok(identities)
    }
}
