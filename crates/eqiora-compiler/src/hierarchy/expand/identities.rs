//! Exact entity and relation identity construction for occurrence expansion.
use super::*;
use eqiora_core::GraphPath;

pub(super) fn one_diagnostic(error: Diagnostic) -> Vec<Diagnostic> {
    vec![error]
}

pub(super) fn contextualize_diagnostic(
    error: Diagnostic,
    instance_path: &InstancePath,
) -> Diagnostic {
    error.with_graph_path(GraphPath::new(instance_path.segments().iter().cloned()))
}

pub(super) fn contextualize_diagnostics(
    errors: Vec<Diagnostic>,
    instance_path: &InstancePath,
) -> Vec<Diagnostic> {
    errors
        .into_iter()
        .map(|error| contextualize_diagnostic(error, instance_path))
        .collect()
}

impl RootExpansion<'_, '_> {
    pub(super) fn register_port_family_member(
        &mut self,
        registration: PortFamilyMemberRegistration<'_>,
        scope: &mut Scope,
    ) -> Result<(), Diagnostic> {
        self.record_notation(
            &registration.display_name,
            registration.identity,
            &SymbolKind::Port {
                activation: eqiora_lang::ActivationSyntax::Continuous,
                quantities: Some(registration.quantities.clone()),
            },
        );
        let symbol = FlatSymbol {
            internal_name: internal_name(registration.identity.full),
            display_name: registration.display_name.clone(),
            full_identity: registration.identity.full,
            kind: SymbolKind::Port {
                activation: eqiora_lang::ActivationSyntax::Continuous,
                quantities: Some(registration.quantities),
            },
        };
        scope.insert_port_family_member(
            registration.file,
            registration.range,
            registration.family_name.to_owned(),
            registration.selector_member,
            registration.boundary,
            symbol.clone(),
        )?;
        if self
            .display_symbols
            .insert(
                registration.display_name,
                DisplayIdentity {
                    full: registration.identity.full,
                    kind: EntityKind::Port,
                },
            )
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened display symbol `{}`",
                symbol.display_name
            )));
        }
        Ok(())
    }

    pub(super) fn register_family_relation_display(
        &mut self,
        display_name: String,
        identity: &RelationIdentity,
    ) -> Result<(), Diagnostic> {
        if self
            .display_symbols
            .insert(
                display_name.clone(),
                DisplayIdentity {
                    full: identity.entity.full,
                    kind: EntityKind::Relation,
                },
            )
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened display symbol `{display_name}`"
            )));
        }
        Ok(())
    }

    pub(super) fn register_symbol(
        &mut self,
        display_name: String,
        local_name: &str,
        identity: &EntityIdentity,
        kind: SymbolKind,
        scope: &mut Scope,
    ) -> Result<(), Diagnostic> {
        self.record_notation(&display_name, identity, &kind);
        let internal_name = internal_name(identity.full);
        if matches!(kind, SymbolKind::Domain) {
            let key = identity.key.support_representation()?;
            let representation = EntityIdentity {
                full: key.full_identity()?,
                key,
                definition: identity.definition.clone(),
                instance: identity.instance.clone(),
                bindings: identity.bindings.clone(),
            };
            self.support_representations
                .insert(internal_name.clone(), (representation, false));
        }
        let symbol = FlatSymbol {
            internal_name,
            display_name: display_name.clone(),
            full_identity: identity.full,
            kind,
        };
        if scope
            .insert_symbol(local_name.to_owned(), symbol.clone())
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened scope symbol `{local_name}`"
            )));
        }
        if self
            .display_symbols
            .insert(
                display_name,
                DisplayIdentity {
                    full: identity.full,
                    kind: identity.key.entity_kind(),
                },
            )
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened display symbol `{}`",
                symbol.display_name
            )));
        }
        Ok(())
    }

    pub(super) fn entity_identity(
        &self,
        instance_path: &InstancePath,
        declaration_path: impl IntoIterator<Item = String>,
        kind: EntityKind,
        definition: SourceLocation,
        instance: SourceLocation,
        bindings: Vec<SourceLocation>,
    ) -> Result<EntityIdentity, Diagnostic> {
        let declaration_path =
            DeclarationPath::with_limits(declaration_path, self.elaborator.limits.identity)?;
        let key = ElaborationKey::entity_with_limits(
            self.namespace.clone(),
            instance_path.clone(),
            declaration_path,
            kind,
            self.elaborator.limits.identity,
        )?;
        let full = key.full_identity()?;
        Ok(EntityIdentity {
            key,
            full,
            definition,
            instance,
            bindings,
        })
    }

    pub(super) fn relation_identity(
        &self,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        definition: SourceLocation,
        instance: SourceLocation,
        bindings: Vec<SourceLocation>,
    ) -> Result<RelationIdentity, Diagnostic> {
        let entity = self.entity_identity(
            instance_path,
            declaration_path.clone(),
            EntityKind::Relation,
            definition,
            instance,
            bindings,
        )?;
        let declaration_path =
            DeclarationPath::with_limits(declaration_path, self.elaborator.limits.identity)?;
        let activation_key = ElaborationKey::generated_with_limits(
            self.namespace.clone(),
            instance_path.clone(),
            declaration_path,
            GeneratedRole::RelationActivation,
            self.elaborator.limits.identity,
        )?;
        let activation_full = activation_key.full_identity()?;
        Ok(RelationIdentity {
            entity,
            activation_key,
            activation_full,
        })
    }

    pub(super) fn boundary_family_entity_identity(
        &self,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        kind: EntityKind,
        boundary: FullElaborationIdentity,
        source: EntitySourceOrigin,
    ) -> Result<EntityIdentity, Diagnostic> {
        let declaration_path =
            DeclarationPath::with_limits(declaration_path, self.elaborator.limits.identity)?;
        let key = ElaborationKey::boundary_family_entity_with_limits(
            self.namespace.clone(),
            instance_path.clone(),
            declaration_path,
            kind,
            boundary,
            self.elaborator.limits.identity,
        )?;
        let full = key.full_identity()?;
        Ok(EntityIdentity {
            key,
            full,
            definition: source.definition,
            instance: source.instance,
            bindings: source.bindings,
        })
    }

    pub(super) fn boundary_family_relation_identity(
        &self,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        boundary: FullElaborationIdentity,
        source: EntitySourceOrigin,
    ) -> Result<RelationIdentity, Diagnostic> {
        let entity = self.boundary_family_entity_identity(
            instance_path,
            declaration_path.clone(),
            EntityKind::Relation,
            boundary,
            source,
        )?;
        let declaration_path =
            DeclarationPath::with_limits(declaration_path, self.elaborator.limits.identity)?;
        let activation_key = ElaborationKey::boundary_family_generated_with_limits(
            self.namespace.clone(),
            instance_path.clone(),
            declaration_path,
            GeneratedRole::RelationActivation,
            boundary,
            self.elaborator.limits.identity,
        )?;
        let activation_full = activation_key.full_identity()?;
        Ok(RelationIdentity {
            entity,
            activation_key,
            activation_full,
        })
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn register_physical_port_occurrence(
        &mut self,
        identity: EntityIdentity,
        display_name: String,
        instance_path: InstancePath,
        exposure_candidate: bool,
        contract: Option<PhysicalExposureContractIdentity>,
    ) -> Result<(), Diagnostic> {
        let full = identity.full;
        let internal_name = internal_name(full);
        if self
            .physical_ports_by_name
            .insert(internal_name, full)
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened physical Port identity {full}"
            )));
        }
        if self
            .physical_ports
            .insert(
                full,
                PhysicalPortOccurrence {
                    identity,
                    display_name,
                    instance_path,
                    exposure_candidate,
                    contract,
                },
            )
            .is_some()
        {
            return Err(hierarchy_error(format!(
                "duplicate flattened physical Port occurrence {full}"
            )));
        }
        Ok(())
    }
}
