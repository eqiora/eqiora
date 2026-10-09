//! Nominal product occurrences reuse the existing typed leaf allocation owners.
use super::super::flat::DisplayIdentity;
use super::*;
use crate::lower::LoweringExpression;

pub(super) struct RecordFieldOccurrence<'a> {
    pub(super) namespace: &'a DefinitionNamespace,
    pub(super) definition_name: &'a str,
    pub(super) instance_path: &'a InstancePath,
    pub(super) display_prefix: &'a str,
    pub(super) file: &'a str,
    pub(super) instance: SourceLocation,
    pub(super) bindings: Vec<SourceLocation>,
}

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_record_definitions(&mut self) -> Result<(), Diagnostic> {
        for (namespace, records) in &self.elaborator.records {
            for (name, value) in records {
                let full = value.key.full_identity()?;
                let identity = EntityIdentity {
                    key: value.key.clone(),
                    full,
                    definition: SourceLocation::new(&value.file, value.range),
                    instance: SourceLocation::new(self.model.file, self.model.range()),
                    bindings: Vec::new(),
                };
                let display = if matches!(namespace, DefinitionNamespace::Local) {
                    name.clone()
                } else {
                    format!("{namespace}.{name}")
                };
                self.display_symbols.insert(
                    display,
                    DisplayIdentity {
                        full,
                        kind: EntityKind::Record,
                    },
                );
                self.items.push(FlatItemBlueprint::Nominal {
                    name: internal_name(full),
                    definition: value.definition.clone().into(),
                    identity,
                });
            }
        }
        for (name, value) in self.elaborator.visible_records(&self.model.namespace) {
            self.display_symbols.insert(
                name,
                DisplayIdentity {
                    full: value.key.full_identity()?,
                    kind: EntityKind::Record,
                },
            );
        }
        Ok(())
    }

    pub(super) fn allocate_record_field(
        &mut self,
        scope: &mut Scope,
        declaration: &eqiora_lang::FieldDecl,
        occurrence: RecordFieldOccurrence<'_>,
        identities: &mut ScopeIdentities,
    ) -> Result<bool, Diagnostic> {
        let RecordFieldOccurrence {
            namespace,
            definition_name,
            instance_path,
            display_prefix,
            file,
            instance: instance_source,
            bindings,
        } = occurrence;
        let Some(record) = self
            .elaborator
            .record_for_type(namespace, declaration.value_type())
            .cloned()
        else {
            return Ok(false);
        };
        let mut path = definition_path(namespace, "record", definition_name, declaration.name());
        let instance = self.entity_identity(
            instance_path,
            path.clone(),
            EntityKind::RecordInstance,
            SourceLocation::new(file, declaration.range()),
            instance_source.clone(),
            bindings.clone(),
        )?;
        let instance_display = if display_prefix.is_empty() {
            declaration.name().to_owned()
        } else {
            format!("{display_prefix}.{}", declaration.name())
        };
        self.display_symbols.insert(
            instance_display.clone(),
            DisplayIdentity {
                full: instance.full,
                kind: EntityKind::RecordInstance,
            },
        );
        let mut members = Vec::new();
        for ((name, _), syntax) in record
            .definition
            .members()
            .iter()
            .zip(&record.member_syntax)
        {
            path.push(name.clone());
            let identity = self.entity_identity(
                instance_path,
                path.clone(),
                EntityKind::Field,
                SourceLocation::new(&record.file, syntax.range()),
                instance_source.clone(),
                bindings.clone(),
            )?;
            path.pop();
            let local = format!("{}.{name}", declaration.name());
            self.register_symbol(
                format!("{instance_display}.{name}"),
                &local,
                &identity,
                SymbolKind::Field,
                scope,
            )?;
            members.push(LoweringExpression::name(
                internal_name(identity.full),
                declaration.range(),
            ));
            identities.entities.insert(local, identity);
        }
        self.items.push(FlatItemBlueprint::RecordInstance {
            name: internal_name(instance.full),
            definition: record.definition.id(),
            members,
            identity: instance,
        });
        Ok(true)
    }
    pub(super) fn emit_record_field(
        &mut self,
        scope: &Scope,
        declaration: &eqiora_lang::FieldDecl,
        record: &crate::record::BoundRecord,
        identities: &ScopeIdentities,
        file: &str,
    ) -> Result<(), Diagnostic> {
        let (domain, activation) = rewrite_field_scope(file, declaration, scope)?;
        for ((name, _), syntax) in record
            .definition
            .members()
            .iter()
            .zip(&record.member_syntax)
        {
            let identity = identities.entities[&format!("{}.{name}", declaration.name())].clone();
            let representation = self.add_support_representation(domain.as_deref())?;
            self.items.push(FlatItemBlueprint::Field {
                name: internal_name(identity.full),
                domain: domain.clone(),
                representation,
                value_type: syntax.clone(),
                role: declaration.role(),
                spatial_regularity: declaration.spatial_regularity(),
                activation: activation.clone(),
                range: declaration.range(),
                identity,
            });
        }
        Ok(())
    }
}
