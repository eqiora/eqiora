//! External Geometry root allocation kept outside the expansion ceiling.

use super::*;
use crate::external::ExternalSupportBinding;

impl<'a, 'd> RootExpansion<'a, 'd> {
    pub(in crate::hierarchy) fn expand_external(
        mut self,
        component: ComponentDefinition<'d>,
        supports: &[ExternalSupportBinding],
        clocks: &[(String, eqiora_schema::kernel::ClockDomainDef)],
        properties: &BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>,
    ) -> Result<ExpandedBlueprint, Vec<Diagnostic>> {
        let model = self.model.clone();
        let mut root_scope = Scope::external_root();
        root_scope.extend_properties(properties);
        root_scope.lexical_namespace = Some(component.namespace.clone());
        root_scope.reduction_terms_limit = self.elaborator.limits.max_parameter_terms;
        root_scope.set_pure_operators(self.elaborator.visible_pure_operators(&model.namespace));
        self.allocate_external_clocks(&mut root_scope, clocks)
            .map_err(one_diagnostic)?;
        self.allocate_external_supports(&mut root_scope, supports)
            .map_err(one_diagnostic)?;
        let identities = self
            .allocate_model_scope(&mut root_scope)
            .map_err(one_diagnostic)?;
        let instance = model
            .items()
            .iter()
            .find_map(|item| match item {
                Item::Instance(instance) => Some(instance),
                _ => None,
            })
            .ok_or_else(|| vec![hierarchy_error("external root has no Component occurrence")])?;
        let instance_path = child_instance_path(
            &self.root_path,
            instance.name(),
            self.elaborator.limits.identity,
        )
        .map_err(one_diagnostic)?;
        let interface = self.expand_component(
            component,
            instance,
            model.file,
            instance_path,
            instance.name().to_owned(),
            &root_scope,
        )?;
        let ports = interface
            .public_ports
            .values()
            .map(|symbol| symbol.internal_name.clone())
            .collect();
        self.items.push(FlatItemBlueprint::Boundary {
            ports,
            range: model.range(),
        });
        self.materialize_model_items(&root_scope, &identities)
            .map_err(one_diagnostic)?;
        self.record_index_dependencies(&root_scope)
            .map_err(one_diagnostic)?;
        self.finalize_physical_connections()
            .map_err(one_diagnostic)?;
        self.items.sort_by_key(FlatItemBlueprint::sort_key);
        Ok(ExpandedBlueprint::new(
            self.model.name().to_owned(),
            SourceLocation::new(self.model.file, self.model.range()),
            (self.model_key, self.model_full),
            self.items,
            self.display_symbols,
            self.physical_exposures,
            self.notation_specs,
        )
        .with_structural_dependencies(self.structural_dependencies))
    }

    pub(super) fn allocate_external_supports(
        &mut self,
        scope: &mut Scope,
        supports: &[ExternalSupportBinding],
    ) -> Result<(), Diagnostic> {
        let mut singular = Vec::new();
        let mut member_slots = BTreeSet::new();
        for support in supports {
            if let ExternalSupportBinding::CompleteExterior {
                slot,
                geometry,
                parent_slot,
                members,
            } = support
            {
                for member in members {
                    if !member_slots.insert(exterior_member_slot(slot, &member.entity_set)) {
                        continue;
                    }
                    singular.push(ExternalSupportBinding::boundary(
                        exterior_member_slot(slot, &member.entity_set),
                        *geometry,
                        member.entity_set.clone(),
                        parent_slot.clone(),
                        member.embedding.clone(),
                    ));
                }
            } else {
                singular.push(support.clone());
            }
        }
        // Existing singular bindings allocate first so exact aliases keep their public identity.
        singular.sort_by_key(|support| support.slot().starts_with('@'));
        for support in &singular {
            let ExternalSupportBinding::CoordinateInterval { slot, bounds } = support else {
                continue;
            };
            let identity = self.external_support_identity(slot)?;
            let internal_name = internal_name(identity.full);
            self.register_symbol(slot.clone(), slot, &identity, SymbolKind::Domain, scope)?;
            scope.insert_spatial_support(
                slot.clone(),
                SpatialSupport::Coordinates {
                    domain: identity.full,
                    factors: vec![(identity.full, bounds.lower().dim())],
                },
            );
            self.items.push(FlatItemBlueprint::Domain {
                name: internal_name,
                contract: LoweringDomainContract::CoordinateInterval(*bounds),
                range: self.model.range(),
                identity,
            });
        }
        let mut regions = BTreeMap::new();
        for support in &singular {
            let ExternalSupportBinding::Region {
                slot,
                geometry,
                entity_set,
                ambient_dimension,
            } = support
            else {
                continue;
            };
            let key = (*geometry, entity_set.clone());
            if let Some((symbol, spatial)) = regions.get(&key) {
                self.alias_external_support(scope, slot, symbol, spatial)?;
                continue;
            }
            let identity = self.external_support_identity(slot)?;
            let internal_name = internal_name(identity.full);
            self.register_symbol(slot.clone(), slot, &identity, SymbolKind::Domain, scope)?;
            scope.insert_spatial_support(
                slot.clone(),
                SpatialSupport::Volume {
                    domain: identity.full,
                    dimensions: *ambient_dimension,
                },
            );
            regions.insert(
                key,
                (
                    scope.symbol(slot).expect("registered support").clone(),
                    scope
                        .spatial_support(slot)
                        .expect("registered support")
                        .clone(),
                ),
            );
            self.items.push(FlatItemBlueprint::Domain {
                name: internal_name,
                contract: LoweringDomainContract::ExternalGeometryRegion {
                    geometry: *geometry,
                    entity_set: entity_set.clone(),
                    dimensions: *ambient_dimension,
                },
                range: self.model.range(),
                identity,
            });
        }
        let mut boundaries = BTreeMap::new();
        for support in &singular {
            let ExternalSupportBinding::Boundary {
                slot,
                geometry,
                entity_set,
                parent_slot,
                embedding,
                ..
            } = support
            else {
                continue;
            };
            let Some(SpatialSupport::Volume {
                domain: parent,
                dimensions,
            }) = scope.spatial_support(parent_slot).cloned()
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.model.file,
                    self.model.range(),
                    format!(
                        "external boundary support `{slot}` has no volume parent `{parent_slot}`"
                    ),
                ));
            };
            let key = (*geometry, entity_set.clone(), parent);
            if let Some((symbol, spatial)) = boundaries.get(&key) {
                self.alias_external_support(scope, slot, symbol, spatial)?;
                continue;
            }
            let parent_name = scope
                .symbol(parent_slot)
                .ok_or_else(|| hierarchy_error("external volume support has no symbol"))?
                .internal_name
                .clone();
            let identity = self.external_support_identity(slot)?;
            let internal_name = internal_name(identity.full);
            self.register_symbol(slot.clone(), slot, &identity, SymbolKind::Domain, scope)?;
            scope.insert_spatial_support(
                slot.clone(),
                SpatialSupport::Boundary {
                    domain: identity.full,
                    parent,
                    dimensions,
                },
            );
            boundaries.insert(
                key,
                (
                    scope.symbol(slot).expect("registered support").clone(),
                    scope
                        .spatial_support(slot)
                        .expect("registered support")
                        .clone(),
                ),
            );
            self.boundary_parents.insert(identity.full, parent);
            // Exact external Geometry owns the boundary metric and orientation;
            // primitive embeddings also feed the existing complete-exterior proof.
            // Other geometry remains subject to Geometry-aware semantic lowering.
            self.boundary_embeddings
                .insert(identity.full, embedding.clone());
            if let Some(embedding) = embedding {
                self.boundary_sides
                    .insert(identity.full, (embedding.normal_axis(), embedding.side()));
            }
            self.items.push(FlatItemBlueprint::Domain {
                name: internal_name,
                contract: LoweringDomainContract::ExternalGeometryBoundary {
                    entity_set: entity_set.clone(),
                    parent: parent_name,
                },
                range: self.model.range(),
                identity,
            });
        }
        for support in supports {
            let ExternalSupportBinding::CompleteExterior {
                slot,
                parent_slot,
                members,
                ..
            } = support
            else {
                continue;
            };
            let Some(SpatialSupport::Volume {
                domain: parent,
                dimensions,
            }) = scope.spatial_support(parent_slot)
            else {
                return Err(hierarchy_error(
                    "complete-exterior binding has no exact volume parent",
                ));
            };
            let resolved = members
                .iter()
                .map(|member| {
                    let name = exterior_member_slot(slot, &member.entity_set);
                    let symbol = scope
                        .symbol(&name)
                        .expect("allocated exact exterior member");
                    (symbol.internal_name.clone(), symbol.full_identity)
                })
                .collect();
            let set = super::super::supports::external_complete_exterior_set(
                self.model.file,
                slot,
                *parent,
                *dimensions,
                resolved,
                |identity| {
                    let SpatialSupport::Boundary {
                        parent, dimensions, ..
                    } = scope.spatial_support_by_identity(*identity)?
                    else {
                        return None;
                    };
                    let (axis, side) = self.boundary_sides.get(identity)?;
                    Some(CartesianDomain::Boundary {
                        exact_parent: *parent,
                        ambient_dimension: *dimensions,
                        axis: *axis,
                        side: *side,
                    })
                },
                &mut self.complete_exterior_memberships,
            )?;
            scope.insert_boundary_set(slot.clone(), set);
        }
        Ok(())
    }

    fn alias_external_support(
        &mut self,
        scope: &mut Scope,
        slot: &str,
        symbol: &FlatSymbol,
        spatial: &SpatialSupport<FullElaborationIdentity>,
    ) -> Result<(), Diagnostic> {
        if scope
            .insert_symbol(slot.to_owned(), symbol.clone())
            .is_some()
            || self
                .display_symbols
                .insert(
                    slot.to_owned(),
                    DisplayIdentity {
                        full: symbol.full_identity,
                        kind: EntityKind::Domain,
                    },
                )
                .is_some()
        {
            return Err(hierarchy_error("duplicate selected support binding name"));
        }
        scope.insert_spatial_support(slot.to_owned(), spatial.clone());
        Ok(())
    }

    fn external_support_identity(&self, slot: &str) -> Result<EntityIdentity, Diagnostic> {
        self.entity_identity(
            &self.root_path,
            definition_path(&self.model.namespace, "model", self.model.name(), slot),
            EntityKind::Domain,
            SourceLocation::new(self.model.file, self.model.range()),
            SourceLocation::new(self.model.file, self.model.range()),
            Vec::new(),
        )
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_external_clocks(
        &mut self,
        scope: &mut Scope,
        clocks: &[(String, eqiora_schema::kernel::ClockDomainDef)],
    ) -> Result<(), Diagnostic> {
        let mut seen =
            BTreeMap::<ulid::Ulid, (eqiora_schema::kernel::ClockDomainDef, FlatSymbol)>::new();
        for (slot, clock) in clocks {
            if let Some((previous, symbol)) = seen.get(&clock.id().ulid()) {
                if previous != clock {
                    return Err(hierarchy_error(
                        "one supplied clock identity has conflicting definitions",
                    ));
                }
                if scope.insert_symbol(slot.clone(), symbol.clone()).is_some()
                    || self
                        .display_symbols
                        .insert(
                            slot.clone(),
                            DisplayIdentity {
                                full: symbol.full_identity,
                                kind: EntityKind::ClockDomain,
                            },
                        )
                        .is_some()
                {
                    return Err(hierarchy_error("duplicate selected clock binding name"));
                }
                continue;
            }
            let eqiora_schema::kernel::ClockKind::Periodic { period, phase } = clock.kind() else {
                return Err(hierarchy_error("clock requirement needs periodic clock"));
            };
            let identity = self.entity_identity(
                &self.root_path,
                definition_path(&self.model.namespace, "model", self.model.name(), slot),
                EntityKind::ClockDomain,
                SourceLocation::new(self.model.file, self.model.range()),
                SourceLocation::new(self.model.file, self.model.range()),
                Vec::new(),
            )?;
            self.register_symbol(
                slot.clone(),
                slot,
                &identity,
                SymbolKind::Clock(period),
                scope,
            )?;
            let symbol = scope.symbol(slot).expect("just registered clock").clone();
            seen.insert(clock.id().ulid(), (clock.clone(), symbol));
            self.items.push(FlatItemBlueprint::Clock {
                supplied_id: Some(clock.id()),
                name: internal_name(identity.full),
                period: time_expression(period)?,
                phase: time_expression(phase)?,
                range: self.model.range(),
                identity,
            });
        }
        Ok(())
    }
}

fn time_expression(
    value: eqiora_schema::kernel::RationalTime,
) -> Result<eqiora_lang::Expr, Diagnostic> {
    use eqiora_lang::{BinaryOp, DecimalLiteral, ExprKind, SourceAstFactory as F, TextRange};
    let range = TextRange::default();
    let leaf = |integer: u64, unit: &str| {
        F::expression(
            ExprKind::Quantity {
                value: DecimalLiteral::parse(&integer.to_string()).expect("u64 decimal"),
                unit: Box::new(
                    F::expression(
                        if unit == "1" {
                            ExprKind::Number(
                                eqiora_lang::DecimalLiteral::parse("1.0").expect("exact literal"),
                            )
                        } else {
                            ExprKind::Name(unit.to_owned())
                        },
                        range,
                    )
                    .expect("canonical unit"),
                ),
            },
            range,
        )
    };
    F::expression(
        ExprKind::Binary {
            op: BinaryOp::Div,
            left: Box::new(
                leaf(value.numerator(), "s").map_err(|error| hierarchy_error(error.message()))?,
            ),
            right: Box::new(
                leaf(value.denominator(), "1").map_err(|error| hierarchy_error(error.message()))?,
            ),
        },
        range,
    )
    .map_err(|error| hierarchy_error(error.message()))
}

fn exterior_member_slot(slot: &str, entity_set: &str) -> String {
    format!("@exterior/{slot}/{entity_set}")
}
