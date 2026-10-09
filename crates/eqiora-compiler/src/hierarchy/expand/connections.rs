//! Resolve ordinary and indexed connection fragments through the shared owner.
use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn add_connection(
        &mut self,
        declaration: &ConnectionDecl,
        scope: &Scope,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        origin: ConnectionOrigin,
    ) -> Result<(), Diagnostic> {
        if let Some(binder) = declaration.binder() {
            let set = scope.index_set(binder.set().as_str()).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    &origin.definition_file,
                    binder.range(),
                    "Connection family requires a resolved exact IndexSet",
                )
            })?;
            for ordinal in 0..set.extent() {
                let member = scope.with_index_member(binder.member(), set, ordinal)?;
                let ports = declaration
                    .port_expressions()
                    .iter()
                    .map(|expression| member.endpoint(&origin.definition_file, expression))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut path = declaration_path.clone();
                path.extend([
                    "index_member".to_owned(),
                    set.id().to_string(),
                    ordinal.to_string(),
                ]);
                self.add_resolved_connection(
                    declaration.syntax(),
                    ports,
                    declaration.range(),
                    instance_path,
                    path,
                    ConnectionOrigin {
                        definition_file: origin.definition_file.clone(),
                        instance: origin.instance.clone(),
                        bindings: origin.bindings.clone(),
                    },
                )?;
            }
            return Ok(());
        }
        let ports = declaration
            .port_expressions()
            .iter()
            .map(|expression| scope.endpoint(&origin.definition_file, expression))
            .collect::<Result<Vec<_>, _>>()?;
        self.add_resolved_connection(
            declaration.syntax(),
            ports,
            declaration.range(),
            instance_path,
            declaration_path,
            origin,
        )
    }

    pub(super) fn add_boundary_connection(
        &mut self,
        declaration: &BoundaryConnectionDecl,
        scope: &Scope,
        active: Option<ActiveBoundaryMember<'_>>,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        origin: ConnectionOrigin,
    ) -> Result<(), Diagnostic> {
        let ports = declaration
            .ports()
            .iter()
            .map(|reference| {
                resolve_boundary_port_reference(&origin.definition_file, reference, scope, active)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ports.len() < 2 {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                &origin.definition_file,
                declaration.range(),
                "Connection requires at least two visible Ports",
            ));
        }
        self.add_resolved_connection(
            declaration.syntax(),
            ports,
            declaration.range(),
            instance_path,
            declaration_path,
            origin,
        )
    }

    pub(super) fn add_resolved_connection(
        &mut self,
        syntax: ConnectionSyntax,
        mut ports: Vec<&FlatSymbol>,
        range: eqiora_lang::TextRange,
        instance_path: &InstancePath,
        declaration_path: Vec<String>,
        origin: ConnectionOrigin,
    ) -> Result<(), Diagnostic> {
        if ports.len() < 2 {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                &origin.definition_file,
                range,
                "Connection requires at least two visible Ports",
            ));
        }
        match syntax {
            ConnectionSyntax::Conserving | ConnectionSyntax::SpatialPeriodic => {
                ports.sort_unstable_by_key(|port| port.full_identity);
            }
            ConnectionSyntax::Signal => {
                if let Some((_, inputs)) = ports.split_first_mut() {
                    inputs.sort_unstable_by_key(|port| port.full_identity);
                }
            }
        }
        let path_display = declaration_path.join("/");
        let path = DeclarationPath::with_limits(declaration_path, self.elaborator.limits.identity)
            .map_err(|diagnostic| {
                source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    &origin.definition_file,
                    range,
                    format!(
                        "cannot identify Connection at declaration path `{path_display}`: {}",
                        diagnostic.message()
                    ),
                )
            })?;
        let source = EntitySourceOrigin {
            definition: SourceLocation::new(&origin.definition_file, range),
            instance: origin.instance,
            bindings: origin.bindings,
        };
        if matches!(
            syntax,
            ConnectionSyntax::Conserving | ConnectionSyntax::SpatialPeriodic
        ) && ports
            .first()
            .is_some_and(|port| self.physical_ports.contains_key(&port.full_identity))
        {
            if let Some(non_physical) = ports
                .iter()
                .find(|port| !self.physical_ports.contains_key(&port.full_identity))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    &origin.definition_file,
                    range,
                    format!(
                        "physical Connection cannot include non-physical Port `{}`",
                        non_physical.display_name
                    ),
                ));
            }
            if syntax == ConnectionSyntax::SpatialPeriodic {
                if ports.len() != 2 {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        &origin.definition_file,
                        range,
                        "spatial-periodic Connection requires exactly two field-physical Ports",
                    ));
                }
                if ports.iter().any(|port| {
                    self.spatial_periodic_ports.contains(&port.full_identity)
                        || self.physical_connections.iter().any(|fragment| {
                            fragment.topology.members().contains(&port.full_identity)
                        })
                }) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        &origin.definition_file,
                        range,
                        "spatial-periodic Port already belongs to another physical Connection",
                    ));
                }
                self.spatial_periodic_ports
                    .extend(ports.iter().map(|port| port.full_identity));
                let key = ElaborationKey::anonymous_connection_with_limits(
                    self.namespace.clone(),
                    instance_path.clone(),
                    path,
                    ports.iter().map(|port| port.full_identity),
                    self.elaborator.limits.identity,
                )?;
                let full = key.full_identity()?;
                self.items.push(FlatItemBlueprint::Connection {
                    syntax,
                    ports: ports
                        .into_iter()
                        .map(|port| port.internal_name.clone())
                        .collect(),
                    range,
                    identity: ConnectionIdentity {
                        key,
                        full,
                        origins: vec![source],
                    },
                });
                return Ok(());
            }
            let topology = ConnectionFragment::try_new(
                ports.iter().map(|port| port.full_identity),
                self.elaborator.limits.connection_sets,
            )
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    &origin.definition_file,
                    range,
                    format!("cannot stage physical Connection fragment: {error}"),
                )
            })?;
            self.physical_connections.push(StagedPhysicalConnection {
                topology,
                origin: PhysicalConnectionOrigin {
                    declaration_path: path,
                    instance_path: instance_path.clone(),
                    source,
                },
            });
            return Ok(());
        }
        let key = ElaborationKey::anonymous_connection_with_limits(
            self.namespace.clone(),
            instance_path.clone(),
            path,
            ports.iter().map(|port| port.full_identity),
            self.elaborator.limits.identity,
        )?;
        let full = key.full_identity()?;
        self.items.push(FlatItemBlueprint::Connection {
            syntax,
            ports: ports
                .into_iter()
                .map(|port| port.internal_name.clone())
                .collect(),
            range,
            identity: ConnectionIdentity {
                key,
                full,
                origins: vec![source],
            },
        });
        Ok(())
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn record_physical_relation_owners(
        &mut self,
        file: &str,
        range: eqiora_lang::TextRange,
        relation: FullElaborationIdentity,
        body: &crate::lower::LoweringRelationBody,
    ) -> Result<(), Diagnostic> {
        let mut names = BTreeSet::new();
        if body
            .expressions()
            .any(|expression| !expression.collect_physical_port_names(&mut names))
        {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                range,
                "Relation expression is newer than physical ownership analysis",
            ));
        }
        let mut selected = BTreeSet::new();
        for name in names {
            if let Some(port) = self.physical_ports_by_name.get(&name) {
                selected.insert(*port);
            }
        }
        for port in selected {
            let owners = self.physical_owner_relations.entry(port).or_default();
            owners.insert(relation);
            if owners.len() > 1 {
                let display = &self.physical_ports[&port].display_name;
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    format!("physical Port `{display}` cannot have more than one owning Relation"),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn finalize_physical_connections(&mut self) -> Result<(), Diagnostic> {
        if self.physical_ports.is_empty() && self.physical_connections.is_empty() {
            return Ok(());
        }
        for fragment in &self.physical_connections {
            if fragment
                .topology
                .members()
                .iter()
                .any(|member| self.spatial_periodic_ports.contains(member))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.model.file,
                    self.model.range(),
                    "one field-physical Port cannot belong to both ordinary and spatial-periodic Connections",
                ));
            }
            let boundary_members = fragment
                .topology
                .members()
                .iter()
                .filter_map(|identity| {
                    self.physical_ports.get(identity).and_then(|port| {
                        let Some(PhysicalExposureContractIdentity::FieldBoundary {
                            connector,
                            boundary,
                        }) = port.contract
                        else {
                            return None;
                        };
                        Some((*identity, connector, boundary))
                    })
                })
                .collect::<Vec<_>>();
            if boundary_members.is_empty() {
                continue;
            }
            if boundary_members.len() != fragment.topology.members().len() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.model.file,
                    self.model.range(),
                    "conserving Connection cannot mix scalar and field-physical Ports",
                ));
            }
            let mut contracts = Vec::with_capacity(boundary_members.len());
            let mut metric_validation_deferred = false;
            for (_, connector, boundary) in &boundary_members {
                let embedding = self.boundary_embeddings.get(boundary).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.model.file,
                        self.model.range(),
                        "field-physical Port boundary has no Cartesian embedding recipe",
                    )
                })?;
                let parent = *self.boundary_parents.get(boundary).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.model.file,
                        self.model.range(),
                        "field-physical Port boundary has no exact parent identity",
                    )
                })?;
                if let Some(embedding) = embedding {
                    contracts.push(BoundaryPhysicalPortContract {
                        connector: *connector,
                        boundary: *boundary,
                        parent,
                        embedding: embedding.clone(),
                    });
                } else {
                    metric_validation_deferred = true;
                }
            }
            if !metric_validation_deferred {
                validate_boundary_physical_connection(&contracts).map_err(|violation| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.model.file,
                        self.model.range(),
                        format!(
                            "field-physical Connection is incompatible before topology normalization: {violation:?}"
                        ),
                    )
                })?;
            }
        }
        let endpoints = self
            .physical_ports
            .iter()
            .filter(|(identity, _)| !self.spatial_periodic_ports.contains(identity))
            .map(|(identity, occurrence)| {
                OccurrencePhysicalEndpoint::new(
                    *identity,
                    occurrence.exposure_candidate,
                    self.physical_owner_relations.contains_key(identity),
                )
            })
            .collect::<Vec<_>>();
        let fragments = self
            .physical_connections
            .iter()
            .map(|fragment| {
                OccurrenceConnectionFragment::new(
                    fragment.topology.clone(),
                    fragment.origin.instance_path.clone(),
                )
            })
            .collect::<Vec<_>>();
        let normalized = normalize_occurrence_connections(
            &endpoints,
            &fragments,
            self.elaborator.limits.connection_sets,
        )
        .map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.model.file,
                self.model.range(),
                format!("invalid occurrence-level physical connection closure: {error}"),
            )
        })?;

        let projection_count = normalized
            .sets()
            .iter()
            .try_fold(0_usize, |count, set| {
                count.checked_add(set.topology().eliminated_exposures().len())
            })
            .ok_or_else(|| hierarchy_error("physical exposure projection count overflows usize"))?;
        let projection_limits = self.elaborator.limits.physical_exposures;
        if projection_count > projection_limits.max_projections {
            return Err(hierarchy_error(format!(
                "physical exposure projections total {projection_count}, exceeding the {} limit",
                projection_limits.max_projections
            )));
        }
        self.physical_exposures
            .try_reserve_exact(projection_count)
            .map_err(|_| hierarchy_error("cannot reserve physical exposure projections"))?;
        let mut cut_graph = ExposureCutIndex::new(
            self.physical_ports.keys().copied(),
            self.physical_ports.len(),
            &fragments,
        )?;
        let mut projection_memberships = 0_usize;

        let mut exposure_connections = BTreeMap::new();
        for (set_index, set) in normalized.sets().iter().enumerate() {
            let topology = set.topology();
            let owner_fragment = set
                .witness()
                .lca_owner_candidate_fragment_indices()
                .iter()
                .copied()
                .min_by(|left, right| {
                    compare_physical_connection_origins(
                        &self.physical_connections[*left].origin,
                        &self.physical_connections[*right].origin,
                    )
                })
                .expect("occurrence normalization proves one explicit LCA fragment");
            let owner = &self.physical_connections[owner_fragment].origin;
            debug_assert_eq!(owner.instance_path, *topology.owner_instance_path());
            let key = ElaborationKey::anonymous_connection_with_limits(
                self.namespace.clone(),
                topology.owner_instance_path().clone(),
                owner.declaration_path.clone(),
                topology.retained_members().iter().copied(),
                self.elaborator.limits.identity,
            )?;
            let full = key.full_identity()?;
            let origins = set
                .witness()
                .contributing_fragment_indices()
                .iter()
                .map(|index| self.physical_connections[*index].origin.source.clone())
                .collect::<Vec<_>>();
            self.items.push(FlatItemBlueprint::Connection {
                syntax: ConnectionSyntax::Conserving,
                ports: topology
                    .retained_members()
                    .iter()
                    .map(|member| self.physical_ports[member].identity.full)
                    .map(internal_name)
                    .collect(),
                range: owner.source.definition.range,
                identity: ConnectionIdentity { key, full, origins },
            });
            for exposure in topology.eliminated_exposures() {
                if exposure_connections.insert(*exposure, full).is_some() {
                    return Err(hierarchy_error(format!(
                        "physical exposure {exposure} projects to more than one canonical Connection"
                    )));
                }
                let occurrence = &self.physical_ports[exposure];
                let interior = cut_graph.derive(
                    *exposure,
                    &occurrence.instance_path,
                    topology.retained_members(),
                    &fragments,
                    projection_limits.max_traversal_memberships,
                )?;
                if interior.is_empty() || interior.len() == topology.retained_members().len() {
                    return Err(hierarchy_error(format!(
                        "physical exposure `{}` does not define a nonempty proper occurrence cut",
                        occurrence.display_name
                    )));
                }
                if interior.len() > projection_limits.max_members_per_cut {
                    return Err(hierarchy_error(format!(
                        "physical exposure `{}` cut has {} members, exceeding the {} limit",
                        occurrence.display_name,
                        interior.len(),
                        projection_limits.max_members_per_cut
                    )));
                }
                projection_memberships = projection_memberships
                    .checked_add(interior.len())
                    .ok_or_else(|| {
                        hierarchy_error("physical exposure cut membership count overflows usize")
                    })?;
                if projection_memberships > projection_limits.max_memberships {
                    return Err(hierarchy_error(format!(
                        "physical exposure cuts total {projection_memberships} memberships, exceeding the {} limit",
                        projection_limits.max_memberships
                    )));
                }
                let contract = occurrence.contract.ok_or_else(|| {
                    hierarchy_error(format!(
                        "physical exposure `{}` has no closed nominal contract",
                        occurrence.display_name
                    ))
                })?;
                self.physical_exposures
                    .push(PhysicalExposureProjectionBlueprint {
                        selector: occurrence.display_name.clone(),
                        exposure: occurrence.identity.clone(),
                        connection: full,
                        interior,
                        contract,
                    });
            }
            debug_assert!(
                normalized
                    .exposure_witnesses()
                    .iter()
                    .filter(|witness| witness.connection_set_index() == set_index)
                    .all(|witness| exposure_connections.contains_key(&witness.exposure()))
            );
        }

        self.items.retain(|item| match item {
            FlatItemBlueprint::Port { identity, .. } => {
                !exposure_connections.contains_key(&identity.full)
            }
            _ => true,
        });
        for exposure in exposure_connections.into_keys() {
            let occurrence = &self.physical_ports[&exposure];
            let removed = self.display_symbols.remove(&occurrence.display_name);
            if !matches!(removed, Some(identity) if identity.full == exposure) {
                return Err(hierarchy_error(format!(
                    "physical exposure `{}` has no exact display-symbol entry",
                    occurrence.display_name
                )));
            }
        }
        self.physical_exposures.sort_unstable_by(|left, right| {
            left.selector
                .cmp(&right.selector)
                .then_with(|| left.exposure.full.cmp(&right.exposure.full))
        });
        Ok(())
    }
}
