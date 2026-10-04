use std::collections::{BTreeMap, VecDeque};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, EntityKind, Id, OntologyId, RawId, Span};
use eqiora_lang::{ActivationSyntax, ConnectionSyntax, TextRange};
use eqiora_schema::Model;

use crate::identity::{
    ElaborationKey, FullElaborationIdentity, ModelViewKey, StagedIdentities, StagingIdAllocator,
};
use crate::lower::{
    CompiledModel, LoweringDomainContract, LoweringIdentities, LoweringItem, LoweringModel,
    LoweringPortContract, ModelSymbols,
};
use crate::projection::{
    PhysicalExposureContract, PhysicalExposureProjection, PhysicalExposureProjectionMap,
};
use crate::provenance::{ElaborationSourceOrigin, ProvenanceBuilder, ProvenanceLimits};

use super::{HierarchyLimits, hierarchy_error};

#[derive(Debug, Clone)]
pub(super) struct SourceLocation {
    pub(super) file: String,
    pub(super) range: TextRange,
}

impl SourceLocation {
    pub(super) fn new(file: &str, range: TextRange) -> Self {
        Self {
            file: file.to_owned(),
            range,
        }
    }

    pub(super) fn span(&self) -> Span {
        span(&self.file, self.range)
    }
}

#[derive(Debug, Clone)]
pub(super) struct EntityIdentity {
    pub(super) key: ElaborationKey,
    pub(super) full: FullElaborationIdentity,
    pub(super) definition: SourceLocation,
    pub(super) instance: SourceLocation,
    pub(super) bindings: Vec<SourceLocation>,
}

#[derive(Debug, Clone)]
pub(super) struct EntitySourceOrigin {
    pub(super) definition: SourceLocation,
    pub(super) instance: SourceLocation,
    pub(super) bindings: Vec<SourceLocation>,
}

#[derive(Debug, Clone)]
pub(super) struct ConnectionIdentity {
    pub(super) key: ElaborationKey,
    pub(super) full: FullElaborationIdentity,
    pub(super) origins: Vec<EntitySourceOrigin>,
}

#[derive(Debug, Clone)]
pub(super) struct RelationIdentity {
    pub(super) entity: EntityIdentity,
    pub(super) activation_key: ElaborationKey,
    pub(super) activation_full: FullElaborationIdentity,
}

#[derive(Debug, Clone)]
pub(super) enum FlatItemBlueprint {
    RecordInstance {
        name: String,
        definition: Id<kinds::Record>,
        members: Vec<crate::lower::LoweringExpression>,
        identity: EntityIdentity,
    },
    Nominal {
        name: String,
        definition: eqiora_schema::kernel::KernelNode,
        identity: EntityIdentity,
    },
    Domain {
        name: String,
        contract: LoweringDomainContract,
        range: TextRange,
        identity: EntityIdentity,
    },
    Representation {
        name: String,
        range: TextRange,
        identity: EntityIdentity,
    },
    Field {
        name: String,
        domain: Option<String>,
        representation: Option<String>,
        value_type: eqiora_lang::ValueTypeSyntax,
        role: eqiora_lang::FieldRoleSyntax,
        activation: ActivationSyntax,
        range: TextRange,
        identity: EntityIdentity,
    },
    Parameter {
        name: String,
        value: eqiora_core::ValueLiteral,
        range: TextRange,
        identity: EntityIdentity,
    },
    Port {
        name: String,
        contract: LoweringPortContract,
        range: TextRange,
        identity: EntityIdentity,
    },
    Clock {
        supplied_id: Option<Id<kinds::ClockDomain>>,
        name: String,
        period: eqiora_lang::Expr,
        phase: eqiora_lang::Expr,
        range: TextRange,
        identity: EntityIdentity,
    },
    Observable {
        name: String,
        value_type: eqiora_lang::ValueTypeSyntax,
        value: crate::lower::LoweringExpression,
        reduction: Option<crate::lower::LoweringIntegral>,
        domain: Option<String>,
        range: TextRange,
        identity: EntityIdentity,
    },
    Event {
        name: String,
        guard: crate::lower::LoweringExpression,
        direction: eqiora_schema::kernel::EventDirection,
        priority: i64,
        range: TextRange,
        identity: EntityIdentity,
    },
    Relation {
        name: String,
        activation: ActivationSyntax,
        domain: Option<String>,
        body: crate::lower::LoweringRelationBody,
        initial: bool,
        range: TextRange,
        identity: RelationIdentity,
    },
    Connection {
        syntax: ConnectionSyntax,
        ports: Vec<String>,
        range: TextRange,
        identity: ConnectionIdentity,
    },
    Boundary {
        ports: Vec<String>,
        range: TextRange,
    },
}

impl FlatItemBlueprint {
    pub(super) fn sort_key(&self) -> (u8, String) {
        match self {
            Self::Nominal { name, .. } | Self::RecordInstance { name, .. } => (0, name.clone()),
            Self::Domain { name, .. } => (0, name.clone()),
            Self::Representation { name, .. } => (1, name.clone()),
            Self::Field { name, .. } => (2, name.clone()),
            Self::Parameter { name, .. } => (3, name.clone()),
            Self::Port { name, .. } => (4, name.clone()),
            Self::Observable { name, .. } => (5, name.clone()),
            Self::Event { name, .. } => (5, name.clone()),
            Self::Clock { name, .. } => (5, name.clone()),
            Self::Relation { name, .. } => (6, name.clone()),
            Self::Connection { identity, .. } => (7, identity.full.to_string()),
            Self::Boundary { .. } => (8, String::new()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DisplayIdentity {
    pub(super) full: FullElaborationIdentity,
    pub(super) kind: EntityKind,
}

pub(super) struct ExpandedBlueprint {
    structural_dependencies: BTreeMap<String, std::collections::BTreeSet<String>>,
    notation_specs: Vec<crate::notation::NotationSpec>,
    model_name: String,
    model_source: SourceLocation,
    model_key: ModelViewKey,
    model_full: FullElaborationIdentity,
    items: Vec<FlatItemBlueprint>,
    display_symbols: BTreeMap<String, DisplayIdentity>,
    physical_exposures: Vec<PhysicalExposureProjectionBlueprint>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PhysicalExposureContractIdentity {
    ScalarPhysical {
        connector: FullElaborationIdentity,
    },
    FieldBoundary {
        connector: FullElaborationIdentity,
        boundary: FullElaborationIdentity,
    },
}

#[derive(Debug, Clone)]
pub(super) struct PhysicalExposureProjectionBlueprint {
    pub(super) selector: String,
    pub(super) exposure: EntityIdentity,
    pub(super) connection: FullElaborationIdentity,
    pub(super) interior: Vec<FullElaborationIdentity>,
    pub(super) contract: PhysicalExposureContractIdentity,
}

impl ExpandedBlueprint {
    pub(super) fn with_structural_dependencies(
        mut self,
        dependencies: BTreeMap<String, std::collections::BTreeSet<String>>,
    ) -> Self {
        self.structural_dependencies = dependencies;
        self
    }
    fn event_identities(&self) -> BTreeMap<&str, &EntityIdentity> {
        self.items
            .iter()
            .filter_map(|item| match item {
                FlatItemBlueprint::Event { name, identity, .. } => Some((name.as_str(), identity)),
                _ => None,
            })
            .collect()
    }

    pub(super) fn new(
        model_name: String,
        model_source: SourceLocation,
        model_identity: (ModelViewKey, FullElaborationIdentity),
        items: Vec<FlatItemBlueprint>,
        display_symbols: BTreeMap<String, DisplayIdentity>,
        physical_exposures: Vec<PhysicalExposureProjectionBlueprint>,
        notation_specs: Vec<crate::notation::NotationSpec>,
    ) -> Self {
        Self {
            notation_specs,
            structural_dependencies: BTreeMap::new(),
            model_name,
            model_source,
            model_key: model_identity.0,
            model_full: model_identity.1,
            items,
            display_symbols,
            physical_exposures,
        }
    }

    pub(super) fn compile(
        &self,
        limits: HierarchyLimits,
    ) -> Result<CompiledModel, Vec<Diagnostic>> {
        let events = self.event_identities();
        let mut allocator = StagingIdAllocator::with_projector_and_limits(
            crate::identity::Sha256PrefixProjector,
            limits.identity,
        );
        allocator
            .stage_model_view(&self.model_key)
            .map_err(|error| vec![error])?;
        for item in &self.items {
            match item {
                FlatItemBlueprint::Relation {
                    identity,
                    activation,
                    ..
                } => {
                    allocator
                        .stage(&identity.entity.key)
                        .map_err(|error| vec![error])?;
                    if !matches!(activation, ActivationSyntax::Named(name) if events.contains_key(name.as_str()))
                    {
                        allocator
                            .stage(&identity.activation_key)
                            .map_err(|error| vec![error])?;
                    }
                }
                FlatItemBlueprint::Domain { identity, .. }
                | FlatItemBlueprint::Representation { identity, .. }
                | FlatItemBlueprint::Field { identity, .. }
                | FlatItemBlueprint::Parameter { identity, .. }
                | FlatItemBlueprint::Port { identity, .. }
                | FlatItemBlueprint::Observable { identity, .. }
                | FlatItemBlueprint::Event { identity, .. }
                | FlatItemBlueprint::RecordInstance { identity, .. } => {
                    allocator
                        .stage(&identity.key)
                        .map_err(|error| vec![error])?;
                }
                FlatItemBlueprint::Nominal {
                    definition,
                    identity,
                    ..
                } => {
                    allocator
                        .stage_bound_entity(&identity.key, definition.id())
                        .map_err(|error| vec![error])?;
                }
                FlatItemBlueprint::Clock {
                    identity,
                    supplied_id,
                    ..
                } => {
                    match supplied_id {
                        Some(id) => allocator.stage_bound_clock(&identity.key, *id),
                        None => allocator.stage(&identity.key),
                    }
                    .map_err(|error| vec![error])?;
                }
                FlatItemBlueprint::Connection { identity, .. } => {
                    allocator
                        .stage(&identity.key)
                        .map_err(|error| vec![error])?;
                }
                FlatItemBlueprint::Boundary { .. } => {}
            }
        }
        let staged = allocator.finish();
        let mut identities =
            AssignedLoweringIdentities::new(self, &staged).map_err(|error| vec![error])?;
        let model = self.lowering_model(&staged).map_err(|error| vec![error])?;
        let compiled =
            crate::lower::lower_typed_model(&self.model_source.file, &model, &mut identities)?;
        let symbols = self
            .display_symbols
            .iter()
            .map(|(name, identity)| {
                resolve_raw(&staged, *identity)
                    .map(|id| (name.clone(), id))
                    .map_err(|error| vec![error])
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let provenance = self
            .provenance(limits.provenance, &staged)
            .map_err(|error| vec![error])?;
        let physical_exposures = self
            .physical_exposure_projections(&staged)
            .map_err(|error| vec![error])?;
        let graph_ids = symbols
            .values()
            .filter_map(|raw| {
                provenance
                    .identity_for_graph_id(*raw)
                    .map(|full| (full, *raw))
            })
            .collect::<BTreeMap<_, _>>();
        let notation = crate::notation::ModelNotation::resolve(
            self.notation_specs
                .iter()
                .cloned()
                .map(|mut spec| {
                    spec.graph_id = spec
                        .graph_identity
                        .and_then(|full| graph_ids.get(&full).copied());
                    spec
                })
                .collect(),
        );
        Ok(compiled.with_elaboration_metadata(
            ModelSymbols::from_map(symbols),
            provenance,
            physical_exposures,
            notation,
        ))
    }

    fn physical_exposure_projections(
        &self,
        staged: &StagedIdentities,
    ) -> Result<PhysicalExposureProjectionMap, Diagnostic> {
        let mut projections = Vec::new();
        projections
            .try_reserve_exact(self.physical_exposures.len())
            .map_err(|_| hierarchy_error("cannot reserve physical exposure projections"))?;
        for blueprint in &self.physical_exposures {
            let contract = match blueprint.contract {
                PhysicalExposureContractIdentity::ScalarPhysical { connector } => {
                    PhysicalExposureContract::ScalarPhysical {
                        connector: staged.resolve::<kinds::Domain>(connector)?,
                    }
                }
                PhysicalExposureContractIdentity::FieldBoundary {
                    connector,
                    boundary,
                } => PhysicalExposureContract::FieldBoundary {
                    connector: staged.resolve::<kinds::Domain>(connector)?,
                    boundary: staged.resolve::<kinds::Domain>(boundary)?,
                },
            };
            let mut interior = Vec::new();
            interior
                .try_reserve_exact(blueprint.interior.len())
                .map_err(|_| hierarchy_error("cannot reserve projected physical exposure cut"))?;
            for identity in &blueprint.interior {
                interior.push(staged.resolve::<kinds::Port>(*identity)?);
            }
            projections.push(PhysicalExposureProjection::new(
                blueprint.selector.clone(),
                blueprint.exposure.full,
                staged.resolve::<kinds::Connection>(blueprint.connection)?,
                interior,
                contract,
            ));
        }
        projections.sort_unstable_by(|left, right| {
            left.selector()
                .cmp(right.selector())
                .then_with(|| left.exposure().cmp(&right.exposure()))
        });
        PhysicalExposureProjectionMap::from_sorted(projections)
    }

    fn lowering_model(&self, staged: &StagedIdentities) -> Result<LoweringModel, Diagnostic> {
        let mut items = Vec::new();
        items
            .try_reserve_exact(self.items.len())
            .map_err(|_| hierarchy_error("cannot reserve typed flattened declarations"))?;
        for item in &self.items {
            let item = match item {
                FlatItemBlueprint::Domain {
                    name,
                    contract,
                    range,
                    ..
                } => LoweringItem::Domain {
                    name: name.clone(),
                    contract: contract.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Representation { name, range, .. } => {
                    LoweringItem::Representation {
                        name: name.clone(),
                        range: *range,
                    }
                }
                FlatItemBlueprint::Field {
                    name,
                    domain,
                    representation,
                    value_type,
                    role,
                    activation,
                    range,
                    ..
                } => LoweringItem::Field {
                    name: name.clone(),
                    domain: domain.clone(),
                    representation: representation.clone(),
                    value_type: value_type.clone(),
                    role: *role,
                    activation: activation.clone(),
                    range: *range,
                },
                FlatItemBlueprint::RecordInstance {
                    definition,
                    members,
                    identity,
                    ..
                } => LoweringItem::RecordInstance {
                    id: staged.resolve::<kinds::RecordInstance>(identity.full)?.id(),
                    definition: *definition,
                    members: members.clone(),
                },
                FlatItemBlueprint::Nominal {
                    name, definition, ..
                } => LoweringItem::Nominal {
                    name: name.clone(),
                    definition: definition.clone(),
                },
                FlatItemBlueprint::Parameter {
                    name, value, range, ..
                } => LoweringItem::Parameter {
                    name: name.clone(),
                    value: value.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Port {
                    name,
                    contract,
                    range,
                    ..
                } => LoweringItem::Port {
                    name: name.clone(),
                    contract: contract.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Observable {
                    name,
                    value_type,
                    value,
                    reduction,
                    domain,
                    range,
                    ..
                } => LoweringItem::Observable {
                    name: name.clone(),
                    value_type: value_type.clone(),
                    value: value.clone(),
                    reduction: reduction.clone(),
                    domain: domain.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Event {
                    name,
                    guard,
                    direction,
                    priority,
                    range,
                    ..
                } => LoweringItem::Event {
                    name: name.clone(),
                    guard: guard.clone(),
                    direction: *direction,
                    priority: *priority,
                    range: *range,
                },
                FlatItemBlueprint::Clock {
                    name,
                    period,
                    phase,
                    range,
                    ..
                } => LoweringItem::Clock {
                    name: name.clone(),
                    period: period.clone(),
                    phase: phase.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Relation {
                    name,
                    activation,
                    domain,
                    body,
                    initial,
                    range,
                    ..
                } => LoweringItem::Relation {
                    name: name.clone(),
                    activation: activation.clone(),
                    domain: domain.clone(),
                    body: body.clone(),
                    initial: *initial,
                    range: *range,
                },
                FlatItemBlueprint::Connection {
                    syntax,
                    ports,
                    range,
                    ..
                } => LoweringItem::Connection {
                    syntax: *syntax,
                    ports: ports.clone(),
                    range: *range,
                },
                FlatItemBlueprint::Boundary { ports, range } => LoweringItem::Boundary {
                    ports: ports.clone(),
                    range: *range,
                },
            };
            items.push(item);
        }
        Ok(LoweringModel {
            structural_dependencies: self.structural_dependencies.clone(),
            name: self.model_name.clone(),
            range: self.model_source.range,
            items,
        })
    }

    fn provenance(
        &self,
        limits: ProvenanceLimits,
        staged: &StagedIdentities,
    ) -> Result<crate::provenance::ProvenanceMap, Diagnostic> {
        let events = self.event_identities();
        let mut builder = ProvenanceBuilder::with_limits(limits);
        builder.insert(
            self.model_full,
            self.model_source.span(),
            self.model_source.span(),
            [],
        )?;
        for item in &self.items {
            match item {
                FlatItemBlueprint::Relation {
                    identity,
                    initial,
                    activation,
                    ..
                } => {
                    insert_provenance(&mut builder, &identity.entity, staged)?;
                    if *initial
                        || matches!(activation, ActivationSyntax::Named(name) if events.contains_key(name.as_str()))
                    {
                        continue;
                    }
                    builder.insert_graph(
                        resolve_entity_raw(
                            staged,
                            EntityKind::Activation,
                            identity.activation_full,
                        )?,
                        identity.activation_full,
                        identity.entity.definition.span(),
                        identity.entity.instance.span(),
                        identity.entity.bindings.iter().map(SourceLocation::span),
                    )?;
                }
                FlatItemBlueprint::Domain { identity, .. }
                | FlatItemBlueprint::Representation { identity, .. }
                | FlatItemBlueprint::Field { identity, .. }
                | FlatItemBlueprint::Parameter { identity, .. }
                | FlatItemBlueprint::Port { identity, .. }
                | FlatItemBlueprint::Nominal { identity, .. }
                | FlatItemBlueprint::Clock { identity, .. }
                | FlatItemBlueprint::Observable { identity, .. }
                | FlatItemBlueprint::Event { identity, .. }
                | FlatItemBlueprint::RecordInstance { identity, .. } => {
                    insert_provenance(&mut builder, identity, staged)?;
                }
                FlatItemBlueprint::Connection { identity, .. } => {
                    insert_connection_provenance(&mut builder, identity, staged)?;
                }
                FlatItemBlueprint::Boundary { .. } => {}
            }
        }
        for projection in &self.physical_exposures {
            let identity = &projection.exposure;
            builder.insert(
                identity.full,
                identity.definition.span(),
                identity.instance.span(),
                identity.bindings.iter().map(SourceLocation::span),
            )?;
        }
        Ok(builder.finish())
    }
}

struct AssignedLoweringIdentities {
    model: OntologyId<Model>,
    domains: BTreeMap<String, Id<kinds::Domain>>,
    representations: BTreeMap<String, Id<kinds::Representation>>,
    fields: BTreeMap<String, Id<kinds::Field>>,
    parameters: BTreeMap<String, Id<kinds::Parameter>>,
    ports: BTreeMap<String, Id<kinds::Port>>,
    clocks: BTreeMap<String, Id<kinds::ClockDomain>>,
    events: BTreeMap<String, Id<kinds::Activation>>,
    observables: BTreeMap<String, Id<kinds::Observable>>,
    relations: BTreeMap<String, (Id<kinds::Relation>, Id<kinds::Activation>)>,
    connections: VecDeque<Id<kinds::Connection>>,
}

impl AssignedLoweringIdentities {
    fn new(blueprint: &ExpandedBlueprint, staged: &StagedIdentities) -> Result<Self, Diagnostic> {
        let events = blueprint.event_identities();
        let model = staged.resolve_model_view(blueprint.model_full)?.id();
        let mut result = Self {
            model,
            domains: BTreeMap::new(),
            representations: BTreeMap::new(),
            fields: BTreeMap::new(),
            parameters: BTreeMap::new(),
            ports: BTreeMap::new(),
            clocks: BTreeMap::new(),
            events: BTreeMap::new(),
            observables: BTreeMap::new(),
            relations: BTreeMap::new(),
            connections: VecDeque::new(),
        };
        for item in &blueprint.items {
            match item {
                FlatItemBlueprint::Domain { name, identity, .. } => {
                    result.domains.insert(
                        name.clone(),
                        staged.resolve::<kinds::Domain>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Representation { name, identity, .. } => {
                    result.representations.insert(
                        name.clone(),
                        staged.resolve::<kinds::Representation>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Field { name, identity, .. } => {
                    result.fields.insert(
                        name.clone(),
                        staged.resolve::<kinds::Field>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Parameter { name, identity, .. } => {
                    result.parameters.insert(
                        name.clone(),
                        staged.resolve::<kinds::Parameter>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Port { name, identity, .. } => {
                    result.ports.insert(
                        name.clone(),
                        staged.resolve::<kinds::Port>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Observable { name, identity, .. } => {
                    result.observables.insert(
                        name.clone(),
                        staged.resolve::<kinds::Observable>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Event { name, identity, .. } => {
                    result.events.insert(
                        name.clone(),
                        staged.resolve::<kinds::Activation>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Clock { name, identity, .. } => {
                    result.clocks.insert(
                        name.clone(),
                        staged.resolve::<kinds::ClockDomain>(identity.full)?.id(),
                    );
                }
                FlatItemBlueprint::Relation {
                    name,
                    identity,
                    activation,
                    ..
                } => {
                    let activation_full = match activation {
                        ActivationSyntax::Named(name) => events
                            .get(name.as_str())
                            .map_or(identity.activation_full, |event| event.full),
                        _ => identity.activation_full,
                    };
                    result.relations.insert(
                        name.clone(),
                        (
                            staged
                                .resolve::<kinds::Relation>(identity.entity.full)?
                                .id(),
                            staged.resolve::<kinds::Activation>(activation_full)?.id(),
                        ),
                    );
                }
                FlatItemBlueprint::Connection { identity, .. } => {
                    result
                        .connections
                        .push_back(staged.resolve::<kinds::Connection>(identity.full)?.id());
                }
                FlatItemBlueprint::Nominal { .. }
                | FlatItemBlueprint::RecordInstance { .. }
                | FlatItemBlueprint::Boundary { .. } => {}
            }
        }
        Ok(result)
    }
}

impl LoweringIdentities for AssignedLoweringIdentities {
    fn model(&mut self, _name: &str) -> OntologyId<Model> {
        self.model
    }

    fn domain(&mut self, name: &str) -> Id<kinds::Domain> {
        self.domains[name]
    }

    fn representation(&mut self, name: &str) -> Id<kinds::Representation> {
        self.representations[name]
    }

    fn field(&mut self, name: &str) -> Id<kinds::Field> {
        self.fields[name]
    }

    fn parameter(&mut self, name: &str) -> Id<kinds::Parameter> {
        self.parameters[name]
    }

    fn port(&mut self, name: &str) -> Id<kinds::Port> {
        self.ports[name]
    }

    fn clock(&mut self, name: &str) -> Id<kinds::ClockDomain> {
        self.clocks[name]
    }

    fn observable(&mut self, name: &str) -> Id<kinds::Observable> {
        self.observables[name]
    }

    fn activation(&mut self, name: &str) -> Id<kinds::Activation> {
        self.events[name]
    }

    fn relation(&mut self, name: &str) -> (Id<kinds::Relation>, Id<kinds::Activation>) {
        self.relations[name]
    }

    fn connection(&mut self) -> Id<kinds::Connection> {
        self.connections
            .pop_front()
            .expect("staged connection order matches flat source order")
    }
}

fn resolve_raw(staged: &StagedIdentities, identity: DisplayIdentity) -> Result<RawId, Diagnostic> {
    resolve_entity_raw(staged, identity.kind, identity.full)
}

fn resolve_entity_raw(
    staged: &StagedIdentities,
    kind: EntityKind,
    identity: FullElaborationIdentity,
) -> Result<RawId, Diagnostic> {
    match kind {
        EntityKind::Domain => Ok(staged.resolve::<kinds::Domain>(identity)?.id().erase()),
        EntityKind::Representation => Ok(staged
            .resolve::<kinds::Representation>(identity)?
            .id()
            .erase()),
        EntityKind::Observable => Ok(staged.resolve::<kinds::Observable>(identity)?.id().erase()),
        EntityKind::Field => Ok(staged.resolve::<kinds::Field>(identity)?.id().erase()),
        EntityKind::Parameter => Ok(staged.resolve::<kinds::Parameter>(identity)?.id().erase()),
        EntityKind::Port => Ok(staged.resolve::<kinds::Port>(identity)?.id().erase()),
        EntityKind::Record => Ok(staged.resolve::<kinds::Record>(identity)?.id().erase()),
        EntityKind::RecordInstance => Ok(staged
            .resolve::<kinds::RecordInstance>(identity)?
            .id()
            .erase()),
        EntityKind::Enum => Ok(staged.resolve::<kinds::Enum>(identity)?.id().erase()),
        EntityKind::FiniteSpace => Ok(staged.resolve::<kinds::FiniteSpace>(identity)?.id().erase()),
        EntityKind::IndexSet => Ok(staged.resolve::<kinds::IndexSet>(identity)?.id().erase()),
        EntityKind::ClockDomain => Ok(staged.resolve::<kinds::ClockDomain>(identity)?.id().erase()),
        EntityKind::Relation => Ok(staged.resolve::<kinds::Relation>(identity)?.id().erase()),
        EntityKind::Activation => Ok(staged.resolve::<kinds::Activation>(identity)?.id().erase()),
        EntityKind::Connection => Ok(staged.resolve::<kinds::Connection>(identity)?.id().erase()),
        _ => Err(hierarchy_error(format!(
            "unsupported elaborated provenance entity kind {kind:?}"
        ))),
    }
}

fn insert_provenance(
    builder: &mut ProvenanceBuilder,
    identity: &EntityIdentity,
    staged: &StagedIdentities,
) -> Result<(), Diagnostic> {
    builder.insert_graph(
        resolve_entity_raw(staged, identity.key.entity_kind(), identity.full)?,
        identity.full,
        identity.definition.span(),
        identity.instance.span(),
        identity.bindings.iter().map(SourceLocation::span),
    )
}

fn insert_connection_provenance(
    builder: &mut ProvenanceBuilder,
    identity: &ConnectionIdentity,
    staged: &StagedIdentities,
) -> Result<(), Diagnostic> {
    builder.insert_graph_origins(
        resolve_entity_raw(staged, EntityKind::Connection, identity.full)?,
        identity.full,
        identity.origins.iter().map(|origin| {
            ElaborationSourceOrigin::new(
                origin.definition.span(),
                origin.instance.span(),
                origin.bindings.iter().map(SourceLocation::span).collect(),
            )
        }),
    )
}

fn span(file: &str, range: TextRange) -> Span {
    Span {
        file: file.to_owned(),
        start: range.start(),
        end: range.end(),
    }
}
