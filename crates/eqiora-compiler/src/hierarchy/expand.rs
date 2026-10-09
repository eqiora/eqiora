mod activations;
mod connections;
mod coordinate_products;
mod identities;
use identities::{contextualize_diagnostic, contextualize_diagnostics, one_diagnostic};
mod indexed_relations;
mod integral_partials;
mod model_scope;
pub(super) mod observable;
mod parameters;
mod record_parameters;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::ValueFrame;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, EntityKind, ValueShape};
use eqiora_lang::{
    BoundaryConnectionDecl, BoundaryPairingSyntax, ComponentItem, ComponentPortDecl,
    ComponentPortFamilyDecl, ConnectionDecl, ConnectionSyntax, ConnectorSyntax, DomainSyntax,
    FrameSyntax, InstanceDecl, Item, PortSyntax, VisibilitySyntax,
};
use eqiora_schema::kernel::typing::SpatialSupport;
use eqiora_schema::kernel::{
    BoundaryPairing, BoundaryPhysicalConnector, BoundaryPhysicalPortContract, BoundarySide,
    CartesianBoundaryEmbedding, validate_boundary_physical_connection,
};

use crate::connection_sets::ConnectionFragment;
use crate::diagnostics::source_error;
use crate::dimensions::lower_dimension;
use crate::identity::{
    DeclarationPath, ElaborationKey, FullElaborationIdentity, GeneratedRole, IdentityNamespace,
    InstancePath, ModelViewKey,
};
use crate::lower::{LoweringDomainContract, LoweringPortContract};

use super::body_check::field_expression_type;
use super::complete_exterior::CartesianDomain;
use super::field_slots::{FieldContract, component_field_interface, resolve_instance_fields};
use super::occurrence_connections::{
    OccurrenceConnectionFragment, OccurrencePhysicalEndpoint, normalize_occurrence_connections,
};
use super::{exposure_cuts::ExposureCutIndex, hierarchy_error};

use super::flat::{
    ConnectionIdentity, DisplayIdentity, EntityIdentity, EntitySourceOrigin, ExpandedBlueprint,
    FlatItemBlueprint, PhysicalExposureContractIdentity, PhysicalExposureProjectionBlueprint,
    RelationIdentity, SourceLocation,
};

mod binding_locations;
mod cartesian;
mod component_items;
mod connector_domain;
mod external;
mod indexed;
mod input_bindings;
mod model_items;
mod model_lets;
pub(super) mod names;
mod nominal;
mod notation;
mod records;
mod structural;

use super::parameters::{ParameterLineage, ParameterResolver, ResolvedParameter};
use super::preflight::{
    ComponentDefinition, ConnectorDefinition, DefinitionKey, DefinitionNamespace, Elaborator,
    ExpansionSize, ModelDefinition,
};
use super::scope::{
    ActiveBoundaryMember, FlatSymbol, InstanceInterface, PhysicalMemberNames, Scope, SymbolKind,
    resolve_boundary_port_reference, resolve_local_kind, rewrite_equations, rewrite_field_scope,
    rewrite_model_port, rewrite_relation,
};
use super::supports::{
    CompleteExteriorMembershipBudget, ResolvedBoundaryTarget, ResolvedSupportBindings,
    component_support_interface, resolve_instance_support_bindings,
};
use binding_locations::{
    boundary_family_bindings, boundary_set_forwarding_locations,
    compare_physical_connection_origins, field_forwarding_locations, instance_binding_locations,
    normalize_binding_locations, parameter_forwarding_locations,
};
use names::{
    boundary_family_display, child_instance_path, definition_path, display_child, internal_name,
};

#[derive(Debug, Default)]
struct ScopeIdentities {
    entities: BTreeMap<String, EntityIdentity>,
    relations: BTreeMap<String, RelationIdentity>,
    boundary_family_entities: BTreeMap<(String, FullElaborationIdentity), EntityIdentity>,
    boundary_family_relations: BTreeMap<(String, FullElaborationIdentity), RelationIdentity>,
}

struct ComponentOccurrence<'a, 'd> {
    definition: &'a ComponentDefinition<'d>,
    instance: &'a InstanceDecl,
    instance_file: &'a str,
    instance_path: &'a InstancePath,
    display_prefix: &'a str,
}

struct ConnectionOrigin {
    instance: SourceLocation,
    bindings: Vec<SourceLocation>,
    definition_file: String,
}

#[derive(Debug, Clone)]
struct PhysicalPortOccurrence {
    identity: EntityIdentity,
    display_name: String,
    instance_path: InstancePath,
    exposure_candidate: bool,
    contract: Option<PhysicalExposureContractIdentity>,
}

struct PhysicalPortMaterialization {
    contract: PhysicalExposureContractIdentity,
}

struct PortFamilyMemberRegistration<'a> {
    quantities: PhysicalMemberNames,
    file: &'a str,
    range: eqiora_lang::TextRange,
    display_name: String,
    family_name: &'a str,
    selector_member: &'a str,
    boundary: FullElaborationIdentity,
    identity: &'a EntityIdentity,
}

#[derive(Debug, Clone)]
struct PhysicalConnectionOrigin {
    declaration_path: DeclarationPath,
    instance_path: InstancePath,
    source: EntitySourceOrigin,
}

#[derive(Debug, Clone)]
struct StagedPhysicalConnection {
    topology: ConnectionFragment<FullElaborationIdentity>,
    origin: PhysicalConnectionOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ConnectorSpecializationKey {
    definition: DefinitionKey,
    shape: ValueShape,
}

pub(super) struct RootExpansion<'a, 'd> {
    structural_dependencies: BTreeMap<String, BTreeSet<String>>,
    structural_dependency_count: usize,
    notation_specs: Vec<crate::notation::NotationSpec>,
    instance_qualifiers: BTreeMap<Vec<String>, eqiora_lang::Notation>,
    elaborator: &'a Elaborator<'d>,
    model: ModelDefinition<'d>,
    namespace: IdentityNamespace,
    root_path: InstancePath,
    model_key: ModelViewKey,
    model_full: FullElaborationIdentity,
    items: Vec<FlatItemBlueprint>,
    declaration_count: usize,
    support_representations: BTreeMap<String, (EntityIdentity, bool)>,
    connector_domains: BTreeMap<ConnectorSpecializationKey, FlatSymbol>,
    display_symbols: BTreeMap<String, DisplayIdentity>,
    physical_ports: BTreeMap<FullElaborationIdentity, PhysicalPortOccurrence>,
    physical_ports_by_name: BTreeMap<String, FullElaborationIdentity>,
    physical_owner_relations: BTreeMap<FullElaborationIdentity, BTreeSet<FullElaborationIdentity>>,
    physical_connections: Vec<StagedPhysicalConnection>,
    spatial_periodic_ports: BTreeSet<FullElaborationIdentity>,
    physical_exposures: Vec<PhysicalExposureProjectionBlueprint>,
    boundary_embeddings: BTreeMap<FullElaborationIdentity, Option<CartesianBoundaryEmbedding>>,
    boundary_parents: BTreeMap<FullElaborationIdentity, FullElaborationIdentity>,
    boundary_sides: BTreeMap<FullElaborationIdentity, (usize, BoundarySide)>,
    complete_exterior_memberships: CompleteExteriorMembershipBudget,
}

impl<'a, 'd> RootExpansion<'a, 'd> {
    pub(super) fn new(
        elaborator: &'a Elaborator<'d>,
        model: ModelDefinition<'d>,
        size: ExpansionSize,
    ) -> Result<Self, Diagnostic> {
        let namespace = elaborator.identity_namespace.clone();
        let root_path = InstancePath::with_limits([model.name()], elaborator.limits.identity)?;
        let model_key = ModelViewKey::with_limits(
            namespace.clone(),
            root_path.clone(),
            elaborator.limits.identity,
        )?;
        let model_full = model_key.full_identity()?;
        let declarations =
            elaborator
                .records
                .values()
                .try_fold(size.declarations, |count, records| {
                    count
                        .checked_add(records.len())
                        .filter(|count| *count <= elaborator.limits.max_declarations)
                        .ok_or_else(|| {
                            hierarchy_error(
                                "record definitions exceed the declaration expansion limit",
                            )
                        })
                })?;
        let item_capacity = declarations
            .checked_add(size.connections)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| hierarchy_error("flat item capacity overflows usize"))?;
        let mut items = Vec::new();
        items
            .try_reserve_exact(item_capacity)
            .map_err(|_| hierarchy_error("cannot reserve flat component expansion"))?;
        let mut expansion = Self {
            notation_specs: Vec::new(),
            instance_qualifiers: BTreeMap::new(),
            elaborator,
            model,
            namespace,
            root_path,
            model_key,
            model_full,
            items,
            declaration_count: declarations,
            structural_dependencies: BTreeMap::new(),
            structural_dependency_count: 0,
            support_representations: BTreeMap::new(),
            connector_domains: BTreeMap::new(),
            display_symbols: BTreeMap::new(),
            physical_ports: BTreeMap::new(),
            physical_ports_by_name: BTreeMap::new(),
            physical_owner_relations: BTreeMap::new(),
            physical_connections: Vec::new(),
            spatial_periodic_ports: BTreeSet::new(),
            physical_exposures: Vec::new(),
            boundary_embeddings: BTreeMap::new(),
            boundary_parents: BTreeMap::new(),
            boundary_sides: BTreeMap::new(),
            complete_exterior_memberships: CompleteExteriorMembershipBudget::new(
                elaborator.limits.complete_exteriors,
            ),
        };
        expansion.allocate_finite_spaces()?;
        Ok(expansion)
    }

    fn add_support_representation(
        &mut self,
        support: Option<&str>,
    ) -> Result<Option<String>, Diagnostic> {
        let Some(support) = support else {
            return Ok(None);
        };
        let (identity, emitted) = self
            .support_representations
            .get_mut(support)
            .ok_or_else(|| hierarchy_error("Field support has no representation owner"))?;
        let name = internal_name(identity.full);
        if !*emitted {
            self.items.push(FlatItemBlueprint::Representation {
                name: name.clone(),
                range: identity.definition.range,
                identity: identity.clone(),
            });
            *emitted = true;
        }
        Ok(Some(name))
    }

    pub(super) fn expand(self) -> Result<ExpandedBlueprint, Vec<Diagnostic>> {
        self.expand_bound(&[], &[], &BTreeMap::new())
    }

    pub(super) fn expand_bound(
        mut self,
        supports: &[crate::external::ExternalSupportBinding],
        clocks: &[(String, eqiora_schema::kernel::ClockDomainDef)],
        properties: &BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>,
    ) -> Result<ExpandedBlueprint, Vec<Diagnostic>> {
        let model = self.model.clone();
        let mut root_scope = Scope::default();
        root_scope.extend_properties(properties);
        root_scope.record_context =
            super::parameters::RecordContext::model(self.elaborator, &model);
        root_scope.reduction_terms_limit = self.elaborator.limits.max_parameter_terms;
        root_scope.lexical_namespace = Some(model.namespace.clone());
        root_scope.set_pure_operators(self.elaborator.visible_pure_operators(&model.namespace));
        self.allocate_external_clocks(&mut root_scope, clocks)
            .map_err(one_diagnostic)?;
        self.allocate_external_supports(&mut root_scope, supports)
            .map_err(one_diagnostic)?;
        let identities = match self.allocate_model_scope(&mut root_scope) {
            Ok(value) => value,
            Err(error) => return Err(vec![error]),
        };

        self.expand_model_children(&model, &mut root_scope)?;

        self.allocate_model_expression_bindings(&mut root_scope, &model)?;

        if let Err(error) = self.materialize_model_items(&root_scope, &identities) {
            return Err(vec![error]);
        }
        self.record_index_dependencies(&root_scope)
            .map_err(one_diagnostic)?;
        if let Err(error) = self.finalize_physical_connections() {
            return Err(vec![error]);
        }
        self.expand_integral_partials().map_err(one_diagnostic)?;
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

    fn expand_component(
        &mut self,
        component: ComponentDefinition<'d>,
        instance: &InstanceDecl,
        instance_file: &str,
        instance_path: InstancePath,
        display_prefix: String,
        parent_scope: &Scope,
    ) -> Result<InstanceInterface, Vec<Diagnostic>> {
        if let Some(notation) = instance.notation() {
            self.instance_qualifiers
                .insert(instance_path.segments().to_vec(), notation.clone());
        }
        let support_interface = component_support_interface(component.file, component.declaration)
            .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        let boundary_sides = &self.boundary_sides;
        let membership_budget = &mut self.complete_exterior_memberships;
        let support_bindings = resolve_instance_support_bindings(
            instance_file,
            component.declaration,
            &support_interface,
            instance,
            |name| parent_scope.spatial_support(name).cloned(),
            |name| {
                let SpatialSupport::Boundary { domain, .. } = parent_scope.spatial_support(name)?
                else {
                    return None;
                };
                let symbol = parent_scope.symbol(name)?;
                Some(ResolvedBoundaryTarget::new(
                    symbol.internal_name.clone(),
                    *domain,
                ))
            },
            |identity| match parent_scope.spatial_support_by_identity(*identity)? {
                SpatialSupport::Volume { dimensions, .. } => Some(CartesianDomain::Volume {
                    ambient_dimension: *dimensions,
                }),
                SpatialSupport::Boundary {
                    parent, dimensions, ..
                } => {
                    let (axis, side) = boundary_sides.get(identity)?;
                    Some(CartesianDomain::Boundary {
                        exact_parent: *parent,
                        ambient_dimension: *dimensions,
                        axis: *axis,
                        side: *side,
                    })
                }
                SpatialSupport::Coordinates { .. }
                | SpatialSupport::Interface { .. }
                | SpatialSupport::PhysicalInterface { .. } => None,
            },
            |name| parent_scope.boundary_set(name).cloned(),
            membership_budget,
        )
        .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        let parameters = parameters::resolve(
            self.elaborator,
            &component,
            instance,
            instance_file,
            &instance_path,
            parent_scope,
        )?;
        let mut bindings = parent_scope
            .forwarded_parameter_resolution_bindings()
            .to_vec();
        bindings.extend_from_slice(parent_scope.forwarded_field_resolution_bindings());
        bindings.extend_from_slice(parent_scope.forwarded_boundary_set_resolution_bindings());
        bindings.extend(instance_binding_locations(instance_file, instance));
        normalize_binding_locations(&mut bindings);
        let mut forwarded_field_resolution_bindings =
            parent_scope.forwarded_field_resolution_bindings().to_vec();
        forwarded_field_resolution_bindings.extend(field_forwarding_locations(
            instance_file,
            instance,
            component.declaration,
        ));
        normalize_binding_locations(&mut forwarded_field_resolution_bindings);
        let mut forwarded_parameter_resolution_bindings = parent_scope
            .forwarded_parameter_resolution_bindings()
            .to_vec();
        forwarded_parameter_resolution_bindings.extend(parameter_forwarding_locations(
            instance_file,
            instance,
            component.declaration,
        ));
        normalize_binding_locations(&mut forwarded_parameter_resolution_bindings);
        let mut forwarded_boundary_set_resolution_bindings = parent_scope
            .forwarded_boundary_set_resolution_bindings()
            .to_vec();
        forwarded_boundary_set_resolution_bindings.extend(boundary_set_forwarding_locations(
            instance_file,
            instance,
            &support_bindings,
        ));
        normalize_binding_locations(&mut forwarded_boundary_set_resolution_bindings);
        let mut scope = Scope::child(parent_scope);
        scope.lexical_namespace = Some(component.namespace.clone());
        scope.bind_properties(
            self.elaborator,
            &component.namespace,
            component.declaration,
            instance,
            parent_scope,
            instance_file,
        )?;
        scope.record_context =
            super::parameters::RecordContext::component(self.elaborator, &component);
        scope.set_pure_operators(self.elaborator.visible_pure_operators(&component.namespace));
        scope.set_occurrence_bindings(bindings.clone());
        scope.set_forwarded_parameter_resolution_bindings(forwarded_parameter_resolution_bindings);
        scope.set_forwarded_field_resolution_bindings(forwarded_field_resolution_bindings);
        scope.set_forwarded_boundary_set_resolution_bindings(
            forwarded_boundary_set_resolution_bindings,
        );
        let mut identities = ScopeIdentities::default();

        for (slot, target) in support_bindings.singular_targets() {
            let Some(symbol) = parent_scope.symbol(target).cloned() else {
                return Err(vec![contextualize_diagnostic(
                    source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        instance_file,
                        instance.range(),
                        format!("resolved support target `{target}` has no flattened symbol"),
                    ),
                    &instance_path,
                )]);
            };
            let support = support_bindings.singular_supports()[slot].clone();
            if scope.insert_symbol(slot.clone(), symbol).is_some()
                || scope
                    .insert_spatial_support(slot.clone(), support)
                    .is_some()
            {
                return Err(vec![contextualize_diagnostic(
                    hierarchy_error(format!("duplicate flattened support alias `{slot}`")),
                    &instance_path,
                )]);
            }
        }
        for (slot, set) in support_bindings.boundary_sets() {
            if scope
                .insert_boundary_set(slot.to_owned(), set.clone())
                .is_some()
            {
                return Err(vec![contextualize_diagnostic(
                    hierarchy_error(format!(
                        "duplicate flattened complete-exterior binding `{slot}`"
                    )),
                    &instance_path,
                )]);
            }
        }

        let clocks = super::field_slots::resolve_instance_clocks(
            instance_file,
            component.declaration,
            instance,
            |name| {
                parent_scope
                    .symbol(name)
                    .filter(|symbol| matches!(symbol.kind, SymbolKind::Clock(_)))
                    .map(|symbol| symbol.internal_name.clone())
            },
        )
        .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        for binding in instance
            .bindings()
            .iter()
            .filter(|binding| clocks.contains_key(binding.name()))
        {
            let eqiora_lang::ExprKind::Name(target) = binding.value().kind() else {
                unreachable!("validated nominal clock reference")
            };
            let symbol = parent_scope
                .symbol(target)
                .expect("validated clock binding")
                .clone();
            scope.insert_symbol(binding.name().to_owned(), symbol);
        }

        let field_interface = component_field_interface(
            component.file,
            component.declaration,
            &support_interface,
            &parameters
                .iter()
                .map(|(name, value)| (name.clone(), value.clone().into()))
                .collect(),
        )
        .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        let field_bindings = resolve_instance_fields(
            instance_file,
            component.declaration,
            &field_interface,
            instance,
            |name| {
                parent_scope
                    .symbol(name)
                    .filter(|symbol| matches!(symbol.kind, SymbolKind::Clock(_)))
                    .map(|symbol| symbol.internal_name.clone())
            },
            |slot| scope.spatial_support(slot).cloned(),
            |target| {
                let symbol = parent_scope.symbol(target)?;
                if !matches!(symbol.kind, SymbolKind::Field) {
                    return None;
                }
                parent_scope.field_type(target).cloned().map(|value| {
                    let (role, activation) = parent_scope.field_evolution[target].clone();
                    FieldContract::continuum(value, role, activation)
                })
            },
        )
        .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        for (slot, target) in field_bindings {
            let symbol = parent_scope.symbol(&target).cloned().ok_or_else(|| {
                vec![contextualize_diagnostic(
                    hierarchy_error(format!(
                        "resolved Field target `{target}` has no flattened symbol"
                    )),
                    &instance_path,
                )]
            })?;
            let field_type = parent_scope.field_type(&target).cloned().ok_or_else(|| {
                vec![contextualize_diagnostic(
                    hierarchy_error(format!(
                        "resolved Field target `{target}` has no exact type"
                    )),
                    &instance_path,
                )]
            })?;
            if scope.symbol(&slot).is_some() || scope.field_type(&slot).is_some() {
                return Err(vec![contextualize_diagnostic(
                    hierarchy_error(format!("duplicate flattened Field alias `{slot}`")),
                    &instance_path,
                )]);
            }
            let declaration = component
                .signature()
                .iter()
                .find_map(|item| match item {
                    eqiora_lang::SignatureItem::Field(value) if value.name() == slot => Some(value),
                    _ => None,
                })
                .expect("bound authored Field requirement");
            self.record_type_structure(
                &symbol.internal_name,
                component.file,
                declaration.value_type(),
                &parameters
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone().into()))
                    .collect(),
            )
            .map_err(one_diagnostic)?;
            scope.insert_symbol(slot.clone(), symbol);
            let requirement = field_interface.field(&slot).expect("resolved requirement");
            let activation = match &requirement.activation {
                eqiora_lang::ActivationSyntax::Named(clock) => {
                    eqiora_lang::ActivationSyntax::Named(clocks[clock].clone())
                }
                other => other.clone(),
            };
            scope
                .field_evolution
                .insert(slot.clone(), (requirement.role, activation));
            scope.insert_field_type(slot, field_type);
        }
        self.record_properties(
            ComponentOccurrence {
                definition: &component,
                instance,
                instance_file,
                instance_path: &instance_path,
                display_prefix: &display_prefix,
            },
            &bindings,
        )
        .map_err(one_diagnostic)?;
        self.record_borrowed_fields(
            ComponentOccurrence {
                definition: &component,
                instance,
                instance_file,
                instance_path: &instance_path,
                display_prefix: &display_prefix,
            },
            &scope,
            &bindings,
        )
        .map_err(one_diagnostic)?;

        self.allocate_component_products(
            ComponentOccurrence {
                definition: &component,
                instance,
                instance_file,
                instance_path: &instance_path,
                display_prefix: &display_prefix,
            },
            &bindings,
            &mut scope,
            &mut identities,
        )?;

        for item in component
            .owned_items()
            .filter(|item| matches!(item, ComponentItem::Clock(_)))
            .chain(
                component
                    .owned_items()
                    .filter(|item| !matches!(item, ComponentItem::Clock(_))),
            )
        {
            match item {
                ComponentItem::Parameter(declaration) => {
                    if self.allocate_component_record_parameter(
                        ComponentOccurrence {
                            definition: &component,
                            instance,
                            instance_file,
                            instance_path: &instance_path,
                            display_prefix: &display_prefix,
                        },
                        declaration,
                        &parameters,
                        &bindings,
                        &mut scope,
                    )? {
                        continue;
                    }
                    self.allocate_parameter(
                        ComponentOccurrence {
                            definition: &component,
                            instance,
                            instance_file,
                            instance_path: &instance_path,
                            display_prefix: &display_prefix,
                        },
                        declaration,
                        &parameters,
                        &bindings,
                        &mut scope,
                    )?;
                }
                ComponentItem::Port(declaration) => {
                    let identity = self
                        .entity_identity(
                            &instance_path,
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                declaration.name(),
                            ),
                            EntityKind::Port,
                            SourceLocation::new(component.file, declaration.range()),
                            SourceLocation::new(instance_file, instance.range()),
                            bindings.clone(),
                        )
                        .map_err(one_diagnostic)?;
                    self.register_symbol(
                        display_child(&display_prefix, declaration.name()),
                        declaration.name(),
                        &identity,
                        SymbolKind::Port {
                            activation: super::scope::port_activation(
                                component.file,
                                declaration.syntax(),
                                declaration.range(),
                                &scope,
                            )
                            .map_err(one_diagnostic)?,
                            quantities: self
                                .port_quantities(
                                    declaration.syntax(),
                                    &component.namespace,
                                    component.file,
                                    declaration.range(),
                                )
                                .map_err(one_diagnostic)?,
                        },
                        &mut scope,
                    )
                    .map_err(one_diagnostic)?;
                    identities
                        .entities
                        .insert(declaration.name().to_owned(), identity);
                }
                ComponentItem::PortFamily(family) => {
                    let declaration = family.port();
                    let set = support_bindings
                        .boundary_set(family.binder().set().as_str())
                        .ok_or_else(|| {
                            vec![contextualize_diagnostic(
                                hierarchy_error(format!(
                                    "Port family `{}` has no resolved complete-exterior binding `{}`",
                                    declaration.name(),
                                    family.binder().set().as_str()
                                )),
                                &instance_path,
                            )]
                        })?;
                    for side in set.witness().sides() {
                        let boundary = *side.boundary();
                        let member = set.member(&boundary).ok_or_else(|| {
                            vec![contextualize_diagnostic(
                                hierarchy_error(
                                    "complete-exterior witness has no identity-keyed member locator",
                                ),
                                &instance_path,
                            )]
                        })?;
                        let member_bindings = boundary_family_bindings(
                            &bindings,
                            instance_file,
                            member.source_range(),
                        );
                        let identity = self
                            .boundary_family_entity_identity(
                                &instance_path,
                                definition_path(
                                    &component.namespace,
                                    "component",
                                    component.name(),
                                    declaration.name(),
                                ),
                                EntityKind::Port,
                                boundary,
                                EntitySourceOrigin {
                                    definition: SourceLocation::new(component.file, family.range()),
                                    instance: SourceLocation::new(instance_file, instance.range()),
                                    bindings: member_bindings,
                                },
                            )
                            .map_err(one_diagnostic)?;
                        self.register_port_family_member(
                            PortFamilyMemberRegistration {
                                quantities: self
                                    .port_quantities(
                                        declaration.syntax(),
                                        &component.namespace,
                                        component.file,
                                        declaration.range(),
                                    )
                                    .map_err(one_diagnostic)?
                                    .ok_or_else(|| {
                                        one_diagnostic(hierarchy_error(
                                            "physical Port family is missing named quantities",
                                        ))
                                    })?,
                                file: component.file,
                                range: family.range(),
                                display_name: boundary_family_display(
                                    &display_prefix,
                                    declaration.name(),
                                    side.axis(),
                                    side.side(),
                                ),
                                family_name: declaration.name(),
                                selector_member: family.binder().member(),
                                boundary,
                                identity: &identity,
                            },
                            &mut scope,
                        )
                        .map_err(one_diagnostic)?;
                        identities
                            .boundary_family_entities
                            .insert((declaration.name().to_owned(), boundary), identity);
                    }
                }
                ComponentItem::Field(declaration) => {
                    if self
                        .allocate_record_field(
                            &mut scope,
                            declaration,
                            records::RecordFieldOccurrence {
                                namespace: &component.namespace,
                                definition_name: component.name(),
                                instance_path: &instance_path,
                                display_prefix: &display_prefix,
                                file: component.file,
                                instance: SourceLocation::new(instance_file, instance.range()),
                                bindings: bindings.clone(),
                            },
                            &mut identities,
                        )
                        .map_err(one_diagnostic)?
                    {
                        let record = self
                            .elaborator
                            .record_for_type(&component.namespace, declaration.value_type())
                            .expect("allocated record");
                        let support = declaration
                            .domain()
                            .and_then(|domain| scope.spatial_support(domain).cloned());
                        for (name, value_type) in record.definition.members() {
                            scope.insert_field_type(
                                format!("{}.{name}", declaration.name()),
                                eqiora_schema::kernel::typing::ExpressionType::new(
                                    value_type.clone(),
                                    support.clone(),
                                ),
                            );
                        }
                        continue;
                    }
                    let support = declaration
                        .domain()
                        .and_then(|domain| scope.spatial_support(domain).cloned());
                    scope.field_evolution.insert(
                        declaration.name().to_owned(),
                        (
                            declaration.role(),
                            super::scope::rewrite_activation(
                                component.file,
                                declaration.activation(),
                                declaration.range(),
                                &scope,
                            )
                            .map_err(one_diagnostic)?,
                        ),
                    );
                    let field_type = field_expression_type(
                        component.file,
                        declaration,
                        support,
                        &scope.symbolic_parameters(),
                    )
                    .map_err(one_diagnostic)?;
                    let identity = self
                        .entity_identity(
                            &instance_path,
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                declaration.name(),
                            ),
                            EntityKind::Field,
                            SourceLocation::new(component.file, declaration.range()),
                            SourceLocation::new(instance_file, instance.range()),
                            bindings.clone(),
                        )
                        .map_err(one_diagnostic)?;
                    self.register_symbol(
                        display_child(&display_prefix, declaration.name()),
                        declaration.name(),
                        &identity,
                        SymbolKind::Field,
                        &mut scope,
                    )
                    .map_err(one_diagnostic)?;
                    identities
                        .entities
                        .insert(declaration.name().to_owned(), identity);
                    if scope
                        .insert_field_type(declaration.name().to_owned(), field_type)
                        .is_some()
                    {
                        return Err(vec![hierarchy_error(format!(
                            "duplicate flattened Field type `{}`",
                            declaration.name()
                        ))]);
                    }
                }
                ComponentItem::Observable(declaration) => {
                    let identity = self
                        .entity_identity(
                            &instance_path,
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                declaration.name(),
                            ),
                            EntityKind::Observable,
                            SourceLocation::new(component.file, declaration.range()),
                            SourceLocation::new(instance_file, instance.range()),
                            bindings.clone(),
                        )
                        .map_err(one_diagnostic)?;
                    self.register_symbol(
                        display_child(&display_prefix, declaration.name()),
                        declaration.name(),
                        &identity,
                        SymbolKind::Observable,
                        &mut scope,
                    )
                    .map_err(one_diagnostic)?;
                    identities
                        .entities
                        .insert(declaration.name().to_owned(), identity);
                }
                ComponentItem::Event(_) | ComponentItem::Clock(_) => {
                    self.allocate_local_activation(
                        item,
                        &ComponentOccurrence {
                            definition: &component,
                            instance,
                            instance_file,
                            instance_path: &instance_path,
                            display_prefix: &display_prefix,
                        },
                        &bindings,
                        &mut scope,
                        &mut identities,
                    )?;
                }
                ComponentItem::Relation(declaration) => {
                    let identity = self
                        .relation_identity(
                            &instance_path,
                            definition_path(
                                &component.namespace,
                                "component",
                                component.name(),
                                declaration.name(),
                            ),
                            SourceLocation::new(component.file, declaration.range()),
                            SourceLocation::new(instance_file, instance.range()),
                            bindings.clone(),
                        )
                        .map_err(one_diagnostic)?;
                    self.register_symbol(
                        display_child(&display_prefix, declaration.name()),
                        declaration.name(),
                        &identity.entity,
                        SymbolKind::Relation,
                        &mut scope,
                    )
                    .map_err(one_diagnostic)?;
                    identities
                        .relations
                        .insert(declaration.name().to_owned(), identity);
                }
                ComponentItem::RelationFamily(family) => {
                    if component.owned_items().any(|item| matches!(item, ComponentItem::IndexSet(set) if set.name() == family.binder().set().as_str())) {
                        continue;
                    }
                    let declaration = family.relation();
                    let set = support_bindings
                        .boundary_set(family.binder().set().as_str())
                        .ok_or_else(|| {
                            vec![contextualize_diagnostic(
                                hierarchy_error(format!(
                                    "Relation family `{}` has no resolved complete-exterior binding `{}`",
                                    declaration.name(),
                                    family.binder().set().as_str()
                                )),
                                &instance_path,
                            )]
                        })?;
                    for side in set.witness().sides() {
                        let boundary = *side.boundary();
                        let member = set.member(&boundary).ok_or_else(|| {
                            vec![contextualize_diagnostic(
                                hierarchy_error(
                                    "complete-exterior witness has no identity-keyed member locator",
                                ),
                                &instance_path,
                            )]
                        })?;
                        let identity = self
                            .boundary_family_relation_identity(
                                &instance_path,
                                definition_path(
                                    &component.namespace,
                                    "component",
                                    component.name(),
                                    declaration.name(),
                                ),
                                boundary,
                                EntitySourceOrigin {
                                    definition: SourceLocation::new(component.file, family.range()),
                                    instance: SourceLocation::new(instance_file, instance.range()),
                                    bindings: boundary_family_bindings(
                                        &bindings,
                                        instance_file,
                                        member.source_range(),
                                    ),
                                },
                            )
                            .map_err(one_diagnostic)?;
                        self.register_family_relation_display(
                            boundary_family_display(
                                &display_prefix,
                                declaration.name(),
                                side.axis(),
                                side.side(),
                            ),
                            &identity,
                        )
                        .map_err(one_diagnostic)?;
                        identities
                            .boundary_family_relations
                            .insert((declaration.name().to_owned(), boundary), identity);
                    }
                }
                ComponentItem::Domain(_)
                | ComponentItem::Coordinate(_)
                | ComponentItem::Let(_)
                | ComponentItem::Initial(_)
                | ComponentItem::Connection(_)
                | ComponentItem::BoundaryConnection(_)
                | ComponentItem::IndexSet(_)
                | ComponentItem::Instance(_) => {}
                _ => {
                    return Err(vec![source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        component.file,
                        component.range(),
                        "component item is newer than hierarchy elaboration",
                    )]);
                }
            }
        }

        for item in component.owned_items() {
            if let ComponentItem::Field(field) = item {
                let activation = super::scope::rewrite_activation(
                    component.file,
                    field.activation(),
                    field.range(),
                    &scope,
                )
                .map_err(one_diagnostic)?;
                if let Some(record) = self
                    .elaborator
                    .record_for_type(&component.namespace, field.value_type())
                {
                    for (name, _) in record.definition.members() {
                        scope.field_evolution.insert(
                            format!("{}.{name}", field.name()),
                            (field.role(), activation.clone()),
                        );
                    }
                } else {
                    scope
                        .field_evolution
                        .insert(field.name().to_owned(), (field.role(), activation));
                }
            }
        }

        for item in component.owned_items() {
            match item {
                ComponentItem::Port(declaration)
                    if matches!(
                        declaration.syntax(),
                        PortSyntax::ScalarPhysicalConnector { .. }
                            | PortSyntax::FieldPhysical { .. }
                    ) =>
                {
                    let identity = identities.entities[declaration.name()].clone();
                    self.register_physical_port_occurrence(
                        identity,
                        display_child(&display_prefix, declaration.name()),
                        instance_path.clone(),
                        declaration.visibility() == VisibilitySyntax::Public,
                        None,
                    )
                    .map_err(one_diagnostic)?;
                }
                ComponentItem::PortFamily(family) => {
                    let declaration = family.port();
                    let set = support_bindings
                        .boundary_set(family.binder().set().as_str())
                        .ok_or_else(|| {
                            vec![contextualize_diagnostic(
                                hierarchy_error(format!(
                                    "Port family `{}` has no resolved complete-exterior binding `{}`",
                                    declaration.name(),
                                    family.binder().set().as_str()
                                )),
                                &instance_path,
                            )]
                        })?;
                    for side in set.witness().sides() {
                        let boundary = *side.boundary();
                        let identity = identities
                            .boundary_family_entities
                            .get(&(declaration.name().to_owned(), boundary))
                            .cloned()
                            .ok_or_else(|| {
                                vec![contextualize_diagnostic(
                                    hierarchy_error(format!(
                                        "Port family `{}` member identity was not allocated",
                                        declaration.name()
                                    )),
                                    &instance_path,
                                )]
                            })?;
                        self.register_physical_port_occurrence(
                            identity,
                            boundary_family_display(
                                &display_prefix,
                                declaration.name(),
                                side.axis(),
                                side.side(),
                            ),
                            instance_path.clone(),
                            declaration.visibility() == VisibilitySyntax::Public,
                            None,
                        )
                        .map_err(one_diagnostic)?;
                    }
                }
                _ => {}
            }
        }

        self.allocate_component_lets(&mut scope, &component)
            .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;
        self.allocate_nominals(
            &mut scope,
            &component.namespace,
            component.name(),
            &instance_path,
            &component
                .owned_items()
                .filter_map(|item| match item {
                    ComponentItem::IndexSet(value) => Some(value),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            component.file,
        )
        .map_err(one_diagnostic)?;

        self.expand_children(
            &component.namespace,
            component.owned_items().filter_map(|item| match item {
                ComponentItem::Instance(instance) => Some(instance),
                _ => None,
            }),
            component.file,
            &instance_path,
            &display_prefix,
            &mut scope,
        )?;

        self.allocate_component_expression_bindings(&mut scope, &component)
            .map_err(|errors| contextualize_diagnostics(errors, &instance_path))?;

        self.materialize_component_items(
            ComponentOccurrence {
                definition: &component,
                instance,
                instance_file,
                instance_path: &instance_path,
                display_prefix: &display_prefix,
            },
            &scope,
            &identities,
            &support_bindings,
        )
        .map_err(|error| vec![contextualize_diagnostic(error, &instance_path)])?;

        let public_ports = component
            .owned_items()
            .filter_map(|item| match item {
                ComponentItem::Port(port) if port.visibility() == VisibilitySyntax::Public => scope
                    .symbol(port.name())
                    .cloned()
                    .map(|symbol| (port.name().to_owned(), symbol)),
                _ => None,
            })
            .collect();
        let public_port_families = component
            .owned_items()
            .filter_map(|item| match item {
                ComponentItem::PortFamily(family)
                    if family.port().visibility() == VisibilitySyntax::Public =>
                {
                    scope
                        .port_family(family.port().name())
                        .cloned()
                        .map(|index| (family.port().name().to_owned(), index))
                }
                _ => None,
            })
            .collect();
        self.record_index_dependencies(&scope)
            .map_err(one_diagnostic)?;
        Ok(InstanceInterface::with_public_port_families(
            public_ports,
            public_port_families,
        ))
    }
}
