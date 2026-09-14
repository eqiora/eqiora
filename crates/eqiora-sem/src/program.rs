//! Whole-model validation and immutable interpreter input.

mod conservation;
mod diagnostics;
use diagnostics::{
    clock_error, expression_path, kernel_error, kernel_path, model_path, relation_dimension_error,
};
pub(crate) mod geometry_admission;
mod nominal_values;
mod numerical_residuals;
mod observables;
mod record_admission;
mod relation_admission;
mod relation_operands;
pub(crate) mod signal_activation;
pub(crate) mod signal_connections;
mod snapshot_admission;
use relation_admission::validate_relations;
mod spatial_domains;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, GraphPath, Id, OntologyId, RawId};
use eqiora_graph::{Edge, EdgeKind, Revision, Snapshot};
use eqiora_schema::Model;
use eqiora_schema::kernel::scalar_connection::{
    ScalarConnectionKind, ScalarConnectionViolation, ScalarPortContract, validate_scalar_connection,
};
use eqiora_schema::kernel::typing::{
    self, ExpressionType, RootContract, SpatialSupport, TypeViolation, TypedResidual,
    TypedResidualError,
};
use eqiora_schema::kernel::{
    ActivationKind, AxisBounds, BoundaryPhysicalConnectionViolation, ClockKind,
    ConnectionSemantics, DomainKind, ExprDag, ExprNode, KernelNode,
    SpatialPeriodicBoundaryViolation, SymbolRef, validate_boundary_physical_connection,
    validate_spatial_periodic_boundary_connection,
};

use geometry_admission::{GeometryBoundaryEmbedding, GeometryBoundaryJunction};
use spatial_domains::field_support;

/// A completely validated, immutable Semantic Kernel model.
///
/// This is the only model form accepted by the reference interpreter. It is
/// compiled from one [`Snapshot`] revision, owns the selected definitions,
/// and cannot observe later Graph Federation commits.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelProgram {
    revision: Revision,
    model: OntologyId<Model>,
    nodes: BTreeMap<RawId, KernelNode>,
    values: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    edges: Vec<Edge>,
    boundary: BTreeSet<RawId>,
    spatial_supports: BTreeMap<RawId, SpatialSupport<RawId>>,
    cartesian_bounds: BTreeMap<RawId, Vec<AxisBounds>>,
    geometry_boundary_junctions: BTreeMap<RawId, GeometryBoundaryJunction>,
}

impl KernelProgram {
    pub(crate) const fn node_definitions(&self) -> &BTreeMap<RawId, KernelNode> {
        &self.nodes
    }

    pub(crate) const fn cartesian_bounds_map(&self) -> &BTreeMap<RawId, Vec<AxisBounds>> {
        &self.cartesian_bounds
    }

    pub(crate) const fn geometry_boundary_junctions(
        &self,
    ) -> &BTreeMap<RawId, GeometryBoundaryJunction> {
        &self.geometry_boundary_junctions
    }

    /// Graph Federation revision captured by this program.
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }

    /// Selected Standard Ontology `ModelView` identifier.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Look up a validated kernel definition.
    #[must_use]
    pub fn node(&self, id: RawId) -> Option<&KernelNode> {
        self.nodes.get(&id)
    }

    /// Revision-local Parameter value.
    ///
    /// This captures `SetValue` operations visible in the source snapshot;
    /// it is intentionally separate from the immutable node definition.
    #[must_use]
    pub fn value(&self, id: RawId) -> Option<DynQuantity> {
        self.typed_value(id)
            .and_then(eqiora_core::ValueLiteral::real_scalar_value)
    }

    /// Complete revision-local Parameter value, retaining every typed component.
    #[must_use]
    pub fn typed_value(&self, id: RawId) -> Option<&eqiora_core::ValueLiteral> {
        self.values.get(&id)
    }

    /// Kernel definitions in deterministic `(kind, ULID)` order.
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = &KernelNode> {
        self.nodes.values()
    }

    /// Validated internal model edges in deterministic order.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Model boundary Ports in deterministic order.
    #[must_use]
    pub fn boundary(&self) -> &BTreeSet<RawId> {
        &self.boundary
    }

    /// Resolve the accepted Cartesian bounds for one Domain.
    ///
    /// This is the single metric projection of fixed and Parameter-backed
    /// coordinate recipes. Callers must not interpret raw sources
    /// independently.
    ///
    /// # Errors
    /// Returns `EQ0302` when `domain` is absent or is not an accepted
    /// Cartesian box.
    pub fn resolved_cartesian_bounds(
        &self,
        domain: Id<kinds::Domain>,
    ) -> Result<&[AxisBounds], Diagnostic> {
        self.cartesian_bounds
            .get(&domain.erase())
            .map(Vec::as_slice)
            .ok_or_else(|| {
                kernel_error(
                    domain.erase(),
                    "selected Domain is not an accepted Cartesian box",
                )
            })
    }

    /// Reconstruct the exact typed residual owned by one accepted Relation.
    ///
    /// The returned proof is the only input accepted by componentwise
    /// Operator scalarization. Symbol types and spatial support are therefore
    /// derived from this immutable program rather than supplied as a parallel
    /// caller-owned shape array.
    ///
    /// # Errors
    /// Returns deterministic diagnostics when `relation` is absent or is not
    /// a Relation in this accepted program. Re-inference failures indicate a
    /// violated internal invariant and retain the original expression path.
    pub fn typed_relation_residual(
        &self,
        relation: Id<kinds::Relation>,
    ) -> Result<TypedResidual<RawId>, Vec<Diagnostic>> {
        let relation_id = relation.erase();
        let Some(KernelNode::Relation(definition)) = self.nodes.get(&relation_id) else {
            return Err(vec![kernel_error(
                relation_id,
                "selected node is not an accepted Relation",
            )]);
        };
        let scopes = edge_targets(&self.edges, relation_id, EdgeKind::AppliesOn);
        let scope = match scopes.len() {
            0 => None,
            1 => scopes.first().copied(),
            _ => {
                return Err(vec![kernel_error(
                    relation_id,
                    "accepted Relation unexpectedly has multiple spatial scopes",
                )]);
            }
        };
        self.type_derived_residual(
            self.numerical_residuals(relation_id)
                .map_err(|error| vec![error])?,
            relation_id,
            scope,
            if definition.is_initial() {
                RootContract::InitialResiduals
            } else {
                RootContract::ComponentwiseResidual
            },
        )
    }

    pub(crate) fn execution_symbol_type(
        &self,
        symbol: SymbolRef,
    ) -> Option<eqiora_core::ValueType> {
        symbol_type(symbol, &self.nodes, &self.edges, &self.spatial_supports)
            .ok()
            .map(|inferred| inferred.value_type)
    }

    pub(crate) fn type_derived_residual(
        &self,
        expression: ExprDag,
        owner: RawId,
        scope: Option<RawId>,
        root_contract: RootContract,
    ) -> Result<TypedResidual<RawId>, Vec<Diagnostic>> {
        let relation_support = scope.and_then(|scope| self.spatial_supports.get(&scope).cloned());
        TypedResidual::infer(expression, relation_support, root_contract, |symbol| {
            symbol_type(symbol, &self.nodes, &self.edges, &self.spatial_supports)
        })
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| typed_residual_diagnostic(owner, error))
                .collect()
        })
    }

    pub(crate) fn type_boundary_junction_residual(
        &self,
        expression: ExprDag,
        connection: Id<kinds::Connection>,
        dimensions: usize,
    ) -> Result<TypedResidual<RawId>, Vec<Diagnostic>> {
        let interface = SpatialSupport::Interface {
            connection: connection.erase(),
            dimensions,
        };
        TypedResidual::infer(
            expression,
            Some(interface.clone()),
            RootContract::ComponentwiseResidual,
            |symbol| {
                let mut inferred =
                    symbol_type(symbol, &self.nodes, &self.edges, &self.spatial_supports)?;
                if matches!(symbol, SymbolRef::PortTrace(_) | SymbolRef::PortFlux(_)) {
                    inferred.support = Some(interface.clone());
                }
                Ok(inferred)
            },
        )
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| typed_residual_diagnostic(connection.erase(), error))
                .collect()
        })
    }
}

fn validate_closed_topology(
    snapshot: &Snapshot,
    members: &BTreeSet<RawId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for &member in members {
        for edge in snapshot.outgoing(member).filter(|edge| {
            matches!(
                edge.kind(),
                EdgeKind::DefinedOn
                    | EdgeKind::AppliesOn
                    | EdgeKind::BoundaryOf
                    | EdgeKind::DependsOn
                    | EdgeKind::StructurallyDependsOn
                    | EdgeKind::HasPort
                    | EdgeKind::Activates
                    | EdgeKind::Connects
                    | EdgeKind::ClockedBy
            )
        }) {
            if !members.contains(&edge.to()) {
                diagnostics.push(kernel_error(
                    member,
                    format!(
                        "{:?} edge from {member} leaves the selected ModelView at {}",
                        edge.kind(),
                        edge.to()
                    ),
                ));
            }
        }
    }
}

fn validate_activations(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    spatial_supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&id, node) in nodes {
        let KernelNode::Activation(activation) = node else {
            continue;
        };

        let relations = edge_targets(edges, id, EdgeKind::Activates);
        if relations.is_empty() {
            diagnostics.push(kernel_error(
                id,
                "Activation must activate at least one Relation",
            ));
        }

        let clocks = edge_targets(edges, id, EdgeKind::ClockedBy);
        match activation.kind() {
            ActivationKind::Continuous => {
                require_no_clocks(id, &clocks, "continuous", diagnostics);
            }
            ActivationKind::Periodic => {
                if clocks.len() != 1 {
                    diagnostics.push(clock_error(
                        id,
                        format!(
                            "periodic Activation requires exactly one ClockDomain, found {}",
                            clocks.len()
                        ),
                    ));
                } else if !matches!(
                    nodes.get(clocks.first().expect("one clock was checked")),
                    Some(KernelNode::ClockDomain(clock))
                        if matches!(clock.kind(), ClockKind::Periodic { .. })
                ) {
                    diagnostics.push(clock_error(
                        id,
                        "periodic Activation must be ClockedBy a periodic ClockDomain",
                    ));
                }
            }
            ActivationKind::Event { guard, .. } => {
                require_no_clocks(id, &clocks, "event", diagnostics);
                validate_expression(
                    guard,
                    id,
                    TypingEnvironment {
                        nodes,
                        edges,
                        spatial_supports,
                    },
                    None,
                    RootContract::ScalarActivation,
                    diagnostics,
                );
            }
            ActivationKind::Guard { guard } => {
                require_no_clocks(id, &clocks, "guard", diagnostics);
                validate_expression(
                    guard,
                    id,
                    TypingEnvironment {
                        nodes,
                        edges,
                        spatial_supports,
                    },
                    None,
                    RootContract::ScalarActivation,
                    diagnostics,
                );
            }
            _ => diagnostics.push(kernel_error(
                id,
                "Activation kind is newer than this semantic interpreter",
            )),
        }
    }
}

fn require_no_clocks(
    activation: RawId,
    clocks: &BTreeSet<RawId>,
    name: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if !clocks.is_empty() {
        diagnostics.push(clock_error(
            activation,
            format!("{name} Activation must not have ClockedBy edges"),
        ));
    }
}

fn validate_connections(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    boundary: &BTreeSet<RawId>,
    cartesian_bounds: &BTreeMap<RawId, Vec<AxisBounds>>,
    geometry_boundary_junctions: &BTreeMap<RawId, GeometryBoundaryJunction>,
    geometry_boundary_embeddings: &BTreeMap<RawId, GeometryBoundaryEmbedding>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if let Err(errors) = signal_connections::sources(nodes, edges, boundary) {
        diagnostics.extend(errors);
    }
    let mut memberships = BTreeMap::new();
    for (&id, node) in nodes {
        let KernelNode::Connection(connection) = node else {
            continue;
        };
        let ports = edge_targets(edges, id, EdgeKind::Connects);
        for &port in &ports {
            let signal_source = matches!(connection.semantics(), ConnectionSemantics::Signal { driver } if driver.erase() == port);
            if !signal_source && let Some(previous) = memberships.insert(port, id) {
                diagnostics.push(kernel_error(
                    id,
                    format!(
                        "Port {port} belongs to both Connection {previous} and Connection {id}"
                    ),
                ));
            }
        }
        let definitions = ports
            .iter()
            .filter_map(|port| match nodes.get(port) {
                Some(KernelNode::Port(definition)) => Some(definition),
                _ => {
                    diagnostics.push(kernel_error(
                        id,
                        format!("Connects target {port} has no Port definition"),
                    ));
                    None
                }
            })
            .collect::<Vec<_>>();
        if definitions.len() != ports.len() {
            continue;
        }

        let boundary_members = definitions
            .iter()
            .filter(|port| port.boundary_physical_contract().is_some())
            .count();
        if boundary_members > 0 {
            if boundary_members != definitions.len() {
                diagnostics.push(kernel_error(
                    id,
                    "conserving Connection cannot mix boundary-physical and scalar Port families",
                ));
                continue;
            }
            let geometry_members = definitions
                .iter()
                .filter_map(|port| port.boundary_physical_contract())
                .filter(|(_, boundary)| {
                    geometry_boundary_embeddings.contains_key(&boundary.erase())
                })
                .count();
            if geometry_members > 0 {
                if geometry_members != definitions.len()
                    || !geometry_boundary_junctions.contains_key(&id)
                {
                    // Closed Geometry admission owns the precise diagnostic.
                }
                continue;
            }
            let contracts = ports
                .iter()
                .filter_map(|port| {
                    crate::boundary_physical::resolve_port_contract(
                        *port,
                        nodes,
                        edges,
                        cartesian_bounds,
                    )
                    .ok()
                    .flatten()
                })
                .collect::<Vec<_>>();
            if contracts.len() != ports.len() {
                // Port-local validation owns the precise contract diagnostic.
                // Connection validation only compares already valid members.
                continue;
            }
            match connection.semantics() {
                ConnectionSemantics::Conserving => {
                    if let Err(violation) = validate_boundary_physical_connection(&contracts) {
                        diagnostics.push(boundary_connection_error(id, violation));
                    }
                }
                ConnectionSemantics::SpatialPeriodic => {
                    if let Err(violation) =
                        validate_spatial_periodic_boundary_connection(&contracts)
                    {
                        diagnostics.push(spatial_periodic_connection_error(id, violation));
                    }
                }
                ConnectionSemantics::Signal { .. } => diagnostics.push(kernel_error(
                    id,
                    "boundary-physical Ports require conserving or spatial-periodic Connection semantics",
                )),
                _ => diagnostics.push(kernel_error(
                    id,
                    "boundary-physical Connection semantics are newer than this semantic validator",
                )),
            }
            continue;
        }

        let kind = match connection.semantics() {
            ConnectionSemantics::Signal { .. } => ScalarConnectionKind::Signal,
            ConnectionSemantics::Conserving => ScalarConnectionKind::Conserving,
            ConnectionSemantics::SpatialPeriodic => {
                diagnostics.push(kernel_error(
                    id,
                    "spatial-periodic Connection requires boundary-physical Ports",
                ));
                continue;
            }
            _ => {
                diagnostics.push(kernel_error(
                    id,
                    "Connection semantics are newer than this semantic interpreter",
                ));
                continue;
            }
        };
        let Some(contracts) = definitions
            .iter()
            .map(|port| {
                let mut contract = semantic_scalar_port_contract(port)?;
                if let ConnectionSemantics::Signal { driver } = connection.semantics()
                    && let ScalarPortContract::Signal { direction, .. } = &mut contract
                {
                    *direction = if port.id() == driver {
                        eqiora_schema::kernel::SignalDirection::Output
                    } else {
                        eqiora_schema::kernel::SignalDirection::Input
                    };
                }
                Some(contract)
            })
            .collect::<Option<Vec<_>>>()
        else {
            diagnostics.push(kernel_error(
                id,
                "Connection contains a Port payload newer than this semantic validator",
            ));
            continue;
        };
        if kind == ScalarConnectionKind::Signal
            && let Some(first) = definitions.first()
        {
            let first = first.id().erase();
            for definition in &definitions[1..] {
                let other = definition.id().erase();
                if edge_targets(edges, first, EdgeKind::ClockedBy)
                    != edge_targets(edges, other, EdgeKind::ClockedBy)
                    || edge_targets(edges, first, EdgeKind::DefinedOn)
                        != edge_targets(edges, other, EdgeKind::DefinedOn)
                {
                    diagnostics.push(kernel_error(id, "signal Connection requires the same exact activation and spatial support; use an explicit transition"));
                }
            }
        }
        if let Err(violation) = validate_scalar_connection(kind, &contracts) {
            diagnostics.push(scalar_connection_error(id, violation));
            continue;
        }

        if kind != ScalarConnectionKind::Conserving {
            continue;
        }
        let Some(domain) = definitions[0]
            .physical_domain()
            .map(|domain| domain.erase())
        else {
            continue;
        };
        if !matches!(
            nodes.get(&domain),
            Some(KernelNode::Domain(domain))
                if matches!(domain.kind(), DomainKind::ScalarPhysical { .. })
        ) {
            diagnostics.push(kernel_error(
                id,
                "scalar physical Connection refers to a non-physical Domain",
            ));
        }
    }

    for (&id, node) in nodes {
        let KernelNode::Port(port) = node else {
            continue;
        };
        if (port.physical_domain().is_some() || port.boundary_physical_contract().is_some())
            && !memberships.contains_key(&id)
        {
            diagnostics.push(kernel_error(
                id,
                "physical Port requires exactly one explicit conserving or spatial-periodic Connection",
            ));
        }
    }
}

fn spatial_periodic_connection_error(
    connection: RawId,
    violation: SpatialPeriodicBoundaryViolation,
) -> Diagnostic {
    match violation {
        SpatialPeriodicBoundaryViolation::WrongPortCount { found } => kernel_error(
            connection,
            format!("spatial-periodic Connection requires exactly two Ports, found {found}"),
        ),
        SpatialPeriodicBoundaryViolation::ConnectorMismatch => kernel_error(
            connection,
            "spatial-periodic Ports must share one exact Connector identity",
        ),
        SpatialPeriodicBoundaryViolation::ParentMismatch => kernel_error(
            connection,
            "spatial-periodic Ports must belong to one exact parent Domain",
        ),
        SpatialPeriodicBoundaryViolation::NormalAxisMismatch => kernel_error(
            connection,
            "spatial-periodic Ports must lie on parallel sides of one Cartesian axis",
        ),
        SpatialPeriodicBoundaryViolation::SidesNotOpposite => kernel_error(
            connection,
            "spatial-periodic Ports require exactly one lower and one upper side",
        ),
        SpatialPeriodicBoundaryViolation::GeometryMismatch => kernel_error(
            connection,
            "spatial-periodic Port supports do not define one exact positive Cartesian translation",
        ),
    }
}

fn boundary_connection_error(
    connection: RawId,
    violation: BoundaryPhysicalConnectionViolation,
) -> Diagnostic {
    match violation {
        BoundaryPhysicalConnectionViolation::TooFewPorts { found } => kernel_error(
            connection,
            format!("boundary-physical Connection requires at least two Ports, found {found}"),
        ),
        BoundaryPhysicalConnectionViolation::ConnectorMismatch => kernel_error(
            connection,
            "boundary-physical Ports must share one exact Connector identity",
        ),
        BoundaryPhysicalConnectionViolation::NoncoincidentBoundaries => kernel_error(
            connection,
            "boundary-physical Ports must lie on one coincident Cartesian boundary",
        ),
    }
}

fn semantic_scalar_port_contract(
    port: &eqiora_schema::kernel::PortDef,
) -> Option<ScalarPortContract<RawId>> {
    if let Some((direction, value_type)) = port.signal_contract() {
        Some(ScalarPortContract::Signal {
            direction,
            value_type: value_type.clone(),
        })
    } else {
        port.physical_domain()
            .map(|domain| ScalarPortContract::ScalarPhysical {
                nominal: domain.erase(),
            })
    }
}

fn scalar_connection_error(connection: RawId, violation: ScalarConnectionViolation) -> Diagnostic {
    match violation {
        ScalarConnectionViolation::SignalTypeMismatch => relation_dimension_error(
            connection,
            "connected signal Ports must have identical physical dimensions and compatible scalar domains, shapes and frames",
        ),
        ScalarConnectionViolation::TooFewPorts { found } => kernel_error(
            connection,
            format!("Connection requires at least two Ports, found {found}"),
        ),
        ScalarConnectionViolation::SignalDirections {
            outputs, inputs, ..
        } => kernel_error(
            connection,
            format!(
                "signal Connection requires one output and one or more inputs; found {outputs} outputs and {inputs} inputs"
            ),
        ),
        ScalarConnectionViolation::MixedConservingFamilies => kernel_error(
            connection,
            "conserving Connection cannot mix signal and scalar physical Ports",
        ),
        ScalarConnectionViolation::PhysicalNominalMismatch => kernel_error(
            connection,
            "scalar physical Ports in a conserving Connection must share one exact Domain",
        ),
    }
}

#[derive(Clone, Copy)]
struct TypingEnvironment<'a> {
    nodes: &'a BTreeMap<RawId, KernelNode>,
    edges: &'a [Edge],
    spatial_supports: &'a BTreeMap<RawId, SpatialSupport<RawId>>,
}

fn validate_expression(
    expression: &ExprDag,
    owner: RawId,
    environment: TypingEnvironment<'_>,
    scope: Option<RawId>,
    root_contract: RootContract,
    diagnostics: &mut Vec<Diagnostic>,
) -> BTreeSet<RawId> {
    for node in expression.nodes() {
        if let ExprNode::Constant(value) = node {
            nominal_values::check_literal(owner, value, environment.nodes, diagnostics);
        }
    }
    let symbols = expression
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(symbol) => symbol_id(*symbol),
            ExprNode::Sample { clock, .. } => Some(clock.erase()),
            _ => None,
        })
        .collect();
    let relation_support =
        scope.and_then(|scope| environment.spatial_supports.get(&scope).cloned());
    match TypedResidual::infer(
        expression.clone(),
        relation_support.clone(),
        root_contract,
        |symbol| {
            symbol_type(
                symbol,
                environment.nodes,
                environment.edges,
                environment.spatial_supports,
            )
        },
    ) {
        Ok(typed) => {
            if let Some(KernelNode::Relation(relation)) = environment.nodes.get(&owner) {
                match relation.meaning() {
                    eqiora_schema::kernel::RelationMeaning::Conditions(_) => {
                        if let Err(error) =
                            relation.validate_conditions(&typed, relation_support.as_ref())
                        {
                            diagnostics.push(typed_residual_diagnostic(
                                owner,
                                TypedResidualError::Type {
                                    node_index: 0,
                                    error,
                                },
                            ));
                        }
                    }
                    eqiora_schema::kernel::RelationMeaning::Conservation(terms) => {
                        conservation::validate_conservation_types(
                            owner,
                            *terms,
                            &typed,
                            relation_support.as_ref(),
                            environment,
                            diagnostics,
                        );
                    }
                }
            }
        }
        Err(errors) => diagnostics.extend(
            errors
                .into_iter()
                .map(|error| typed_residual_diagnostic(owner, error)),
        ),
    }
    symbols
}

fn typed_residual_diagnostic(
    owner: RawId,
    error: TypedResidualError<RawId, SymbolTypeError>,
) -> Diagnostic {
    match error {
        TypedResidualError::Symbol {
            node_index,
            symbol,
            error: SymbolTypeError::Missing,
        } => {
            let message = symbol_id(symbol).map_or_else(
                || "expression symbol is not supported by this semantic interpreter".to_owned(),
                |id| format!("expression symbol {id} is outside the selected ModelView"),
            );
            Diagnostic::error(codes::UNRESOLVED_SYMBOL, message)
                .with_graph_path(expression_path(owner, node_index))
        }
        TypedResidualError::Symbol {
            node_index,
            symbol: _,
            error: SymbolTypeError::Typing(error),
        }
        | TypedResidualError::Type { node_index, error } => {
            type_violation_diagnostic(owner, node_index, &error)
        }
        TypedResidualError::Symbol {
            node_index,
            error: SymbolTypeError::WrongFieldRole,
            ..
        } => kernel_error(
            owner,
            "Derivative requires continuous state; Pre/Next require state",
        )
        .with_graph_path(expression_path(owner, node_index)),
        TypedResidualError::Symbol {
            node_index,
            symbol: _,
            error: SymbolTypeError::WrongPortContract,
        } => kernel_error(
            owner,
            "expression symbol does not satisfy its signal or physical Port contract",
        )
        .with_graph_path(expression_path(owner, node_index)),
    }
}

fn type_violation_diagnostic(
    owner: RawId,
    expression_id: u32,
    error: &TypeViolation<RawId>,
) -> Diagnostic {
    let diagnostic = if error.is_dimension_or_shape() {
        relation_dimension_error(owner, error.to_string())
    } else {
        kernel_error(owner, error.to_string())
    };
    diagnostic.with_graph_path(expression_path(owner, expression_id))
}

fn symbol_id(symbol: SymbolRef) -> Option<RawId> {
    match symbol {
        SymbolRef::Field(id)
        | SymbolRef::Derivative(id)
        | SymbolRef::Pre(id)
        | SymbolRef::Next(id) => Some(id.erase()),
        SymbolRef::Parameter(id) => Some(id.erase()),
        SymbolRef::Port(id)
        | SymbolRef::Across(id)
        | SymbolRef::Through(id)
        | SymbolRef::PortTrace(id)
        | SymbolRef::PortFlux(id) => Some(id.erase()),
        SymbolRef::Time => None,
        _ => None,
    }
}

fn symbol_type(
    symbol: SymbolRef,
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    spatial_supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
) -> Result<ExpressionType<RawId>, SymbolTypeError> {
    match symbol {
        SymbolRef::Field(id) | SymbolRef::Pre(id) | SymbolRef::Next(id) => {
            match nodes.get(&id.erase()) {
                Some(KernelNode::Field(field)) => {
                    if matches!(symbol, SymbolRef::Pre(_) | SymbolRef::Next(_))
                        && (field.role() != eqiora_schema::kernel::FieldRole::State)
                    {
                        return Err(SymbolTypeError::WrongFieldRole);
                    }
                    Ok(ExpressionType::new(
                        field.value_type().clone(),
                        field_support(id.erase(), edges, spatial_supports),
                    ))
                }
                _ => Err(SymbolTypeError::Missing),
            }
        }
        SymbolRef::Derivative(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Field(field))
                if field.role() != eqiora_schema::kernel::FieldRole::State
                    || !edge_targets(edges, id.erase(), EdgeKind::ClockedBy).is_empty() =>
            {
                Err(SymbolTypeError::WrongFieldRole)
            }
            Some(KernelNode::Field(field)) => typing::time_derivative(&ExpressionType::new(
                field.value_type().clone(),
                field_support(id.erase(), edges, spatial_supports),
            ))
            .map_err(SymbolTypeError::Typing),
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Parameter(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Parameter(parameter)) => {
                Ok(ExpressionType::new(parameter.value_type().clone(), None))
            }
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Port(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Port(port)) => port
                .signal_contract()
                .map(|(_, value_type)| {
                    ExpressionType::new(
                        value_type.clone(),
                        field_support(id.erase(), edges, spatial_supports),
                    )
                })
                .ok_or(SymbolTypeError::WrongPortContract),
            _ => Err(SymbolTypeError::Missing),
        },
        SymbolRef::Across(id) | SymbolRef::Through(id) => {
            let Some(KernelNode::Port(port)) = nodes.get(&id.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let Some(domain) = port.physical_domain() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(KernelNode::Domain(domain)) = nodes.get(&domain.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let DomainKind::ScalarPhysical {
                across_type,
                through_type,
            } = domain.kind()
            else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let value_type = if matches!(symbol, SymbolRef::Across(_)) {
                across_type
            } else {
                through_type
            };
            Ok(ExpressionType::new(value_type.clone(), None))
        }
        SymbolRef::PortTrace(id) | SymbolRef::PortFlux(id) => {
            let Some(KernelNode::Port(port)) = nodes.get(&id.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let Some((connector, boundary)) = port.boundary_physical_contract() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(KernelNode::Domain(connector)) = nodes.get(&connector.erase()) else {
                return Err(SymbolTypeError::Missing);
            };
            let DomainKind::BoundaryPhysical { connector } = connector.kind() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let Some(support) = spatial_supports.get(&boundary.erase()).cloned() else {
                return Err(SymbolTypeError::WrongPortContract);
            };
            let value_type = if matches!(symbol, SymbolRef::PortTrace(_)) {
                connector.trace_type()
            } else {
                connector.flux_type()
            };
            Ok(ExpressionType::new(value_type.clone(), Some(support)))
        }
        SymbolRef::Time => Ok(ExpressionType::scalar(
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension"),
            None,
        )),
        _ => Err(SymbolTypeError::Missing),
    }
}

enum SymbolTypeError {
    Missing,
    Typing(TypeViolation<RawId>),
    WrongPortContract,
    WrongFieldRole,
}

fn edge_targets(edges: &[Edge], from: RawId, kind: EdgeKind) -> BTreeSet<RawId> {
    edges
        .iter()
        .filter(|edge| edge.from() == from && edge.kind() == kind)
        .map(Edge::to)
        .collect()
}
