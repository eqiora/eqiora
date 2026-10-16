use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::ValueFrame;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id, RawId};
use eqiora_graph::{Edge, EdgeKind};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{
    ActivationKind, AxisBounds, BoundaryPhysicalPortContract, CartesianBoundaryEmbedding,
    CartesianPeriodicBoundaryIdentification, ConnectionSemantics, DomainKind, ExprDag,
    ExprDagBuilder, ExprNode, KernelNode, SymbolRef,
};

use crate::KernelProgram;

/// Derived pointwise junction law for one validated field-valued Connection.
///
/// Every shaped root means componentwise zero; Operator lowering owns the
/// scalar expansion. Coincident junctions use the smallest Port ID as anchor.
/// Spatial-periodic pairs use the lower-side Port and interpret the upper
/// symbols after pullback through the derived translation.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryJunctionResidual {
    connection: Id<kinds::Connection>,
    geometry: BoundaryJunctionGeometry,
    typed: TypedResidual<RawId>,
}

/// Mesh-independent chart relation used by one boundary junction residual.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryJunctionGeometry {
    /// All Ports denote one coincident point set.
    Coincident,
    /// Upper-side values are pulled back through one derived translation.
    CartesianPeriodic(CartesianPeriodicBoundaryIdentification),
}

impl BoundaryJunctionResidual {
    /// Exact Connection whose closed law was derived.
    #[must_use]
    pub const fn connection(&self) -> Id<kinds::Connection> {
        self.connection
    }

    /// Exact chart relation under which the pointwise residual is evaluated.
    #[must_use]
    pub const fn geometry(&self) -> &BoundaryJunctionGeometry {
        &self.geometry
    }

    /// Pointwise shaped expression DAG.
    #[must_use]
    pub const fn dag(&self) -> &ExprDag {
        self.typed.expression()
    }

    /// Exact semantic typing proof consumed by Operator lowering.
    #[must_use]
    pub const fn typed(&self) -> &TypedResidual<RawId> {
        &self.typed
    }
}

impl KernelProgram {
    /// Compose trace-continuity and outward-flux-balance roots for one
    /// validated boundary-physical Connection.
    ///
    /// # Errors
    /// Returns `EQ0302` if `connection` is not a closed field-valued junction
    /// in this immutable program.
    pub fn compose_boundary_physical_junction(
        &self,
        connection: Id<kinds::Connection>,
    ) -> Result<BoundaryJunctionResidual, Diagnostic> {
        let connection_id = connection.erase();
        let Some(KernelNode::Connection(connection_definition)) = self.node(connection_id) else {
            return Err(port_error(
                connection_id,
                "selected node is not a boundary Connection",
            ));
        };
        if !matches!(
            connection_definition.semantics(),
            ConnectionSemantics::Conserving | ConnectionSemantics::SpatialPeriodic
        ) {
            return Err(port_error(
                connection_id,
                "selected node is not a conserving boundary Connection",
            ));
        }
        let mut ports = edge_targets(self.edges(), connection_id, EdgeKind::Connects)
            .into_iter()
            .map(|port| {
                port.downcast::<kinds::Port>().ok_or_else(|| {
                    port_error(
                        connection_id,
                        "boundary junction contains a non-Port member",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let geometry_junction = self.geometry_boundary_junctions().get(&connection_id);
        let contracts = if geometry_junction.is_some() {
            Vec::new()
        } else {
            ports
                .iter()
                .map(|port| {
                    resolve_port_contract(
                        port.erase(),
                        self.node_definitions(),
                        self.edges(),
                        self.cartesian_bounds_map(),
                    )?
                    .ok_or_else(|| {
                        port_error(
                            port.erase(),
                            "Connection member is not a boundary-physical Port",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let geometry = match (connection_definition.semantics(), geometry_junction) {
            (ConnectionSemantics::Conserving, Some(_)) => BoundaryJunctionGeometry::Coincident,
            (ConnectionSemantics::Conserving, None) => {
                BoundaryPhysicalPortContract::validate_connection(&contracts).map_err(|_| {
                    port_error(connection_id, "coincident boundary Connection is not valid")
                })?;
                BoundaryJunctionGeometry::Coincident
            }
            (ConnectionSemantics::SpatialPeriodic, Some(junction)) => {
                let (identification, lower_port) = junction.periodic.as_ref().ok_or_else(|| {
                    port_error(
                        connection_id,
                        "Geometry junction has no admitted periodic chart",
                    )
                })?;
                let lower = ports
                    .iter()
                    .position(|port| port.erase() == *lower_port)
                    .expect("admitted periodic lower Port belongs to the Connection");
                ports.swap(0, lower);
                BoundaryJunctionGeometry::CartesianPeriodic(identification.clone())
            }
            (ConnectionSemantics::SpatialPeriodic, None) => {
                let identification = CartesianPeriodicBoundaryIdentification::derive(&contracts)
                    .map_err(|_| {
                        port_error(connection_id, "spatial-periodic boundary pair is not valid")
                    })?;
                let lower = contracts
                    .iter()
                    .position(|contract| {
                        contract.embedding.side() == eqiora_schema::kernel::BoundarySide::Lower
                    })
                    .expect("validated periodic pair owns one lower side");
                if lower != 0 {
                    ports.swap(0, lower);
                }
                BoundaryJunctionGeometry::CartesianPeriodic(identification)
            }
            _ => unreachable!("boundary Connection semantics were checked above"),
        };
        let anchor = *ports
            .first()
            .ok_or_else(|| port_error(connection_id, "boundary junction has no anchor Port"))?;
        let shapes = ports
            .iter()
            .map(|port| {
                let Some(KernelNode::Port(definition)) = self.node(port.erase()) else {
                    return Err(port_error(
                        port.erase(),
                        "boundary Port definition is missing",
                    ));
                };
                let Some((connector, _)) = definition.boundary_physical_contract() else {
                    return Err(port_error(
                        port.erase(),
                        "boundary junction contains another Port family",
                    ));
                };
                let Some(KernelNode::Domain(domain)) = self.node(connector.erase()) else {
                    return Err(port_error(
                        port.erase(),
                        "boundary Connector definition is missing",
                    ));
                };
                let DomainKind::BoundaryPhysical { connector } = domain.kind() else {
                    return Err(port_error(
                        port.erase(),
                        "boundary Port names another Connector family",
                    ));
                };
                Ok(connector.shape().clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = shapes
            .first()
            .cloned()
            .ok_or_else(|| port_error(connection_id, "boundary junction has no shape"))?;
        if shapes.iter().any(|candidate| candidate != &shape) {
            return Err(port_error(
                connection_id,
                "validated boundary junction has inconsistent shapes",
            ));
        }
        let Some(KernelNode::Port(anchor_definition)) = self.node(anchor.erase()) else {
            return Err(port_error(
                anchor.erase(),
                "boundary Port definition is missing",
            ));
        };
        let Some((_, anchor_boundary)) = anchor_definition.boundary_physical_contract() else {
            return Err(port_error(
                anchor.erase(),
                "boundary junction anchor has another Port family",
            ));
        };
        let parents = edge_targets(self.edges(), anchor_boundary.erase(), EdgeKind::BoundaryOf);
        let Some(parent) = parents.first().copied().filter(|_| parents.len() == 1) else {
            return Err(port_error(
                anchor.erase(),
                "boundary junction anchor has no unique parent Domain",
            ));
        };
        let dimensions = if let Some(junction) = geometry_junction {
            junction.dimensions
        } else {
            let Some(bounds) = self.cartesian_bounds_map().get(&parent) else {
                return Err(port_error(
                    anchor.erase(),
                    "boundary junction parent is not a Cartesian volume",
                ));
            };
            bounds.len()
        };

        let mut builder = ExprDagBuilder::new();
        let anchor_trace = builder.symbol(SymbolRef::PortTrace(anchor))?;
        let mut roots = Vec::with_capacity(ports.len());
        for port in ports.iter().skip(1) {
            let trace = builder.symbol(SymbolRef::PortTrace(*port))?;
            let difference = builder.sub(trace, anchor_trace)?;
            roots.push(difference);
        }
        let mut flux = builder.symbol(SymbolRef::PortFlux(anchor))?;
        for port in ports.iter().skip(1) {
            let next = builder.symbol(SymbolRef::PortFlux(*port))?;
            flux = builder.add(flux, next)?;
        }
        roots.push(flux);
        let dag = builder.finish(roots)?;
        let typed = self
            .type_boundary_junction_residual(dag, connection, dimensions)
            .map_err(|diagnostics| {
                diagnostics.into_iter().next().unwrap_or_else(|| {
                    port_error(
                        connection_id,
                        "boundary junction typing failed without a diagnostic",
                    )
                })
            })?;
        Ok(BoundaryJunctionResidual {
            connection,
            geometry,
            typed,
        })
    }
}

pub(crate) fn resolve_port_contract(
    port_id: RawId,
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    cartesian_bounds: &BTreeMap<RawId, Vec<AxisBounds>>,
) -> Result<Option<BoundaryPhysicalPortContract<RawId>>, Diagnostic> {
    let Some(KernelNode::Port(port)) = nodes.get(&port_id) else {
        return Ok(None);
    };
    let Some((connector, boundary)) = port.boundary_physical_contract() else {
        return Ok(None);
    };
    let Some(KernelNode::Domain(connector_definition)) = nodes.get(&connector.erase()) else {
        return Err(port_error(
            port_id,
            "boundary-physical Port names a missing Connector Domain",
        ));
    };
    let DomainKind::BoundaryPhysical {
        connector: connector_contract,
    } = connector_definition.kind()
    else {
        return Err(port_error(
            port_id,
            "boundary-physical Port must name a boundary-physical Connector Domain",
        ));
    };

    let Some(KernelNode::Domain(boundary_definition)) = nodes.get(&boundary.erase()) else {
        return Err(port_error(
            port_id,
            "boundary-physical Port names a missing boundary Domain",
        ));
    };
    let DomainKind::CartesianBoundary { axis, side } = boundary_definition.kind() else {
        return Err(port_error(
            port_id,
            "boundary-physical Port support must be a Cartesian boundary Domain",
        ));
    };
    let parents = edge_targets(edges, boundary.erase(), EdgeKind::BoundaryOf);
    if parents.len() != 1 {
        return Err(port_error(
            port_id,
            format!(
                "boundary-physical Port boundary requires exactly one parent, found {}",
                parents.len()
            ),
        ));
    }
    let parent = *parents.first().expect("one boundary parent was checked");
    let Some(bounds) = cartesian_bounds.get(&parent) else {
        return Err(port_error(
            port_id,
            "boundary-physical Port parent must be a Cartesian volume Domain",
        ));
    };
    let embedding = CartesianBoundaryEmbedding::derive(bounds, *axis, *side).ok_or_else(|| {
        port_error(
            port_id,
            "boundary-physical Port boundary axis is outside its parent Domain",
        )
    })?;

    if connector_contract.frame() == ValueFrame::SpatialCartesian
        && connector_contract
            .shape()
            .extents()
            .iter()
            .skip(connector_contract.trace_type().array_rank())
            .any(|extent| usize::try_from(extent.get()).ok() != Some(embedding.ambient_dimension()))
    {
        return Err(port_error(
            port_id,
            "Cartesian boundary Connector extents must equal the parent ambient dimension",
        ));
    }

    Ok(Some(BoundaryPhysicalPortContract {
        connector: connector.erase(),
        boundary: boundary.erase(),
        parent,
        embedding,
    }))
}

pub(crate) fn validate_networks(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    cartesian_bounds: &BTreeMap<RawId, Vec<AxisBounds>>,
    geometry_boundary_embeddings: &BTreeMap<
        RawId,
        crate::program::geometry_admission::GeometryBoundaryEmbedding,
    >,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&id, node) in nodes {
        let KernelNode::Port(port) = node else {
            continue;
        };
        if port.boundary_physical_contract().is_none() {
            continue;
        }
        if let Some((_, boundary)) = port.boundary_physical_contract()
            && geometry_boundary_embeddings.contains_key(&boundary.erase())
        {
            validate_port_topology(id, boundary.erase(), nodes, edges, diagnostics);
            continue;
        }
        let contract = match resolve_port_contract(id, nodes, edges, cartesian_bounds) {
            Ok(Some(contract)) => contract,
            Ok(None) => continue,
            Err(error) => {
                diagnostics.push(error);
                continue;
            }
        };

        let owners = edge_sources(edges, id, EdgeKind::HasPort);
        if owners.len() != 1 {
            diagnostics.push(port_error(
                id,
                format!(
                    "boundary-physical Port requires exactly one owning Relation, found {}",
                    owners.len()
                ),
            ));
        } else {
            let owner = *owners.first().expect("one Port owner was checked");
            validate_owner(owner, id, contract.boundary, nodes, edges, diagnostics);
        }

        let memberships = edge_sources(edges, id, EdgeKind::Connects);
        if memberships.len() != 1 {
            diagnostics.push(port_error(
                id,
                format!(
                    "boundary-physical Port requires exactly one conserving membership, found {}",
                    memberships.len()
                ),
            ));
        }
    }

    for (&relation_id, node) in nodes {
        let KernelNode::Relation(relation) = node else {
            continue;
        };
        let owned = edge_targets(edges, relation_id, EdgeKind::HasPort);
        for expression_node in relation.expression().nodes() {
            let ExprNode::Symbol(SymbolRef::PortTrace(port) | SymbolRef::PortFlux(port)) =
                expression_node
            else {
                continue;
            };
            if !owned.contains(&port.erase()) {
                diagnostics.push(port_error(
                    relation_id,
                    format!(
                        "boundary-physical symbol for Port {} is outside the owning Relation",
                        port.erase()
                    ),
                ));
            }
        }
    }
}

fn validate_port_topology(
    port: RawId,
    boundary: RawId,
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let owners = edge_sources(edges, port, EdgeKind::HasPort);
    if owners.len() != 1 {
        diagnostics.push(port_error(
            port,
            format!(
                "boundary-physical Port requires exactly one owning Relation, found {}",
                owners.len()
            ),
        ));
    } else {
        validate_owner(
            *owners.first().expect("one Port owner was checked"),
            port,
            boundary,
            nodes,
            edges,
            diagnostics,
        );
    }
    let memberships = edge_sources(edges, port, EdgeKind::Connects);
    if memberships.len() != 1 {
        diagnostics.push(port_error(
            port,
            format!(
                "boundary-physical Port requires exactly one conserving membership, found {}",
                memberships.len()
            ),
        ));
    }
}

fn validate_owner(
    owner: RawId,
    port: RawId,
    boundary: RawId,
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    diagnostics: &mut Vec<Diagnostic>,
) {
    if !matches!(nodes.get(&owner), Some(KernelNode::Relation(_))) {
        diagnostics.push(port_error(
            port,
            "boundary-physical Port owner must be a Relation",
        ));
        return;
    }
    let scopes = edge_targets(edges, owner, EdgeKind::AppliesOn);
    if scopes != BTreeSet::from([boundary]) {
        diagnostics.push(port_error(
            port,
            "boundary-physical owning Relation must apply on the exact Port boundary",
        ));
    }
    let activations = edge_sources(edges, owner, EdgeKind::Activates);
    if activations.len() != 1
        || !matches!(
            activations.first().and_then(|activation| nodes.get(activation)),
            Some(KernelNode::Activation(activation))
                if matches!(activation.kind(), ActivationKind::Continuous)
        )
    {
        diagnostics.push(port_error(
            port,
            "boundary-physical owning Relation requires exactly one continuous Activation",
        ));
    }
}

fn edge_targets(edges: &[Edge], from: RawId, kind: EdgeKind) -> BTreeSet<RawId> {
    edges
        .iter()
        .filter(|edge| edge.from() == from && edge.kind() == kind)
        .map(Edge::to)
        .collect()
}

fn edge_sources(edges: &[Edge], to: RawId, kind: EdgeKind) -> BTreeSet<RawId> {
    edges
        .iter()
        .filter(|edge| edge.to() == to && edge.kind() == kind)
        .map(Edge::from)
        .collect()
}

fn port_error(id: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message).with_graph_path(GraphPath::new([
        "semantic",
        "boundary-physical",
        &id.to_string(),
    ]))
}
