//! Package-neutral normalization of exact field-physical boundary meaning.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, RawId};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{
    BoundarySide, ConnectionSemantics, DomainKind, ExprNode, KernelNode, SymbolRef,
};
use eqiora_sem::KernelProgram;

/// Exact support binding for one Relation admitted by boundary normalization.
///
/// Ordering is identity-only and therefore independent of declaration or
/// package traversal order. Numerical realizations can retain complete
/// boundary closure without recovering support from source structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BoundaryRelationBinding {
    boundary: RawId,
    relation: RawId,
}

impl BoundaryRelationBinding {
    pub(crate) const fn new(boundary: RawId, relation: RawId) -> Self {
        Self { boundary, relation }
    }

    pub(crate) const fn boundary(self) -> RawId {
        self.boundary
    }

    pub(crate) const fn relation(self) -> RawId {
        self.relation
    }
}

/// Method-neutral disposition of one exact field-physical boundary.
///
/// Trace and flux are canonical quantities. Whether a Realization treats
/// either condition as essential or natural is deliberately not repeated in
/// this semantic projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalBoundaryDisposition {
    /// The complete trace quantity is prescribed to zero.
    TraceZero,
    /// The complete parent-outward flux quantity is prescribed to zero.
    FluxZero,
    /// One closed nontrivial Relation prescribes a trace or flux quantity.
    ///
    /// The shared projection records only which canonical quantity the law
    /// constrains and the exact Relation witnessing it. Equation-specific
    /// lowering owns the law's mathematical interpretation.
    Prescribed(PrescribedBoundaryLaw),
    /// A live field-valued physical connection remains to be realized.
    PortBinding {
        /// Exact conserving Connection carrying the unresolved interface.
        connection: RawId,
        /// Exact interface Port owned by this boundary law.
        port: RawId,
    },
}

/// Canonical quantity constrained by one explicit boundary law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalBoundaryQuantity {
    /// Field trace on the exact Boundary.
    Trace,
    /// Parent-outward flux on the exact Boundary.
    Flux,
}

/// Physics-free identity of one closed nontrivial boundary prescription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrescribedBoundaryLaw {
    quantity: PhysicalBoundaryQuantity,
    relation: RawId,
}

impl PrescribedBoundaryLaw {
    pub(crate) const fn new(quantity: PhysicalBoundaryQuantity, relation: RawId) -> Self {
        Self { quantity, relation }
    }

    /// Canonical trace or flux quantity constrained by the Relation.
    #[must_use]
    pub const fn quantity(self) -> PhysicalBoundaryQuantity {
        self.quantity
    }

    /// Exact semantic Relation that owns the prescription.
    #[must_use]
    pub const fn relation(self) -> RawId {
        self.relation
    }
}

/// Authored support key; mesh membership remains a Realization concern.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PhysicalBoundarySupport {
    CartesianSide { axis: usize, side: BoundarySide },
    NamedEntitySet(String),
}

/// One exact semantic Boundary and its package-neutral disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalBoundaryEntry {
    boundary: RawId,
    disposition: PhysicalBoundaryDisposition,
}

impl PhysicalBoundaryEntry {
    pub(crate) const fn new(boundary: RawId, disposition: PhysicalBoundaryDisposition) -> Self {
        Self {
            boundary,
            disposition,
        }
    }

    /// Exact Boundary Domain identity in the admitted Semantic Model.
    #[must_use]
    pub const fn boundary(&self) -> RawId {
        self.boundary
    }

    /// Normalized boundary meaning independent of source or package spelling.
    #[must_use]
    pub const fn disposition(&self) -> PhysicalBoundaryDisposition {
        self.disposition
    }
}

/// Complete exact side inventory for one dimension-typed Cartesian body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartesianBoundaryInventory<const D: usize> {
    entries: BTreeMap<(usize, BoundarySide), PhysicalBoundaryEntry>,
}

impl<const D: usize> CartesianBoundaryInventory<D> {
    pub(crate) fn new(entries: BTreeMap<(usize, BoundarySide), PhysicalBoundaryEntry>) -> Self {
        Self { entries }
    }

    /// Exact entry for one physical axis and outward side.
    #[must_use]
    pub fn boundary(&self, axis: usize, side: BoundarySide) -> Option<&PhysicalBoundaryEntry> {
        self.entries.get(&(axis, side))
    }

    pub(crate) fn entries(
        &self,
    ) -> impl Iterator<Item = (&(usize, BoundarySide), &PhysicalBoundaryEntry)> + Clone {
        self.entries.iter()
    }
}

/// Identities admitted while one field-physical junction is normalized.
#[derive(Debug)]
pub(crate) struct NormalizedFieldPhysicalInterface {
    pub(crate) disposition: PhysicalBoundaryDisposition,
    pub(crate) interface_port: RawId,
    pub(crate) relations: BTreeSet<RawId>,
    pub(crate) ports: BTreeSet<RawId>,
    pub(crate) connection: RawId,
    pub(crate) connector_domains: BTreeSet<RawId>,
    pub(crate) uninterpreted_live_relations: BTreeSet<RawId>,
}

/// Enumerate the exact complete Cartesian exterior of one dimension-typed volume.
///
/// # Errors
/// Rejects an out-of-dimension, duplicated, or incomplete side inventory.
pub(crate) fn exact_cartesian_boundaries<const D: usize>(
    program: &KernelProgram,
    domain: RawId,
) -> Result<BTreeMap<(usize, BoundarySide), RawId>, Diagnostic> {
    if !matches!(D, 2 | 3) {
        return Err(lowering_error(
            domain,
            format!("Cartesian boundary lowering supports dimension two or three, received {D}"),
        ));
    }
    let mut boundaries = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        let DomainKind::CartesianBoundary { axis, side } = boundary.kind() else {
            continue;
        };
        if boundary_parent(program, boundary.id().erase()) != Some(domain) {
            continue;
        }
        if *axis >= D {
            return Err(lowering_error(
                boundary.id().erase(),
                format!("{D}D Cartesian boundary lies outside its ambient dimension"),
            ));
        }
        if boundaries
            .insert((*axis, *side), boundary.id().erase())
            .is_some()
        {
            return Err(lowering_error(
                boundary.id().erase(),
                format!("{D}D Cartesian boundary side is duplicated"),
            ));
        }
    }
    for axis in 0..D {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            if !boundaries.contains_key(&(axis, side)) {
                return Err(lowering_error(
                    domain,
                    format!("missing Cartesian boundary on axis {axis} {side:?}"),
                ));
            }
        }
    }
    Ok(boundaries)
}

/// Normalize one already-recognized field-physical interface Relation.
///
/// Equation-specific code must first prove that `interface_relation` pairs
/// the exact field trace and constitutive outward flux with `interface_port`.
/// This function owns only junction structure and exact zero-terminal
/// elimination.
///
/// # Errors
/// Rejects support drift, non-conserving or non-closed Connections, another
/// Port family, or an ambiguous zero-terminal elimination.
pub(crate) fn normalize_field_physical_interface(
    program: &KernelProgram,
    boundary: RawId,
    boundary_relations: &[RawId],
    interface_relation: RawId,
    interface_port: RawId,
) -> Result<NormalizedFieldPhysicalInterface, Diagnostic> {
    let Some(KernelNode::Port(interface_definition)) = program.node(interface_port) else {
        return Err(lowering_error(
            interface_port,
            "field-physical interface Port is missing",
        ));
    };
    let Some((_, interface_boundary)) = interface_definition.boundary_physical_contract() else {
        return Err(lowering_error(
            interface_port,
            "interface requires a field-valued boundary Port",
        ));
    };
    if interface_boundary.erase() != boundary {
        return Err(lowering_error(
            interface_port,
            "interface Port does not use the Relation's exact Boundary",
        ));
    }

    let connections = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.to() == interface_port)
        .map(|edge| edge.from())
        .collect::<Vec<_>>();
    if connections.len() != 1 {
        return Err(lowering_error(
            interface_port,
            "field-physical interface Port requires exactly one conserving Connection",
        ));
    }
    let connection = connections[0];
    let Some(KernelNode::Connection(connection_definition)) = program.node(connection) else {
        return Err(lowering_error(
            connection,
            "field-physical interface Connection is missing",
        ));
    };
    if connection_definition.semantics() != ConnectionSemantics::Conserving {
        return Err(lowering_error(
            connection,
            "field-physical interface requires conserving Connection semantics",
        ));
    }
    let typed_connection = connection.downcast::<kinds::Connection>().ok_or_else(|| {
        lowering_error(
            connection,
            "field-physical interface has the wrong Connection identity kind",
        )
    })?;
    program
        .compose_boundary_physical_junction(typed_connection)
        .map_err(|_| lowering_error(connection, "field-physical junction is not closed"))?;

    let ports = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == connection)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if !ports.contains(&interface_port) {
        return Err(lowering_error(
            connection,
            "field-physical junction omits its recognized interface Port",
        ));
    }
    let mut connector_domains = BTreeSet::new();
    for port in &ports {
        let Some(KernelNode::Port(definition)) = program.node(*port) else {
            return Err(lowering_error(
                *port,
                "field-physical junction member Port is missing",
            ));
        };
        let Some((connector, _)) = definition.boundary_physical_contract() else {
            return Err(lowering_error(
                *port,
                "field-physical junction contains another Port family",
            ));
        };
        connector_domains.insert(connector.erase());
    }

    let disposition = if ports.len() == 2 {
        let Some(terminal_port) = ports.iter().copied().find(|port| *port != interface_port) else {
            return Err(lowering_error(
                connection,
                "two-Port field-physical junction has no distinct terminal peer",
            ));
        };
        let terminal_relations = boundary_relations
            .iter()
            .copied()
            .filter(|relation| {
                *relation != interface_relation
                    && relation_references_port(program, *relation, terminal_port)
            })
            .collect::<Vec<_>>();
        if terminal_relations.len() == 1 {
            terminal_disposition(program, terminal_relations[0], terminal_port)?.unwrap_or(
                PhysicalBoundaryDisposition::PortBinding {
                    connection,
                    port: interface_port,
                },
            )
        } else {
            PhysicalBoundaryDisposition::PortBinding {
                connection,
                port: interface_port,
            }
        }
    } else {
        PhysicalBoundaryDisposition::PortBinding {
            connection,
            port: interface_port,
        }
    };

    let mut relations = BTreeSet::from([interface_relation]);
    let mut uninterpreted_live_relations = BTreeSet::new();
    match disposition {
        PhysicalBoundaryDisposition::TraceZero
        | PhysicalBoundaryDisposition::FluxZero
        | PhysicalBoundaryDisposition::Prescribed(_) => {
            let peer_relations = boundary_relations.iter().copied().filter(|relation| {
                *relation != interface_relation
                    && ports
                        .iter()
                        .any(|port| relation_references_port(program, *relation, *port))
            });
            relations.extend(peer_relations);
            if relations.len() != 2 {
                return Err(lowering_error(
                    boundary,
                    "two-Port terminal elimination requires exactly one interface and one terminal Relation",
                ));
            }
        }
        PhysicalBoundaryDisposition::PortBinding { .. } => {
            let additional = boundary_relations.iter().copied().filter(|relation| {
                ports
                    .iter()
                    .any(|port| relation_references_port(program, *relation, *port))
            });
            relations.extend(additional.clone());
            uninterpreted_live_relations
                .extend(additional.filter(|relation| *relation != interface_relation));
        }
    }

    Ok(NormalizedFieldPhysicalInterface {
        disposition,
        interface_port,
        relations,
        ports,
        connection,
        connector_domains,
        uninterpreted_live_relations,
    })
}

fn terminal_disposition(
    program: &KernelProgram,
    relation: RawId,
    port: RawId,
) -> Result<Option<PhysicalBoundaryDisposition>, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let mut trace_occurrences = 0;
    let mut flux_occurrences = 0;
    let mut another_port = false;
    for node in expression.nodes() {
        match node {
            ExprNode::Symbol(SymbolRef::PortTrace(id)) if id.erase() == port => {
                trace_occurrences += 1;
            }
            ExprNode::Symbol(SymbolRef::PortFlux(id)) if id.erase() == port => {
                flux_occurrences += 1;
            }
            ExprNode::Symbol(SymbolRef::PortTrace(_) | SymbolRef::PortFlux(_)) => {
                another_port = true;
            }
            _ => {}
        }
    }
    if trace_occurrences > 0 && flux_occurrences > 0 {
        let directly_prescribes_trace = expression
            .roots()
            .iter()
            .any(|root| port_trace(expression, *root) == Some(port));
        let directly_prescribes_flux = expression
            .roots()
            .iter()
            .any(|root| port_flux(expression, *root) == Some(port));
        if directly_prescribes_trace && directly_prescribes_flux {
            return Err(lowering_error(
                relation,
                "field-physical terminal cannot prescribe zero trace and zero flux simultaneously",
            ));
        }
        return Ok(None);
    }
    if expression.roots().len() != 1 || another_port {
        return Ok(None);
    }
    let root = expression.roots()[0];
    Ok(match (trace_occurrences, flux_occurrences) {
        (1, 0) if port_trace(expression, root) == Some(port) => {
            Some(PhysicalBoundaryDisposition::TraceZero)
        }
        (0, 1) if port_flux(expression, root) == Some(port) => {
            Some(PhysicalBoundaryDisposition::FluxZero)
        }
        (1, 0) => Some(PhysicalBoundaryDisposition::Prescribed(
            PrescribedBoundaryLaw::new(PhysicalBoundaryQuantity::Trace, relation),
        )),
        (0, 1) => Some(PhysicalBoundaryDisposition::Prescribed(
            PrescribedBoundaryLaw::new(PhysicalBoundaryQuantity::Flux, relation),
        )),
        _ => None,
    })
}

fn relation_references_port(program: &KernelProgram, relation: RawId, port: RawId) -> bool {
    relation_expression(program, relation).is_ok_and(|expression| {
        expression.nodes().iter().any(|node| {
            matches!(
                node,
                ExprNode::Symbol(SymbolRef::PortTrace(id) | SymbolRef::PortFlux(id))
                    if id.erase() == port
            )
        })
    })
}

fn port_trace(
    expression: &eqiora_schema::kernel::ExprDag,
    value: eqiora_schema::kernel::ExprId,
) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortTrace(port))) => Some(port.erase()),
        _ => None,
    }
}

fn port_flux(
    expression: &eqiora_schema::kernel::ExprDag,
    value: eqiora_schema::kernel::ExprId,
) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortFlux(port))) => Some(port.erase()),
        _ => None,
    }
}

fn relation_expression(
    program: &KernelProgram,
    relation: RawId,
) -> Result<eqiora_schema::kernel::ExprDag, Diagnostic> {
    match program.node(relation) {
        Some(KernelNode::Relation(_)) => program.numerical_residuals(relation),
        _ => Err(lowering_error(
            relation,
            "field-physical boundary owner has no Relation definition",
        )),
    }
}

fn boundary_parent(program: &KernelProgram, boundary: RawId) -> Option<RawId> {
    let parents = program
        .edges()
        .iter()
        .filter(|edge| edge.from() == boundary && edge.kind() == EdgeKind::BoundaryOf)
        .map(|edge| edge.to())
        .collect::<Vec<_>>();
    (parents.len() == 1).then(|| parents[0])
}

fn lowering_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_SPATIAL_LOWERING, message).with_graph_path(GraphPath::new([
        owner.kind().graph().name().to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
    ]))
}
