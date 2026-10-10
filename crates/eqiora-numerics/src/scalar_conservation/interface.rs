use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn recognize_interface_side(
    program: &KernelProgram,
    domain: RawId,
    boundary: RawId,
    axis: usize,
    side: BoundarySide,
    relations: &[RawId],
    field: RawId,
    volume_coefficient: &ScalarSpatialExpression<f64>,
    dimensions: usize,
) -> Result<Option<PendingInterfaceSide>, Diagnostic> {
    let candidates = relations
        .iter()
        .copied()
        .filter(|relation| {
            relation_expression(program, *relation).is_ok_and(|expression| {
                expression.nodes().iter().any(|node| {
                    matches!(
                        node,
                        ExprNode::Symbol(SymbolRef::PortTrace(_) | SymbolRef::PortFlux(_))
                    )
                })
            })
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(None);
    }
    let [relation] = candidates.as_slice() else {
        return Err(lowering_error(
            boundary,
            "scalar interface boundary has multiple Port carrier Relations",
        ));
    };
    require_continuous_relation(program, *relation)?;
    if relations.len() != 1 {
        return Err(lowering_error(
            boundary,
            "scalar interface boundary contains an overlapping exterior Relation",
        ));
    }
    let expression = &relation_expression(program, *relation)?;
    if expression.roots().len() != 2 {
        return Err(lowering_error(
            *relation,
            "scalar interface carrier requires exactly trace and outward-flux roots",
        ));
    }
    let mut trace_binding = None;
    let mut flux_binding = None;
    for root in expression.roots() {
        let view = AdditiveResidualView::derive(expression, *root, *relation)?;
        if view.leaves().len() != 2 {
            return Err(view.mismatch("interface carrier root requires exactly two opposite terms"));
        }
        let left = &view.leaves()[0];
        let right = &view.leaves()[1];
        if left.sign() == right.sign() {
            return Err(view.mismatch("interface carrier terms must have opposite signs"));
        }
        for (physical, port) in [(left, right), (right, left)] {
            match (
                expression.node(physical.value()),
                expression.node(port.value()),
            ) {
                (
                    Some(ExprNode::Trace { value, .. }),
                    Some(ExprNode::Symbol(SymbolRef::PortTrace(id))),
                ) if is_field(expression, *value, field) => {
                    trace_binding = Some((*root, id.erase()))
                }
                (
                    Some(ExprNode::NormalComponent { .. }),
                    Some(ExprNode::Symbol(SymbolRef::PortFlux(id))),
                ) => {
                    validate_normal_flux(
                        program,
                        expression,
                        physical.value(),
                        field,
                        volume_coefficient,
                        *relation,
                        dimensions,
                    )?;
                    flux_binding = Some((*root, id.erase()));
                }
                _ => {}
            }
        }
    }
    let Some((trace_relation_root, port)) = trace_binding else {
        return Err(lowering_error(
            *relation,
            "scalar interface carrier is missing exact trace continuity",
        ));
    };
    let Some((flux_relation_root, flux_port)) = flux_binding else {
        return Err(lowering_error(
            *relation,
            "scalar interface carrier is missing exact outward-flux continuity",
        ));
    };
    if port != flux_port {
        return Err(lowering_error(
            *relation,
            "scalar interface trace and flux use different Ports",
        ));
    }
    let owned = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::HasPort && edge.from() == *relation)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if owned != BTreeSet::from([port]) {
        return Err(lowering_error(
            *relation,
            "scalar interface Relation must own exactly its bound Port",
        ));
    }
    let Some(KernelNode::Port(port_definition)) = program.node(port) else {
        return Err(lowering_error(port, "scalar interface Port is missing"));
    };
    let Some((connector, port_boundary)) = port_definition.boundary_physical_contract() else {
        return Err(lowering_error(
            port,
            "scalar interface requires a field-valued boundary Port",
        ));
    };
    if port_boundary.erase() != boundary {
        return Err(lowering_error(
            port,
            "scalar interface Port is bound to the wrong parent Boundary",
        ));
    }
    let parent_bounds =
        program.resolved_cartesian_bounds(domain.downcast().expect("box Domain identity"))?;
    let embedding = CartesianBoundaryEmbedding::derive(parent_bounds, axis, side)
        .ok_or_else(|| lowering_error(boundary, "scalar interface embedding is invalid"))?;
    Ok(Some(PendingInterfaceSide {
        side: ScalarInterfaceSide {
            domain,
            boundary,
            axis,
            side,
            trace: ScalarTermLineage {
                relation: *relation,
                expression: trace_relation_root,
            },
            flux: ScalarTermLineage {
                relation: *relation,
                expression: flux_relation_root,
            },
        },
        embedding,
        port,
        connector: connector.erase(),
    }))
}

pub(super) fn validate_interface_pair(
    program: &KernelProgram,
    connection: RawId,
    first: &PendingInterfaceSide,
    second: &PendingInterfaceSide,
) -> Result<(), Diagnostic> {
    let Some(KernelNode::Connection(definition)) = program.node(connection) else {
        return Err(lowering_error(
            connection,
            "scalar interface Connection is missing",
        ));
    };
    if definition.semantics() != ConnectionSemantics::Conserving {
        return Err(lowering_error(
            connection,
            "scalar material interface requires conserving Connection semantics",
        ));
    }
    let ports = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == connection)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if ports != BTreeSet::from([first.port, second.port]) {
        return Err(lowering_error(
            connection,
            "scalar material interface must contain exactly its two recognized Ports",
        ));
    }
    if first.connector != second.connector || first.side.domain == second.side.domain {
        return Err(lowering_error(
            connection,
            "scalar material interface requires one connector across two distinct parent Domains",
        ));
    }
    if first.embedding != second.embedding
        || first.side.axis != second.side.axis
        || first.side.side == second.side.side
    {
        return Err(lowering_error(
            connection,
            "scalar material interface sides must be coincident with opposite parent-outward orientation",
        ));
    }
    let typed = connection
        .downcast::<kinds::Connection>()
        .ok_or_else(|| lowering_error(connection, "scalar interface has wrong identity kind"))?;
    program
        .compose_boundary_physical_junction(typed)
        .map_err(|_| {
            lowering_error(
                connection,
                "scalar material interface junction is not closed",
            )
        })?;
    Ok(())
}
#[derive(Debug)]
pub(super) struct PendingInterfaceSide {
    pub(super) port: RawId,
    pub(super) side: ScalarInterfaceSide,
    pub(super) embedding: CartesianBoundaryEmbedding,
    pub(super) connector: RawId,
}
