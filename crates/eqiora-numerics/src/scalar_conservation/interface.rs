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
    if relations.len() != 1 {
        return Err(lowering_error(
            boundary,
            "scalar interface boundary contains an overlapping exterior Relation",
        ));
    }
    let pending = recognize_carrier(program, domain, boundary, axis, side, *relation)?;
    if pending.side.field != field {
        return Err(lowering_error(
            *relation,
            "interface trace uses a foreign Field",
        ));
    }
    validate_normal_flux(
        program,
        &relation_expression(program, *relation)?,
        pending.normal,
        field,
        volume_coefficient,
        *relation,
        dimensions,
    )?;
    Ok(Some(pending))
}

/// Parse exact carrier structure; the caller must prove its constitutive flux
/// against the owning volume equation before admitting an executable interface.
pub(crate) fn recognize_carrier(
    program: &KernelProgram,
    domain: RawId,
    boundary: RawId,
    axis: usize,
    side: BoundarySide,
    relation: RawId,
) -> Result<PendingInterfaceSide, Diagnostic> {
    require_continuous_relation(program, relation)?;
    let relation = &relation;
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
                    Some(ExprNode::Trace { value, on }),
                    Some(ExprNode::Symbol(SymbolRef::PortTrace(id))),
                ) if on.erase() == boundary => {
                    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = expression.node(*value)
                    else {
                        return Err(view.mismatch("interface trace must name an exact Field"));
                    };
                    if !program.edges().iter().any(|edge| {
                        edge.kind() == EdgeKind::DefinedOn
                            && edge.from() == field.erase()
                            && edge.to() == domain
                    }) {
                        return Err(
                            view.mismatch("interface trace Field is outside its exact Region")
                        );
                    }
                    if trace_binding
                        .replace((*root, id.erase(), field.erase()))
                        .is_some()
                    {
                        return Err(view.mismatch("interface carrier repeats trace continuity"));
                    }
                }
                (
                    Some(ExprNode::NormalComponent { on, .. }),
                    Some(ExprNode::Symbol(SymbolRef::PortFlux(id))),
                ) if on.erase() == boundary => {
                    let previous = flux_binding.replace((*root, id.erase(), physical.value()));
                    if previous.is_some() {
                        return Err(view.mismatch("interface carrier repeats flux continuity"));
                    }
                }
                _ => {}
            }
        }
    }
    let Some((trace_relation_root, port, field)) = trace_binding else {
        return Err(lowering_error(
            *relation,
            "scalar interface carrier is missing exact trace continuity",
        ));
    };
    let Some((flux_relation_root, flux_port, normal)) = flux_binding else {
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
    Ok(PendingInterfaceSide {
        normal,
        side: ScalarInterfaceSide {
            domain,
            field,
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
    })
}

pub(crate) fn validate_interface_pair(
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
pub(crate) struct PendingInterfaceSide {
    pub(crate) normal: ExprId,
    pub(crate) port: RawId,
    pub(crate) side: ScalarInterfaceSide,
    pub(super) embedding: CartesianBoundaryEmbedding,
    pub(super) connector: RawId,
}

/// Finish exact two-sided geometric and Port closure after each constitutive
/// witness has been checked by the caller's volume compiler.
pub(crate) fn close_connections(
    program: &KernelProgram,
    pending: BTreeMap<RawId, Vec<PendingInterfaceSide>>,
) -> Result<Vec<ScalarMaterialInterface>, Diagnostic> {
    let mut interfaces = Vec::with_capacity(pending.len());
    for (connection, mut members) in pending {
        members.sort_by_key(|member| member.side.boundary);
        let [first, second] = members.as_slice() else {
            return Err(lowering_error(
                connection,
                format!(
                    "scalar material interface requires exactly two recognized sides, found {}",
                    members.len()
                ),
            ));
        };
        validate_interface_pair(program, connection, first, second)?;
        interfaces.push(ScalarMaterialInterface {
            source: eqiora_realization::ConformingTraceSource::ConservingConnection(
                connection.downcast().expect("validated Connection"),
            ),
            physical_support: None,
            sides: [first.side.clone(), second.side.clone()],
        });
    }

    Ok(interfaces)
}
