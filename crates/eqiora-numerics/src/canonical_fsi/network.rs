//! Exact Region-local mathematical witnesses and complete Connection closure.
use super::*;

type Region = (
    RawId,
    CartesianBounds<2>,
    Option<BTreeMap<(usize, BoundarySide), RawId>>,
);
type Endpoint = (RawId, RawId, CartesianBounds<2>, LiveSide);

pub(super) fn lower(
    program: &KernelProgram,
    regions: Vec<Region>,
) -> Result<FixedReferenceFsiCartesianModel2d, Diagnostic> {
    let mut fluids = BTreeMap::new();
    let mut solids = BTreeMap::new();
    let mut test_orientations = BTreeMap::new();
    let mut allowed = BTreeSet::new();
    let mut endpoints = BTreeMap::<RawId, Vec<Endpoint>>::new();
    for (domain, bounds, sides) in regions {
        let fluid = lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries(
            program,
            domain,
            bounds,
            sides.as_ref().map(|sides| {
                sides
                    .iter()
                    .map(|(&(axis, side), &boundary)| ((side, axis), boundary))
                    .collect()
            }),
        );
        let solid = match &sides {
            Some(sides) => lower_isotropic_elastodynamics_subdomain_2d_with_boundaries(
                program,
                domain,
                bounds,
                sides.clone(),
            ),
            None => lower_isotropic_elastodynamics_subdomain_2d(program, domain, bounds),
        };
        match (fluid, solid) {
            (Ok(fluid), Err(_)) => {
                if !fluid.boundary.uninterpreted_live_relations.is_empty() {
                    return Err(lowering_error(
                        domain,
                        "Region has uninterpreted live Port Relations",
                    ));
                }
                allowed.extend([
                    domain,
                    fluid.representation,
                    fluid.model.velocity(),
                    fluid.model.pressure(),
                    fluid.model.force_potential(),
                ]);
                allowed.extend(fluid.volume_relations.iter().copied());
                allowed.extend(fluid.boundary.relations.iter().copied());
                allowed.extend(fluid.boundary.ports.iter().copied());
                allowed.extend(fluid.boundary.connections.iter().copied());
                allowed.extend(fluid.boundary.connector_domains.iter().copied());
                collect_boundaries(
                    domain,
                    fluid.model.velocity(),
                    bounds,
                    fluid.model.boundary_inventory(),
                    &mut allowed,
                    &mut endpoints,
                );
                test_orientations
                    .insert(fluid.model.velocity(), fluid.model.momentum_orientation());
                test_orientations.insert(fluid.model.pressure(), -1.0);
                if fluids.insert(domain, fluid.model).is_some() {
                    return Err(lowering_error(
                        domain,
                        "Region inventory repeats an exact Domain",
                    ));
                }
            }
            (Err(_), Ok(solid)) => {
                if !solid.boundary.uninterpreted_live_relations.is_empty() {
                    return Err(lowering_error(
                        domain,
                        "Region has uninterpreted live Port Relations",
                    ));
                }
                allowed.extend([
                    domain,
                    solid.representation,
                    solid.model.velocity(),
                    solid.model.continuum().displacement(),
                    solid.model.continuum().load_potential(),
                ]);
                allowed.extend(solid.volume_relations.iter().copied());
                allowed.extend(solid.boundary.relations.iter().copied());
                allowed.extend(solid.boundary.ports.iter().copied());
                allowed.extend(solid.boundary.connections.iter().copied());
                allowed.extend(solid.boundary.connector_domains.iter().copied());
                collect_boundaries(
                    domain,
                    solid.model.velocity(),
                    bounds,
                    solid.model.continuum().boundary_inventory(),
                    &mut allowed,
                    &mut endpoints,
                );
                test_orientations
                    .insert(solid.model.velocity(), solid.model.momentum_orientation());
                if solids.insert(domain, solid.model).is_some() {
                    return Err(lowering_error(
                        domain,
                        "Region inventory repeats an exact Domain",
                    ));
                }
            }
            (Err(_), Err(_)) => {
                return Err(lowering_error(
                    domain,
                    "Region has no complete admitted inertial-constraint or first-order elastic meaning",
                ));
            }
            (Ok(_), Ok(_)) => {
                return Err(lowering_error(
                    domain,
                    "Region has ambiguous mathematical execution witnesses",
                ));
            }
        }
    }
    if fluids.is_empty() || solids.is_empty() || endpoints.is_empty() {
        return Err(model_lowering_error(
            program,
            "coupled transient lowering requires inertial constraint, eliminated state and live Connection inventories",
        ));
    }
    let mut interfaces = BTreeMap::new();
    for (connection, mut sides) in endpoints {
        sides.sort_by_key(|side| side.0);
        let [left, right] = sides.as_slice() else {
            return Err(lowering_error(
                connection,
                "Connection requires two exact Region-relative endpoints",
            ));
        };
        if left.0 == right.0 {
            return Err(lowering_error(
                connection,
                "Connection endpoints must own distinct Regions",
            ));
        }
        require_exact_interface(program, left.3, right.3)?;
        require_coincident_bounds(&left.2, &right.2, left.3, right.3)?;
        let endpoint = |side: &Endpoint| FsiInterfaceSide {
            domain: side.0,
            field: side.1,
            boundary: side.3.boundary,
            port: side.3.port,
            side: side.3.side,
        };
        interfaces.insert(
            connection,
            FsiInterface {
                connection,
                axis: left.3.axis,
                endpoints: [endpoint(left), endpoint(right)],
            },
        );
    }
    require_closed(program, allowed, "Region network")?;
    let domains = fluids
        .keys()
        .chain(solids.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let equation_roles = crate::form_compiler::equation_roles::EquationRoles::derive(
        program,
        domains.iter().copied(),
    )?;
    let region_forms = domains
        .into_iter()
        .map(|domain| {
            crate::form_compiler::region::CompiledRegionForm::derive(program, domain, 2)
                .map(|form| (domain, form))
        })
        .collect::<Result<_, _>>()?;
    Ok(FixedReferenceFsiCartesianModel2d {
        model: program.model(),
        semantic_revision: program.revision().0,
        fluids,
        solids,
        interfaces,
        test_orientations,
        equation_roles,
        region_forms,
    })
}

fn collect_boundaries(
    domain: RawId,
    field: RawId,
    bounds: CartesianBounds<2>,
    inventory: &CartesianBoundaryInventory<2>,
    allowed: &mut BTreeSet<RawId>,
    endpoints: &mut BTreeMap<RawId, Vec<Endpoint>>,
) {
    for (&(axis, side), entry) in inventory.entries() {
        allowed.insert(entry.boundary());
        if let PhysicalBoundaryDisposition::PortBinding { connection, port } = entry.disposition() {
            endpoints.entry(connection).or_default().push((
                domain,
                field,
                bounds,
                LiveSide {
                    axis,
                    side,
                    boundary: entry.boundary(),
                    connection,
                    port,
                },
            ));
        }
    }
}

pub(super) fn require_closed(
    program: &KernelProgram,
    mut allowed: BTreeSet<RawId>,
    projection: &str,
) -> Result<(), Diagnostic> {
    let relations = allowed
        .iter()
        .copied()
        .filter(|&id| matches!(program.node(id), Some(KernelNode::Relation(_))))
        .collect::<Vec<_>>();
    allowed.extend(
        program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Activates && relations.contains(&edge.to()))
            .map(|edge| edge.from()),
    );
    for relation in relations {
        let Some(KernelNode::Relation(definition)) = program.node(relation) else {
            unreachable!()
        };
        allowed.extend(
            definition
                .expression()
                .nodes()
                .iter()
                .filter_map(|node| match node {
                    ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
                    _ => None,
                }),
        );
    }
    for node in program.nodes() {
        if !allowed.contains(&node.id()) {
            return Err(model_lowering_error(
                program,
                format!(
                    "closed {projection} lowering would ignore unexpected {:?} node {}",
                    node.kind(),
                    node.id()
                ),
            ));
        }
    }
    Ok(())
}
