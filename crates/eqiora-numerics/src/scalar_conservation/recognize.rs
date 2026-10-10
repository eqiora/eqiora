use super::*;

/// Project the first bounded scalar-conservation profile from exact Model meaning.
///
/// The projection is private and method-neutral. It accepts one or more 1D--3D
/// Cartesian scalar regions, complete exterior/interface closure, optional
/// positive constant storage, positive affine isotropic constitutive
/// coefficients, and state-independent volumetric sources.
pub(crate) fn recognize_scalar_conservation(
    program: &KernelProgram,
) -> Result<ScalarConservationDescriptor, Diagnostic> {
    recognize_scalar_conservation_on_supports(program, cartesian_region_supports(program)?)
}

/// Exact geometric support, independent of the equation coefficient domain.
pub(crate) fn cartesian_region_supports(
    program: &KernelProgram,
) -> Result<Vec<ScalarRegionSupport>, Diagnostic> {
    let mut supports = Vec::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        if !matches!(
            domain.kind(),
            eqiora_schema::kernel::DomainKind::CartesianBox { .. }
        ) {
            continue;
        }
        let bounds = program.resolved_cartesian_bounds(domain.id())?;
        if !(1..=3).contains(&bounds.len()) {
            return Err(lowering_error(
                domain.id().erase(),
                format!(
                    "scalar conservation supports 1D--3D Cartesian boxes, found {}D",
                    bounds.len()
                ),
            ));
        }
        let bounds = bounds
            .iter()
            .map(|axis| [axis.lower().value(), axis.upper().value()])
            .collect::<Vec<_>>();
        let boundaries = exact_boundaries(program, domain.id().erase(), bounds.len())?;
        supports.push(ScalarRegionSupport::new(
            domain.id().erase(),
            bounds,
            boundaries,
        ));
    }
    Ok(supports)
}

pub(crate) fn recognize_scalar_conservation_on_supports(
    program: &KernelProgram,
    mut supports: Vec<ScalarRegionSupport>,
) -> Result<ScalarConservationDescriptor, Diagnostic> {
    supports.sort_by_key(|support| support.domain);
    if supports.is_empty() {
        return Err(model_error(
            program,
            "scalar conservation requires at least one Cartesian volume Domain",
        ));
    }
    if supports
        .windows(2)
        .any(|pair| pair[0].domain == pair[1].domain)
    {
        return Err(model_error(
            program,
            "scalar conservation has duplicate Cartesian region support",
        ));
    }

    let physical = physical_interface::candidates(program)?;
    let physical_boundaries = physical
        .iter()
        .flat_map(|(_, sides)| *sides)
        .collect::<BTreeSet<_>>();
    let mut regions = Vec::with_capacity(supports.len());
    let mut pending = BTreeMap::<RawId, Vec<PendingInterfaceSide>>::new();
    let mut parameters = Vec::new();
    for support in supports {
        let (region, region_interfaces) = recognize_region(program, support, &physical_boundaries)?;
        collect_region_parameters(&region, &mut parameters);
        for member in region_interfaces {
            pending
                .entry(connection_of(program, member.port)?)
                .or_default()
                .push(member);
        }
        regions.push(region);
    }

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

    interfaces.extend(physical_interface::recognize(program, &physical, &regions)?);
    interfaces.sort_by_key(ScalarMaterialInterface::source);
    Ok(ScalarConservationDescriptor {
        model: program.model(),
        semantic_revision: program.revision().0,
        regions,
        interfaces,
        parameters,
    })
}

fn recognize_region(
    program: &KernelProgram,
    support: ScalarRegionSupport,
    physical_boundaries: &BTreeSet<RawId>,
) -> Result<(ScalarConservationRegion, Vec<PendingInterfaceSide>), Diagnostic> {
    let ScalarRegionSupport {
        domain,
        bounds,
        boundaries: boundary_domains,
    } = support;
    let dimensions = bounds.len();
    if !(1..=3).contains(&dimensions) {
        return Err(lowering_error(
            domain,
            format!("scalar conservation supports 1D--3D Cartesian boxes, found {dimensions}D"),
        ));
    }
    if bounds
        .iter()
        .any(|[lower, upper]| !lower.is_finite() || !upper.is_finite() || lower >= upper)
    {
        return Err(lowering_error(
            domain,
            "scalar-conservation support requires finite strict Cartesian bounds",
        ));
    }
    if boundary_domains.len() != dimensions * 2
        || boundary_domains.iter().any(|((axis, _), boundary)| {
            *axis >= dimensions || boundary_parent(program, *boundary) != Some(domain)
        })
    {
        return Err(lowering_error(
            domain,
            "scalar-conservation support requires one exact child boundary per Cartesian side",
        ));
    }
    for axis in 0..dimensions {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            if !boundary_domains.contains_key(&(axis, side)) {
                return Err(lowering_error(
                    domain,
                    format!("scalar-conservation support is missing boundary axis {axis} {side:?}"),
                ));
            }
        }
    }
    let fields = continuum_fields_on(program, domain);
    let [field] = fields.as_slice() else {
        return Err(lowering_error(
            domain,
            format!(
                "scalar conservation requires exactly one continuum Field, found {}",
                fields.len()
            ),
        ));
    };
    let Some(KernelNode::Field(field_definition)) = program.node(*field) else {
        return Err(lowering_error(
            *field,
            "scalar conservation Field is missing",
        ));
    };
    if !field_definition.shape().is_scalar() || field_definition.frame() != ValueFrame::Invariant {
        return Err(lowering_error(
            *field,
            "scalar conservation requires an invariant scalar Field",
        ));
    }
    let volume_relations = relations_on(program, domain);
    let [balance_relation] = volume_relations.as_slice() else {
        return Err(lowering_error(
            domain,
            format!(
                "scalar conservation requires exactly one volume Relation, found {}",
                volume_relations.len()
            ),
        ));
    };
    let (balance_dimension, storage, flux, source) =
        recognize_balance(program, *balance_relation, *field, dimensions, &bounds)?;

    let mut exterior = BTreeMap::new();
    let mut interfaces = Vec::new();
    for ((axis, side), boundary) in boundary_domains {
        let relations = relations_on(program, boundary);
        if physical_boundaries.contains(&boundary) {
            if !relations.is_empty() {
                return Err(lowering_error(
                    boundary,
                    "physical Interface overlaps a separate boundary Relation",
                ));
            }
            continue; // The descriptor returns only after the explicit interface laws are proved.
        }
        if relations.is_empty() {
            return Err(lowering_error(
                boundary,
                "scalar conservation boundary has no Relation",
            ));
        }
        if let Some(interface) = recognize_interface_side(
            program,
            domain,
            boundary,
            axis,
            side,
            &relations,
            *field,
            &flux.coefficient,
            dimensions,
        )? {
            interfaces.push(interface);
            continue;
        }
        let [relation] = relations.as_slice() else {
            return Err(lowering_error(
                boundary,
                format!(
                    "scalar exterior boundary requires exactly one Relation, found {}",
                    relations.len()
                ),
            ));
        };
        let law = recognize_exterior_law(
            program,
            *relation,
            *field,
            &flux.coefficient,
            dimensions,
            matches!(program.node(*balance_relation), Some(KernelNode::Relation(value)) if matches!(value.meaning(), eqiora_schema::kernel::RelationMeaning::Conservation(_))),
        )?;
        exterior.insert(
            (axis, side),
            ScalarExteriorBoundary {
                boundary,
                parent: domain,
                axis,
                side,
                law,
            },
        );
    }

    Ok((
        ScalarConservationRegion {
            domain,
            field: *field,
            field_dimension: field_definition.dimension(),
            dimensions,
            bounds,
            balance_relation: *balance_relation,
            balance_dimension,
            storage,
            flux,
            source,
            exterior,
        },
        interfaces,
    ))
}
