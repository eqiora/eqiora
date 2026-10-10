//! Exact authored trace equality and common-normal flux balance on physical interfaces.
use super::*;
use eqiora_schema::kernel::DomainKind;

pub(super) fn candidates(program: &KernelProgram) -> Result<Vec<(RawId, [RawId; 2])>, Diagnostic> {
    let mut interfaces = Vec::new();
    let mut used = BTreeSet::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        let DomainKind::PhysicalInterface { boundaries } = domain.kind() else {
            continue;
        };
        let boundaries = boundaries.map(Id::erase);
        if boundaries.iter().any(|side| !used.insert(*side)) {
            return Err(lowering_error(
                domain.id().erase(),
                "physical Interfaces overlap on an exact boundary",
            ));
        }
        interfaces.push((domain.id().erase(), boundaries));
    }
    interfaces.sort_by_key(|(domain, _)| *domain);
    Ok(interfaces)
}

pub(super) fn recognize(
    program: &KernelProgram,
    candidates: &[(RawId, [RawId; 2])],
    regions: &[ScalarConservationRegion],
) -> Result<Vec<ScalarMaterialInterface>, Diagnostic> {
    candidates
        .iter()
        .map(|&(domain, boundaries)| {
            let sides = [
                side(program, boundaries[0], regions)?,
                side(program, boundaries[1], regions)?,
            ];
            if sides[0].0.domain == sides[1].0.domain
                || sides[0].1 != sides[1].1
                || sides[0].2 == sides[1].2
            {
                return Err(lowering_error(
                    domain,
                    "physical Interface requires distinct Regions and opposite Cartesian sides",
                ));
            }
            let embeddings = sides.map(|(region, axis, side)| {
                let bounds = program
                    .resolved_cartesian_bounds(region.domain.downcast().expect("Region Domain"))?;
                CartesianBoundaryEmbedding::derive(bounds, axis, side).ok_or_else(|| {
                    lowering_error(
                        domain,
                        "physical Interface has an invalid Cartesian embedding",
                    )
                })
            });
            let [first, second] = embeddings;
            if first? != second? {
                return Err(lowering_error(
                    domain,
                    "physical Interface sides are not exactly coincident",
                ));
            }
            let pair = [sides[0].0, sides[1].0];
            let (trace, flux) = laws(program, domain, pair)?;
            Ok(ScalarMaterialInterface {
                source: eqiora_realization::ConformingTraceSource::RelationEquality {
                    relation: trace.relation.downcast().expect("authored Relation"),
                    root_index: trace.expression.index(),
                },
                physical_support: Some(domain),
                sides: std::array::from_fn(|index| ScalarInterfaceSide {
                    domain: sides[index].0.domain,
                    boundary: boundaries[index],
                    axis: sides[index].1,
                    side: sides[index].2,
                    trace,
                    flux,
                }),
            })
        })
        .collect()
}

fn side<'a>(
    program: &KernelProgram,
    boundary: RawId,
    regions: &'a [ScalarConservationRegion],
) -> Result<(&'a ScalarConservationRegion, usize, BoundarySide), Diagnostic> {
    let Some(KernelNode::Domain(definition)) = program.node(boundary) else {
        return Err(lowering_error(
            boundary,
            "physical Interface boundary is absent",
        ));
    };
    let DomainKind::CartesianBoundary { axis, side } = definition.kind() else {
        return Err(lowering_error(
            boundary,
            "scalar physical Interface requires exact Cartesian boundaries",
        ));
    };
    let parent = boundary_parent(program, boundary).ok_or_else(|| {
        lowering_error(boundary, "physical Interface boundary has no exact parent")
    })?;
    let region = regions
        .iter()
        .find(|region| region.domain == parent)
        .ok_or_else(|| {
            lowering_error(
                boundary,
                "physical Interface parent is outside the admitted Regions",
            )
        })?;
    Ok((region, *axis, *side))
}

fn laws(
    program: &KernelProgram,
    domain: RawId,
    regions: [&ScalarConservationRegion; 2],
) -> Result<(ScalarTermLineage, ScalarTermLineage), Diagnostic> {
    let mut trace = None;
    let mut flux = None;
    for relation in relations_on(program, domain) {
        require_continuous_relation(program, relation)?;
        let expression = relation_expression(program, relation)?;
        for &root in expression.roots() {
            let view = AdditiveResidualView::derive(&expression, root, relation)?;
            let [left, right] = view.leaves() else {
                return Err(lowering_error(
                    relation,
                    "physical Interface law requires exactly two opposite trace terms",
                ));
            };
            if left.sign() == right.sign() {
                return Err(lowering_error(
                    relation,
                    "physical Interface equality requires opposite term signs",
                ));
            }
            let terms = [left.value(), right.value()];
            let is_trace = [[0, 1], [1, 0]].into_iter().any(|order| {
                terms.into_iter().zip(order).all(|(term, index)| {
                    matches!(expression.node(term), Some(ExprNode::Trace { value, on })
                        if on.erase() == domain && is_field(&expression, *value, regions[index].field))
                })
            });
            let destination = if is_trace {
                &mut trace
            } else {
                let normals = terms.iter().all(|term| {
                    matches!(expression.node(*term),
                    Some(ExprNode::NormalComponent { on, .. }) if on.erase() == domain)
                });
                let balanced = normals
                    && [[0, 1], [1, 0]].into_iter().any(|order| {
                        terms.into_iter().zip(order).all(|(term, index)| {
                            let region = regions[index];
                            validate_normal_flux(
                                program,
                                &expression,
                                term,
                                region.field,
                                &region.flux.coefficient,
                                relation,
                                region.dimensions,
                            )
                            .is_ok()
                        })
                    });
                if !balanced {
                    return Err(lowering_error(
                        relation,
                        "physical Interface requires exact trace equality and flux balance matching both volume coefficients",
                    ));
                }
                &mut flux
            };
            if destination
                .replace(ScalarTermLineage {
                    relation,
                    expression: root,
                })
                .is_some()
            {
                return Err(lowering_error(
                    relation,
                    "physical Interface repeats a trace equality or flux balance",
                ));
            }
        }
    }
    match (trace, flux) {
        (Some(trace), Some(flux)) => Ok((trace, flux)),
        _ => Err(lowering_error(
            domain,
            "physical Interface requires explicit trace equality and flux balance; its declaration supplies neither",
        )),
    }
}
