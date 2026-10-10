use super::*;
use eqiora_schema::kernel::DomainKind;

#[derive(Debug, Clone, Copy)]
struct Side {
    domain: RawId,
    boundary: RawId,
    axis: usize,
    orientation: BoundarySide,
}

struct TracePair {
    fields: [RawId; 2],
    lineage: ScalarTermLineage,
}

struct FluxPair {
    terms: [ExprId; 2],
    lineage: ScalarTermLineage,
}

pub(super) struct Candidate {
    support: RawId,
    sides: [Side; 2],
    traces: Vec<TracePair>,
    fluxes: Vec<FluxPair>,
}

pub(super) fn discover(
    program: &KernelProgram,
    supports: &[ScalarRegionSupport],
) -> Result<Vec<Candidate>, Diagnostic> {
    let mut candidates = Vec::new();
    for node in program.nodes() {
        let KernelNode::Domain(definition) = node else {
            continue;
        };
        let DomainKind::PhysicalInterface { boundaries } = definition.kind() else {
            continue;
        };
        let support = definition.id().erase();
        let mut sides = Vec::new();
        let mut embeddings = Vec::new();
        for boundary in boundaries {
            let boundary = boundary.erase();
            let (region, axis, orientation) = supports
                .iter()
                .find_map(|region| {
                    region
                        .boundaries
                        .iter()
                        .find(|(_, value)| **value == boundary)
                        .map(|(&(axis, orientation), _)| (region, axis, orientation))
                })
                .ok_or_else(|| {
                    lowering_error(support, "physical Interface has a foreign Region Boundary")
                })?;
            let bounds =
                program.resolved_cartesian_bounds(region.domain.downcast().expect("Region"))?;
            embeddings.push(
                CartesianBoundaryEmbedding::derive(bounds, axis, orientation).ok_or_else(|| {
                    lowering_error(support, "invalid physical Interface embedding")
                })?,
            );
            sides.push(Side {
                domain: region.domain,
                boundary,
                axis,
                orientation,
            });
        }
        let sides: [Side; 2] = sides.try_into().expect("two declared boundaries");
        if sides[0].domain == sides[1].domain
            || sides[0].axis != sides[1].axis
            || sides[0].orientation == sides[1].orientation
            || embeddings[0] != embeddings[1]
        {
            return Err(lowering_error(
                support,
                "physical Interface requires coincident opposite sides of distinct Regions",
            ));
        }
        let mut candidate = Candidate {
            support,
            sides,
            traces: Vec::new(),
            fluxes: Vec::new(),
        };
        for relation in relations_on(program, support) {
            require_continuous_relation(program, relation)?;
            let typed = typed_relation(program, relation)?;
            let expression = typed.expression();
            for &root in expression.roots() {
                let view = AdditiveResidualView::derive(expression, root, relation)?;
                let [left, right] = view.leaves() else {
                    return Err(
                        view.mismatch("physical Interface requires two exact opposite terms")
                    );
                };
                if left.sign() == right.sign() {
                    return Err(
                        view.mismatch("physical Interface requires opposite equality signs")
                    );
                }
                let terms = [left.value(), right.value()];
                let lineage = ScalarTermLineage {
                    relation,
                    expression: root,
                };
                let trace_fields = terms.map(|term| match expression.node(term) {
                    Some(ExprNode::Trace { value, on }) if on.erase() == support => {
                        match expression.node(*value) {
                            Some(ExprNode::Symbol(SymbolRef::Field(field))) => Some(field.erase()),
                            _ => None,
                        }
                    }
                    _ => None,
                });
                if let [Some(first), Some(second)] = trace_fields {
                    let owns = |index: usize, field| {
                        program.edges().iter().any(|edge| {
                            edge.kind() == EdgeKind::DefinedOn
                                && edge.from() == field
                                && edge.to() == sides[index].domain
                        })
                    };
                    let fields = if owns(0, first) && owns(1, second) {
                        [first, second]
                    } else if owns(0, second) && owns(1, first) {
                        [second, first]
                    } else {
                        return Err(view.mismatch("interface trace pair has foreign Field support"));
                    };
                    candidate.traces.push(TracePair { fields, lineage });
                } else if terms.iter().all(|term| {
                    matches!(expression.node(*term),
                    Some(ExprNode::NormalComponent { on, .. }) if on.erase() == support)
                }) {
                    candidate.fluxes.push(FluxPair { terms, lineage });
                } else {
                    return Err(view.mismatch(
                        "physical Interface requires exact Field traces or common-normal fluxes",
                    ));
                }
            }
        }
        if candidate.traces.is_empty() || candidate.traces.len() != candidate.fluxes.len() {
            return Err(lowering_error(
                support,
                "physical Interface requires one flux balance for each trace equality",
            ));
        }
        candidates.push(candidate);
    }
    Ok(candidates)
}

impl Candidate {
    pub(super) fn boundaries(&self) -> Vec<InterfaceBoundary> {
        self.traces
            .iter()
            .flat_map(|trace| {
                self.sides
                    .iter()
                    .zip(trace.fields)
                    .map(|(side, field)| InterfaceBoundary {
                        boundary: side.boundary,
                        field,
                        carrier: None,
                    })
            })
            .collect()
    }

    pub(super) fn finish(
        self,
        check: &impl Fn(RawId, RawId, RawId, RawId, ExprId) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarMaterialInterface>, Diagnostic> {
        let mut used = BTreeSet::new();
        let mut interfaces = Vec::new();
        for trace in self.traces {
            let matches = self
                .fluxes
                .iter()
                .enumerate()
                .filter_map(|(index, flux)| {
                    [[0, 1], [1, 0]]
                        .into_iter()
                        .any(|order| {
                            (0..2).all(|side| {
                                check(
                                    self.sides[side].domain,
                                    self.support,
                                    flux.lineage.relation,
                                    trace.fields[side],
                                    flux.terms[order[side]],
                                )
                                .is_ok()
                            })
                        })
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            let [index] = matches.as_slice() else {
                return Err(lowering_error(
                    self.support,
                    "physical Interface requires one exact constitutive flux balance for each Field pair",
                ));
            };
            if !used.insert(*index) {
                return Err(lowering_error(
                    self.support,
                    "physical Interface reuses a flux balance",
                ));
            }
            let flux = self.fluxes[*index].lineage;
            interfaces.push(ScalarMaterialInterface {
                source: eqiora_realization::ConformingTraceSource::RelationEquality {
                    relation: trace.lineage.relation.downcast().expect("Relation"),
                    root_index: trace.lineage.expression.index(),
                },
                physical_support: Some(self.support),
                sides: std::array::from_fn(|index| ScalarInterfaceSide {
                    domain: self.sides[index].domain,
                    field: trace.fields[index],
                    boundary: self.sides[index].boundary,
                    axis: self.sides[index].axis,
                    side: self.sides[index].orientation,
                    trace: trace.lineage,
                    flux,
                }),
            });
        }
        Ok(interfaces)
    }
}
