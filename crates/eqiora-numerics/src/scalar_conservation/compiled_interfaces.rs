//! Interface syntax is discovered before, and proved against, compiled Region rows.
use super::*;
use crate::form_compiler::linear::InterfaceBoundary;

mod physical;

pub(crate) struct CompiledInterfaceCandidates {
    connections: BTreeMap<RawId, Vec<PendingInterfaceSide>>,
    physical: Vec<physical::Candidate>,
}

impl CompiledInterfaceCandidates {
    pub(crate) fn discover(
        program: &KernelProgram,
        supports: &[ScalarRegionSupport],
    ) -> Result<Self, Diagnostic> {
        let mut connections = BTreeMap::<RawId, Vec<PendingInterfaceSide>>::new();
        for support in supports {
            for (&(axis, side), &boundary) in &support.boundaries {
                for relation in relations_on(program, boundary) {
                    let expression = relation_expression(program, relation)?;
                    if !expression.nodes().iter().any(|node| {
                        matches!(
                            node,
                            ExprNode::Symbol(SymbolRef::PortTrace(_) | SymbolRef::PortFlux(_))
                        )
                    }) {
                        continue;
                    }
                    let candidate = interface::recognize_carrier(
                        program,
                        support.domain,
                        boundary,
                        axis,
                        side,
                        relation,
                    )?;
                    connections
                        .entry(connection_of(program, candidate.port)?)
                        .or_default()
                        .push(candidate);
                }
            }
        }
        for node in program.nodes() {
            if let KernelNode::Connection(connection) = node
                && !connections.contains_key(&connection.id().erase())
            {
                return Err(lowering_error(
                    connection.id().erase(),
                    "Connection has no complete compiled boundary carrier",
                ));
            }
        }
        Ok(Self {
            connections,
            physical: physical::discover(program, supports)?,
        })
    }

    pub(crate) fn boundaries(&self) -> Result<BTreeSet<InterfaceBoundary>, Diagnostic> {
        let mut coverage = Vec::new();
        for side in self.connections.values().flatten() {
            coverage.push(InterfaceBoundary {
                boundary: side.side.boundary,
                field: side.side.field,
                carrier: Some(side.side.trace.relation),
            });
        }
        for candidate in &self.physical {
            coverage.extend(candidate.boundaries());
        }
        let mut endpoints = BTreeSet::new();
        for endpoint in &coverage {
            if !endpoints.insert((endpoint.boundary, endpoint.field)) {
                return Err(lowering_error(
                    endpoint.boundary,
                    "Field has duplicate interface coverage on its exact Boundary",
                ));
            }
        }
        Ok(coverage.into_iter().collect())
    }

    pub(crate) fn finish(
        self,
        program: &KernelProgram,
        check: impl Fn(RawId, RawId, RawId, RawId, ExprId) -> Result<(), Diagnostic>,
    ) -> Result<Vec<ScalarMaterialInterface>, Diagnostic> {
        for candidate in self.connections.values().flatten() {
            let side = &candidate.side;
            check(
                side.domain,
                side.boundary,
                side.flux.relation,
                side.field,
                candidate.normal,
            )?;
        }
        let mut interfaces = close_connections(program, self.connections)?;
        for candidate in self.physical {
            interfaces.extend(candidate.finish(&check)?);
        }
        interfaces.sort_by_key(ScalarMaterialInterface::source);
        Ok(interfaces)
    }
}
