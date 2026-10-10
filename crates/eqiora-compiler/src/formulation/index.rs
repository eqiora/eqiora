//! Exact fresh-compile definitions and support edges shared by scalar forms.
use super::*;

pub(super) struct KernelIndex<'a> {
    pub(super) coefficients: BTreeMap<String, AuthoredFormExpression>,
    pub(super) nodes: BTreeMap<RawId, &'a KernelNode>,
    pub(super) applies_on: BTreeMap<RawId, RawId>,
    pub(super) defined_on: BTreeMap<RawId, RawId>,
    pub(super) clocked_by: BTreeMap<RawId, RawId>,
    pub(super) boundary_of: BTreeMap<RawId, RawId>,
}

impl<'a> KernelIndex<'a> {
    pub(super) fn interface_sides(&self, domain: RawId) -> Option<([RawId; 2], [RawId; 2])> {
        let KernelNode::Domain(definition) = self.nodes.get(&domain).copied()? else {
            return None;
        };
        let eqiora_schema::kernel::DomainKind::PhysicalInterface { boundaries } = definition.kind()
        else {
            return None;
        };
        let boundaries = boundaries.map(Id::erase);
        let parents = [
            *self.boundary_of.get(&boundaries[0])?,
            *self.boundary_of.get(&boundaries[1])?,
        ];
        Some((boundaries, parents))
    }

    pub(super) fn new(transaction: &'a Transaction) -> Self {
        let mut nodes = BTreeMap::new();
        let mut applies_on = BTreeMap::new();
        let mut defined_on = BTreeMap::new();
        let mut boundary_of = BTreeMap::new();
        let mut clocked_by = BTreeMap::new();
        for op in transaction.ops() {
            match op {
                Op::DefineKernelNode { node } => {
                    nodes.insert(node.id(), node);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::AppliesOn,
                } => {
                    applies_on.insert(*from, *to);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::DefinedOn,
                } if to.downcast::<kinds::Domain>().is_some() => {
                    defined_on.insert(*from, *to);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::BoundaryOf,
                } => {
                    boundary_of.insert(*from, *to);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::ClockedBy,
                } => {
                    clocked_by.insert(*from, *to);
                }
                _ => {}
            }
        }
        Self {
            coefficients: BTreeMap::new(),
            nodes,
            applies_on,
            defined_on,
            boundary_of,
            clocked_by,
        }
    }
}
