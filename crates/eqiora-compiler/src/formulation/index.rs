//! Exact fresh-compile definitions and support edges shared by scalar forms.
use super::*;

pub(super) struct KernelIndex<'a> {
    pub(super) nodes: BTreeMap<RawId, &'a KernelNode>,
    pub(super) applies_on: BTreeMap<RawId, RawId>,
    pub(super) defined_on: BTreeMap<RawId, RawId>,
    pub(super) boundary_of: BTreeMap<RawId, RawId>,
}

impl<'a> KernelIndex<'a> {
    pub(super) fn new(transaction: &'a Transaction) -> Self {
        let mut nodes = BTreeMap::new();
        let mut applies_on = BTreeMap::new();
        let mut defined_on = BTreeMap::new();
        let mut boundary_of = BTreeMap::new();
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
                _ => {}
            }
        }
        Self {
            nodes,
            applies_on,
            defined_on,
            boundary_of,
        }
    }
}
