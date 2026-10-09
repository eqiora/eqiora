//! Physical interface geometry without a conserving Connection or authored law.
use super::*;
use eqiora_schema::kernel::CartesianBoundaryEmbedding;

pub(super) fn validate(
    interface: RawId,
    boundaries: [RawId; 2],
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    bounds: &BTreeMap<RawId, Vec<AxisBounds>>,
) -> Result<(), Diagnostic> {
    let reject = |message| kernel_error(interface, message);
    if boundaries[0] == boundaries[1] || boundaries.contains(&interface) {
        return Err(reject(
            "physical interface requires two distinct boundary Domains",
        ));
    }
    if !edge_targets(edges, interface, EdgeKind::BoundaryOf).is_empty() {
        return Err(reject("physical interface cannot have a BoundaryOf parent"));
    }
    if edge_targets(edges, interface, EdgeKind::DependsOn) != BTreeSet::from(boundaries) {
        return Err(reject(
            "physical interface dependencies must equal its ordered boundary pair",
        ));
    }
    let side = |boundary| {
        let Some(KernelNode::Domain(definition)) = nodes.get(&boundary) else {
            return Err(reject(
                "physical interface references an absent boundary Domain",
            ));
        };
        let DomainKind::CartesianBoundary { axis, side } = definition.kind() else {
            return Err(reject(
                "physical interface requires admitted boundary geometry",
            ));
        };
        let parents = edge_targets(edges, boundary, EdgeKind::BoundaryOf);
        let Some(&parent) = parents.first().filter(|_| parents.len() == 1) else {
            return Err(reject(
                "physical interface side requires one exact boundary parent",
            ));
        };
        let parent_bounds = bounds
            .get(&parent)
            .ok_or_else(|| reject("physical interface parent has no admitted Cartesian bounds"))?;
        let embedding = CartesianBoundaryEmbedding::derive(parent_bounds, *axis, *side)
            .ok_or_else(|| reject("physical interface boundary embedding is invalid"))?;
        Ok((parent, embedding))
    };
    let (first_parent, first) = side(boundaries[0])?;
    let (second_parent, second) = side(boundaries[1])?;
    if first_parent == second_parent {
        return Err(reject(
            "physical interface sides must belong to distinct regions",
        ));
    }
    if first != second || first.side() == second.side() {
        return Err(reject(
            "physical interface boundaries must coincide with opposite outward normals",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{Id, entity::kinds};
    use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::kernel::{BoundarySide, DomainDef};

    fn fixture(right: [f64; 2], right_side: BoundarySide, reverse: bool) -> Result<(), Diagnostic> {
        let regions = [Id::<kinds::Domain>::new(), Id::new()];
        let faces = [Id::<kinds::Domain>::new(), Id::new()];
        let interface = Id::<kinds::Domain>::new();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let intervals = [[0.0, 1.0], right].map(|[lower, upper]| {
            vec![
                AxisBounds::new(
                    DynQuantity::new(lower, length),
                    DynQuantity::new(upper, length),
                )
                .unwrap(),
            ]
        });
        let ordered = if reverse { [faces[1], faces[0]] } else { faces };
        let definitions = [
            DomainDef::cartesian_box(regions[0], intervals[0].clone()).unwrap(),
            DomainDef::cartesian_box(regions[1], intervals[1].clone()).unwrap(),
            DomainDef::cartesian_boundary(faces[0], 0, BoundarySide::Upper),
            DomainDef::cartesian_boundary(faces[1], 0, right_side),
            DomainDef::physical_interface(interface, ordered).unwrap(),
        ];
        let mut nodes = BTreeMap::new();
        let mut transaction = Transaction::new("physical interface geometry");
        for definition in definitions {
            let node = KernelNode::from(definition);
            nodes.insert(node.id(), node.clone());
            transaction.push(Op::DefineKernelNode { node });
        }
        for index in 0..2 {
            transaction.push(Op::Connect {
                from: faces[index].erase(),
                to: regions[index].erase(),
                edge: EdgeKind::BoundaryOf,
            });
            transaction.push(Op::Connect {
                from: interface.erase(),
                to: faces[index].erase(),
                edge: EdgeKind::DependsOn,
            });
        }
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let edges = store.snapshot().edges().copied().collect::<Vec<_>>();
        let bounds = BTreeMap::from([
            (regions[0].erase(), intervals[0].clone()),
            (regions[1].erase(), intervals[1].clone()),
        ]);
        validate(
            interface.erase(),
            ordered.map(Id::erase),
            &nodes,
            &edges,
            &bounds,
        )
    }

    #[test]
    fn coincident_opposite_sides_admit_both_authored_orientations() {
        for reverse in [false, true] {
            fixture([1.0, 3.0], BoundarySide::Lower, reverse).unwrap();
        }
    }

    #[test]
    fn separation_and_equal_outward_normals_are_rejected() {
        let separated = fixture([2.0, 3.0], BoundarySide::Lower, false).unwrap_err();
        assert!(
            separated
                .message()
                .contains("coincide with opposite outward normals")
        );
        let aligned = fixture([-1.0, 1.0], BoundarySide::Upper, false).unwrap_err();
        assert!(
            aligned
                .message()
                .contains("coincide with opposite outward normals")
        );
    }
}
