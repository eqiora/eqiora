use eqiora::RawId;
use eqiora::graph::EdgeKind;
use eqiora::kernel::{BoundarySide, DomainKind, KernelNode};
use eqiora::sem::KernelProgram;

/// Resolve the fixture's authored support independently of the lowered model.
pub(crate) fn cartesian_boundary_id(
    program: &KernelProgram,
    parent: RawId,
    axis: usize,
    side: BoundarySide,
) -> RawId {
    let boundaries = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBoundary { axis: actual_axis, side: actual_side }
                    if *actual_axis == axis && *actual_side == side) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .filter(|boundary| {
            program.edges().iter().any(|edge| {
                edge.kind() == EdgeKind::BoundaryOf
                    && edge.from() == *boundary
                    && edge.to() == parent
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(boundaries.len(), 1, "one authored Cartesian boundary");
    boundaries[0]
}
