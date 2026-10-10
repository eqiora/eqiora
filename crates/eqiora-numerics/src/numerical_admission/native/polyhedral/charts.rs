//! A retained storage map's coordinate-only target is not a meshed PDE region.
use super::*;
use eqiora_graph::EdgeKind;

pub(super) fn mapped_storage_targets(
    program: &KernelProgram,
    primary_geometry: [u8; 32],
) -> Result<BTreeSet<RawId>, Diagnostic> {
    let mut targets = BTreeSet::new();
    for node in program.nodes() {
        let KernelNode::Domain(source) = node else {
            continue;
        };
        if !matches!(source.kind(), DomainKind::GeometryRegion { geometry, .. } if geometry.bytes() == primary_geometry)
        {
            continue;
        }
        let Some(motion) = crate::form_compiler::linear::motion::StorageMotion::select(
            program,
            source.id().erase(),
        )?
        else {
            continue;
        };
        let target = motion.target;
        if target == source.id().erase() {
            continue;
        }
        // A coordinate target cannot hide any solved Field, equation or frontier.
        let owns_mathematics = program.edges().iter().any(|edge| {
            edge.to() == target
                && match program.node(edge.from()) {
                    Some(KernelNode::Field(_)) => edge.kind() == EdgeKind::DefinedOn,
                    Some(KernelNode::Relation(_)) => edge.kind() == EdgeKind::AppliesOn,
                    _ => false,
                }
        });
        let owns_boundary = program.nodes().any(|node| matches!(node,
            KernelNode::Domain(domain) if boundary_parent(program, domain.id().erase()) == Some(target)));
        if !owns_mathematics && !owns_boundary {
            targets.insert(target);
        }
    }
    Ok(targets)
}
