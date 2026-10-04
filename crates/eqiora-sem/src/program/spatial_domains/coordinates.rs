//! Exact ordered factor closure; no ambient Cartesian frame is inferred.
use super::*;

pub(in crate::program) fn admit(
    nodes: &BTreeMap<RawId, KernelNode>,
    supports: &mut BTreeMap<RawId, SpatialSupport<RawId>>,
    cartesian_geometry: &BTreeSet<RawId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&domain, node) in nodes {
        if !matches!(node, KernelNode::Domain(value) if matches!(value.kind(),
            DomainKind::CoordinateInterval { .. } | DomainKind::CoordinateProduct { .. }))
        {
            continue;
        }
        match factors(nodes, domain, supports, cartesian_geometry) {
            Ok(factors) => {
                supports.insert(domain, SpatialSupport::Coordinates { domain, factors });
            }
            Err(error) => diagnostics.push(error),
        }
    }
}

fn factors(
    nodes: &BTreeMap<RawId, KernelNode>,
    root: RawId,
    supports: &BTreeMap<RawId, SpatialSupport<RawId>>,
    cartesian_geometry: &BTreeSet<RawId>,
) -> Result<Vec<(RawId, DimExponents, usize)>, Diagnostic> {
    let mut pending = vec![root];
    let mut remaining = nodes.len().saturating_sub(1);
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            return Err(kernel_error(
                root,
                "coordinate product repeats an exact factor or contains a cycle",
            ));
        }
        let Some(KernelNode::Domain(domain)) = nodes.get(&id) else {
            return Err(kernel_error(
                root,
                "coordinate factor is outside the exact Model Domain closure",
            ));
        };
        match domain.kind() {
            DomainKind::CoordinateInterval { bounds } => result.push((id, bounds.lower().dim(), 1)),
            DomainKind::CartesianBox { .. } | DomainKind::GeometryRegion { .. } => {
                if matches!(domain.kind(), DomainKind::GeometryRegion { .. })
                    && !cartesian_geometry.contains(&id)
                {
                    return Err(kernel_error(
                        root,
                        "physical coordinate factor requires an exact admitted Cartesian Geometry region",
                    ));
                }
                let Some(SpatialSupport::Volume { dimensions, .. }) = supports.get(&id) else {
                    return Err(kernel_error(
                        root,
                        "physical coordinate factor has no admitted volume support",
                    ));
                };
                result.push((
                    id,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length dimension"),
                    *dimensions,
                ));
            }
            DomainKind::CoordinateProduct { factors } => {
                remaining = remaining.checked_sub(factors.len()).ok_or_else(|| {
                    kernel_error(
                        root,
                        "coordinate product exceeds the unique-Domain work bound",
                    )
                })?;
                pending.extend(factors.iter().rev().map(|factor| factor.erase()));
            }
            _ => {
                return Err(kernel_error(
                    root,
                    "coordinate product factor must be an interval, Cartesian volume, or coordinate product",
                ));
            }
        }
    }
    Ok(result)
}
