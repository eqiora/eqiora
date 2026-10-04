//! Exact ordered factor closure; no ambient Cartesian frame is inferred.
use super::*;

pub(super) fn factors(
    nodes: &BTreeMap<RawId, KernelNode>,
    root: RawId,
) -> Result<Vec<(RawId, DimExponents)>, Diagnostic> {
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
            DomainKind::CoordinateInterval { bounds } => result.push((id, bounds.lower().dim())),
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
                    "coordinate product factor must be an interval or coordinate product",
                ));
            }
        }
    }
    Ok(result)
}
