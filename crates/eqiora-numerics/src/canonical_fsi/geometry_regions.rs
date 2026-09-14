//! Exact geometric support binding; names address selections but never imply roles.

use super::*;

type RegionBoundary = (
    RawId,
    CartesianBounds<2>,
    BTreeMap<(usize, BoundarySide), RawId>,
);

pub(super) fn cartesian_regions(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<Vec<RegionBoundary>, Diagnostic> {
    let mut regions = Vec::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        let DomainKind::GeometryRegion {
            geometry: identity,
            entity_set,
        } = domain.kind()
        else {
            continue;
        };
        let owner = domain.id().erase();
        if identity.bytes() != geometry.digest_bytes() {
            return Err(lowering_error(
                owner,
                "Region has a foreign exact Geometry identity",
            ));
        }
        let parent = geometry.entity_set(entity_set).ok_or_else(|| {
            lowering_error(owner, "Region references a missing Geometry selection")
        })?;
        let mut sides = BTreeMap::new();
        let mut bounds = [[f64::NAN; 2]; 2];
        let mut embeddings = Vec::new();
        for node in program.nodes() {
            let KernelNode::Domain(boundary) = node else {
                continue;
            };
            let DomainKind::GeometryBoundary { entity_set } = boundary.kind() else {
                continue;
            };
            if !program.edges().iter().any(|edge| {
                edge.kind() == EdgeKind::BoundaryOf
                    && edge.from() == boundary.id().erase()
                    && edge.to() == owner
            }) {
                continue;
            }
            let support = geometry.entity_set(entity_set).ok_or_else(|| {
                lowering_error(
                    boundary.id().erase(),
                    "Boundary references a missing Geometry selection",
                )
            })?;
            let embedding = geometry
                .cartesian_boundary_embedding(support, parent)
                .filter(|embedding| embedding.ambient_dimension() == 2)
                .ok_or_else(|| {
                    lowering_error(
                        boundary.id().erase(),
                        "Boundary is not one exact complete Cartesian side of its Region",
                    )
                })?;
            let axis = embedding.normal_axis();
            let side = embedding.side();
            if sides.insert((axis, side), boundary.id().erase()).is_some() {
                return Err(lowering_error(
                    owner,
                    "Region duplicates one exact Cartesian side",
                ));
            }
            bounds[axis][match side {
                BoundarySide::Lower => 0,
                BoundarySide::Upper => 1,
            }] = embedding.coordinate();
            embeddings.push(embedding);
        }
        if sides.len() != 4
            || bounds
                .iter()
                .any(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        {
            return Err(lowering_error(
                owner,
                "Region requires four complete exact Cartesian sides",
            ));
        }
        for embedding in embeddings {
            let [lo, hi] = bounds[1 - embedding.normal_axis()];
            if embedding.tangential_intervals() != [(lo, hi)] {
                return Err(lowering_error(
                    owner,
                    "Boundary tangential support differs from its exact Region",
                ));
            }
        }
        regions.push((owner, bounds, sides));
    }
    Ok(regions)
}
