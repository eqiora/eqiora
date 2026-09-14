//! Exact Region ownership and quotient facet witnesses for every mesh consumer.
use super::*;
use eqiora_meshing::CellId;

/// Geometry admission supplies exact cell membership; topology supplies orientation.
/// Field/space/scale authentication remains in `RegionDofMap::new`.
pub(crate) fn bind_region_topology(
    mesh: &dyn MeshTopology,
    membership: impl IntoIterator<Item = (CellId, RawId)>,
    quotients: &[ConformingTraceQuotient],
) -> Result<(Vec<RawId>, Vec<TraceBinding>), Diagnostic> {
    let dimension = mesh.topological_dimension();
    let count = mesh.entity_count(dimension).unwrap_or(0);
    if dimension == 0 || count == 0 {
        return Err(invalid(
            "Region binding requires nonempty positive-dimensional cells",
        ));
    }
    let mut owners = vec![None; count];
    for (cell, domain) in membership {
        let owner = owners
            .get_mut(cell.index())
            .ok_or_else(|| invalid("Region binding cell is outside the exact mesh"))?;
        if owner.replace(domain).is_some() {
            return Err(invalid("Region binding assigns a mesh cell more than once"));
        }
    }
    let owners = owners
        .into_iter()
        .map(|owner| owner.ok_or_else(|| invalid("Region binding omits a mesh cell")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut interfaces = BTreeMap::<[RawId; 2], Vec<TraceFacet>>::new();
    for index in 0..mesh
        .entity_count(dimension - 1)
        .ok_or_else(|| invalid("Region binding mesh has no facet stratum"))?
    {
        let facet = MeshEntity::new(dimension - 1, index);
        let sides = mesh
            .incidence(facet, dimension)
            .ok_or_else(|| invalid("Region binding facet has no exact cell incidence"))?;
        if sides.is_empty() || sides.len() > 2 {
            return Err(invalid("Region binding requires manifold facet ownership"));
        }
        let domains = sides
            .iter()
            .map(|side| {
                if side.entity.dimension() != dimension {
                    return Err(invalid(
                        "Region binding facet points outside the cell stratum",
                    ));
                }
                owners
                    .get(side.entity.index())
                    .copied()
                    .ok_or_else(|| invalid("Region binding facet references an absent cell"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if domains.len() == 2 && domains[0] != domains[1] {
            let (domains, sides) = if domains[0] < domains[1] {
                ([domains[0], domains[1]], [sides[0], sides[1]])
            } else {
                ([domains[1], domains[0]], [sides[1], sides[0]])
            };
            interfaces
                .entry(domains)
                .or_default()
                .push(TraceFacet { facet, sides });
        }
    }
    let mut covered = BTreeSet::new();
    let mut identities = BTreeSet::new();
    let mut traces = Vec::new();
    for &quotient in quotients {
        let endpoints = quotient.endpoints();
        if !identities.insert((
            quotient.connection().erase(),
            endpoints.map(|endpoint| endpoint.field().erase()),
        )) {
            return Err(invalid("Region binding repeats an exact trace quotient"));
        }
        let domains = endpoints.map(|endpoint| endpoint.domain().erase());
        let mut key = domains;
        key.sort();
        let facets = interfaces
            .get(&key)
            .ok_or_else(|| invalid("trace quotient has no exact cross-Region facet coverage"))?;
        covered.insert(key);
        let facets = facets
            .iter()
            .cloned()
            .map(|mut witness| {
                if domains != key {
                    witness.sides.swap(0, 1);
                }
                witness
            })
            .collect();
        traces.push(TraceBinding { quotient, facets });
    }
    if covered.len() != interfaces.len() {
        return Err(invalid(
            "cross-Region facets lack complete trace quotient coverage",
        ));
    }
    Ok((owners, traces))
}
