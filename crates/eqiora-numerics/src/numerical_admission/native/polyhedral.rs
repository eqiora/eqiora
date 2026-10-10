//! Model selections must cover the authenticated physical cells and frontier.
use super::*;
use crate::canonical::boundary_parent;
use crate::region_assembly::mapping::bind_region_topology;
use eqiora_core::RawId;
use eqiora_schema::kernel::{DomainKind, KernelNode};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) struct SimplicialRegionSupport {
    pub(in crate::numerical_admission) mesh: eqiora_artifact::ArtifactDigest,
    pub(in crate::numerical_admission) cells: Vec<CellId>,
    pub(in crate::numerical_admission) boundaries: BTreeSet<RawId>,
    pub(in crate::numerical_admission) facets: BTreeMap<RawId, Vec<MeshEntity>>,
}

pub(in crate::numerical_admission) fn bind_model_support(
    program: &KernelProgram,
    resources: &NativeMeshResources,
) -> Result<BTreeMap<RawId, SimplicialRegionSupport>, Diagnostic> {
    let NativeMeshResources::GmshSimplicial {
        geometry,
        mesh,
        correspondence,
        ..
    } = resources
    else {
        return Err(invalid(
            "simplicial Model binding requires authenticated simplicial resources",
        ));
    };
    let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(geometry)?;
    let dimension = mesh.dimension();
    match dimension {
        2 => correspondence.validate_against_region(&definition, mesh)?,
        3 => correspondence.validate_against_polyhedra(&definition, mesh)?,
        _ => {
            return Err(invalid(
                "linear simplicial support requires planar or volume cells",
            ));
        }
    }
    let entities = |name: &str| match dimension {
        2 => correspondence.region_entity_set_entities(&definition, name),
        _ => correspondence.polyhedral_entity_set_entities(&definition, name),
    };
    let mut regions = BTreeMap::new();
    let mesh_identity = mesh.digest()?;
    let mut membership = Vec::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        let DomainKind::GeometryRegion {
            geometry: digest,
            entity_set,
        } = domain.kind()
        else {
            continue;
        };
        if digest.bytes() != geometry.digest_bytes() {
            return Err(invalid(
                "simplicial Model region refers to a foreign Geometry",
            ));
        }
        let cells = entities(entity_set)?;
        if cells.is_empty() || cells.iter().any(|cell| cell.dimension() != dimension) {
            return Err(invalid(
                "simplicial Model region requires nonempty volume membership",
            ));
        }
        regions.insert(
            domain.id().erase(),
            SimplicialRegionSupport {
                mesh: mesh_identity.clone(),
                cells: cells.iter().map(|cell| CellId::new(cell.index())).collect(),
                boundaries: BTreeSet::new(),
                facets: BTreeMap::new(),
            },
        );
        membership.extend(
            cells
                .into_iter()
                .map(|cell| (CellId::new(cell.index()), domain.id().erase())),
        );
    }
    // Compatible vector interfaces are not yet admitted. The existing topology
    // owner rejects cross-Region facets without an authenticated trace quotient.
    let (owners, _) = bind_region_topology(mesh.mesh(), membership, &[])?;
    let mut expected = BTreeMap::<RawId, BTreeSet<MeshEntity>>::new();
    for index in 0..mesh
        .mesh()
        .entity_count(dimension - 1)
        .expect("authenticated simplicial facets")
    {
        let facet = MeshEntity::new(dimension - 1, index);
        let sides = mesh
            .mesh()
            .incidence(facet, dimension)
            .expect("authenticated cell incidence");
        if let [side] = sides.as_slice() {
            expected
                .entry(owners[side.entity.index()])
                .or_default()
                .insert(facet);
        }
    }
    let mut covered = BTreeMap::<RawId, BTreeSet<MeshEntity>>::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        let DomainKind::GeometryBoundary { entity_set } = boundary.kind() else {
            continue;
        };
        let parent = boundary_parent(program, boundary.id().erase())
            .filter(|parent| regions.contains_key(parent))
            .ok_or_else(|| invalid("simplicial Model boundary has no selected parent region"))?;
        regions
            .get_mut(&parent)
            .expect("validated parent")
            .boundaries
            .insert(boundary.id().erase());
        let facets = entities(entity_set)?;
        if facets.is_empty() {
            return Err(invalid("simplicial Model boundary membership is empty"));
        }
        regions
            .get_mut(&parent)
            .expect("validated parent")
            .facets
            .insert(boundary.id().erase(), facets.clone());
        for facet in facets {
            if facet.dimension() != dimension - 1
                || !expected
                    .get(&parent)
                    .is_some_and(|outer| outer.contains(&facet))
            {
                return Err(invalid(
                    "simplicial Model boundary is not on its exact parent frontier",
                ));
            }
            if !covered.entry(parent).or_default().insert(facet) {
                return Err(invalid(
                    "simplicial Model boundary selections overlap on a mesh facet",
                ));
            }
        }
    }
    if covered != expected {
        return Err(invalid(
            "simplicial Model boundaries do not completely cover the physical frontier",
        ));
    }
    Ok(regions)
}

#[cfg(test)]
mod tests;
