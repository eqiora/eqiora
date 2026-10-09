//! Model selections must cover the authenticated physical cells and frontier.
use super::*;
use crate::canonical::boundary_parent;
use crate::region_assembly::mapping::bind_region_topology;
use eqiora_core::RawId;
use eqiora_schema::kernel::{DomainKind, KernelNode};

pub(super) fn validate_model_support(
    program: &KernelProgram,
    resources: &NativeMeshResources,
) -> Result<(), Diagnostic> {
    let NativeMeshResources::GmshSimplicial {
        geometry,
        mesh,
        correspondence,
        ..
    } = resources
    else {
        return Err(invalid(
            "polyhedral Model binding requires authenticated simplicial resources",
        ));
    };
    let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(geometry)?;
    correspondence.validate_against_polyhedra(&definition, mesh)?;
    let mut regions = BTreeSet::new();
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
                "polyhedral Model region refers to a foreign Geometry",
            ));
        }
        let cells = correspondence.polyhedral_entity_set_entities(&definition, entity_set)?;
        if cells.is_empty() || cells.iter().any(|cell| cell.dimension() != 3) {
            return Err(invalid(
                "polyhedral Model region requires nonempty volume membership",
            ));
        }
        regions.insert(domain.id().erase());
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
        .entity_count(2)
        .expect("authenticated tetrahedral faces")
    {
        let facet = MeshEntity::new(2, index);
        let sides = mesh
            .mesh()
            .incidence(facet, 3)
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
            .filter(|parent| regions.contains(parent))
            .ok_or_else(|| invalid("polyhedral Model boundary has no selected parent region"))?;
        let facets = correspondence.polyhedral_entity_set_entities(&definition, entity_set)?;
        if facets.is_empty() {
            return Err(invalid("polyhedral Model boundary membership is empty"));
        }
        for facet in facets {
            if facet.dimension() != 2
                || !expected
                    .get(&parent)
                    .is_some_and(|outer| outer.contains(&facet))
            {
                return Err(invalid(
                    "polyhedral Model boundary is not on its exact parent frontier",
                ));
            }
            if !covered.entry(parent).or_default().insert(facet) {
                return Err(invalid(
                    "polyhedral Model boundary selections overlap on a mesh facet",
                ));
            }
        }
    }
    if covered != expected {
        return Err(invalid(
            "polyhedral Model boundaries do not completely cover the physical frontier",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
