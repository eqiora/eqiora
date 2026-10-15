//! Model selections must cover the authenticated physical cells and frontier.
use super::*;
use crate::canonical::boundary_parent;
use crate::region_assembly::mapping::bind_region_cells;
use eqiora_core::RawId;
use eqiora_schema::kernel::{DomainKind, KernelNode};

mod charts;

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
    let (geometry, mesh, correspondence) = match resources {
        NativeMeshResources::GmshSimplicial {
            geometry,
            mesh,
            correspondence,
            ..
        }
        | NativeMeshResources::AffineTriangleSimplicial {
            geometry,
            mesh,
            correspondence,
            ..
        } => (geometry, mesh, correspondence),
        _ => {
            return Err(invalid(
                "simplicial Model binding requires authenticated simplicial resources",
            ));
        }
    };
    let dimension = mesh.dimension();
    let definition = match resources {
        NativeMeshResources::GmshSimplicial { .. } => Some(
            eqiora_artifact::GeometryDefinitionV1::from_canonical(geometry)?,
        ),
        _ => None,
    };
    match (&definition, dimension) {
        (None, 2) => validate_simplicial_resources(resources)?,
        (Some(definition), 2) => correspondence.validate_against_region(definition, mesh)?,
        (Some(definition), 3) => correspondence.validate_against_polyhedra(definition, mesh)?,
        _ => {
            return Err(invalid(
                "linear simplicial support requires planar or volume cells",
            ));
        }
    }
    let entities = |name: &str| match (&definition, dimension) {
        (None, _) if geometry.planar_rectangle_bounds().is_some() => {
            correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
        }
        (None, _) => {
            correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
        }
        (Some(definition), 2) => correspondence.region_entity_set_entities(definition, name),
        (Some(definition), _) => correspondence.polyhedral_entity_set_entities(definition, name),
    };
    let mut regions = BTreeMap::new();
    let mesh_identity = mesh.digest()?;
    let charts = charts::mapped_storage_targets(program, geometry.digest_bytes())?;
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
        if charts.contains(&domain.id().erase()) {
            continue;
        }
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
    let owners = bind_region_cells(mesh.mesh(), membership)?;
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
        if sides.is_empty() || sides.len() > 2 {
            return Err(invalid(
                "simplicial Model requires manifold facet ownership",
            ));
        }
        for side in &sides {
            let parent = owners[side.entity.index()];
            if sides.len() == 1
                || sides
                    .iter()
                    .any(|other| owners[other.entity.index()] != parent)
            {
                expected.entry(parent).or_default().insert(facet);
            }
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

/// Bind the exact selected interface patch, including parent-outward orientation.
pub(in crate::numerical_admission) fn validate_interface_support(
    supports: &BTreeMap<RawId, SimplicialRegionSupport>,
    resources: &NativeMeshResources,
    sides: [(RawId, RawId); 2],
) -> Result<(), Diagnostic> {
    let mesh = match resources {
        NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mesh.mesh(),
        _ => {
            return Err(invalid(
                "interface requires authenticated simplicial resources",
            ));
        }
    };
    let selected = sides
        .iter()
        .map(|(domain, boundary)| {
            let support = supports
                .get(domain)
                .ok_or_else(|| invalid("interface has a foreign Region"))?;
            let facets = support
                .facets
                .get(boundary)
                .ok_or_else(|| invalid("interface has a foreign Boundary"))?;
            Ok((support, facets.iter().copied().collect::<BTreeSet<_>>()))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    if sides[0].0 == sides[1].0 || selected[0].1.is_empty() || selected[0].1 != selected[1].1 {
        return Err(invalid(
            "interface requires the same nonempty facet set on distinct Regions",
        ));
    }
    for &facet in &selected[0].1 {
        let parents = mesh
            .incidence(facet, mesh.topological_dimension())
            .ok_or_else(|| invalid("interface facet has no cell incidence"))?;
        if parents.len() != 2 {
            return Err(invalid(
                "interface requires exactly two incident parent cells",
            ));
        }
        let mut signs = Vec::new();
        for (support, _) in &selected {
            let owned = parents
                .iter()
                .filter(|parent| support.cells.contains(&CellId::new(parent.entity.index())))
                .collect::<Vec<_>>();
            let [parent] = owned.as_slice() else {
                return Err(invalid(
                    "interface has no unique incident cell in its exact Region",
                ));
            };
            let sign = mesh
                .signed_boundary(parent.entity)
                .and_then(|boundary| boundary.into_iter().find(|(entity, _)| *entity == facet))
                .ok_or_else(|| invalid("interface parent has no oriented facet closure"))?
                .1;
            signs.push(sign);
        }
        if signs[0] != -signs[1] {
            return Err(invalid(
                "interface parent-outward orientations are not opposite",
            ));
        }
    }
    Ok(())
}
