use std::collections::BTreeMap;

use eqiora_artifact::{CartesianMeshEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1};
use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::{CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION};
use eqiora_meshing::{MeshTopology, OrientationCode};
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::{boundary_parent, lowering_error, model_lowering_error};

pub(super) type ScalarCartesianSupport =
    (RawId, Vec<[f64; 2]>, BTreeMap<(usize, BoundarySide), RawId>);

pub(crate) fn geometry_cartesian_support(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
    mesh: &CartesianMeshEnvelopeV1,
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
) -> Result<ScalarCartesianSupport, Diagnostic> {
    if let Some(bounds) = geometry.cartesian_box_bounds() {
        geometry_box_cartesian_support(program, geometry, correspondence, bounds)
    } else {
        geometry_rectangle_cartesian_support(program, geometry, mesh, correspondence)
    }
}

pub(crate) fn geometry_rectangle_cartesian_support(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
    mesh: &CartesianMeshEnvelopeV1,
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
) -> Result<ScalarCartesianSupport, Diagnostic> {
    let bounds = geometry.planar_rectangle_bounds().ok_or_else(|| {
        model_lowering_error(
            program,
            "geometry-backed Cartesian lowering requires exact PlanarRectangleV2",
        )
    })?;
    let regions = geometry_regions(program);
    if regions.len() != 1 {
        return Err(region_count_error(program, regions.len()));
    }
    let region = regions[0];
    let DomainKind::GeometryRegion {
        geometry: digest,
        entity_set,
    } = region.kind()
    else {
        unreachable!("GeometryRegion filter is exact")
    };
    let region_set = geometry.entity_set(entity_set).ok_or_else(|| {
        lowering_error(
            region.id().erase(),
            "Model GeometryRegion entity set is absent from exact rectangle Geometry",
        )
    })?;
    if digest.bytes() != geometry.digest_bytes()
        || region_set.dimension() != FACE_DIMENSION
        || region_set.members() != [0]
    {
        return Err(lowering_error(
            region.id().erase(),
            "Model GeometryRegion differs from the exact rectangle source face",
        ));
    }
    let domain = region.id().erase();
    let mut boundary_domains = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        let DomainKind::GeometryBoundary { entity_set } = boundary.kind() else {
            continue;
        };
        if boundary_parent(program, boundary.id().erase()) != Some(domain) {
            continue;
        }
        let facets = correspondence.planar_entity_set_entities(geometry, entity_set)?;
        let mut side = None;
        for facet in facets {
            if facet.dimension() != EDGE_DIMENSION {
                return Err(lowering_error(
                    boundary.id().erase(),
                    "rectangle GeometryBoundary correspondence contains a non-facet entity",
                ));
            }
            let adjacent = mesh
                .mesh()
                .incidence(facet, FACE_DIMENSION)
                .ok_or_else(|| {
                    lowering_error(
                        boundary.id().erase(),
                        "rectangle boundary facet has no parent-cell incidence",
                    )
                })?;
            let [parent] = adjacent.as_slice() else {
                return Err(lowering_error(
                    boundary.id().erase(),
                    "rectangle boundary facet does not have exactly one parent cell",
                ));
            };
            if parent.orientation != OrientationCode::identity() {
                return Err(lowering_error(
                    boundary.id().erase(),
                    "rectangle boundary facet has noncanonical orientation",
                ));
            }
            let facet_side = match parent.local_ordinal {
                0 => (1, BoundarySide::Lower),
                1 => (1, BoundarySide::Upper),
                2 => (0, BoundarySide::Lower),
                3 => (0, BoundarySide::Upper),
                _ => {
                    return Err(lowering_error(
                        boundary.id().erase(),
                        "rectangle boundary facet has an unsupported local side ordinal",
                    ));
                }
            };
            if side
                .replace(facet_side)
                .is_some_and(|old| old != facet_side)
            {
                return Err(lowering_error(
                    boundary.id().erase(),
                    "one rectangle source boundary maps to multiple topology sides",
                ));
            }
        }
        let Some(side) = side else {
            return Err(lowering_error(
                boundary.id().erase(),
                "rectangle GeometryBoundary correspondence is empty",
            ));
        };
        if boundary_domains
            .insert(side, boundary.id().erase())
            .is_some()
        {
            return Err(lowering_error(
                boundary.id().erase(),
                "rectangle GeometryBoundary side is duplicated",
            ));
        }
    }
    Ok((domain, bounds.to_vec(), boundary_domains))
}

fn geometry_box_cartesian_support(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    bounds: &[[f64; 2]],
) -> Result<ScalarCartesianSupport, Diagnostic> {
    let regions = geometry_regions(program);
    let [region] = regions.as_slice() else {
        return Err(region_count_error(program, regions.len()));
    };
    let DomainKind::GeometryRegion {
        geometry: digest,
        entity_set,
    } = region.kind()
    else {
        unreachable!("GeometryRegion filter is exact")
    };
    let region_set = geometry.entity_set(entity_set).ok_or_else(|| {
        lowering_error(
            region.id().erase(),
            "Model GeometryRegion entity set is absent from exact Cartesian-box Geometry",
        )
    })?;
    if digest.bytes() != geometry.digest_bytes()
        || region_set.dimension() != bounds.len()
        || region_set.members() != [0]
    {
        return Err(lowering_error(
            region.id().erase(),
            "Model GeometryRegion differs from the exact Cartesian-box body",
        ));
    }

    let domain = region.id().erase();
    let mut boundary_domains = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        let DomainKind::GeometryBoundary { entity_set } = boundary.kind() else {
            continue;
        };
        if boundary_parent(program, boundary.id().erase()) != Some(domain) {
            continue;
        }
        let set = geometry.entity_set(entity_set).ok_or_else(|| {
            lowering_error(
                boundary.id().erase(),
                "Model GeometryBoundary entity set is absent from exact Cartesian-box Geometry",
            )
        })?;
        let [member] = set.members() else {
            return Err(lowering_error(
                boundary.id().erase(),
                "one Cartesian-box boundary support must name exactly one side",
            ));
        };
        let facets = correspondence.cartesian_box_v1_entity_set_entities(geometry, entity_set)?;
        if set.dimension().checked_add(1) != Some(bounds.len())
            || facets.is_empty()
            || facets
                .iter()
                .any(|facet| facet.dimension() != bounds.len() - 1)
        {
            return Err(lowering_error(
                boundary.id().erase(),
                "Cartesian-box GeometryBoundary differs from one nonempty codimension-one side",
            ));
        }
        let axis = member / 2;
        let side = if member % 2 == 0 {
            BoundarySide::Lower
        } else {
            BoundarySide::Upper
        };
        if axis >= bounds.len()
            || boundary_domains
                .insert((axis, side), boundary.id().erase())
                .is_some()
        {
            return Err(lowering_error(
                boundary.id().erase(),
                "Cartesian-box GeometryBoundary side is invalid or duplicated",
            ));
        }
    }
    Ok((domain, bounds.to_vec(), boundary_domains))
}

fn geometry_regions(program: &KernelProgram) -> Vec<&eqiora_schema::kernel::DomainDef> {
    program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::GeometryRegion { .. }) =>
            {
                Some(domain)
            }
            _ => None,
        })
        .collect()
}

fn region_count_error(program: &KernelProgram, count: usize) -> Diagnostic {
    model_lowering_error(
        program,
        format!("geometry-backed Cartesian lowering requires one GeometryRegion, found {count}"),
    )
}
