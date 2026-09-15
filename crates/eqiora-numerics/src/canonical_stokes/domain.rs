use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::support::{has_edge, lowering_error, model_lowering_error};

pub(super) fn unique_bound_geometry_domain_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<(RawId, BTreeMap<String, RawId>), Diagnostic> {
    let regions = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::GeometryRegion { .. }) =>
            {
                Some(domain)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [region] = regions.as_slice() else {
        return Err(model_lowering_error(
            program,
            "Geometry binding requires exactly one GeometryRegion",
        ));
    };
    let DomainKind::GeometryRegion {
        geometry: digest,
        entity_set,
    } = region.kind()
    else {
        unreachable!()
    };
    let selected_region = geometry
        .entity_set(entity_set)
        .filter(|set| set.dimension() == 2)
        .ok_or_else(|| {
            lowering_error(
                region.id().erase(),
                "GeometryRegion does not select a two-dimensional source region",
            )
        })?;
    if digest.bytes() != geometry.digest_bytes() {
        return Err(lowering_error(
            region.id().erase(),
            "GeometryRegion belongs to another exact source revision",
        ));
    }
    let domain = region.id().erase();
    let mut boundaries = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        if !has_edge(program, boundary.id().erase(), domain, EdgeKind::BoundaryOf) {
            continue;
        }
        let DomainKind::GeometryBoundary { entity_set } = boundary.kind() else {
            return Err(lowering_error(
                boundary.id().erase(),
                "GeometryRegion requires GeometryBoundary supports",
            ));
        };
        let selected = geometry
            .entity_set(entity_set)
            .filter(|set| geometry.selection_is_boundary_of(set, selected_region))
            .ok_or_else(|| {
                lowering_error(
                    boundary.id().erase(),
                    "GeometryBoundary is not a boundary selection of its source region",
                )
            })?;
        if boundaries
            .insert(selected.name().to_owned(), boundary.id().erase())
            .is_some()
        {
            return Err(lowering_error(
                boundary.id().erase(),
                "duplicate GeometryBoundary entity-set selection",
            ));
        }
    }
    if boundaries.is_empty() {
        return Err(lowering_error(
            domain,
            "GeometryRegion requires boundary supports",
        ));
    }
    Ok((domain, boundaries))
}

pub(super) fn unique_named_geometry_domain(
    program: &KernelProgram,
    geometry_digest: [u8; 32],
    region_set: &str,
    required_boundaries: &BTreeSet<String>,
) -> Result<(RawId, BTreeMap<String, RawId>), Diagnostic> {
    let regions = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::GeometryRegion { .. }) =>
            {
                Some(domain)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if regions.len() != 1 {
        return Err(model_lowering_error(
            program,
            format!(
                "geometry-backed 2D Stokes requires exactly one GeometryRegion, found {}",
                regions.len()
            ),
        ));
    }
    let region = regions[0];
    let DomainKind::GeometryRegion {
        geometry,
        entity_set,
    } = region.kind()
    else {
        unreachable!("GeometryRegion filter is exact");
    };
    if geometry.bytes() != geometry_digest || entity_set != region_set {
        return Err(lowering_error(
            region.id().erase(),
            "Model GeometryRegion digest or exact entity-set identity differs from the bound chordal geometry",
        ));
    }
    let domain = region.id().erase();
    let boundaries = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(boundary)
                if has_edge(program, boundary.id().erase(), domain, EdgeKind::BoundaryOf) =>
            {
                match boundary.kind() {
                    DomainKind::GeometryBoundary { entity_set } => {
                        Some((entity_set.clone(), boundary.id().erase()))
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    if boundaries.keys().cloned().collect::<BTreeSet<_>>() != *required_boundaries {
        return Err(lowering_error(
            domain,
            "geometry-backed Stokes boundary entity-set inventory differs from the exact product contract",
        ));
    }
    Ok((domain, boundaries))
}

pub(super) fn unique_box_2d(program: &KernelProgram) -> Result<(RawId, [[f64; 2]; 2]), Diagnostic> {
    unique_box::<2>(program)
}

pub(super) fn unique_box<const D: usize>(
    program: &KernelProgram,
) -> Result<(RawId, [[f64; 2]; D]), Diagnostic> {
    if !matches!(D, 2 | 3) {
        return Err(model_lowering_error(
            program,
            format!(
                "canonical Cartesian fluid lowering supports dimension two or three, received {D}"
            ),
        ));
    }
    let boxes = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if boxes.len() != 1 {
        return Err(model_lowering_error(
            program,
            format!(
                "canonical {D}D fluid lowering requires exactly one Cartesian box, found {}",
                boxes.len()
            ),
        ));
    }
    let domain = boxes[0];
    let bounds = program.resolved_cartesian_bounds(domain.id())?;
    if bounds.len() != D {
        return Err(lowering_error(
            domain.id().erase(),
            format!(
                "canonical Cartesian fluid lowering requires dimension {D}, received {}",
                bounds.len()
            ),
        ));
    }
    let bounds = bounds
        .iter()
        .map(|bound| [bound.lower().value(), bound.upper().value()])
        .collect::<Vec<_>>()
        .try_into()
        .expect("dimension equality establishes Cartesian bound count");
    Ok((domain.id().erase(), bounds))
}
