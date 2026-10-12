//! Authenticate mesh partition and exterior coverage against semantic Domains.

use super::*;

pub(super) fn require_mesh_partition<const D: usize>(
    model: &AleFsiCartesianModel<D>,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
) -> Result<(), Diagnostic> {
    require_supported_dimension::<D>()?;
    if mesh.topological_dimension() != D || mesh.vertices().iter().any(|point| point.len() != D) {
        return Err(invalid_realization(format!(
            "fixed-topology ALE FSI canonical bridge requires one intrinsic {D}D mesh"
        )));
    }
    let replayed = FixedReferenceFsiPartition::<D>::new(
        mesh,
        partition.domains().map(|domain| {
            (
                domain,
                partition
                    .domain_cells(domain)
                    .expect("exact Domain")
                    .to_vec(),
            )
        }),
        &partition.quotients().collect::<Vec<_>>(),
    )?;
    if &replayed != partition {
        return Err(invalid_realization(
            "fixed-topology ALE FSI partition cache differs from exact mesh replay",
        ));
    }
    require_cells_in_bounds(
        mesh,
        partition
            .domain_cells(fluid_domain(model))
            .expect("lowered ALE fluid Domain"),
        model.fluid().bounds(),
        "fluid",
    )?;
    require_cells_in_bounds(
        mesh,
        partition
            .domain_cells(solid_domain(model))
            .expect("lowered ALE solid Domain"),
        model.solid().continuum().bounds(),
        "solid",
    )?;

    let interface = model.interface();
    let fluid_side = interface
        .endpoint(model.fluid().domain())
        .expect("lowered ALE interface owns the fluid Domain");
    let solid_side = interface
        .endpoint(model.solid().continuum().domain())
        .expect("lowered ALE interface owns the solid Domain");
    let interface_coordinate = match fluid_side.side() {
        BoundarySide::Lower => model.fluid().bounds()[interface.axis()][0],
        BoundarySide::Upper => model.fluid().bounds()[interface.axis()][1],
    };
    let interface_facets = partition
        .traces()
        .iter()
        .find(|trace| trace.quotient.source().owner() == interface.connection())
        .expect("lowered ALE quotient")
        .facets
        .iter()
        .map(|witness| FacetId::new(witness.facet.index()))
        .collect::<Vec<_>>();
    for facet in &interface_facets {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(D - 1, facet.index()))
            .ok_or_else(|| invalid_realization("ALE FSI interface facet is outside the mesh"))?;
        if vertices
            .iter()
            .any(|vertex| mesh.vertices()[vertex.index()][interface.axis()] != interface_coordinate)
        {
            return Err(invalid_realization(
                "fixed-topology ALE FSI partition interface does not lie on the exact semantic interface",
            ));
        }
    }

    let fluid_cells = partition
        .domain_cells(fluid_domain(model))
        .expect("lowered ALE fluid Domain")
        .iter()
        .map(|cell| cell.index())
        .collect::<BTreeSet<_>>();
    let mut fluid_coverage = [[false; 2]; D];
    let mut solid_coverage = [[false; 2]; D];
    let facet_count = mesh
        .entity_count(D - 1)
        .ok_or_else(|| invalid_realization("ALE FSI mesh omits its facet stratum"))?;
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(D - 1, facet_index);
        if !mesh
            .is_boundary_entity(facet)
            .ok_or_else(|| invalid_realization("ALE FSI facet is outside the mesh"))?
        {
            continue;
        }
        let adjacent = mesh
            .incidence(facet, D)
            .ok_or_else(|| invalid_realization("ALE FSI exterior facet has no cell incidence"))?;
        let [cell] = adjacent.as_slice() else {
            return Err(invalid_realization(
                "ALE FSI exterior facet must own exactly one adjacent cell",
            ));
        };
        let (bounds, coverage, interface_side) = if fluid_cells.contains(&cell.entity.index()) {
            (
                model.fluid().bounds(),
                &mut fluid_coverage,
                fluid_side.side(),
            )
        } else {
            (
                model.solid().continuum().bounds(),
                &mut solid_coverage,
                solid_side.side(),
            )
        };
        let vertices = mesh
            .entity_vertices(facet)
            .ok_or_else(|| invalid_realization("ALE FSI exterior facet has no vertex closure"))?;
        let mut matched = None;
        for (axis, axis_bounds) in bounds.iter().enumerate() {
            for (side_index, bound) in axis_bounds.iter().enumerate() {
                if vertices
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][axis] == *bound)
                {
                    if matched.is_some() {
                        return Err(invalid_realization(
                            "ALE FSI exterior facet ambiguously belongs to multiple semantic sides",
                        ));
                    }
                    matched = Some((axis, side_index));
                }
            }
        }
        let Some((axis, side_index)) = matched else {
            return Err(invalid_realization(
                "ALE FSI mesh exterior does not lie on an exact semantic side",
            ));
        };
        let side = if side_index == 0 {
            BoundarySide::Lower
        } else {
            BoundarySide::Upper
        };
        if axis == interface.axis() && side == interface_side {
            return Err(invalid_realization(
                "ALE FSI semantic interface appeared on the mesh exterior",
            ));
        }
        coverage[axis][side_index] = true;
    }
    require_exterior_coverage(fluid_coverage, interface.axis(), fluid_side.side(), "fluid")?;
    require_exterior_coverage(solid_coverage, interface.axis(), solid_side.side(), "solid")
}

fn require_cells_in_bounds<const D: usize>(
    mesh: &SimplicialMesh,
    cells: &[CellId],
    bounds: &[[f64; 2]; D],
    physics: &str,
) -> Result<(), Diagnostic> {
    for cell in cells {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(D, cell.index()))
            .ok_or_else(|| {
                invalid_realization(format!(
                    "fixed-topology ALE FSI {physics} cell is outside the mesh"
                ))
            })?;
        if vertices.iter().any(|vertex| {
            mesh.vertices()[vertex.index()]
                .iter()
                .enumerate()
                .any(|(axis, value)| *value < bounds[axis][0] || *value > bounds[axis][1])
        }) {
            return Err(invalid_realization(format!(
                "fixed-topology ALE FSI {physics} cell lies outside its exact semantic Domain"
            )));
        }
    }
    Ok(())
}

fn require_exterior_coverage<const D: usize>(
    coverage: [[bool; 2]; D],
    interface_axis: usize,
    interface_side: BoundarySide,
    physics: &str,
) -> Result<(), Diagnostic> {
    for (axis, sides) in coverage.iter().enumerate() {
        for (side_index, covered) in sides.iter().enumerate() {
            let side = if side_index == 0 {
                BoundarySide::Lower
            } else {
                BoundarySide::Upper
            };
            if !(*covered || axis == interface_axis && side == interface_side) {
                return Err(invalid_realization(format!(
                    "ALE FSI mesh does not cover the exact exterior {physics} side on axis {axis} {side:?}"
                )));
            }
        }
    }
    Ok(())
}
