//! One-sided reconstruction uses only cells incident to the admitted interface facet.
use super::*;
use eqiora_meshing::CartesianMesh;

pub(super) fn adjacent_cells(
    mesh: &CartesianMesh,
    cell: MeshEntity,
    axis: usize,
    coordinate: f64,
) -> Result<Vec<MeshEntity>, Diagnostic> {
    let dimension = mesh.topological_dimension();
    let facets = mesh.incidence(cell, dimension - 1).expect("cell facets");
    let facets = facets
        .into_iter()
        .filter(|facet| {
            mesh.entity_vertices(facet.entity)
                .expect("facet vertices")
                .iter()
                .all(|vertex| {
                    mesh.vertex_coordinates(*vertex).expect("facet vertex")[axis] == coordinate
                })
        })
        .collect::<Vec<_>>();
    let [facet] = facets.as_slice() else {
        return Err(invalid(
            "Interface cell requires exactly one retained interface facet",
        ));
    };
    let adjacent = mesh
        .incidence(facet.entity, dimension)
        .expect("facet cells");
    if adjacent.len() != 2 {
        return Err(invalid(
            "Interface observation requires exactly two adjacent cells",
        ));
    }
    Ok(adjacent.into_iter().map(|entry| entry.entity).collect())
}

pub(super) fn owned_cell(
    mesh: &CartesianMesh,
    adjacent: &[MeshEntity],
    owned: &[usize],
) -> Result<Option<MeshEntity>, Diagnostic> {
    let mut matches = adjacent.iter().copied().filter(|cell| {
        mesh.entity_vertices(*cell)
            .expect("adjacent vertices")
            .iter()
            .all(|vertex| owned.binary_search(&vertex.index()).is_ok())
    });
    let cell = matches.next();
    if matches.next().is_some() {
        return Err(invalid(
            "Interface Field does not select one exact side cell",
        ));
    }
    Ok(cell)
}

pub(super) fn tabulate(
    mesh: &CartesianMesh,
    cell: MeshEntity,
    coordinates: &[f64],
    space: &DiscreteSpace,
) -> Result<(crate::discrete_space::BasisTabulation, Vec<f64>), Diagnostic> {
    let geometry = mesh.geometry_map(cell).expect("adjacent cell geometry");
    let dimension = mesh.topological_dimension();
    let inverse = geometry.inverse_jacobian()?;
    let mut center = vec![0.0; dimension];
    geometry.map_point(&vec![0.0; dimension], &mut center)?;
    let reference = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| {
                    inverse[row * dimension + column] * (coordinates[column] - center[column])
                })
                .sum()
        })
        .collect::<Vec<_>>();
    Ok((space.tabulate(&reference)?, inverse))
}
