//! One-sided reconstruction uses only cells incident to the admitted interface facet.
use super::*;

pub(super) fn owned_cell(
    mesh: &impl MeshTopology,
    adjacent: &[MeshEntity],
    owned: &[usize],
) -> Result<Option<MeshEntity>, Diagnostic> {
    let mut matches = adjacent.iter().copied().filter(|cell| {
        super::measure::vertices(mesh, *cell)
            .expect("authenticated adjacent vertices")
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

pub(super) fn tabulate<'mesh>(
    mesh: &'mesh impl MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
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
