//! Exact cell-local MINI coefficients and algebraic numbering for remesh projection.
use super::*;
use crate::simplicial_ale_remesh::invalid;

pub(super) fn velocity_scalar_dofs(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cell: CellId,
    bubble: bool,
) -> Result<Vec<usize>, Diagnostic> {
    let mut dofs = cell_vertex_indices(mesh, cell)?.to_vec();
    if bubble {
        let position = partition.fluid_cells().binary_search(&cell).map_err(|_| {
            invalid("ALE FSI remesh fluid cell lacks a canonical MINI bubble position")
        })?;
        dofs.push(mesh.vertices().len() + position);
    }
    Ok(dofs)
}

pub(super) fn evaluate_velocity_cell(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cell: CellId,
    point: [f64; DIMENSION],
    vertex: &[[f64; COMPONENTS]],
    bubbles: Option<&std::collections::BTreeMap<CellId, [f64; COMPONENTS]>>,
) -> Result<[f64; COMPONENTS], Diagnostic> {
    if bubbles.is_some() && partition.fluid_cells().binary_search(&cell).is_err() {
        return Err(invalid("remesh bubble history selected a foreign cell"));
    }
    let vertices = cell_vertex_indices(mesh, cell)?;
    let basis = cell_basis(mesh, cell, point, bubbles.is_some())?;
    let mut value = std::array::from_fn(|component| {
        vertices
            .iter()
            .enumerate()
            .map(|(local, &vertex_index)| basis.values[local] * vertex[vertex_index][component])
            .sum::<f64>()
    });
    if let Some(bubbles) = bubbles {
        let bubble = bubbles
            .get(&cell)
            .ok_or_else(|| invalid("ALE FSI remesh history omits its exact MINI bubble cell"))?;
        for component in 0..COMPONENTS {
            value[component] += basis.values[3] * bubble[component];
        }
    }
    Ok(value)
}
