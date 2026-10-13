//! Compose common equation actions with exact Field and geometry directions.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use std::collections::BTreeMap;

struct CurrentCell<'a> {
    form: &'a BoundRegionForm<f64>,
    previous: BTreeMap<RawId, Vec<f64>>,
    point: Vec<f64>,
}

fn bind_cell<'a, const D: usize>(
    layout: &'a FsiLayout<D>,
    cell: usize,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
) -> Result<CurrentCell<'a>, Diagnostic> {
    let domain = layout.cell_domain(cell)?;
    let form = layout.region_form(domain)?;
    let previous = layout.mapping().physical_cell_history(
        cell,
        form,
        Some(&previous.physical_state().fields),
    )?;
    let point =
        layout
            .mapping()
            .physical_cell_point(cell, form, &current.physical_state().fields)?;
    Ok(CurrentCell {
        form,
        previous,
        point,
    })
}

pub(super) fn evaluate_current_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    cell: &PreparedAleFsiCell,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
) -> Result<MappedCellResidual, Diagnostic> {
    let PreparedCellGeometry::Current { pullbacks } = &cell.geometry else {
        return Err(invalid(
            "current cell lacks its exact geometry configuration",
        ));
    };
    let binding = bind_cell(layout, cell_index, previous, current)?;
    let (action, _) =
        binding
            .form
            .prepare_gcl_cell(pullbacks, geometry_action, cell_index, None, quadrature)?;
    let residual = action.residual(&binding.previous, &binding.point)?;
    Ok((
        residual,
        cell.reduced_map.clone(),
        cell.full_map.clone(),
        CellResidualSource::CurrentGeometry,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_current_jacobian<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    cell: &PreparedAleFsiCell,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    candidate_width: usize,
    directions: &[RegionDirection<D>],
) -> Result<Vec<f64>, Diagnostic> {
    if cell.cell.index() != cell_index || directions.len() != candidate_width {
        return Err(invalid("ALE Jacobian cell or direction inventory changed"));
    }
    let PreparedCellGeometry::Current { pullbacks } = &cell.geometry else {
        return Err(invalid(
            "current cell lacks its exact geometry configuration",
        ));
    };
    let binding = bind_cell(layout, cell_index, previous, current)?;
    layout.mapping().linearize_geometry_cell(
        cell_index,
        binding.form,
        geometry_action.current_mesh(),
        &binding.previous,
        &binding.point,
        directions,
        |direction| {
            binding.form.prepare_gcl_cell(
                pullbacks,
                geometry_action,
                cell_index,
                direction,
                quadrature,
            )
        },
    )
}
