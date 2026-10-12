//! Compose common equation actions with exact Field and geometry directions.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use eqiora_meshing::AffineGeometryLinearization;
use std::collections::BTreeMap;

struct FluidCell<'a> {
    form: &'a BoundRegionForm<f64>,
    field: RawId,
    previous: BTreeMap<RawId, Vec<f64>>,
    point: Vec<f64>,
}

fn bind_cell<'a, const D: usize>(
    layout: &'a FsiLayout<D>,
    cell: usize,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
) -> Result<FluidCell<'a>, Diagnostic> {
    let domain = layout.cell_domain(cell)?;
    if layout.pressure_field(domain).is_none() {
        return Err(invalid(
            "ALE transported cell has no exact constraint witness",
        ));
    }
    let form = layout.region_form(domain)?;
    let previous = layout.mapping().physical_cell_history(
        cell,
        form,
        Some(&previous.physical_state().fields),
    )?;
    let mut point = Vec::new();
    for field in form.fields() {
        point.extend(
            layout
                .mapping()
                .physical_cell_coefficients(
                    cell,
                    field.field,
                    field.field,
                    &current.physical_state().fields,
                )?
                .into_iter()
                .map(|value| value / field.scale),
        );
    }
    Ok(FluidCell {
        form,
        field: layout.velocity_field(domain)?,
        previous,
        point,
    })
}

pub(super) fn evaluate_fluid_residual<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    cell: &PreparedAleFsiCell,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
) -> Result<MappedCellResidual, Diagnostic> {
    let bubble_cell = CellId::new(cell_index);
    if cell.bubble_cell != Some(bubble_cell) {
        return Err(invalid(
            "ALE cell identity changed after structural preparation",
        ));
    }
    let binding = bind_cell(layout, cell_index, previous, current)?;
    let (action, _) = binding.form.prepare_gcl_cell(
        binding.field,
        geometry_action,
        cell_index,
        None,
        quadrature,
    )?;
    let residual = action.residual(&binding.previous, &binding.point)?;
    Ok((
        residual,
        cell.reduced_map.clone(),
        cell.full_map.clone(),
        CellResidualSource::Fluid { bubble_cell },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_fluid_jacobian<const D: usize>(
    layout: &FsiLayout<D>,
    cell_index: usize,
    expected_bubble_cell: CellId,
    reference: &SimplicialMesh,
    quadrature: &QuadratureRule,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    geometry_action: &FixedTopologyGeometryAction<D>,
    candidate_width: usize,
    directions: &[AlgebraicDirection<D>],
) -> Result<Vec<f64>, Diagnostic> {
    if expected_bubble_cell.index() != cell_index || directions.len() != candidate_width {
        return Err(invalid("ALE Jacobian cell or direction inventory changed"));
    }
    let binding = bind_cell(layout, cell_index, previous, current)?;
    let (action, _) = binding.form.prepare_gcl_cell(
        binding.field,
        geometry_action,
        cell_index,
        None,
        quadrature,
    )?;
    let action = action.linearize(&binding.previous, &binding.point)?;
    let count = binding.point.len();
    let map = geometry_action
        .cell(cell_index)
        .ok_or_else(|| invalid("ALE Jacobian has a foreign cell"))?
        .current_map();
    // Differentiate the intrinsic affine coordinates once per local action.
    // Global mesh-motion columns then compose with these columns; they do not
    // reintegrate a local operator for every global unknown.
    let mut geometry_columns = Vec::with_capacity(D + D * D);
    for axis in 0..D + D * D {
        let mut origin = vec![0.; D];
        let mut jacobian = vec![0.; D * D];
        if axis < D {
            origin[axis] = 1.;
        } else {
            jacobian[axis - D] = 1.;
        }
        let direction = AffineGeometryLinearization::new(map.clone(), origin, jacobian)?;
        let (_, tangent) = binding.form.prepare_gcl_cell(
            binding.field,
            geometry_action,
            cell_index,
            Some(&direction),
            quadrature,
        )?;
        geometry_columns.push(
            tangent
                .expect("requested geometry derivative")
                .residual(&binding.previous, &binding.point)?,
        );
    }
    let mut matrix = zeroed(
        count
            .checked_mul(candidate_width)
            .ok_or_else(|| invalid("ALE Jacobian size overflows usize"))?,
        "fluid cell Jacobian",
    )?;
    let signs = layout.mapping().cell_signs(cell_index)?;
    for (column, direction) in directions.iter().enumerate() {
        let mut local_direction = Vec::with_capacity(count);
        for field in binding.form.fields() {
            for (key, sign) in layout
                .mapping()
                .cell_field_keys(cell_index, field.field)?
                .iter()
                .zip(&signs[field.range.clone()])
            {
                let value = direction
                    .physical
                    .get(key)
                    .ok_or_else(|| invalid("ALE direction omits an exact Field coefficient"))?;
                local_direction.push(value * f64::from(*sign) / field.scale);
            }
        }
        if direction.coordinate.len() != reference.vertices().len() {
            return Err(invalid(
                "ALE coordinate direction differs from its reference vertex inventory",
            ));
        }
        let coordinate_direction = direction
            .coordinate
            .iter()
            .map(|value| value.to_vec())
            .collect::<Vec<_>>();
        let geometry_direction = geometry_action
            .current_mesh()
            .linearized_geometry_map(MeshEntity::new(D, cell_index), &coordinate_direction)?;
        for row in 0..count {
            let state = action.jacobian[row * count..(row + 1) * count]
                .iter()
                .zip(&local_direction)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let shape = geometry_columns
                .iter()
                .zip(
                    geometry_direction
                        .origin_tangent()
                        .iter()
                        .chain(geometry_direction.jacobian_tangent()),
                )
                .map(|(values, direction)| values[row] * direction)
                .sum::<f64>();
            matrix[row * candidate_width + column] = state + shape;
        }
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid("ALE Jacobian is non-finite"));
    }
    Ok(matrix)
}
