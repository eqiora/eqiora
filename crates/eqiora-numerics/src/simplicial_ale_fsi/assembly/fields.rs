//! Local Field projections and their power normalization.

use super::*;

pub(super) fn local_velocity_coefficients<const D: usize>(
    vertices: &[MeshEntity],
    field: eqiora_core::RawId,
    values: &std::collections::BTreeMap<crate::region_assembly::mapping::FieldDof, f64>,
    cell: CellId,
) -> Result<Vec<[f64; D]>, Diagnostic> {
    if vertices.len() != D + 1 {
        return Err(invalid(
            "ALE velocity simplex has an incomplete vertex closure",
        ));
    }
    vertices
        .iter()
        .copied()
        .chain(std::iter::once(MeshEntity::new(D, cell.index())))
        .map(|entity| {
            let mut vector = [0.0; D];
            for (component, value) in vector.iter_mut().enumerate() {
                *value = *values
                    .get(&crate::region_assembly::mapping::FieldDof {
                        field,
                        entity,
                        slot: 0,
                        component,
                    })
                    .ok_or_else(|| {
                        invalid("ALE velocity local projection omits exact Field/entity/component")
                    })?;
            }
            Ok(vector)
        })
        .collect()
}

pub(super) fn local_pressure_coefficients<const D: usize>(
    vertices: &[MeshEntity],
    field: eqiora_core::RawId,
    values: &std::collections::BTreeMap<crate::region_assembly::mapping::FieldDof, f64>,
) -> Result<Vec<f64>, Diagnostic> {
    if vertices.len() != D + 1 {
        return Err(invalid(
            "ALE constraint simplex has an incomplete vertex closure",
        ));
    }
    vertices
        .iter()
        .map(|&entity| {
            values
                .get(&crate::region_assembly::mapping::FieldDof {
                    field,
                    entity,
                    slot: 0,
                    component: 0,
                })
                .copied()
                .ok_or_else(|| {
                    invalid("ALE constraint projection omits exact Field/entity coordinate")
                })
        })
        .collect()
}

pub(super) fn fluid_row_scales<const D: usize>(plan: &AleFsiStepPlan<D>) -> Vec<f64> {
    let scale = plan.scale();
    let power = scale.power();
    (0..fluid_local_size::<D>())
        .map(|row| {
            if row < fluid_pressure_offset::<D>() {
                scale.velocity() / power
            } else {
                scale.pressure() / power
            }
        })
        .collect()
}

pub(super) const fn fluid_pressure_offset<const D: usize>() -> usize {
    (D + 2) * D
}

pub(super) const fn fluid_local_size<const D: usize>() -> usize {
    fluid_pressure_offset::<D>() + D + 1
}

pub(super) const fn solid_local_size<const D: usize>() -> usize {
    (D + 1) * D
}
