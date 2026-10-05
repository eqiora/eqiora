//! Cartesian design-coordinate admission and geometry actions.

use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryLinearization, CartesianMesh, MeshEntity, MeshGeometry};
use eqiora_schema::kernel::BoundarySide;

use super::{ScalarEllipticCartesianModel, invalid};
use crate::spatial_design::SpatialDesignCoordinate;

pub(super) struct SelectedDesignCoordinates {
    pub(super) coordinates: Vec<SpatialDesignCoordinate>,
    pub(super) values: Vec<f64>,
    pub(super) actions: Vec<SpatialDesignAction>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum SpatialDesignAction {
    ModelParameter(usize),
    CartesianBound { axis: usize, side: BoundarySide },
}

pub(super) fn select_design_coordinates(
    model: &ScalarEllipticCartesianModel,
    selected: &[SpatialDesignCoordinate],
) -> Result<SelectedDesignCoordinates, Diagnostic> {
    if selected.is_empty() {
        return Err(invalid(
            "Cartesian differentiation requires at least one explicitly selected design coordinate",
        ));
    }
    if selected
        .iter()
        .enumerate()
        .any(|(index, field)| selected[..index].contains(field))
    {
        return Err(invalid(
            "Cartesian differentiation design selection contains a duplicate",
        ));
    }
    let mut values = Vec::with_capacity(selected.len());
    let mut actions = Vec::with_capacity(selected.len());
    for coordinate in selected {
        match *coordinate {
            SpatialDesignCoordinate::ModelParameter {
                parameter: field,
                component,
                imaginary,
            } => {
                if component != 0 || imaginary {
                    return Err(invalid(
                        "Cartesian scalar Parameter requires its real scalar coordinate",
                    ));
                }
                let Some(index) = model
                    .parameter_fields()
                    .iter()
                    .position(|candidate| candidate == &field)
                else {
                    return Err(invalid(
                        "selected model Parameter does not affect the lowered Cartesian relation",
                    ));
                };
                values.push(model.parameter_values()[index]);
                actions.push(SpatialDesignAction::ModelParameter(index));
            }
            SpatialDesignCoordinate::CartesianBound { domain, axis, side } => {
                if domain != model.domain_id() || axis >= model.dimension() {
                    return Err(invalid(
                        "selected Cartesian bound does not belong to the lowered Domain",
                    ));
                }
                values.push(match side {
                    BoundarySide::Lower => model.bounds()[axis][0],
                    BoundarySide::Upper => model.bounds()[axis][1],
                });
                actions.push(SpatialDesignAction::CartesianBound { axis, side });
            }
        }
    }
    Ok(SelectedDesignCoordinates {
        coordinates: selected.to_vec(),
        values,
        actions,
    })
}

pub(super) fn activate_model_parameter(action: SpatialDesignAction, tangent: &mut [f64]) {
    if let SpatialDesignAction::ModelParameter(index) = action {
        tangent[index] = 1.0;
    }
}

pub(super) fn design_geometry(
    mesh: &CartesianMesh,
    entity: MeshEntity,
    action: SpatialDesignAction,
) -> Result<AffineGeometryLinearization, Diagnostic> {
    match action {
        SpatialDesignAction::ModelParameter(_) => AffineGeometryLinearization::stationary(
            mesh.geometry_map(entity)
                .ok_or_else(|| invalid("Cartesian entity geometry is unavailable"))?,
        ),
        SpatialDesignAction::CartesianBound { axis, side } => mesh.linearize_axis_endpoints(
            entity,
            axis,
            match side {
                BoundarySide::Lower => [1.0, 0.0],
                BoundarySide::Upper => [0.0, 1.0],
            },
        ),
    }
}
