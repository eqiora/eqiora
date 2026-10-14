use eqiora_core::Diagnostic;
use eqiora_meshing::MeshGeometry;
use eqiora_schema::kernel::BoundarySide;

use super::{IncompressibleFlowScaleProfile2d, invalid_realization};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::canonical_stokes::TransientIncompressibleNavierStokesModel;
use crate::canonical_stokes::realization::NormalizedCartesianSimplicialMesh2d;
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d,
};

const DIMENSION: usize = 2;

pub(crate) fn pressure_uses_gauge(
    model: &TransientIncompressibleNavierStokesModel<2>,
) -> Result<bool, Diagnostic> {
    let mut essential = 0_usize;
    let mut traction = 0_usize;
    for disposition in model.boundaries().map(|entry| entry.disposition()) {
        match disposition {
            PhysicalBoundaryDisposition::TraceZero => essential += 1,
            PhysicalBoundaryDisposition::FluxZero => traction += 1,
            PhysicalBoundaryDisposition::Prescribed(law) => match law.quantity() {
                PhysicalBoundaryQuantity::Trace => essential += 1,
                PhysicalBoundaryQuantity::Flux => traction += 1,
            },
            PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                return Err(invalid_realization(format!(
                    "live transient PortBinding {connection} through Port {port} requires an explicit trace-space interface Realization"
                )));
            }
        }
    }
    match (essential, traction) {
        (count, 0) if count > 0 && count == model.boundary_entries.len() => Ok(true),
        (essential, traction)
            if essential > 0
                && traction > 0
                && essential + traction == model.boundary_entries.len() =>
        {
            Ok(false)
        }
        (0, count) if count == model.boundary_entries.len() => Err(invalid_realization(
            "all-traction transient boundary is invalid because the velocity is otherwise determined only up to a constant",
        )),
        _ => Err(invalid_realization(
            "transient boundary must contain both essential-velocity and constant-traction meaning, or a complete essential partition",
        )),
    }
}

pub(super) fn numerical_boundary(
    model: &TransientIncompressibleNavierStokesModel<2>,
    normalized: &NormalizedCartesianSimplicialMesh2d,
) -> Result<SimplicialMiniStokesBoundary2d, Diagnostic> {
    let facets = normalized
        .boundary_facets
        .iter()
        .map(|(facet, axis, side)| {
            let disposition = model
                .cartesian_boundary(*axis, *side)
                .ok_or_else(|| {
                    invalid_realization(format!(
                        "transient boundary inventory omits axis {axis} {side:?}"
                    ))
                })?
                .disposition();
            let condition = match disposition {
                PhysicalBoundaryDisposition::TraceZero => {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::FluxZero => {
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                        value: [0.0; DIMENSION],
                    }
                }
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::Prescribed(law) => {
                    return Err(invalid_realization(format!(
                        "transient prescribed traction Relation {} is not a spatially constant traction admitted by this slice",
                        law.relation()
                    )));
                }
                PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                    return Err(invalid_realization(format!(
                        "live transient PortBinding {connection} through Port {port} requires an explicit trace-space interface Realization"
                    )));
                }
            };
            Ok(SimplicialMiniStokesBoundaryFacet2d::new(
                *facet, condition,
            ))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    SimplicialMiniStokesBoundary2d::new(&normalized.mesh, facets)
        .map_err(|error| invalid_realization(error.message()))
}

pub(super) fn essential_velocity(
    model: &TransientIncompressibleNavierStokesModel<2>,
    scales: IncompressibleFlowScaleProfile2d,
    coordinate_hat: [f64; DIMENSION],
) -> Result<[f64; DIMENSION], Diagnostic> {
    let lower = [model.bounds()[0][0], model.bounds()[1][0]];
    let length = scales.length_value();
    let physical = [
        lower[0] + length * coordinate_hat[0],
        lower[1] + length * coordinate_hat[1],
    ];
    let mut selected = None;
    for axis in 0..DIMENSION {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let side_index = usize::from(side == BoundarySide::Upper);
            let normalized_bound = (model.bounds()[axis][side_index] - lower[axis]) / length;
            if coordinate_hat[axis] != normalized_bound {
                continue;
            }
            let disposition = model
                .cartesian_boundary(axis, side)
                .expect("lowered transient model owns every exact side")
                .disposition();
            let value = match disposition {
                PhysicalBoundaryDisposition::TraceZero => Some([0.0; DIMENSION]),
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    Some(
                        model
                            .prescribed_cartesian_normal_velocity(axis, side, &physical)?
                            .ok_or_else(|| {
                                invalid_realization(format!(
                                    "prescribed velocity Relation {} has no retained normal-velocity expression",
                                    law.relation()
                                ))
                            })?,
                    )
                }
                _ => None,
            };
            let Some(value) = value else {
                continue;
            };
            if selected.is_some_and(|existing| existing != value) {
                return Err(invalid_realization(
                    "essential velocity prescriptions disagree at a shared Cartesian corner",
                ));
            }
            selected = Some(value);
        }
    }
    selected
        .map(|value| {
            [
                value[0] / scales.velocity_value(),
                value[1] / scales.velocity_value(),
            ]
        })
        .ok_or_else(|| {
            invalid_realization(
                "an essential boundary vertex is absent from the canonical trace inventory",
            )
        })
}

pub(super) fn require_compatible_complete_trace(
    model: &TransientIncompressibleNavierStokesModel<2>,
    normalized: &NormalizedCartesianSimplicialMesh2d,
    scales: IncompressibleFlowScaleProfile2d,
) -> Result<(), Diagnostic> {
    let mut net_flux = CompensatedSum::default();
    let mut flux_scale = CompensatedSum::default();
    for (facet, axis, side) in &normalized.boundary_facets {
        let vertices = normalized
            .mesh
            .entity_vertices(*facet)
            .expect("validated boundary facet owns two vertices");
        let left = &normalized.mesh.vertices()[vertices[0].index()];
        let right = &normalized.mesh.vertices()[vertices[1].index()];
        let left_velocity = essential_velocity(model, scales, [left[0], left[1]])?;
        let right_velocity = essential_velocity(model, scales, [right[0], right[1]])?;
        let normal_sign = if *side == BoundarySide::Lower {
            -1.0
        } else {
            1.0
        };
        let contribution = 0.5
            * normalized
                .mesh
                .geometry_map(*facet)
                .expect("validated boundary facet owns affine geometry")
                .measure_scale()
            * normal_sign
            * (left_velocity[*axis] + right_velocity[*axis]);
        net_flux.add(contribution);
        flux_scale.add(contribution.abs());
    }
    let count = normalized.boundary_facets.len() as f64;
    let accumulated_roundoff = count * f64::EPSILON;
    if accumulated_roundoff >= 1.0 {
        return Err(invalid_realization(
            "transient boundary inventory exceeds the floating-point validation limit",
        ));
    }
    let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
    let tolerance = 128.0 * gamma * flux_scale.total().max(1.0);
    let net_flux = net_flux.total();
    if net_flux.abs() > tolerance {
        return Err(invalid_realization(format!(
            "prescribed velocity has non-zero net parent-outward flux {net_flux:e} (tolerance {tolerance:e})"
        )));
    }
    Ok(())
}

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum
    }
}
