use std::collections::BTreeMap;

use eqiora_assembly::{AssemblyMap, DofId, LocalContribution, LocalUnknown};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath};
use eqiora_realization::CellCenteredConvectionScheme;
use eqiora_schema::kernel::BoundarySide;

use super::api::{ScalarTransportBoundaryRole, TransportFace2d};
use super::periodic::periodic_seam_faces;
use super::reconstruction::{AffineFaceTrace, FaceReconstructor, ReconstructionSummary};
use crate::canonical_transport::{
    ScalarTransportCartesianBoundary, ScalarTransportCartesianModel2d,
};
use crate::cartesian_fvm_geometry::{CartesianFacetAdjacency, cartesian_fvm_geometry};
use eqiora_meshing::CartesianMesh;

const DIMENSION: usize = 2;
pub(super) type BoundaryRoles2d = BTreeMap<(usize, BoundarySide), ScalarTransportBoundaryRole>;

pub(super) fn cell_geometry(mesh: &CartesianMesh) -> Result<(Vec<[f64; 2]>, Vec<f64>), Diagnostic> {
    let (cells, _) = cartesian_fvm_geometry::<2>(mesh)?;
    let mut centers = Vec::with_capacity(cells.len());
    let mut measures = Vec::with_capacity(cells.len());
    for cell in cells {
        centers.push(cell.center);
        measures.push(cell.measure);
    }
    Ok((centers, measures))
}

pub(super) fn transport_faces(
    model: &ScalarTransportCartesianModel2d,
    mesh: &CartesianMesh,
    centers: &[[f64; 2]],
    previous: &[f64],
    scheme: CellCenteredConvectionScheme,
    duration: f64,
) -> Result<
    (
        Vec<TransportFace2d>,
        BoundaryRoles2d,
        ReconstructionSummary,
        usize,
    ),
    Diagnostic,
> {
    let mut reconstructor =
        FaceReconstructor::new(model, mesh, centers, previous, scheme, duration)?;
    let (_, facets) = cartesian_fvm_geometry::<2>(mesh)?;
    let mut faces = Vec::with_capacity(facets.len());
    let mut role_by_side = BTreeMap::new();
    for facet in facets {
        let center = facet.center;
        let area = facet.measure;
        let normal_axis = facet.normal_axis;
        let velocity = model.advecting_velocity(&center)?;
        if velocity.iter().any(|value| !value.is_finite()) {
            return Err(invalid_numerics(
                "transport advecting velocity is non-finite",
            ));
        }
        match facet.adjacency {
            CartesianFacetAdjacency::Interior {
                lower,
                upper,
                center_distance,
            } => {
                let transmissibility = model.diffusivity() * area / center_distance;
                if !transmissibility.is_finite() || transmissibility <= 0.0 {
                    return Err(invalid_numerics(
                        "transport interior two-point geometry is invalid",
                    ));
                }
                faces.push(TransportFace2d::Interior {
                    lower,
                    upper,
                    outward_from_lower_flux: velocity[normal_axis] * area,
                    transmissibility,
                    advective_trace: reconstructor.interior_trace(
                        normal_axis,
                        lower,
                        upper,
                        velocity[normal_axis] * area,
                    )?,
                });
            }
            CartesianFacetAdjacency::Boundary {
                cell,
                side,
                center_distance,
            } => {
                let normal_sign = if side == BoundarySide::Lower {
                    -1.0
                } else {
                    1.0
                };
                let law = model
                    .boundary(normal_axis, side)
                    .expect("canonical transport lowerer produces a complete boundary set");
                if matches!(
                    law,
                    ScalarTransportCartesianBoundary::SpatialPeriodic { .. }
                ) {
                    continue;
                }
                let outward_volume_flux = normal_sign * velocity[normal_axis] * area;
                let role = if outward_volume_flux < 0.0 {
                    ScalarTransportBoundaryRole::Inflow
                } else if outward_volume_flux > 0.0 {
                    ScalarTransportBoundaryRole::Outflow
                } else {
                    ScalarTransportBoundaryRole::ImpermeableWall
                };
                if let Some(previous) = role_by_side.insert((normal_axis, side), role)
                    && previous != role
                {
                    return Err(invalid_realization(
                        "transport outward velocity changes boundary role along one Cartesian side",
                    ));
                }
                match (role, law) {
                    (
                        ScalarTransportBoundaryRole::Inflow,
                        ScalarTransportCartesianBoundary::PrescribedTrace(value),
                    ) => {
                        let trace = value.evaluate(&center)?;
                        let transmissibility = model.diffusivity() * area / center_distance;
                        if !trace.is_finite()
                            || !transmissibility.is_finite()
                            || transmissibility <= 0.0
                        {
                            return Err(invalid_numerics(
                                "transport inflow trace or two-point geometry is invalid",
                            ));
                        }
                        faces.push(TransportFace2d::PrescribedTrace {
                            cell,
                            outward_volume_flux,
                            transmissibility,
                            trace,
                            advective_trace: AffineFaceTrace::exact_boundary(trace)?,
                        });
                    }
                    (
                        ScalarTransportBoundaryRole::Outflow
                        | ScalarTransportBoundaryRole::ImpermeableWall,
                        ScalarTransportCartesianBoundary::PrescribedDiffusiveFlux(value),
                    ) => {
                        let flux = value.evaluate(&center)? * area;
                        if !flux.is_finite() {
                            return Err(invalid_numerics(
                                "transport prescribed diffusive flux is non-finite",
                            ));
                        }
                        faces.push(TransportFace2d::PrescribedDiffusiveFlux {
                            cell,
                            outward_volume_flux,
                            diffusive_flux_integral: flux,
                            role,
                            advective_trace: reconstructor.boundary_trace(
                                normal_axis,
                                side,
                                cell,
                                role,
                            )?,
                        });
                    }
                    (ScalarTransportBoundaryRole::Inflow, _) => {
                        return Err(invalid_realization(
                            "transport inflow requires an exact prescribed trace before assembly",
                        ));
                    }
                    (
                        ScalarTransportBoundaryRole::Outflow
                        | ScalarTransportBoundaryRole::ImpermeableWall,
                        _,
                    ) => {
                        return Err(invalid_realization(
                            "transport outflow or wall requires an exact prescribed diffusive flux before assembly",
                        ));
                    }
                }
            }
        }
    }
    let mut periodic_face_count = 0_usize;
    let probe = [
        0.5 * (model.bounds()[0][0] + model.bounds()[0][1]),
        0.5 * (model.bounds()[1][0] + model.bounds()[1][1]),
    ];
    let velocity = model.advecting_velocity(&probe)?;
    for (axis, normal_velocity) in velocity.into_iter().enumerate() {
        let lower = model
            .boundary(axis, BoundarySide::Lower)
            .and_then(ScalarTransportCartesianBoundary::spatial_periodic_binding);
        let upper = model
            .boundary(axis, BoundarySide::Upper)
            .and_then(ScalarTransportCartesianBoundary::spatial_periodic_binding);
        match (lower, upper) {
            (Some((lower_connection, _)), Some((upper_connection, _)))
                if lower_connection == upper_connection =>
            {
                let seam =
                    periodic_seam_faces(mesh, axis, normal_velocity, model.diffusivity(), scheme)?;
                periodic_face_count =
                    periodic_face_count.checked_add(seam.len()).ok_or_else(|| {
                        invalid_numerics("transport periodic face count overflows usize")
                    })?;
                faces.extend(seam);
            }
            (None, None) => {}
            _ => {
                return Err(invalid_realization(
                    "transport periodic sides do not form one exact axis pair",
                ));
            }
        }
    }
    Ok((
        faces,
        role_by_side,
        reconstructor.summary(),
        periodic_face_count,
    ))
}

pub(super) fn validate_side_roles(
    model: &ScalarTransportCartesianModel2d,
    roles: &BoundaryRoles2d,
) -> Result<(), Diagnostic> {
    for axis in 0..DIMENSION {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let periodic = matches!(
                model.boundary(axis, side),
                Some(ScalarTransportCartesianBoundary::SpatialPeriodic { .. })
            );
            if periodic == roles.contains_key(&(axis, side)) {
                return Err(invalid_realization(format!(
                    "transport boundary role inventory disagrees with canonical meaning on axis {axis} {side:?}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn face_packet(
    face: &TransportFace2d,
    matrix_scale: f64,
    row_scale: f64,
) -> Result<(LocalContribution<f64>, AssemblyMap<f64>), Diagnostic> {
    match *face {
        TransportFace2d::Interior {
            lower,
            upper,
            outward_from_lower_flux,
            transmissibility,
            ref advective_trace,
        } => {
            let columns = face_columns([lower, upper], advective_trace);
            let mut matrix = vec![0.0; 2 * columns.len()];
            add_local(
                &mut matrix,
                columns.len(),
                0,
                &columns,
                lower,
                matrix_scale * transmissibility,
            );
            add_local(
                &mut matrix,
                columns.len(),
                0,
                &columns,
                upper,
                -matrix_scale * transmissibility,
            );
            add_local(
                &mut matrix,
                columns.len(),
                1,
                &columns,
                lower,
                -matrix_scale * transmissibility,
            );
            add_local(
                &mut matrix,
                columns.len(),
                1,
                &columns,
                upper,
                matrix_scale * transmissibility,
            );
            for &(cell, coefficient) in advective_trace.terms() {
                add_local(
                    &mut matrix,
                    columns.len(),
                    0,
                    &columns,
                    cell.index(),
                    matrix_scale * outward_from_lower_flux * coefficient,
                );
                add_local(
                    &mut matrix,
                    columns.len(),
                    1,
                    &columns,
                    cell.index(),
                    -matrix_scale * outward_from_lower_flux * coefficient,
                );
            }
            let explicit_flux = outward_from_lower_flux * advective_trace.offset();
            Ok((
                LocalContribution::new(
                    2,
                    columns.len(),
                    matrix,
                    vec![-row_scale * explicit_flux, row_scale * explicit_flux],
                )?,
                AssemblyMap::new(
                    vec![Some(DofId::new(lower)), Some(DofId::new(upper))],
                    columns
                        .keys()
                        .map(|&cell| LocalUnknown::Free(DofId::new(cell)))
                        .collect(),
                )?,
            ))
        }
        TransportFace2d::PrescribedTrace {
            cell,
            outward_volume_flux,
            transmissibility,
            trace,
            ref advective_trace,
        } => {
            let columns = face_columns([cell], advective_trace);
            let mut matrix = vec![0.0; columns.len()];
            add_local(
                &mut matrix,
                columns.len(),
                0,
                &columns,
                cell,
                matrix_scale * transmissibility,
            );
            for &(unknown, coefficient) in advective_trace.terms() {
                add_local(
                    &mut matrix,
                    columns.len(),
                    0,
                    &columns,
                    unknown.index(),
                    matrix_scale * outward_volume_flux * coefficient,
                );
            }
            Ok((
                LocalContribution::new(
                    1,
                    columns.len(),
                    matrix,
                    vec![
                        row_scale
                            * (transmissibility * trace
                                - outward_volume_flux * advective_trace.offset()),
                    ],
                )?,
                AssemblyMap::new(
                    vec![Some(DofId::new(cell))],
                    columns
                        .keys()
                        .map(|&cell| LocalUnknown::Free(DofId::new(cell)))
                        .collect(),
                )?,
            ))
        }
        TransportFace2d::PrescribedDiffusiveFlux {
            cell,
            outward_volume_flux,
            diffusive_flux_integral,
            ref advective_trace,
            ..
        } => {
            let columns = face_columns([cell], advective_trace);
            let mut matrix = vec![0.0; columns.len()];
            for &(unknown, coefficient) in advective_trace.terms() {
                add_local(
                    &mut matrix,
                    columns.len(),
                    0,
                    &columns,
                    unknown.index(),
                    matrix_scale * outward_volume_flux * coefficient,
                );
            }
            Ok((
                LocalContribution::new(
                    1,
                    columns.len(),
                    matrix,
                    vec![
                        row_scale
                            * (diffusive_flux_integral
                                - outward_volume_flux * advective_trace.offset()),
                    ],
                )?,
                AssemblyMap::new(
                    vec![Some(DofId::new(cell))],
                    columns
                        .keys()
                        .map(|&cell| LocalUnknown::Free(DofId::new(cell)))
                        .collect(),
                )?,
            ))
        }
    }
}

fn face_columns<const N: usize>(
    required: [usize; N],
    trace: &AffineFaceTrace,
) -> BTreeMap<usize, usize> {
    let mut cells = required.into_iter().collect::<Vec<_>>();
    cells.extend(trace.terms().iter().map(|(cell, _)| cell.index()));
    cells.sort_unstable();
    cells.dedup();
    cells
        .into_iter()
        .enumerate()
        .map(|(local, cell)| (cell, local))
        .collect()
}

fn add_local(
    matrix: &mut [f64],
    columns: usize,
    row: usize,
    local_columns: &BTreeMap<usize, usize>,
    cell: usize,
    value: f64,
) {
    let column = local_columns[&cell];
    matrix[row * columns + column] += value;
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message).with_graph_path(GraphPath::new([
        "realization".to_owned(),
        "scalar-transport-fvm-2d".to_owned(),
        "faces".to_owned(),
    ]))
}

fn invalid_numerics(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message).with_graph_path(GraphPath::new([
        "numerics".to_owned(),
        "scalar-transport-fvm-2d".to_owned(),
        "faces".to_owned(),
    ]))
}
