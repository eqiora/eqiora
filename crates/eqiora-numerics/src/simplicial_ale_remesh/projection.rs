mod fields;
mod geometry;
mod normalization;
mod results;
mod velocity;
use crate::canonical_fsi::FinalizedResolvedFixedTopologyAleFsi;
use fields::{ProjectionFields, source_coefficients, target_state};
use geometry::derive_target_geometry;
use normalization::{
    RemeshNormalization2d, divide_scalars, divide_vectors, divided_row_unchecked, divided_rows,
    dot, finite_sqrt, integer_sqrt,
};
use results::{PressureProjection, VectorP1Projection, VelocityProjection};
use velocity::{evaluate_velocity_cell, velocity_scalar_dofs};

use crate::simplicial_fsi::FixedReferenceFsiState;
use eqiora_core::{Id, entity::kinds};
use eqiora_realization::P1HarmonicMeshMotionPolicy;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshTopology, OverlapCoordinateChart2d, QuadratureRule,
    RetainedFacetSide2d, SimplicialMesh, SimplicialRevisionOverlap2d, VertexId,
};
use eqiora_solver::{LinearOperatorProperties, LinearSolveRequest};

use crate::simplicial_ale_fsi::{AleFsiState, P1HarmonicMeshMotionAction};
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiPartition, FixedReferenceFsiScale,
};

use super::contract::{AcceptedAleFsiRemeshProjection2d, AleFsiRemeshProjectionEvidence2d};
use super::integration::{
    cell_basis, checked_product, checked_sum, dense_zeroed, euclidean_norm,
    independent_constraints, integrate_cell, integrate_physical_triangle, require_auxiliary_budget,
    require_constraint_work_budget, require_dense_dimension, require_projection_solver,
    require_quadrature, solve_dense,
};

const DIMENSION: usize = 2;
const COMPONENTS: usize = 2;

/// Project one accepted ALE FSI state onto a topology-distinct target mesh.
///
/// The transition has exact zero model duration.  Absolute solid displacement
/// is projected in the material chart before target harmonic geometry is
/// derived.  Fluid velocity/pressure are then projected through a newly built
/// current-spatial overlap, while solid velocity remains in the material
/// chart.  The returned finalizer input exists only after independent replay
/// of every admitted constraint and moment.
///
/// # Errors
/// Returns a structured discretization, geometry, realization, or solve
/// diagnostic for an incompatible source/target root, incomplete overlap,
/// unsupported solver/quadrature, failed trace/constraint projection, bad
/// target geometry, or failed independent numerical replay.
#[allow(clippy::too_many_arguments)]
pub fn project_simplicial_ale_fsi_remesh_2d(
    source_plan: &FinalizedResolvedFixedTopologyAleFsi<2>,
    source_state: &AleFsiState<2>,
    target_reference: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    target_motion: &P1HarmonicMeshMotionAction<2>,
    scale: FixedReferenceFsiScale<2>,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<AcceptedAleFsiRemeshProjection2d, Diagnostic> {
    require_quadrature(quadrature)?;
    require_projection_solver(solver)?;
    let source_reference = source_plan.reference();
    let source_partition = source_plan.partition();
    let source_motion = source_plan.motion();
    let policy = source_motion.policy();
    let material = source_plan.step_plan().material();
    source_motion.validate_reference(source_reference, source_partition)?;
    target_motion.validate_reference(target_reference, target_partition)?;
    source_state.validate_against(source_reference, source_partition, source_motion)?;
    let fields = ProjectionFields::admit(source_plan, target_partition, target_motion)?;
    source_plan.layout().mapping().validate_step_history(
        &source_state.physical_state().fields,
        source_plan.layout().time_step(),
    )?;
    let source = source_coefficients(
        source_reference,
        source_partition,
        source_state.physical_state(),
        fields,
    )?;
    let normalization = RemeshNormalization2d::new(
        scale,
        material
            .density(fields.fluid_velocity)
            .ok_or_else(|| super::invalid("remesh velocity lacks exact density"))?,
        material
            .density(fields.solid_velocity)
            .ok_or_else(|| super::invalid("remesh rate lacks exact density"))?,
    )?;
    require_exact_zero_exterior_velocity(source_reference, &source.velocity)?;
    require_dense_dimension(target_reference.vertices().len())?;

    let solid_reference_overlap = material_overlap(
        source_reference,
        source_partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
        target_reference,
        target_partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
    )?;
    let displacement_prescribed = replay_solid_boundary_trace(
        policy,
        &solid_reference_overlap,
        source_reference,
        &source.displacement,
        target_reference,
        target_partition,
    )?;
    let mut displacement_mass = assemble_p1_mass(
        target_reference,
        target_partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
        quadrature,
        1.0,
    )?;
    let mut displacement_mixed = assemble_p1_mixed(
        policy,
        &solid_reference_overlap,
        source_reference,
        target_reference,
        source_partition,
        target_partition,
        &source.displacement,
        quadrature,
    )?;
    divide_scalars(&mut displacement_mass, normalization.area)?;
    divide_vectors(&mut displacement_mixed, normalization.displacement_rhs()?)?;
    let displacement_projection = project_vector_p1_with_trace(
        target_reference.vertices().len(),
        &displacement_mass,
        &displacement_mixed,
        &displacement_prescribed,
        scale.length(),
        solver,
    )?;
    let target_displacement = displacement_projection.coefficients;
    let maximum_displacement_trace_defect =
        trace_defect_vector(&target_displacement, &displacement_prescribed)?.max(
            retained_p1_trace_defect(
                &solid_reference_overlap,
                source_reference,
                &source.displacement,
                target_reference,
                &target_displacement,
            )?,
        );
    let displacement_l2_error = vector_p1_l2_error(
        &solid_reference_overlap,
        source_reference,
        target_reference,
        &source.displacement,
        &target_displacement,
        quadrature,
        1.0,
    )?;

    let target_geometry = derive_target_geometry(
        policy,
        target_reference,
        target_partition,
        target_motion,
        &target_displacement,
    )?;
    let source_current = source_state.geometry().reconstruct_mesh(source_reference)?;
    let target_current = target_geometry.reconstruct_mesh(target_reference)?;
    let fluid_current_overlap = current_fluid_overlap(
        &source_current,
        source_partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain"),
        &target_current,
        target_partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain"),
    )?;

    let velocity = project_velocity(
        policy,
        source_reference,
        &source_current,
        source_partition,
        &source.velocity,
        &source.bubbles,
        target_reference,
        &target_current,
        target_partition,
        &solid_reference_overlap,
        &fluid_current_overlap,
        normalization,
        quadrature,
        solver,
    )?;
    let pressure = project_pressure(
        policy,
        &source_current,
        source_partition,
        &source.pressure,
        &target_current,
        target_partition,
        &fluid_current_overlap,
        normalization,
        quadrature,
        solver,
    )?;

    let evidence = AleFsiRemeshProjectionEvidence2d::new(
        solid_reference_overlap,
        fluid_current_overlap,
        target_geometry,
        displacement_projection.reports,
        displacement_projection.right_hand_side_norms,
        velocity.report,
        velocity.right_hand_side_norm,
        pressure.report,
        pressure.right_hand_side_norm,
        scale,
        normalization.fluid_density.max(normalization.solid_density),
        velocity.independent_constraint_count,
        displacement_l2_error,
        velocity.fluid_l2_error,
        velocity.solid_l2_error,
        pressure.l2_error,
        displacement_projection.residual_norm,
        velocity.residual_norm,
        pressure.residual_norm,
        maximum_displacement_trace_defect,
        velocity.maximum_shared_trace_defect,
        velocity.maximum_exterior_trace_defect,
        velocity.weak_divergence_norm,
        velocity.source_momentum,
        velocity.target_momentum,
        pressure.source_moment,
        pressure.target_moment,
    )?;

    Ok(AcceptedAleFsiRemeshProjection2d::new(
        source_state.time(),
        target_state(
            target_reference,
            source_state.physical_state(),
            target_partition,
            fields,
            &velocity.vertex,
            &velocity.bubble,
            &pressure.coefficients,
            &target_displacement,
        )?,
        evidence,
    ))
}

pub(super) fn material_overlap(
    source: &SimplicialMesh,
    source_cells: &[CellId],
    target: &SimplicialMesh,
    target_cells: &[CellId],
) -> Result<SimplicialRevisionOverlap2d, Diagnostic> {
    let source_sides = material_boundary_sides(source, source_cells)?;
    let target_sides = material_boundary_sides(target, target_cells)?;
    SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        source,
        source_cells,
        target,
        target_cells,
    )?
    .with_retained_facets(source, &source_sides, target, &target_sides)
}

fn current_fluid_overlap(
    source: &SimplicialMesh,
    source_cells: &[CellId],
    target: &SimplicialMesh,
    target_cells: &[CellId],
) -> Result<SimplicialRevisionOverlap2d, Diagnostic> {
    let source_sides = material_boundary_sides(source, source_cells)?;
    let target_sides = material_boundary_sides(target, target_cells)?;
    SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::CurrentSpatial,
        source,
        source_cells,
        target,
        target_cells,
    )?
    .with_retained_facets(source, &source_sides, target, &target_sides)
}

fn material_boundary_sides(
    mesh: &SimplicialMesh,
    cells: &[CellId],
) -> Result<Vec<RetainedFacetSide2d>, Diagnostic> {
    let cell_set = cells
        .iter()
        .map(|cell| cell.index())
        .collect::<BTreeSet<_>>();
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .ok_or_else(|| super::invalid("ALE FSI remesh mesh lacks a facet stratum"))?;
    let mut sides = Vec::new();
    for index in 0..facet_count {
        let facet = MeshEntity::new(DIMENSION - 1, index);
        let parents = mesh
            .incidence(facet, DIMENSION)
            .ok_or_else(|| super::invalid("ALE FSI remesh facet lacks parent incidence"))?;
        let selected = parents
            .iter()
            .filter(|parent| cell_set.contains(&parent.entity.index()))
            .map(|parent| CellId::new(parent.entity.index()))
            .collect::<Vec<_>>();
        if selected.len() == 1 {
            sides.push(RetainedFacetSide2d::new(FacetId::new(index), selected[0]));
        }
    }
    if sides.is_empty() {
        return Err(super::invalid(
            "ALE FSI remesh material subset has no retained boundary facets",
        ));
    }
    Ok(sides)
}

fn assemble_p1_mass(
    mesh: &SimplicialMesh,
    cells: &[CellId],
    quadrature: &QuadratureRule,
    weight: f64,
) -> Result<Vec<f64>, Diagnostic> {
    let dimension = mesh.vertices().len();
    let mut mass = dense_zeroed(dimension)?;
    for &cell in cells {
        let vertices = cell_vertex_indices(mesh, cell)?;
        integrate_cell(mesh, cell, quadrature, |physical, measure| {
            let basis = cell_basis(mesh, cell, physical, false)?;
            for (local_row, &row) in vertices.iter().enumerate() {
                for (local_column, &column) in vertices.iter().enumerate() {
                    mass[row * dimension + column] +=
                        weight * measure * basis.values[local_row] * basis.values[local_column];
                }
            }
            Ok(())
        })?;
    }
    Ok(mass)
}

#[allow(clippy::too_many_arguments)]
fn assemble_p1_mixed(
    _policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    target_mesh: &SimplicialMesh,
    _source_partition: &FixedReferenceFsiPartition<2>,
    _target_partition: &FixedReferenceFsiPartition<2>,
    source_values: &[[f64; COMPONENTS]],
    quadrature: &QuadratureRule,
) -> Result<Vec<[f64; COMPONENTS]>, Diagnostic> {
    let mut mixed = vec![[0.0; COMPONENTS]; target_mesh.vertices().len()];
    for fragment in overlap.cell_fragments() {
        let source_cell = fragment.source_cell();
        let target_cell = fragment.target_cell();
        let source_vertices = cell_vertex_indices(source_mesh, source_cell)?;
        let target_vertices = cell_vertex_indices(target_mesh, target_cell)?;
        integrate_physical_triangle(fragment, quadrature, |physical, measure| {
            let source_basis = cell_basis(source_mesh, source_cell, physical, false)?;
            let target_basis = cell_basis(target_mesh, target_cell, physical, false)?;
            let source_value: [f64; COMPONENTS] = std::array::from_fn(|component| {
                source_vertices
                    .iter()
                    .enumerate()
                    .map(|(local, vertex)| {
                        source_basis.values[local] * source_values[*vertex][component]
                    })
                    .sum::<f64>()
            });
            for (local, &target_vertex) in target_vertices.iter().enumerate() {
                for component in 0..COMPONENTS {
                    mixed[target_vertex][component] +=
                        measure * target_basis.values[local] * source_value[component];
                }
            }
            Ok(())
        })?;
    }
    Ok(mixed)
}

fn project_vector_p1_with_trace(
    dimension: usize,
    mass: &[f64],
    mixed: &[[f64; COMPONENTS]],
    prescribed: &[Option<[f64; COMPONENTS]>],
    field_scale: f64,
    solver: LinearSolveRequest<'_>,
) -> Result<VectorP1Projection, Diagnostic> {
    if mass.len() != dimension * dimension
        || mixed.len() != dimension
        || prescribed.len() != dimension
    {
        return Err(super::invalid(
            "ALE FSI remesh displacement projection shapes are incompatible",
        ));
    }
    let prescribed = prescribed
        .iter()
        .map(|value| value.map(|value| value.map(|component| component / field_scale)))
        .collect::<Vec<_>>();
    let free = prescribed
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_none().then_some(index))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return Ok(VectorP1Projection {
            coefficients: prescribed
                .iter()
                .map(|value| {
                    value
                        .map(|value| value.map(|component| component * field_scale))
                        .ok_or_else(|| {
                            super::invalid("ALE FSI remesh prescribed projection is incomplete")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
            reports: Vec::new(),
            right_hand_side_norms: Vec::new(),
            residual_norm: 0.0,
        });
    }
    let mut coefficients = vec![[0.0; COMPONENTS]; dimension];
    for (index, value) in prescribed.iter().enumerate() {
        if let Some(value) = value {
            coefficients[index] = *value;
        }
    }
    let reduced_mass = free
        .iter()
        .flat_map(|&row| {
            free.iter()
                .map(move |&column| mass[row * dimension + column])
        })
        .collect::<Vec<_>>();
    let mut reports = Vec::with_capacity(COMPONENTS);
    let mut right_hand_side_norms = Vec::with_capacity(COMPONENTS);
    let mut residuals = Vec::with_capacity(COMPONENTS);
    for component in 0..COMPONENTS {
        let rhs = free
            .iter()
            .map(|&row| {
                mixed[row][component]
                    - prescribed
                        .iter()
                        .enumerate()
                        .filter_map(|(column, value)| {
                            value.map(|value| mass[row * dimension + column] * value[component])
                        })
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let (solution, report, residual) = solve_dense(
            reduced_mass.clone(),
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            solver,
        )?;
        right_hand_side_norms.push(euclidean_norm(&rhs)?);
        for (&index, value) in free.iter().zip(solution) {
            coefficients[index][component] = value;
        }
        reports.push(report);
        residuals.push(residual);
    }
    for value in &mut coefficients {
        for component in value {
            *component *= field_scale;
        }
    }
    Ok(VectorP1Projection {
        coefficients,
        reports,
        right_hand_side_norms,
        residual_norm: euclidean_norm(&residuals)?,
    })
}

fn require_exact_zero_exterior_velocity(
    mesh: &SimplicialMesh,
    velocity: &[[f64; COMPONENTS]],
) -> Result<(), Diagnostic> {
    if velocity.len() != mesh.vertices().len() {
        return Err(super::invalid(
            "ALE FSI remesh source velocity does not match the source vertex inventory",
        ));
    }
    if homogeneous_exterior_velocity_trace_defect(mesh, velocity)? != 0.0 {
        return Err(super::invalid(
            "ALE FSI remesh requires the admitted exact-zero exterior velocity trace",
        ));
    }
    Ok(())
}

/// Replay the complete homogeneous physical trace from its conforming P1
/// vertex coefficients. The MINI enrichment is cell-interior and therefore
/// vanishes identically on every exterior facet.
pub(super) fn homogeneous_exterior_velocity_trace_defect(
    mesh: &SimplicialMesh,
    velocity: &[[f64; COMPONENTS]],
) -> Result<f64, Diagnostic> {
    if velocity.len() != mesh.vertices().len() {
        return Err(super::invalid(
            "ALE FSI remesh exterior velocity does not match the mesh vertex inventory",
        ));
    }
    FixedReferenceFsiBoundary::<2>::homogeneous_exterior(mesh)?
        .fixed_zero_velocity_vertices()
        .iter()
        .map(|vertex| vector_distance(velocity[vertex.index()], [0.0; COMPONENTS]))
        .try_fold(0.0_f64, |maximum, defect| {
            defect
                .is_finite()
                .then_some(maximum.max(defect))
                .ok_or_else(|| {
                    super::invalid("ALE FSI remesh exterior velocity trace defect is non-finite")
                })
        })
}

fn replay_solid_boundary_trace(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_values: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
) -> Result<Vec<Option<[f64; COMPONENTS]>>, Diagnostic> {
    let solid = target_partition
        .domain_vertices(policy.solid_domain())
        .expect("authenticated motion Domain")
        .iter()
        .map(|vertex| vertex.index())
        .collect::<BTreeSet<_>>();
    let mut boundary = BTreeSet::new();
    for side in overlap.target_retained_facets() {
        boundary.extend(facet_vertex_indices(target_mesh, side.facet())?);
    }
    let mut prescribed = vec![None; target_mesh.vertices().len()];
    for (vertex, value) in prescribed.iter_mut().enumerate() {
        if !solid.contains(&vertex) {
            *value = Some([0.0; COMPONENTS]);
        } else if boundary.contains(&vertex) {
            *value = Some(replay_vector_trace_at(
                overlap,
                source_mesh,
                source_values,
                target_mesh.vertices()[vertex].as_slice(),
                None,
            )?);
        }
    }
    Ok(prescribed)
}

fn replay_interface_velocity_trace(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_values: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
) -> Result<Vec<Option<[f64; COMPONENTS]>>, Diagnostic> {
    let mut prescribed = vec![None; target_mesh.vertices().len()];
    for vertex in FixedReferenceFsiBoundary::<2>::homogeneous_exterior(target_mesh)?
        .fixed_zero_velocity_vertices()
    {
        prescribed[vertex.index()] = Some([0.0; COMPONENTS]);
    }
    let source_interface = interface_facet_indices(policy, source_partition);
    let target_interface = interface_facet_indices(policy, target_partition);
    for vertex in target_interface
        .iter()
        .flat_map(|&facet| {
            target_mesh
                .entity_vertices(MeshEntity::new(1, facet))
                .expect("authenticated interface facet")
        })
        .map(|entity| VertexId::new(entity.index()))
        .collect::<BTreeSet<_>>()
    {
        let value = replay_vector_trace_at(
            overlap,
            source_mesh,
            source_values,
            &target_mesh.vertices()[vertex.index()],
            Some((&source_interface, &target_interface)),
        )?;
        if let Some(existing) = prescribed[vertex.index()] {
            if existing != value {
                return Err(super::invalid(
                    "ALE FSI remesh shared/exterior velocity traces are inconsistent",
                ));
            }
        } else {
            prescribed[vertex.index()] = Some(value);
        }
    }
    Ok(prescribed)
}

fn interface_facet_indices(
    policy: P1HarmonicMeshMotionPolicy,
    partition: &FixedReferenceFsiPartition<2>,
) -> BTreeSet<usize> {
    partition
        .traces()
        .iter()
        .filter(|trace| trace.quotient.connection() == policy.interface())
        .flat_map(|trace| trace.facets.iter().map(|witness| witness.facet.index()))
        .collect()
}

fn replay_vector_trace_at(
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_values: &[[f64; COMPONENTS]],
    point: &[f64],
    interface: Option<(&BTreeSet<usize>, &BTreeSet<usize>)>,
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let point = point2(point)?;
    let mut replay = None;
    for fragment in overlap.retained_facet_fragments() {
        if interface.is_some_and(|(source, target)| {
            !source.contains(&fragment.source_facet().index())
                || !target.contains(&fragment.target_facet().index())
        }) || !point_on_segment(point, *fragment.segment())
        {
            continue;
        }
        let value =
            evaluate_vertex_vector(source_mesh, fragment.source_parent(), point, source_values)?;
        if let Some(previous) = replay {
            if vector_distance(previous, value) > trace_tolerance(point, previous, value) {
                return Err(super::invalid(
                    "ALE FSI remesh source boundary trace is not single valued",
                ));
            }
        } else {
            replay = Some(value);
        }
    }
    replay.ok_or_else(|| {
        super::invalid("ALE FSI remesh retained boundary does not cover a target trace vertex")
    })
}

fn point2(point: &[f64]) -> Result<[f64; DIMENSION], Diagnostic> {
    if point.len() != DIMENSION || point.iter().any(|value| !value.is_finite()) {
        return Err(super::invalid(
            "ALE FSI remesh expected one finite two-dimensional point",
        ));
    }
    Ok([point[0], point[1]])
}

fn point_on_segment(point: [f64; DIMENSION], segment: [[f64; DIMENSION]; 2]) -> bool {
    let direction = [segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]];
    let offset = [point[0] - segment[0][0], point[1] - segment[0][1]];
    let length = direction[0].hypot(direction[1]);
    let scale = point
        .into_iter()
        .chain(segment.into_iter().flatten())
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    let tolerance = 262_144.0 * f64::EPSILON * scale;
    let cross = direction[0] * offset[1] - direction[1] * offset[0];
    let dot = offset[0] * direction[0] + offset[1] * direction[1];
    cross.abs() <= tolerance * (1.0 + length)
        && dot >= -tolerance
        && dot <= length * length + tolerance
}

fn evaluate_vertex_vector(
    mesh: &SimplicialMesh,
    cell: CellId,
    point: [f64; DIMENSION],
    values: &[[f64; COMPONENTS]],
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let vertices = cell_vertex_indices(mesh, cell)?;
    let basis = cell_basis(mesh, cell, point, false)?;
    Ok(std::array::from_fn(|component| {
        vertices
            .iter()
            .enumerate()
            .map(|(local, &vertex)| basis.values[local] * values[vertex][component])
            .sum()
    }))
}

fn trace_defect_vector(
    values: &[[f64; COMPONENTS]],
    prescribed: &[Option<[f64; COMPONENTS]>],
) -> Result<f64, Diagnostic> {
    if values.len() != prescribed.len() {
        return Err(super::invalid(
            "ALE FSI remesh trace replay received incompatible shapes",
        ));
    }
    values
        .iter()
        .zip(prescribed)
        .filter_map(|(actual, expected)| {
            expected.map(|expected| vector_distance(*actual, expected))
        })
        .try_fold(0.0_f64, |maximum, defect| {
            defect
                .is_finite()
                .then_some(maximum.max(defect))
                .ok_or_else(|| super::invalid("ALE FSI remesh trace defect is non-finite"))
        })
}

pub(super) fn retained_p1_trace_defect(
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_values: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_values: &[[f64; COMPONENTS]],
) -> Result<f64, Diagnostic> {
    retained_p1_trace_defect_impl(
        overlap,
        source_mesh,
        source_values,
        target_mesh,
        target_values,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn retained_interface_p1_trace_defect(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_values: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    target_values: &[[f64; COMPONENTS]],
) -> Result<f64, Diagnostic> {
    let source_interface = interface_facet_indices(policy, source_partition);
    let target_interface = interface_facet_indices(policy, target_partition);
    retained_p1_trace_defect_impl(
        overlap,
        source_mesh,
        source_values,
        target_mesh,
        target_values,
        Some((&source_interface, &target_interface)),
    )
}

fn retained_p1_trace_defect_impl(
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_values: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_values: &[[f64; COMPONENTS]],
    facets: Option<(&BTreeSet<usize>, &BTreeSet<usize>)>,
) -> Result<f64, Diagnostic> {
    let mut maximum = 0.0_f64;
    let mut matched = false;
    for fragment in overlap.retained_facet_fragments() {
        if facets.is_some_and(|(source, target)| {
            !source.contains(&fragment.source_facet().index())
                || !target.contains(&fragment.target_facet().index())
        }) {
            continue;
        }
        matched = true;
        for point in fragment.segment() {
            let source = evaluate_vertex_vector(
                source_mesh,
                fragment.source_parent(),
                *point,
                source_values,
            )?;
            let target = evaluate_vertex_vector(
                target_mesh,
                fragment.target_parent(),
                *point,
                target_values,
            )?;
            let defect = vector_distance(source, target);
            if !defect.is_finite() {
                return Err(super::invalid(
                    "ALE FSI remesh retained-fragment trace defect is non-finite",
                ));
            }
            maximum = maximum.max(defect);
        }
    }
    matched.then_some(maximum).ok_or_else(|| {
        super::invalid("ALE FSI remesh retained trace has no role-compatible facet fragment")
    })
}

fn vector_distance(left: [f64; COMPONENTS], right: [f64; COMPONENTS]) -> f64 {
    (left[0] - right[0]).hypot(left[1] - right[1])
}

fn trace_tolerance(
    point: [f64; DIMENSION],
    left: [f64; COMPONENTS],
    right: [f64; COMPONENTS],
) -> f64 {
    let scale = point
        .into_iter()
        .chain(left)
        .chain(right)
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    262_144.0 * f64::EPSILON * scale
}

fn vector_p1_l2_error(
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    target_mesh: &SimplicialMesh,
    source_values: &[[f64; COMPONENTS]],
    target_values: &[[f64; COMPONENTS]],
    quadrature: &QuadratureRule,
    weight: f64,
) -> Result<f64, Diagnostic> {
    let mut squared = 0.0;
    for fragment in overlap.cell_fragments() {
        integrate_physical_triangle(fragment, quadrature, |point, measure| {
            let source =
                evaluate_vertex_vector(source_mesh, fragment.source_cell(), point, source_values)?;
            let target =
                evaluate_vertex_vector(target_mesh, fragment.target_cell(), point, target_values)?;
            squared += weight
                * measure
                * (0..COMPONENTS)
                    .map(|component| (source[component] - target[component]).powi(2))
                    .sum::<f64>();
            Ok(())
        })?;
    }
    finite_sqrt(squared, "solid-displacement L2 error")
}

fn cell_vertex_indices(mesh: &SimplicialMesh, cell: CellId) -> Result<[usize; 3], Diagnostic> {
    let vertices = mesh
        .entity_vertices(MeshEntity::new(DIMENSION, cell.index()))
        .ok_or_else(|| super::invalid("ALE FSI remesh names an absent triangle cell"))?;
    vertices
        .iter()
        .map(|entity| entity.index())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| super::invalid("ALE FSI remesh cell is not a triangle"))
}

fn facet_vertex_indices(mesh: &SimplicialMesh, facet: FacetId) -> Result<[usize; 2], Diagnostic> {
    let vertices = mesh
        .entity_vertices(MeshEntity::new(DIMENSION - 1, facet.index()))
        .ok_or_else(|| super::invalid("ALE FSI remesh names an absent edge facet"))?;
    vertices
        .iter()
        .map(|entity| entity.index())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| super::invalid("ALE FSI remesh facet is not an edge"))
}

#[allow(clippy::too_many_arguments)]
fn project_velocity(
    policy: P1HarmonicMeshMotionPolicy,
    source_reference: &SimplicialMesh,
    source_current: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_velocity: &[[f64; COMPONENTS]],
    source_bubbles: &BTreeMap<CellId, [f64; COMPONENTS]>,
    target_reference: &SimplicialMesh,
    target_current: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    solid_overlap: &SimplicialRevisionOverlap2d,
    fluid_overlap: &SimplicialRevisionOverlap2d,
    normalization: RemeshNormalization2d,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<VelocityProjection, Diagnostic> {
    let vertex_count = target_reference.vertices().len();
    let scalar_dimension = checked_sum(
        vertex_count,
        target_partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain")
            .len(),
        "target MINI scalar dimension",
    )?;
    let mut mass = dense_zeroed(scalar_dimension)?;
    assemble_velocity_mass_region(
        policy,
        target_current,
        target_partition,
        target_partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain"),
        true,
        normalization.fluid_density,
        quadrature,
        &mut mass,
    )?;
    assemble_velocity_mass_region(
        policy,
        target_reference,
        target_partition,
        target_partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
        false,
        normalization.solid_density,
        quadrature,
        &mut mass,
    )?;
    divide_scalars(&mut mass, normalization.velocity_mass)?;

    let mut mixed = vec![[0.0; COMPONENTS]; scalar_dimension];
    assemble_velocity_mixed_region(
        policy,
        fluid_overlap,
        source_current,
        source_partition,
        source_velocity,
        Some(source_bubbles),
        target_current,
        target_partition,
        true,
        normalization.fluid_density,
        quadrature,
        &mut mixed,
    )?;
    assemble_velocity_mixed_region(
        policy,
        solid_overlap,
        source_reference,
        source_partition,
        source_velocity,
        None,
        target_reference,
        target_partition,
        false,
        normalization.solid_density,
        quadrature,
        &mut mixed,
    )?;
    divide_vectors(&mut mixed, normalization.velocity_rhs()?)?;

    let mut prescribed = replay_interface_velocity_trace(
        policy,
        solid_overlap,
        source_reference,
        source_partition,
        source_velocity,
        target_reference,
        target_partition,
    )?;
    prescribed.resize(scalar_dimension, None);
    let prescribed = prescribed
        .into_iter()
        .map(|value| {
            value.map(|value| value.map(|component| component / normalization.physical.velocity()))
        })
        .collect::<Vec<_>>();
    let free = prescribed
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_none().then_some(index))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return Err(super::invalid(
            "ALE FSI remesh velocity projection has no free MINI coefficient",
        ));
    }

    let source_momentum = total_velocity_momentum(
        policy,
        source_reference,
        source_current,
        source_partition,
        source_velocity,
        source_bubbles,
        normalization,
        quadrature,
    )?;
    let divergence_rows = weak_divergence_rows(
        policy,
        target_current,
        target_partition,
        scalar_dimension,
        quadrature,
    )?;
    let momentum_rows = momentum_rows(
        policy,
        target_reference,
        target_current,
        target_partition,
        scalar_dimension,
        normalization,
        quadrature,
    )?;
    let dimensionless_divergence_rows = divided_rows(
        &divergence_rows,
        normalization.physical.length(),
        "weak-divergence rows",
    )?;
    let dimensionless_momentum_rows: [Vec<f64>; COMPONENTS] = std::array::from_fn(|component| {
        divided_row_unchecked(&momentum_rows[component], normalization.velocity_mass)
    });

    let free_vector_dimension =
        checked_product(COMPONENTS, free.len(), "free coupled-velocity dimension")?;
    let reduced_mass_count = checked_product(free.len(), free.len(), "reduced mass shape")?;
    let mut reduced_mass = Vec::with_capacity(reduced_mass_count);
    for &row in &free {
        for &column in &free {
            reduced_mass.push(mass[row * scalar_dimension + column]);
        }
    }
    let mut objective_rhs = vec![0.0; free_vector_dimension];
    for component in 0..COMPONENTS {
        for (position, &row) in free.iter().enumerate() {
            objective_rhs[component * free.len() + position] = mixed[row][component]
                - prescribed
                    .iter()
                    .enumerate()
                    .filter_map(|(column, value)| {
                        value.map(|value| mass[row * scalar_dimension + column] * value[component])
                    })
                    .sum::<f64>();
        }
    }

    let constraint_row_count = checked_sum(
        divergence_rows.len(),
        COMPONENTS,
        "physical velocity constraint count",
    )?;
    let reduced_constraint_coefficients = checked_product(
        constraint_row_count,
        free_vector_dimension,
        "reduced velocity constraint coefficient count",
    )?;
    require_auxiliary_budget(
        reduced_constraint_coefficients,
        "reduced velocity constraints",
    )?;
    require_constraint_work_budget(constraint_row_count, free_vector_dimension)?;
    let mut reduced_constraints = dimensionless_divergence_rows
        .iter()
        .map(|row| reduce_vector_constraint(row, 0.0, scalar_dimension, &free, &prescribed))
        .collect::<Result<Vec<_>, _>>()?;
    for component in 0..COMPONENTS {
        reduced_constraints.push(reduce_vector_constraint(
            &dimensionless_momentum_rows[component],
            source_momentum[component] / normalization.velocity_rhs()?,
            scalar_dimension,
            &free,
            &prescribed,
        )?);
    }
    let independent = independent_constraints(reduced_constraints)?;
    let constraint_count = independent.len();
    let kkt_dimension = checked_sum(
        free_vector_dimension,
        constraint_count,
        "constrained velocity KKT dimension",
    )?;
    let mut kkt = dense_zeroed(kkt_dimension)?;
    for component in 0..COMPONENTS {
        for row in 0..free.len() {
            for column in 0..free.len() {
                let global_row = component * free.len() + row;
                let global_column = component * free.len() + column;
                kkt[global_row * kkt_dimension + global_column] =
                    reduced_mass[row * free.len() + column];
            }
        }
    }
    let mut kkt_rhs = vec![0.0; kkt_dimension];
    kkt_rhs[..free_vector_dimension].copy_from_slice(&objective_rhs);
    for (constraint, (row, rhs)) in independent.iter().enumerate() {
        let multiplier = free_vector_dimension + constraint;
        kkt_rhs[multiplier] = *rhs;
        for (column, coefficient) in row.iter().copied().enumerate() {
            kkt[column * kkt_dimension + multiplier] = coefficient;
            kkt[multiplier * kkt_dimension + column] = coefficient;
        }
    }
    let right_hand_side_norm = euclidean_norm(&kkt_rhs)?;
    let (solution, report, residual_norm) = solve_dense(
        kkt,
        &kkt_rhs,
        LinearOperatorProperties::SymmetricIndefinite,
        solver,
    )?;

    let mut coefficients = vec![[0.0; COMPONENTS]; scalar_dimension];
    for (index, value) in prescribed.iter().enumerate() {
        if let Some(value) = value {
            coefficients[index] = *value;
        }
    }
    for component in 0..COMPONENTS {
        for (position, &index) in free.iter().enumerate() {
            coefficients[index][component] = solution[component * free.len() + position];
        }
    }
    for value in &mut coefficients {
        for component in value {
            *component *= normalization.physical.velocity();
        }
    }
    let vertex = coefficients[..vertex_count].to_vec();
    let bubble = target_partition
        .domain_cells(policy.fluid_domain())
        .expect("authenticated motion Domain")
        .iter()
        .copied()
        .zip(coefficients[vertex_count..].iter().copied())
        .collect();
    let full = flatten_vector_coefficients(&coefficients);
    let weak_divergence_norm = euclidean_norm(
        &divergence_rows
            .iter()
            .map(|row| dot(row, &full))
            .collect::<Vec<_>>(),
    )?;
    let target_momentum = std::array::from_fn(|component| dot(&momentum_rows[component], &full));
    let maximum_shared_trace_defect = retained_interface_p1_trace_defect(
        policy,
        solid_overlap,
        source_reference,
        source_partition,
        source_velocity,
        target_reference,
        target_partition,
        &vertex,
    )?;
    let maximum_exterior_trace_defect =
        homogeneous_exterior_velocity_trace_defect(source_reference, source_velocity)?.max(
            homogeneous_exterior_velocity_trace_defect(target_reference, &vertex)?,
        );
    let (fluid_l2_error, solid_l2_error) = velocity_l2_error(
        policy,
        solid_overlap,
        fluid_overlap,
        source_reference,
        source_current,
        source_partition,
        source_velocity,
        source_bubbles,
        target_reference,
        target_current,
        target_partition,
        &vertex,
        &bubble,
        normalization,
        quadrature,
    )?;
    Ok(VelocityProjection {
        vertex,
        bubble,
        report,
        right_hand_side_norm,
        residual_norm,
        independent_constraint_count: constraint_count,
        maximum_shared_trace_defect,
        maximum_exterior_trace_defect,
        weak_divergence_norm,
        source_momentum,
        target_momentum,
        fluid_l2_error,
        solid_l2_error,
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_velocity_mass_region(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cells: &[CellId],
    bubble: bool,
    density: f64,
    quadrature: &QuadratureRule,
    mass: &mut [f64],
) -> Result<(), Diagnostic> {
    let dimension = integer_sqrt(mass.len())?;
    for &cell in cells {
        let dofs = velocity_scalar_dofs(policy, mesh, partition, cell, bubble)?;
        integrate_cell(mesh, cell, quadrature, |point, measure| {
            let basis = cell_basis(mesh, cell, point, bubble)?;
            for (local_row, &row) in dofs.iter().enumerate() {
                for (local_column, &column) in dofs.iter().enumerate() {
                    mass[row * dimension + column] +=
                        density * measure * basis.values[local_row] * basis.values[local_column];
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assemble_velocity_mixed_region(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_vertex: &[[f64; COMPONENTS]],
    source_bubble: Option<&std::collections::BTreeMap<CellId, [f64; COMPONENTS]>>,
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    bubbles: bool,
    density: f64,
    quadrature: &QuadratureRule,
    mixed: &mut [[f64; COMPONENTS]],
) -> Result<(), Diagnostic> {
    for fragment in overlap.cell_fragments() {
        let source_cell = fragment.source_cell();
        let target_cell = fragment.target_cell();
        let target_dofs =
            velocity_scalar_dofs(policy, target_mesh, target_partition, target_cell, bubbles)?;
        integrate_physical_triangle(fragment, quadrature, |point, measure| {
            let source = evaluate_velocity_cell(
                policy,
                source_mesh,
                source_partition,
                source_cell,
                point,
                source_vertex,
                source_bubble,
            )?;
            let target_basis = cell_basis(target_mesh, target_cell, point, bubbles)?;
            for (local, &dof) in target_dofs.iter().enumerate() {
                for component in 0..COMPONENTS {
                    mixed[dof][component] +=
                        density * measure * target_basis.values[local] * source[component];
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn weak_divergence_rows(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    scalar_dimension: usize,
    quadrature: &QuadratureRule,
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    let row_width = checked_product(COMPONENTS, scalar_dimension, "weak-divergence row width")?;
    let coefficient_count = checked_product(
        partition
            .domain_vertices(policy.fluid_domain())
            .expect("authenticated motion Domain")
            .len(),
        row_width,
        "weak-divergence coefficient count",
    )?;
    require_auxiliary_budget(coefficient_count, "weak-divergence rows")?;
    let mut rows = vec![
        vec![0.0; row_width];
        partition
            .domain_vertices(policy.fluid_domain())
            .expect("authenticated motion Domain")
            .len()
    ];
    for &cell in partition
        .domain_cells(policy.fluid_domain())
        .expect("authenticated motion Domain")
    {
        let vertices = cell_vertex_indices(mesh, cell)?;
        let dofs = velocity_scalar_dofs(policy, mesh, partition, cell, true)?;
        integrate_cell(mesh, cell, quadrature, |point, measure| {
            let velocity_basis = cell_basis(mesh, cell, point, true)?;
            let pressure_basis = cell_basis(mesh, cell, point, false)?;
            for (test_local, &test_vertex) in vertices.iter().enumerate() {
                let test = pressure_position(policy, partition, test_vertex)?;
                for (trial_local, &trial_dof) in dofs.iter().enumerate() {
                    for component in 0..COMPONENTS {
                        rows[test][component * scalar_dimension + trial_dof] += measure
                            * pressure_basis.values[test_local]
                            * velocity_basis.gradients[trial_local][component];
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(rows)
}

fn momentum_rows(
    policy: P1HarmonicMeshMotionPolicy,
    reference: &SimplicialMesh,
    current: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    scalar_dimension: usize,
    normalization: RemeshNormalization2d,
    quadrature: &QuadratureRule,
) -> Result<[Vec<f64>; COMPONENTS], Diagnostic> {
    let mut scalar = vec![0.0; scalar_dimension];
    assemble_basis_moment(
        policy,
        current,
        partition,
        partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain"),
        true,
        normalization.fluid_density,
        quadrature,
        &mut scalar,
    )?;
    assemble_basis_moment(
        policy,
        reference,
        partition,
        partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
        false,
        normalization.solid_density,
        quadrature,
        &mut scalar,
    )?;
    let width = checked_product(
        COMPONENTS,
        scalar_dimension,
        "momentum constraint row width",
    )?;
    Ok(std::array::from_fn(|component| {
        let mut row = vec![0.0; width];
        row[component * scalar_dimension..(component + 1) * scalar_dimension]
            .copy_from_slice(&scalar);
        row
    }))
}

#[allow(clippy::too_many_arguments)]
fn assemble_basis_moment(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cells: &[CellId],
    bubble: bool,
    density: f64,
    quadrature: &QuadratureRule,
    moment: &mut [f64],
) -> Result<(), Diagnostic> {
    for &cell in cells {
        let dofs = velocity_scalar_dofs(policy, mesh, partition, cell, bubble)?;
        integrate_cell(mesh, cell, quadrature, |point, measure| {
            let basis = cell_basis(mesh, cell, point, bubble)?;
            for (local, &dof) in dofs.iter().enumerate() {
                moment[dof] += density * measure * basis.values[local];
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn reduce_vector_constraint(
    row: &[f64],
    mut rhs: f64,
    scalar_dimension: usize,
    free: &[usize],
    prescribed: &[Option<[f64; COMPONENTS]>],
) -> Result<(Vec<f64>, f64), Diagnostic> {
    if row.len() != COMPONENTS * scalar_dimension || prescribed.len() != scalar_dimension {
        return Err(super::invalid(
            "ALE FSI remesh velocity constraint has an incompatible shape",
        ));
    }
    let mut reduced = vec![0.0; COMPONENTS * free.len()];
    for component in 0..COMPONENTS {
        for (position, &dof) in free.iter().enumerate() {
            reduced[component * free.len() + position] = row[component * scalar_dimension + dof];
        }
        for (dof, value) in prescribed.iter().enumerate() {
            if let Some(value) = value {
                rhs -= row[component * scalar_dimension + dof] * value[component];
            }
        }
    }
    Ok((reduced, rhs))
}

fn flatten_vector_coefficients(values: &[[f64; COMPONENTS]]) -> Vec<f64> {
    (0..COMPONENTS)
        .flat_map(|component| values.iter().map(move |value| value[component]))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn total_velocity_momentum(
    policy: P1HarmonicMeshMotionPolicy,
    reference: &SimplicialMesh,
    current: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    vertex: &[[f64; COMPONENTS]],
    bubbles: &std::collections::BTreeMap<CellId, [f64; COMPONENTS]>,
    normalization: RemeshNormalization2d,
    quadrature: &QuadratureRule,
) -> Result<[f64; COMPONENTS], Diagnostic> {
    let mut momentum = [0.0; COMPONENTS];
    integrate_velocity_momentum_region(
        policy,
        current,
        partition,
        partition
            .domain_cells(policy.fluid_domain())
            .expect("authenticated motion Domain"),
        vertex,
        Some(bubbles),
        normalization.fluid_density,
        quadrature,
        &mut momentum,
    )?;
    integrate_velocity_momentum_region(
        policy,
        reference,
        partition,
        partition
            .domain_cells(policy.solid_domain())
            .expect("authenticated motion Domain"),
        vertex,
        None,
        normalization.solid_density,
        quadrature,
        &mut momentum,
    )?;
    Ok(momentum)
}

#[allow(clippy::too_many_arguments)]
fn integrate_velocity_momentum_region(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cells: &[CellId],
    vertex: &[[f64; COMPONENTS]],
    bubbles: Option<&std::collections::BTreeMap<CellId, [f64; COMPONENTS]>>,
    density: f64,
    quadrature: &QuadratureRule,
    momentum: &mut [f64; COMPONENTS],
) -> Result<(), Diagnostic> {
    for &cell in cells {
        integrate_cell(mesh, cell, quadrature, |point, measure| {
            let value =
                evaluate_velocity_cell(policy, mesh, partition, cell, point, vertex, bubbles)?;
            for component in 0..COMPONENTS {
                momentum[component] += density * measure * value[component];
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn velocity_l2_error(
    policy: P1HarmonicMeshMotionPolicy,
    solid_overlap: &SimplicialRevisionOverlap2d,
    fluid_overlap: &SimplicialRevisionOverlap2d,
    source_reference: &SimplicialMesh,
    source_current: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_vertex: &[[f64; COMPONENTS]],
    source_bubble: &std::collections::BTreeMap<CellId, [f64; COMPONENTS]>,
    target_reference: &SimplicialMesh,
    target_current: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    target_vertex: &[[f64; COMPONENTS]],
    target_bubble: &std::collections::BTreeMap<CellId, [f64; COMPONENTS]>,
    normalization: RemeshNormalization2d,
    quadrature: &QuadratureRule,
) -> Result<(f64, f64), Diagnostic> {
    let mut fluid_squared = 0.0;
    accumulate_velocity_l2_error(
        policy,
        fluid_overlap,
        source_current,
        source_partition,
        source_vertex,
        Some(source_bubble),
        target_current,
        target_partition,
        target_vertex,
        Some(target_bubble),
        normalization.fluid_density,
        quadrature,
        &mut fluid_squared,
    )?;
    let mut solid_squared = 0.0;
    accumulate_velocity_l2_error(
        policy,
        solid_overlap,
        source_reference,
        source_partition,
        source_vertex,
        None,
        target_reference,
        target_partition,
        target_vertex,
        None,
        normalization.solid_density,
        quadrature,
        &mut solid_squared,
    )?;
    Ok((
        finite_sqrt(
            fluid_squared,
            "fluid-current density-weighted velocity L2 error",
        )?,
        finite_sqrt(
            solid_squared,
            "solid-material density-weighted velocity L2 error",
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_velocity_l2_error(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_vertex: &[[f64; COMPONENTS]],
    source_bubble: Option<&std::collections::BTreeMap<CellId, [f64; COMPONENTS]>>,
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    target_vertex: &[[f64; COMPONENTS]],
    target_bubble: Option<&std::collections::BTreeMap<CellId, [f64; COMPONENTS]>>,
    density: f64,
    quadrature: &QuadratureRule,
    squared: &mut f64,
) -> Result<(), Diagnostic> {
    for fragment in overlap.cell_fragments() {
        integrate_physical_triangle(fragment, quadrature, |point, measure| {
            let source = evaluate_velocity_cell(
                policy,
                source_mesh,
                source_partition,
                fragment.source_cell(),
                point,
                source_vertex,
                source_bubble,
            )?;
            let target = evaluate_velocity_cell(
                policy,
                target_mesh,
                target_partition,
                fragment.target_cell(),
                point,
                target_vertex,
                target_bubble,
            )?;
            *squared += density
                * measure
                * (0..COMPONENTS)
                    .map(|component| (source[component] - target[component]).powi(2))
                    .sum::<f64>();
            Ok(())
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn project_pressure(
    policy: P1HarmonicMeshMotionPolicy,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source_pressure: &[f64],
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    overlap: &SimplicialRevisionOverlap2d,
    normalization: RemeshNormalization2d,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<PressureProjection, Diagnostic> {
    let dimension = target_partition
        .domain_vertices(policy.fluid_domain())
        .expect("authenticated motion Domain")
        .len();
    let mut mass = dense_zeroed(dimension)?;
    for &cell in target_partition
        .domain_cells(policy.fluid_domain())
        .expect("authenticated motion Domain")
    {
        let vertices = cell_vertex_indices(target_mesh, cell)?;
        integrate_cell(target_mesh, cell, quadrature, |point, measure| {
            let basis = cell_basis(target_mesh, cell, point, false)?;
            for (local_row, &row_vertex) in vertices.iter().enumerate() {
                let row = pressure_position(policy, target_partition, row_vertex)?;
                for (local_column, &column_vertex) in vertices.iter().enumerate() {
                    let column = pressure_position(policy, target_partition, column_vertex)?;
                    mass[row * dimension + column] +=
                        measure * basis.values[local_row] * basis.values[local_column];
                }
            }
            Ok(())
        })?;
    }
    let mut rhs = vec![0.0; dimension];
    for fragment in overlap.cell_fragments() {
        let source_vertices = cell_vertex_indices(source_mesh, fragment.source_cell())?;
        let target_vertices = cell_vertex_indices(target_mesh, fragment.target_cell())?;
        integrate_physical_triangle(fragment, quadrature, |point, measure| {
            let source_basis = cell_basis(source_mesh, fragment.source_cell(), point, false)?;
            let target_basis = cell_basis(target_mesh, fragment.target_cell(), point, false)?;
            let source_value = source_vertices
                .iter()
                .enumerate()
                .map(|(local, &vertex)| {
                    pressure_position(policy, source_partition, vertex)
                        .map(|position| source_basis.values[local] * source_pressure[position])
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<f64>();
            for (local, &vertex) in target_vertices.iter().enumerate() {
                let position = pressure_position(policy, target_partition, vertex)?;
                rhs[position] += measure * target_basis.values[local] * source_value;
            }
            Ok(())
        })?;
    }
    divide_scalars(&mut mass, normalization.area)?;
    divide_scalars(&mut rhs, normalization.pressure_rhs()?)?;
    let right_hand_side_norm = euclidean_norm(&rhs)?;
    let (mut coefficients, report, residual_norm) = solve_dense(
        mass,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver,
    )?;
    for value in &mut coefficients {
        *value *= normalization.physical.pressure();
    }
    let source_moment = pressure_moment(
        policy,
        source_mesh,
        source_partition,
        source_pressure,
        quadrature,
    )?;
    let target_moment = pressure_moment(
        policy,
        target_mesh,
        target_partition,
        &coefficients,
        quadrature,
    )?;
    let l2_error = pressure_l2_error(
        policy,
        overlap,
        source_mesh,
        source_partition,
        source_pressure,
        target_mesh,
        target_partition,
        &coefficients,
        quadrature,
    )?;
    Ok(PressureProjection {
        coefficients,
        report,
        right_hand_side_norm,
        residual_norm,
        source_moment,
        target_moment,
        l2_error,
    })
}

fn pressure_position(
    policy: P1HarmonicMeshMotionPolicy,
    partition: &FixedReferenceFsiPartition<2>,
    vertex: usize,
) -> Result<usize, Diagnostic> {
    partition
        .domain_vertices(policy.fluid_domain())
        .expect("authenticated motion Domain")
        .binary_search(&VertexId::new(vertex))
        .map_err(|_| super::invalid("ALE FSI remesh fluid P1 vertex is absent from pressure space"))
}

fn pressure_moment(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    pressure: &[f64],
    quadrature: &QuadratureRule,
) -> Result<f64, Diagnostic> {
    if pressure.len()
        != partition
            .domain_vertices(policy.fluid_domain())
            .expect("authenticated motion Domain")
            .len()
    {
        return Err(super::invalid(
            "ALE FSI remesh pressure moment has an incompatible coefficient shape",
        ));
    }
    let mut moment = 0.0;
    for &cell in partition
        .domain_cells(policy.fluid_domain())
        .expect("authenticated motion Domain")
    {
        integrate_cell(mesh, cell, quadrature, |point, measure| {
            moment +=
                measure * evaluate_pressure_cell(policy, mesh, partition, cell, point, pressure)?;
            Ok(())
        })?;
    }
    moment
        .is_finite()
        .then_some(moment)
        .ok_or_else(|| super::invalid("ALE FSI remesh absolute pressure moment is non-finite"))
}

fn evaluate_pressure_cell(
    policy: P1HarmonicMeshMotionPolicy,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    cell: CellId,
    point: [f64; DIMENSION],
    pressure: &[f64],
) -> Result<f64, Diagnostic> {
    let vertices = cell_vertex_indices(mesh, cell)?;
    let basis = cell_basis(mesh, cell, point, false)?;
    vertices
        .iter()
        .enumerate()
        .map(|(local, &vertex)| {
            pressure_position(policy, partition, vertex)
                .map(|position| basis.values[local] * pressure[position])
        })
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn pressure_l2_error(
    policy: P1HarmonicMeshMotionPolicy,
    overlap: &SimplicialRevisionOverlap2d,
    source_mesh: &SimplicialMesh,
    source_partition: &FixedReferenceFsiPartition<2>,
    source: &[f64],
    target_mesh: &SimplicialMesh,
    target_partition: &FixedReferenceFsiPartition<2>,
    target: &[f64],
    quadrature: &QuadratureRule,
) -> Result<f64, Diagnostic> {
    let mut squared = 0.0;
    for fragment in overlap.cell_fragments() {
        integrate_physical_triangle(fragment, quadrature, |point, measure| {
            let source_value = evaluate_pressure_cell(
                policy,
                source_mesh,
                source_partition,
                fragment.source_cell(),
                point,
                source,
            )?;
            let target_value = evaluate_pressure_cell(
                policy,
                target_mesh,
                target_partition,
                fragment.target_cell(),
                point,
                target,
            )?;
            squared += measure * (source_value - target_value).powi(2);
            Ok(())
        })?;
    }
    finite_sqrt(squared, "absolute pressure L2 error")
}
