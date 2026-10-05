use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPlan, AssemblyTarget, IndexedAssemblyWork,
    LinearSystem, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh, simplex_centroid_rule,
};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{LinearSolveRequest, SolverPlan};

use super::acceptance::{integrate_body_force, require_compatible_boundary_flux, validate_problem};
use super::api::SimplicialMiniStokesSolution2d;
use super::boundary::{PressureReferenceKind2d, SimplicialMiniStokesBoundary2d};
use super::constraint::MiniPressureMeanConstraintCell;
use super::element::MiniStokesCell;
use super::facet::MiniConstantTractionFacet;
use super::finalized::FinalizedMiniStokesAssembly;
use super::layout::MixedLayout;
use super::{COMPONENTS, DIMENSION, invalid};
use crate::finalized_spatial::FinalizedSimplicialMiniStokes2dProblem;
use crate::operator::LocalOperator;

/// Finalize one complete-essential numerical MINI Stokes problem.
///
/// This compatibility entry point constructs a complete facet-derived
/// essential closure and one zero-integral pressure constraint. New
/// mixed-boundary callers should use
/// [`finalize_simplicial_mini_stokes_2d_with_boundary`].
///
/// # Errors
/// Returns a structured diagnostic for incompatible mesh, data, quadrature,
/// plan, or finalized CSR state.
#[allow(clippy::too_many_arguments)]
pub fn finalize_simplicial_mini_stokes_2d<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    essential_velocity: &B,
    quadrature: &QuadratureRule,
    solver: SolverPlan,
    vector_layout: VectorLayoutKind,
    target: Target,
) -> Result<FinalizedSimplicialMiniStokes2dProblem, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    finalize_simplicial_mini_stokes_2d_with_assembly(
        mesh,
        viscosity,
        body_force,
        essential_velocity,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
        vector_layout,
        target,
    )
}

/// Finalize one complete-essential problem through an explicit assembler.
///
/// # Errors
/// Preserves numerical admission and captured-CSR diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn finalize_simplicial_mini_stokes_2d_with_assembly<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    essential_velocity: &B,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: SolverPlan,
    vector_layout: VectorLayoutKind,
    target: Target,
) -> Result<FinalizedSimplicialMiniStokes2dProblem, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let boundary = SimplicialMiniStokesBoundary2d::all_essential(mesh)?;
    let facet_quadrature = simplex_centroid_rule(DIMENSION - 1)?;
    finalize_simplicial_mini_stokes_2d_with_boundary_and_assembly(
        mesh,
        viscosity,
        body_force,
        &boundary,
        essential_velocity,
        quadrature,
        &facet_quadrature,
        assembly,
        solver,
        vector_layout,
        target,
    )
}

/// Finalize one complete facet-derived essential/traction boundary closure.
///
/// An all-essential closure receives one independent zero-integral pressure
/// constraint. A mixed closure must contain both essential and prescribed-
/// traction facets and receives no global gauge because the traction boundary
/// fixes the constant pressure mode.
///
/// # Errors
/// Rejects incomplete boundary classification, incompatible pressure/velocity
/// closure, invalid cell or facet quadrature, or finalized CSR state.
#[allow(clippy::too_many_arguments)]
pub fn finalize_simplicial_mini_stokes_2d_with_boundary<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    solver: SolverPlan,
    vector_layout: VectorLayoutKind,
    target: Target,
) -> Result<FinalizedSimplicialMiniStokes2dProblem, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    finalize_simplicial_mini_stokes_2d_with_boundary_and_assembly(
        mesh,
        viscosity,
        body_force,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
        vector_layout,
        target,
    )
}

/// Finalize a boundary-aware problem through one explicit assembly backend.
///
/// # Errors
/// Preserves every boundary, local-operator, assembly, and handoff failure.
#[allow(clippy::too_many_arguments)]
pub fn finalize_simplicial_mini_stokes_2d_with_boundary_and_assembly<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: SolverPlan,
    vector_layout: VectorLayoutKind,
    target: Target,
) -> Result<FinalizedSimplicialMiniStokes2dProblem, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let assembled = assemble_simplicial_mini_stokes_2d(
        mesh,
        viscosity,
        body_force,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
        assembly,
    )?;
    FinalizedSimplicialMiniStokes2dProblem::new(solver, vector_layout, target, assembled)
}

/// Solve one complete-essential MINI problem with reference assembly.
///
/// # Errors
/// Preserves finalization, solver, and solution-acceptance diagnostics.
pub fn solve_simplicial_mini_stokes_2d<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    essential_velocity: &B,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<SimplicialMiniStokesSolution2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    solve_simplicial_mini_stokes_2d_with_assembly(
        mesh,
        viscosity,
        body_force,
        essential_velocity,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Solve one complete-essential problem through an explicit assembler.
///
/// # Errors
/// Preserves numerical, assembly, solver, and evidence diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn solve_simplicial_mini_stokes_2d_with_assembly<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    essential_velocity: &B,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: LinearSolveRequest<'_>,
) -> Result<SimplicialMiniStokesSolution2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let boundary = SimplicialMiniStokesBoundary2d::all_essential(mesh)?;
    let facet_quadrature = simplex_centroid_rule(DIMENSION - 1)?;
    solve_simplicial_mini_stokes_2d_with_boundary_and_assembly(
        mesh,
        viscosity,
        body_force,
        &boundary,
        essential_velocity,
        quadrature,
        &facet_quadrature,
        assembly,
        solver,
    )
}

/// Solve one complete facet-derived mixed boundary problem.
///
/// # Errors
/// Preserves boundary-aware finalization and selected solver diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn solve_simplicial_mini_stokes_2d_with_boundary<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<SimplicialMiniStokesSolution2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    solve_simplicial_mini_stokes_2d_with_boundary_and_assembly(
        mesh,
        viscosity,
        body_force,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Solve a boundary-aware problem through explicit assembly and solver paths.
///
/// # Errors
/// Preserves all selected adapter diagnostics without fallback.
#[allow(clippy::too_many_arguments)]
pub fn solve_simplicial_mini_stokes_2d_with_boundary_and_assembly<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: LinearSolveRequest<'_>,
) -> Result<SimplicialMiniStokesSolution2d, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    let assembled = assemble_simplicial_mini_stokes_2d(
        mesh,
        viscosity,
        body_force,
        boundary,
        essential_velocity,
        cell_quadrature,
        facet_quadrature,
        assembly,
    )?;
    let (canonical_system, state) = assembled.into_canonical()?;
    let solved = solver.solve(&canonical_system.linear_problem()?)?;
    state.finish(solved, canonical_system)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_simplicial_mini_stokes_2d<F, B>(
    mesh: &SimplicialMesh,
    viscosity: f64,
    body_force: &F,
    boundary: &SimplicialMiniStokesBoundary2d,
    essential_velocity: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<FinalizedMiniStokesAssembly, Diagnostic>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
    B: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    validate_problem(mesh, viscosity, cell_quadrature)?;
    let boundary = boundary.validated_for(mesh)?;
    let named_reaction_vertices = boundary.named_reaction_vertices(mesh);
    let prepared = boundary.prepare(mesh, essential_velocity)?;
    let with_gauge = prepared.pressure_reference == PressureReferenceKind2d::ZeroIntegral;
    if with_gauge {
        require_compatible_boundary_flux(mesh, &prepared.fixed_velocity)?;
    }
    if !prepared.traction_facets.is_empty() {
        super::acceptance::require_facet_geometry(
            &mesh
                .geometry_map(prepared.traction_facets[0].facet)
                .expect("validated traction facet owns geometry"),
            facet_quadrature,
        )?;
    }
    let layout = MixedLayout::new(mesh, &prepared.fixed_velocity, with_gauge)?;
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("2D simplex mesh owns cells");
    let constraint_count = if with_gauge { cell_count } else { 0 };
    let constraint_end = cell_count
        .checked_add(constraint_count)
        .ok_or_else(|| invalid("MINI Stokes constraint packet count overflows usize"))?;
    let packet_count = constraint_end
        .checked_add(prepared.traction_facets.len())
        .ok_or_else(|| invalid("MINI Stokes packet count overflows usize"))?;
    let cell_operator = MiniStokesCell {
        viscosity,
        body_force,
    };
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(layout.reduced_size)?,
        AssemblyTarget::new(layout.full_size)?,
        AssemblyTarget::new(layout.full_size)?,
    ])?;
    let reduced_target = plan.target_id(0).expect("three-target plan owns reduced");
    let full_target = plan.target_id(1).expect("three-target plan owns full");
    let volume_target = plan.target_id(2).expect("three-target plan owns volume");
    let work = IndexedAssemblyWork::new(packet_count, |packet| {
        if packet < cell_count {
            let cell = MeshEntity::new(DIMENSION, packet);
            let geometry = mesh
                .geometry_map(cell)
                .expect("accepted simplex cell owns geometry");
            let local = cell_operator.evaluate(&geometry, cell_quadrature)?;
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let reduced = layout.reduced_cell_map(packet, &vertices, &prepared.fixed_velocity)?;
            let full = layout.full_cell_map(packet, &vertices)?;
            AssemblyPacket::new(
                local,
                vec![
                    TargetAssemblyMap::new(reduced_target, reduced),
                    TargetAssemblyMap::new(full_target, full.clone()),
                    TargetAssemblyMap::new(volume_target, full),
                ],
            )
        } else if packet < constraint_end {
            let cell_index = packet - cell_count;
            let cell = MeshEntity::new(DIMENSION, cell_index);
            let geometry = mesh
                .geometry_map(cell)
                .expect("accepted simplex cell owns geometry");
            let local = MiniPressureMeanConstraintCell.evaluate(&geometry, cell_quadrature)?;
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let reduced = layout.reduced_constraint_map(&vertices)?;
            let full = layout.full_constraint_map(&vertices)?;
            AssemblyPacket::new(
                local,
                vec![
                    TargetAssemblyMap::new(reduced_target, reduced),
                    TargetAssemblyMap::new(full_target, full.clone()),
                    TargetAssemblyMap::new(volume_target, full),
                ],
            )
        } else {
            let facet = prepared.traction_facets[packet - constraint_end];
            let geometry = mesh
                .geometry_map(facet.facet)
                .expect("validated traction facet owns geometry");
            let local = MiniConstantTractionFacet {
                traction: facet.value,
            }
            .evaluate(&geometry, facet_quadrature)?;
            let vertices = mesh
                .entity_vertices(facet.facet)
                .expect("accepted boundary facet owns vertices");
            let reduced = layout.reduced_facet_map(&vertices, &prepared.fixed_velocity)?;
            let full = layout.full_facet_map(&vertices)?;
            AssemblyPacket::new(
                local,
                vec![
                    TargetAssemblyMap::new(reduced_target, reduced),
                    TargetAssemblyMap::new(full_target, full),
                ],
            )
        }
    });
    let (systems, assembly_report) = assembly.assemble(&plan, &work)?.into_parts();
    let [linear_system, full_system, volume_only_system]: [LinearSystem<f64>; 3] =
        systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "three-target MINI assembly returned {} systems",
                systems.len()
            ))
        })?;
    let integrated_body_force = integrate_body_force(mesh, cell_quadrature, body_force)?;
    let integrated_boundary_traction = prepared.integrated_traction(mesh)?;
    Ok(FinalizedMiniStokesAssembly {
        mesh: mesh.clone(),
        layout,
        fixed_velocity: prepared.fixed_velocity,
        named_reaction_vertices,
        linear_system,
        full_system,
        volume_only_system,
        integrated_body_force,
        integrated_boundary_traction,
        quadrature: cell_quadrature.clone(),
        assembly_report,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_meshing::{MeshQualityGate, triangle_duffy_gauss_legendre};
    use eqiora_solver::{
        LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
        ReductionPolicy, SolverPlan,
    };

    use super::super::{
        SimplicialMiniStokesBoundaryCondition2d, SimplicialMiniStokesBoundaryFacet2d,
        SimplicialMiniStokesPressureReference2d,
    };
    use super::*;

    #[test]
    fn mixed_static_pressure_has_no_gauge_and_retains_three_action_targets() {
        let mesh = rectangle_triangles(2, 2);
        let facet_count = mesh.entity_count(1).unwrap();
        let boundary = SimplicialMiniStokesBoundary2d::new(
            &mesh,
            (0..facet_count).filter_map(|index| {
                let facet = MeshEntity::new(1, index);
                mesh.is_boundary_entity(facet).unwrap().then(|| {
                    let vertices = mesh.entity_vertices(facet).unwrap();
                    let on_right = vertices
                        .iter()
                        .all(|vertex| mesh.vertices()[vertex.index()][0] == 4.0);
                    let condition = if on_right {
                        SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                            value: [-4.5, 0.0],
                        }
                    } else {
                        SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                    };
                    SimplicialMiniStokesBoundaryFacet2d::new(facet, condition)
                })
            }),
        )
        .unwrap();
        let cell_quadrature = triangle_duffy_gauss_legendre(3).unwrap();
        let facet_quadrature = simplex_centroid_rule(DIMENSION - 1).unwrap();
        let solution = solve_simplicial_mini_stokes_2d_with_boundary(
            &mesh,
            6.0,
            &|_| Ok([0.75, 0.0]),
            &boundary,
            &|_| Ok([0.0, 0.0]),
            &cell_quadrature,
            &facet_quadrature,
            reference_solver(),
        )
        .unwrap();

        assert_eq!(
            solution.pressure_reference(),
            SimplicialMiniStokesPressureReference2d::BoundaryTraction
        );
        assert_eq!(solution.gauge_multiplier(), None);
        assert_close(solution.pressure_integral(), 24.0, 2.0e-9);
        assert_vector_close(solution.integrated_body_force(), [6.0, 0.0], 2.0e-10);
        assert_vector_close(
            solution.integrated_boundary_traction(),
            [-9.0, 0.0],
            2.0e-10,
        );
        assert_vector_close(solution.boundary_reaction(), [3.0, 0.0], 2.0e-9);
        assert_eq!(
            solution.full_system().matrix(),
            solution.volume_only_system().matrix()
        );
        assert_eq!(solution.assembly_report().target_count(), 3);
        assert_eq!(solution.assembly_report().packet_count(), 10);
        let midpoint_x_velocity = 2 * 5;
        assert_close(
            solution.full_system().rhs()[midpoint_x_velocity]
                - solution.volume_only_system().rhs()[midpoint_x_velocity],
            -4.5,
            2.0e-14,
        );
        for (coordinates, pressure) in mesh
            .vertices()
            .iter()
            .zip(solution.pressure().vertex_values())
        {
            assert_close(*pressure, 0.75 * coordinates[0] + 1.5, 2.0e-9);
        }
    }

    fn rectangle_triangles(horizontal: usize, vertical: usize) -> SimplicialMesh {
        let width = horizontal + 1;
        let vertices = (0..=vertical)
            .flat_map(|j| {
                (0..=horizontal).map(move |i| {
                    vec![
                        4.0 * i as f64 / horizontal as f64,
                        2.0 * j as f64 / vertical as f64,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let mut cells = Vec::with_capacity(2 * horizontal * vertical);
        for j in 0..vertical {
            for i in 0..horizontal {
                let lower_left = j * width + i;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                cells.push(vec![lower_left, lower_right, upper_right]);
                cells.push(vec![lower_left, upper_right, upper_left]);
            }
        }
        SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.4).unwrap()).unwrap()
    }

    fn reference_solver() -> LinearSolveRequest<'static> {
        let plan = SolverPlan::new(
            LinearSolver::MinimumResidual,
            1.0e-11,
            1.0e-13,
            NonZeroUsize::new(10_000).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Reproducible);
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
    }

    fn assert_vector_close(actual: [f64; 2], expected: [f64; 2], tolerance: f64) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:e}, received {actual:e}"
        );
    }
}
