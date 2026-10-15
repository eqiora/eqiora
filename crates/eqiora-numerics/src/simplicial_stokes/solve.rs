use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPlan, AssemblyTarget, IndexedAssemblyWork,
    LinearSystem, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{LinearSolveRequest, SolverPlan};

use super::acceptance::{require_compatible_boundary_flux, validate_problem};
use super::api::SimplicialMiniStokesSolution2d;
use super::boundary::{PressureReferenceKind2d, SimplicialMiniStokesBoundary2d};
use super::element::MiniStokesCell;
use super::facet::MiniConstantTractionFacet;
use super::finalized::FinalizedMiniStokesAssembly;
use super::layout::MixedLayout;
use super::{COMPONENTS, DIMENSION, invalid};
use crate::finalized_spatial::FinalizedSimplicialMiniStokes2dProblem;
use crate::form_compiler::integral_constraint::zero_integral_constraint;
use crate::operator::LocalOperator;

/// Finalize one numerical MINI Stokes problem with explicit boundary and assembly policy.
///
/// A complete essential trace receives one zero-integral pressure constraint.
/// A mixed essential/traction closure fixes the constant pressure mode through
/// the traction boundary and adds no global gauge.
///
/// # Errors
/// Preserves every boundary, local-operator, assembly, and handoff failure.
#[allow(clippy::too_many_arguments)]
pub fn finalize_simplicial_mini_stokes_2d<F, B>(
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

/// Solve a boundary-aware problem through explicit assembly and solver paths.
///
/// # Errors
/// Preserves all selected adapter diagnostics without fallback.
#[allow(clippy::too_many_arguments)]
pub fn solve_simplicial_mini_stokes_2d<F, B>(
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
            super::acceptance::require_local_geometry(&geometry, cell_quadrature)?;
            let local = zero_integral_constraint(
                eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                &geometry,
                cell_quadrature,
            )?;
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
    Ok(FinalizedMiniStokesAssembly {
        mesh: mesh.clone(),
        layout,
        fixed_velocity: prepared.fixed_velocity,
        named_reaction_vertices,
        linear_system,
        full_system,
        volume_only_system,
        traction_facets: prepared.traction_facets,
        quadrature: cell_quadrature.clone(),
        assembly_report,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
    use eqiora_meshing::{MeshQualityGate, simplex_centroid_rule, simplex_duffy_gauss_legendre};
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
        let cell_quadrature = simplex_duffy_gauss_legendre(2, 3).unwrap();
        let facet_quadrature = simplex_centroid_rule(DIMENSION - 1).unwrap();
        let solution = solve_simplicial_mini_stokes_2d(
            &mesh,
            6.0,
            &|_| Ok([0.75, 0.0]),
            &boundary,
            &|_| Ok([0.0, 0.0]),
            &cell_quadrature,
            &facet_quadrature,
            &REFERENCE_ASSEMBLY_BACKEND,
            reference_solver(),
        )
        .unwrap();

        assert_eq!(
            solution.pressure_reference(),
            SimplicialMiniStokesPressureReference2d::BoundaryTraction
        );
        assert_eq!(solution.gauge_multiplier(), None);
        assert_close(solution.pressure_integral().unwrap(), 24.0, 2.0e-9);
        assert_vector_close(
            solution
                .integrated_body_force(&|_| Ok([0.75, 0.0]))
                .unwrap(),
            [6.0, 0.0],
            2.0e-10,
        );
        assert_vector_close(
            solution.integrated_boundary_traction().unwrap(),
            [-9.0, 0.0],
            2.0e-10,
        );
        assert_vector_close(solution.boundary_reaction().unwrap(), [3.0, 0.0], 2.0e-9);
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
