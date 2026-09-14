//! Focused falsifiers for the fixed-reference CPU realization.

use std::num::NonZeroUsize;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyResult, AssemblyTarget,
    DofId, IndexedAssemblyWork, LocalContribution, LocalUnknown, REFERENCE_ASSEMBLY_BACKEND,
    TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, QuadratureRule, SimplicialMesh,
    simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};

use super::*;

mod resolved;

#[test]
fn exact_partition_rejects_missing_and_extra_interface_facets() {
    let mesh = two_domain_mesh();
    let (fluid, solid, interface) = inventories(&mesh);
    assert!(
        FixedReferenceFsiPartition::<2>::new(
            &mesh,
            fluid.clone(),
            solid.clone(),
            interface.clone()
        )
        .is_ok()
    );
    assert!(
        FixedReferenceFsiPartition::<2>::new(&mesh, fluid.clone(), solid.clone(), Vec::new())
            .is_err()
    );
    let mut incomplete_cells = fluid;
    incomplete_cells.pop();
    assert!(
        FixedReferenceFsiPartition::<2>::new(&mesh, incomplete_cells, solid, interface).is_err()
    );
}

#[test]
fn bubble_history_uses_complete_exact_cell_keys_independent_of_insertion_order() {
    let fixture = fixture_problem();
    let cells = fixture.partition.fluid_cells();
    assert!(cells.len() > 1);
    let make = |bubbles| {
        FixedReferenceFsiState::<2>::new(
            &fixture.mesh,
            &fixture.partition,
            fixture.previous.vertex_velocity().to_vec(),
            bubbles,
            fixture.previous.solid_displacement().to_vec(),
        )
    };
    let entries = cells
        .iter()
        .copied()
        .map(|cell| (cell, [cell.index() as f64, -0.25]))
        .collect::<Vec<_>>();
    let accepted = make(entries.iter().copied().collect()).unwrap();
    assert_eq!(
        accepted,
        make(entries.iter().rev().copied().collect()).unwrap()
    );
    for (cell, expected) in &entries {
        assert_eq!(accepted.fluid_cell_bubble_velocity()[cell], *expected);
    }
    let mut missing = accepted.fluid_cell_bubble_velocity().clone();
    missing.remove(&cells[0]);
    assert!(make(missing).is_err());
    let mut foreign = accepted.fluid_cell_bubble_velocity().clone();
    let value = foreign.remove(&cells[0]).unwrap();
    foreign.insert(fixture.partition.solid_cells()[0], value);
    assert!(make(foreign).is_err());
    let mut nonfinite = accepted.fluid_cell_bubble_velocity().clone();
    nonfinite.get_mut(&cells[0]).unwrap()[1] = f64::NAN;
    assert!(make(nonfinite).is_err());
}

#[test]
fn tetrahedral_contract_replays_one_exact_interface_and_dimensioned_state() {
    let mesh = two_tetrahedron_mesh();
    let interface = shared_tetrahedron_interface(&mesh);
    let partition = FixedReferenceFsiPartition::<3>::new(
        &mesh,
        vec![CellId::new(0)],
        vec![CellId::new(1)],
        vec![interface],
    )
    .unwrap();
    let witness = partition.interface_witnesses()[0];
    assert_eq!(witness.facet(), interface);
    assert_eq!(witness.fluid_cell(), CellId::new(0));
    assert_eq!(witness.solid_cell(), CellId::new(1));
    let boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    let previous = FixedReferenceFsiState::<3>::new(
        &mesh,
        &partition,
        vec![[0.0; 3]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![[0.0; 3]; mesh.vertices().len()],
    )
    .unwrap();
    let material = FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 2.0, 3.0, 1.0).unwrap();
    let scale = FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap();
    assert_eq!(scale.action(), 12.0);
    assert_eq!(scale.energy(), 24.0);
    assert_eq!(scale.power(), 60.0);
    let config =
        FixedReferenceFsiStepConfig::<3>::new(0.25, material, scale, FixedReferenceFsiLoad::Zero)
            .unwrap();
    let under_integrated = validate_problem(
        &mesh,
        &partition,
        &boundary,
        &previous,
        config,
        &simplex_duffy_gauss_legendre(3, 5).unwrap(),
    )
    .unwrap_err();
    assert!(under_integrated.message().contains("degree 8"));
    validate_problem(
        &mesh,
        &partition,
        &boundary,
        &previous,
        config,
        &simplex_duffy_gauss_legendre(3, 6).unwrap(),
    )
    .unwrap();
    let problem_2d = fixture_problem();
    assert_eq!(problem_2d.quadrature.polynomial_exactness(), Some(6));
    validate_problem(
        &problem_2d.mesh,
        &problem_2d.partition,
        &problem_2d.boundary,
        &problem_2d.previous,
        problem_2d.config,
        &problem_2d.quadrature,
    )
    .unwrap();

    assert!(
        FixedReferenceFsiPartition::<3>::new(
            &mesh,
            vec![CellId::new(0)],
            vec![CellId::new(1)],
            Vec::new(),
        )
        .is_err()
    );
    assert!(
        FixedReferenceFsiPartition::<3>::new(
            &two_domain_mesh(),
            vec![CellId::new(0)],
            vec![CellId::new(1)],
            vec![FacetId::new(0)],
        )
        .is_err()
    );
}

#[test]
fn material_coercivity_depends_on_the_admitted_spatial_dimension() {
    assert!(FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 1.0, 1.0, -0.8).is_ok());
    let error = FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 1.0, 1.0, -0.8).unwrap_err();
    assert!(error.message().contains("coercive"));
}

#[test]
fn finalized_operator_is_symmetric_and_constant_pressure_is_closed_by_interface_action() {
    let problem = fixture_problem();
    let finalized =
        resolved::finalize(&problem, problem.config, &REFERENCE_ASSEMBLY_BACKEND).unwrap();
    assert_eq!(
        finalized.linear_system().properties(),
        LinearOperatorProperties::SymmetricIndefinite
    );
    let system = finalized.linear_system();
    for row in 0..system.rows() {
        for column in 0..system.columns() {
            let left = canonical_entry(system, row, column);
            let right = canonical_entry(system, column, row);
            assert!((left - right).abs() < 2.0e-13);
        }
    }
}

#[test]
fn finalized_step_retains_exact_reduced_and_full_target_roles() {
    let problem = fixture_problem();
    let finalized =
        resolved::finalize(&problem, problem.config, &REFERENCE_ASSEMBLY_BACKEND).unwrap();
    let roles = finalized.assembly_target_roles();
    assert_eq!(roles.reduced().index(), 0);
    assert_eq!(roles.full().index(), 1);
    assert_ne!(roles.reduced(), roles.full());
}

#[test]
fn monolithic_step_closes_residual_kinematics_interface_and_energy() {
    let problem = fixture_problem();
    let solution = resolved::finalize(&problem, problem.config, &REFERENCE_ASSEMBLY_BACKEND)
        .unwrap()
        .solve(&REFERENCE_LINEAR_SOLVER)
        .unwrap()
        .into_numerical_evidence();
    assert!(solution.residual_norm() < 1.0e-9);
    assert!(solution.continuity_residual_norm() < 1.0e-9);
    assert!(solution.kinematic_residual_norm() < 1.0e-14);
    assert_eq!(solution.interface_velocity_jump_norm(), 0.0);
    assert!(!solution.interface_actions().is_empty());
    assert!(solution.interface_action_imbalance_norm() < 1.0e-9);
    assert!(solution.energy_balance().defect().abs() < 1.0e-9);
    assert!(
        solution
            .vertex_velocity()
            .iter()
            .flatten()
            .any(|value| value.abs() > 1.0e-10)
    );
}

#[test]
fn tetrahedral_monolithic_step_closes_the_same_physical_acceptance() {
    let problem = fixture_problem_3d();
    let solution = solve_fixed_reference_fsi_step_3d(
        &problem.mesh,
        &problem.partition,
        &problem.boundary,
        &problem.previous,
        problem.config,
        &problem.quadrature,
        reference_solver(),
        &problem.layout,
    )
    .unwrap();

    assert_eq!(solution.vertex_velocity().first().unwrap().len(), 3);
    assert!(solution.residual_norm() < 1.0e-9);
    assert!(solution.continuity_residual_norm() < 1.0e-9);
    assert!(solution.kinematic_residual_norm() < 1.0e-14);
    assert_eq!(solution.interface_velocity_jump_norm(), 0.0);
    assert_eq!(solution.interface_actions().len(), 1);
    assert!(solution.interface_action_imbalance_norm() < 1.0e-9);
    assert!(solution.energy_balance().defect().abs() < 1.0e-9);
    assert!(
        solution
            .vertex_velocity()
            .iter()
            .flatten()
            .any(|value| value.abs() > 1.0e-10)
    );
}

#[test]
fn tetrahedral_physical_step_is_invariant_under_dimensioned_scale_profiles() {
    let problem = fixture_problem_3d();
    let reference = solve_fixed_reference_fsi_step_3d(
        &problem.mesh,
        &problem.partition,
        &problem.boundary,
        &problem.previous,
        problem.config,
        &problem.quadrature,
        reference_solver(),
        &problem.layout,
    )
    .unwrap();
    let rescaled_config = FixedReferenceFsiStepConfig::<3>::new(
        problem.config.time_step(),
        problem.config.material(),
        FixedReferenceFsiScale::<3>::new(4.0, 0.25, 3.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
    )
    .unwrap();
    let rescaled_layout = super::test_model::polyhedra::polyhedral_layout(
        &super::test_model::polyhedra::bipyramid_geometry(),
        &problem.mesh,
        &problem.partition,
        &problem.boundary,
        rescaled_config,
        reference_solver().plan(),
        false,
    );
    let rescaled = solve_fixed_reference_fsi_step_3d(
        &problem.mesh,
        &problem.partition,
        &problem.boundary,
        &problem.previous,
        rescaled_config,
        &problem.quadrature,
        reference_solver(),
        &rescaled_layout,
    )
    .unwrap();

    assert_close(
        reference.vertex_velocity().iter().flatten().copied(),
        rescaled.vertex_velocity().iter().flatten().copied(),
    );
    assert_close(
        reference.fluid_pressure().iter().copied(),
        rescaled.fluid_pressure().iter().copied(),
    );
    assert_close(
        reference.solid_displacement().iter().flatten().copied(),
        rescaled.solid_displacement().iter().flatten().copied(),
    );
    assert_eq!(
        reference
            .interface_actions()
            .iter()
            .map(|action| action.vertex())
            .collect::<Vec<_>>(),
        rescaled
            .interface_actions()
            .iter()
            .map(|action| action.vertex())
            .collect::<Vec<_>>()
    );
    assert_close(
        reference
            .interface_actions()
            .iter()
            .flat_map(|action| action.fluid().into_iter().chain(action.solid())),
        rescaled
            .interface_actions()
            .iter()
            .flat_map(|action| action.fluid().into_iter().chain(action.solid())),
    );
}

#[test]
fn physical_step_is_invariant_under_admitted_scale_profiles() {
    let problem = fixture_problem();
    let reference = resolved::finalize(&problem, problem.config, &REFERENCE_ASSEMBLY_BACKEND)
        .unwrap()
        .solve(&REFERENCE_LINEAR_SOLVER)
        .unwrap()
        .into_numerical_evidence();
    let rescaled_config = FixedReferenceFsiStepConfig::<2>::new(
        problem.config.time_step(),
        problem.config.material(),
        FixedReferenceFsiScale::<2>::new(2.0, 0.25, 3.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
    )
    .unwrap();
    let rescaled = resolved::finalize(&problem, rescaled_config, &REFERENCE_ASSEMBLY_BACKEND)
        .unwrap()
        .solve(&REFERENCE_LINEAR_SOLVER)
        .unwrap()
        .into_numerical_evidence();
    for (left, right) in reference
        .vertex_velocity()
        .iter()
        .flatten()
        .zip(rescaled.vertex_velocity().iter().flatten())
    {
        assert!((left - right).abs() < 2.0e-9);
    }
    for (left, right) in reference
        .fluid_pressure()
        .iter()
        .zip(rescaled.fluid_pressure())
    {
        assert!((left - right).abs() < 2.0e-9);
    }
    for (left, right) in reference
        .solid_displacement()
        .iter()
        .flatten()
        .zip(rescaled.solid_displacement().iter().flatten())
    {
        assert!((left - right).abs() < 2.0e-9);
    }
}

#[test]
fn finalization_rejects_a_symmetric_operator_with_constant_pressure_nullspace() {
    let problem = fixture_problem();
    let error = resolved::finalize_pressure_nullspace(&problem).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_DISCRETIZATION);
    assert!(error.message().contains("constant pressure unclosed"));
}

#[test]
fn finalization_rejects_a_backend_result_outside_the_prepared_target_shape() {
    let problem = fixture_problem();
    let error = resolved::finalize(&problem, problem.config, &WrongShapeAssemblyBackend)
        .expect_err("an assembly backend cannot replace the prepared target shapes");
    assert_eq!(error.code(), codes::INVALID_DISCRETIZATION);
    assert!(error.message().contains("prepared target shape"));
}

#[derive(Debug)]
struct WrongShapeAssemblyBackend;

impl AssemblyBackend for WrongShapeAssemblyBackend {
    fn assemble(
        &self,
        _plan: &AssemblyPlan,
        original_work: &dyn eqiora_assembly::AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1)?, AssemblyTarget::new(1)?])?;
        let reduced = plan
            .target_id(0)
            .expect("two-target malformed test plan owns reduced");
        let full = plan
            .target_id(1)
            .expect("two-target malformed test plan owns full");
        let dof = DofId::new(0);
        let work = IndexedAssemblyWork::new(original_work.packet_count(), move |_| {
            let local = LocalContribution::new(1, 1, vec![1.0], vec![0.0])?;
            let map = || AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)]);
            AssemblyPacket::new(
                local,
                vec![
                    TargetAssemblyMap::new(reduced, map()?),
                    TargetAssemblyMap::new(full, map()?),
                ],
            )
        });
        REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work)
    }
}

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    boundary: FixedReferenceFsiBoundary<2>,
    previous: FixedReferenceFsiState<2>,
    config: FixedReferenceFsiStepConfig<2>,
    quadrature: QuadratureRule,
}

struct Fixture3d {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    layout: super::layout::FsiLayout<3>,
    boundary: FixedReferenceFsiBoundary<3>,
    previous: FixedReferenceFsiState<3>,
    config: FixedReferenceFsiStepConfig<3>,
    quadrature: QuadratureRule,
}

fn fixture_problem() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid, interface) = inventories(&mesh);
    let partition = FixedReferenceFsiPartition::<2>::new(&mesh, fluid, solid, interface).unwrap();
    let boundary = FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
    let mut displacement = vec![[0.0; 2]; mesh.vertices().len()];
    let interface_midpoint = find_vertex(&mesh, [1.0, 0.5]);
    displacement[interface_midpoint][0] = 0.02;
    let previous = FixedReferenceFsiState::<2>::new(
        &mesh,
        &partition,
        vec![[0.0; 2]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 2]))
            .collect(),
        displacement,
    )
    .unwrap();
    let material = FixedReferenceFsiMaterial::<2>::new(1.0, 0.05, 1.5, 2.0, 3.0).unwrap();
    let scale = FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap();
    let config =
        FixedReferenceFsiStepConfig::<2>::new(0.05, material, scale, FixedReferenceFsiLoad::Zero)
            .unwrap();
    Fixture {
        mesh,
        partition,
        boundary,
        previous,
        config,
        quadrature: triangle_duffy_gauss_legendre(4).unwrap(),
    }
}

fn fixture_problem_3d() -> Fixture3d {
    let mesh = interface_bipyramid_mesh();
    let fluid = (0..4).map(CellId::new).collect::<Vec<_>>();
    let solid = (4..8).map(CellId::new).collect::<Vec<_>>();
    let interface = (0..mesh.entity_count(2).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 0.0)
        })
        .map(FacetId::new)
        .collect::<Vec<_>>();
    let partition = FixedReferenceFsiPartition::<3>::new(&mesh, fluid, solid, interface).unwrap();
    let boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    let mut displacement = vec![[0.0; 3]; mesh.vertices().len()];
    displacement[2] = [0.015, -0.01, 0.005];
    let previous = FixedReferenceFsiState::<3>::new(
        &mesh,
        &partition,
        vec![[0.0; 3]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        displacement,
    )
    .unwrap();
    let material = FixedReferenceFsiMaterial::<3>::new(1.0, 0.05, 1.5, 2.0, 3.0).unwrap();
    let scale = FixedReferenceFsiScale::<3>::new(2.0, 1.0, 1.0).unwrap();
    let config =
        FixedReferenceFsiStepConfig::<3>::new(0.05, material, scale, FixedReferenceFsiLoad::Zero)
            .unwrap();
    let layout = super::test_model::polyhedra::polyhedral_layout(
        &super::test_model::polyhedra::bipyramid_geometry(),
        &mesh,
        &partition,
        &boundary,
        config,
        reference_solver().plan(),
        false,
    );
    Fixture3d {
        layout,
        mesh,
        partition,
        boundary,
        previous,
        config,
        quadrature: simplex_duffy_gauss_legendre(3, 6).unwrap(),
    }
}

fn two_domain_mesh() -> SimplicialMesh {
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in [0.0, 1.0, 2.0] {
            vertices.push(vec![x, y]);
        }
    }
    let width = 3;
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..2 {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn two_tetrahedron_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, -1.0],
        ],
        vec![vec![0, 1, 2, 3], vec![0, 2, 1, 4]],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap()
}

fn interface_bipyramid_mesh() -> SimplicialMesh {
    let vertices = vec![
        vec![-1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, -1.0, 0.0],
        vec![0.0, 0.0, -1.0],
    ];
    let ring = [3, 4, 5, 6];
    let mut cells = Vec::new();
    for apex in [0, 1] {
        for edge in 0..ring.len() {
            let mut cell = vec![apex, 2, ring[edge], ring[(edge + 1) % ring.len()]];
            if signed_tetrahedron_jacobian(&vertices, &cell) < 0.0 {
                cell.swap(1, 2);
            }
            cells.push(cell);
        }
    }
    SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.05).unwrap()).unwrap()
}

fn signed_tetrahedron_jacobian(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(2, 2) * column(3, 1))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(1, 2) * column(3, 1))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(1, 2) * column(2, 1))
}

fn shared_tetrahedron_interface(mesh: &SimplicialMesh) -> FacetId {
    (0..mesh.entity_count(2).unwrap())
        .find(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .unwrap()
                .iter()
                .map(|vertex| vertex.index())
                .collect::<Vec<_>>()
                == [0, 1, 2]
        })
        .map(FacetId::new)
        .unwrap()
}

fn inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 3.0;
        if centroid_x < 1.0 {
            fluid.push(CellId::new(index));
        } else {
            solid.push(CellId::new(index));
        }
    }
    let interface = (0..mesh.entity_count(1).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(1, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

fn find_vertex(mesh: &SimplicialMesh, target: [f64; 2]) -> usize {
    mesh.vertices()
        .iter()
        .position(|coordinates| coordinates.as_slice() == target)
        .unwrap()
}

fn reference_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn canonical_entry(system: &CanonicalCsrSystemView, row: usize, column: usize) -> f64 {
    let start = system.row_offsets()[row];
    let end = system.row_offsets()[row + 1];
    match system.column_indices()[start..end].binary_search(&column) {
        Ok(position) => system.values()[start + position],
        Err(_) => 0.0,
    }
}

fn assert_close(left: impl IntoIterator<Item = f64>, right: impl IntoIterator<Item = f64>) {
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    loop {
        match (left.next(), right.next()) {
            (Some(left), Some(right)) => {
                let tolerance = 2.0e-9 + 2.0e-9 * left.abs().max(right.abs());
                assert!((left - right).abs() <= tolerance, "{left:e} != {right:e}");
            }
            (None, None) => break,
            _ => panic!("compared physical fields have different coefficient counts"),
        }
    }
}
