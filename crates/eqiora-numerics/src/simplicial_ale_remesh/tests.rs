use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, OverlapCoordinateChart2d,
    SimplicialMesh, SimplicialRevisionOverlap2d, triangle_duffy_gauss_legendre,
};
use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

use crate::simplicial_ale_fsi::{AleFsiState, P1HarmonicMeshMotionAction};
use crate::simplicial_fsi::{
    FixedReferenceFsiMaterial, FixedReferenceFsiPartition, FixedReferenceFsiScale,
};

use super::integration::{cell_basis, dense_zeroed, integrate_physical_triangle};
use super::project_simplicial_ale_fsi_remesh_2d;
use super::projection::{
    homogeneous_exterior_velocity_trace_defect, material_overlap,
    retained_interface_p1_trace_defect, retained_p1_trace_defect,
};

const COMPONENTS: usize = 2;

#[test]
fn forward_overlap_map_integrates_a_positive_skinny_fragment() {
    let source = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.2).unwrap(),
    )
    .unwrap();
    let adjacent_half = f64::from_bits(0.5_f64.to_bits() + 1);
    let target = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, adjacent_half],
        ],
        vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]],
        MeshQualityGate::new(0.2).unwrap(),
    )
    .unwrap();
    let overlap = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &source,
        &[CellId::new(0), CellId::new(1)],
        &target,
        &[
            CellId::new(0),
            CellId::new(1),
            CellId::new(2),
            CellId::new(3),
        ],
    )
    .unwrap();
    let fragment = overlap
        .cell_fragments()
        .iter()
        .min_by(|left, right| left.area().total_cmp(&right.area()))
        .unwrap();
    assert!(
        eqiora_meshing::AffineGeometryMap::from_simplex_vertices(
            fragment
                .triangle()
                .iter()
                .map(|point| point.to_vec())
                .collect(),
        )
        .is_err(),
        "a common-refinement integration fragment is not a mesh-quality object"
    );

    let mut integral = [0.0; 3];
    integrate_physical_triangle(
        fragment,
        &triangle_duffy_gauss_legendre(5).unwrap(),
        |point, weight| {
            integral[0] += weight;
            integral[1] += weight * point[0];
            integral[2] += weight * point[1];
            Ok(())
        },
    )
    .unwrap();
    let expected = [
        fragment.area(),
        fragment.first_moment()[0],
        fragment.first_moment()[1],
    ];
    for (actual, expected) in integral.into_iter().zip(expected) {
        let tolerance = 512.0 * f64::EPSILON * (fragment.area() + expected.abs());
        assert!((actual - expected).abs() <= tolerance);
    }
}

#[test]
fn topology_distinct_remesh_reproduces_affine_absolute_fields() {
    let source_mesh = two_domain_mesh(false);
    let target_mesh = two_domain_mesh(true);
    assert_ne!(source_mesh.cells(), target_mesh.cells());
    let source_partition = partition(&source_mesh);
    let target_partition = partition(&target_mesh);
    let source_motion =
        P1HarmonicMeshMotionAction::<2>::new(&source_mesh, &source_partition, harmonic_solver())
            .unwrap();
    let target_motion =
        P1HarmonicMeshMotionAction::<2>::new(&target_mesh, &target_partition, harmonic_solver())
            .unwrap();

    let mut displacement = vec![[0.0; COMPONENTS]; source_mesh.vertices().len()];
    for vertex in source_partition.solid_vertices() {
        let point = &source_mesh.vertices()[vertex.index()];
        displacement[vertex.index()] = [0.01 * point[1], 0.005 * (point[0] - 1.0)];
    }
    let provisional = AleFsiState::<2>::new(
        0.75,
        &source_mesh,
        &source_partition,
        &source_motion,
        vec![[0.0; COMPONENTS]; source_mesh.vertices().len()],
        source_partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        vec![0.0; source_partition.fluid_vertices().len()],
        displacement.clone(),
    )
    .unwrap();
    let current = provisional
        .geometry()
        .reconstruct_mesh(&source_mesh)
        .unwrap();
    let pressure = source_partition
        .fluid_vertices()
        .iter()
        .map(|vertex| {
            let point = &current.vertices()[vertex.index()];
            2.0 + point[0] - 3.0 * point[1]
        })
        .collect();
    let source_state = AleFsiState::<2>::new(
        0.75,
        &source_mesh,
        &source_partition,
        &source_motion,
        vec![[0.0; COMPONENTS]; source_mesh.vertices().len()],
        source_partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        pressure,
        displacement,
    )
    .unwrap();

    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let accepted = project_simplicial_ale_fsi_remesh_2d(
        &source_mesh,
        &source_partition,
        &source_motion,
        &source_state,
        &target_mesh,
        &target_partition,
        &target_motion,
        FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 2.0, 2.0, 1.0).unwrap(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 2.0).unwrap(),
        &quadrature,
        projection_solver(),
    )
    .unwrap();

    assert_eq!(accepted.time(), source_state.time());
    assert_eq!(
        accepted.evidence().displacement_solve_reports().len(),
        COMPONENTS
    );
    assert_eq!(
        accepted
            .evidence()
            .dimensionless_displacement_right_hand_side_norms()
            .len(),
        COMPONENTS
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_velocity_right_hand_side_norm()
            .is_finite()
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_pressure_right_hand_side_norm()
            .is_finite()
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_displacement_projection_residual_norm()
            <= accepted
                .evidence()
                .dimensionless_displacement_projection_acceptance_limit()
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_velocity_projection_residual_norm()
            <= accepted
                .evidence()
                .dimensionless_velocity_projection_acceptance_limit()
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_pressure_projection_residual_norm()
            <= accepted
                .evidence()
                .dimensionless_pressure_projection_acceptance_limit()
    );
    assert!(accepted.evidence().displacement_l2_error() < 1.0e-10);
    assert!(
        accepted
            .evidence()
            .fluid_current_density_weighted_velocity_l2_error()
            < 1.0e-10
    );
    assert!(
        accepted
            .evidence()
            .solid_material_density_weighted_velocity_l2_error()
            < 1.0e-10
    );
    assert!(accepted.evidence().pressure_l2_error() < 1.0e-9);
    assert!(
        accepted
            .evidence()
            .dimensionless_displacement_trace_defect()
            < 1.0e-12
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_shared_velocity_trace_defect()
            < 1.0e-12
    );
    assert_eq!(
        accepted
            .evidence()
            .dimensionless_exterior_velocity_trace_defect(),
        0.0
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_weak_incompressibility_defect()
            < 1.0e-9
    );
    assert!(
        accepted
            .evidence()
            .dimensionless_pressure_zeroth_moment_defect()
            < 1.0e-9
    );
    accepted.initial_physical_state().unwrap();
}

#[test]
fn mini_transfer_basis_is_a_cubic_bubble_not_a_cell_constant() {
    let mesh = two_domain_mesh(false);
    let cell = CellId::new(0);
    let vertices = &mesh.cells()[cell.index()];
    let centroid = [
        vertices
            .iter()
            .map(|&vertex| mesh.vertices()[vertex][0])
            .sum::<f64>()
            / 3.0,
        vertices
            .iter()
            .map(|&vertex| mesh.vertices()[vertex][1])
            .sum::<f64>()
            / 3.0,
    ];
    let near_vertex = [
        0.8 * mesh.vertices()[vertices[0]][0]
            + 0.1 * mesh.vertices()[vertices[1]][0]
            + 0.1 * mesh.vertices()[vertices[2]][0],
        0.8 * mesh.vertices()[vertices[0]][1]
            + 0.1 * mesh.vertices()[vertices[1]][1]
            + 0.1 * mesh.vertices()[vertices[2]][1],
    ];
    let center = cell_basis(&mesh, cell, centroid, true).unwrap();
    let off_center = cell_basis(&mesh, cell, near_vertex, true).unwrap();
    assert!((center.values[3] - 1.0).abs() < 1.0e-12);
    assert!((off_center.values[3] - 0.216).abs() < 1.0e-12);
    assert_ne!(center.values[3], off_center.values[3]);
}

#[test]
fn retained_fragment_endpoints_expose_a_coarsened_trace_kink() {
    let source = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 1.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0, 1, 4], vec![0, 4, 3], vec![1, 2, 5], vec![1, 5, 4]],
        MeshQualityGate::new(0.3).unwrap(),
    )
    .unwrap();
    let target = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.3).unwrap(),
    )
    .unwrap();
    let source_cells = (0..source.cells().len())
        .map(CellId::new)
        .collect::<Vec<_>>();
    let target_cells = (0..target.cells().len())
        .map(CellId::new)
        .collect::<Vec<_>>();
    let overlap = material_overlap(&source, &source_cells, &target, &target_cells).unwrap();
    let mut source_trace = vec![[0.0; COMPONENTS]; source.vertices().len()];
    source_trace[1] = [1.0, 0.0];
    let target_trace = vec![[0.0; COMPONENTS]; target.vertices().len()];
    let defect =
        retained_p1_trace_defect(&overlap, &source, &source_trace, &target, &target_trace).unwrap();
    assert!((defect - 1.0).abs() < 1.0e-12);
}

#[test]
fn shared_interface_and_physical_exterior_are_distinct_trace_obligations() {
    let source = two_domain_mesh(false);
    let target = two_domain_mesh(true);
    let source_partition = partition(&source);
    let target_partition = partition(&target);
    let overlap = material_overlap(
        &source,
        source_partition.solid_cells(),
        &target,
        target_partition.solid_cells(),
    )
    .unwrap();
    let source_velocity = vec![[0.0; COMPONENTS]; source.vertices().len()];
    let mut target_velocity = vec![[0.0; COMPONENTS]; target.vertices().len()];
    let interior_interface = target_partition
        .interface_vertices()
        .iter()
        .find(|vertex| target.vertices()[vertex.index()][1] == 0.5)
        .expect("fixture owns one non-exterior interface vertex");
    target_velocity[interior_interface.index()] = [1.0, 0.0];

    let shared = retained_interface_p1_trace_defect(
        &overlap,
        &source,
        &source_partition,
        &source_velocity,
        &target,
        &target_partition,
        &target_velocity,
    )
    .unwrap();
    let exterior = homogeneous_exterior_velocity_trace_defect(&source, &source_velocity)
        .unwrap()
        .max(homogeneous_exterior_velocity_trace_defect(&target, &target_velocity).unwrap());

    assert!((shared - 1.0).abs() < 1.0e-12);
    assert_eq!(exterior, 0.0);
}

#[test]
fn dense_reference_allocation_fails_before_shape_overflow() {
    assert!(dense_zeroed(usize::MAX).is_err());
}

#[test]
fn dimensionless_admission_is_invariant_under_physical_rescaling() {
    let base = scaled_projection_case(1.0, 0.25, 3.0, false).unwrap();
    let rescaled = scaled_projection_case(1.0e3, 2.5e-4, 3.0e6, false).unwrap();
    let base_values = dimensionless_obligations(base.evidence());
    let rescaled_values = dimensionless_obligations(rescaled.evidence());
    for (left, right) in base_values.into_iter().zip(rescaled_values) {
        assert!((left - right).abs() < 1.0e-8);
    }
    assert!(scaled_projection_case(1.0, 0.25, 3.0, true).is_err());
    assert!(scaled_projection_case(1.0e3, 2.5e-4, 3.0e6, true).is_err());
}

fn scaled_projection_case(
    coordinate_factor: f64,
    velocity_scale: f64,
    pressure_scale: f64,
    violate_exterior: bool,
) -> Result<crate::simplicial_ale_remesh::AcceptedAleFsiRemeshProjection2d, Diagnostic> {
    let source_mesh = scaled_two_domain_mesh(false, coordinate_factor);
    let target_mesh = scaled_two_domain_mesh(true, coordinate_factor);
    let source_partition = partition(&source_mesh);
    let target_partition = partition(&target_mesh);
    let source_motion =
        P1HarmonicMeshMotionAction::<2>::new(&source_mesh, &source_partition, harmonic_solver())?;
    let target_motion =
        P1HarmonicMeshMotionAction::<2>::new(&target_mesh, &target_partition, harmonic_solver())?;
    let length_scale = 2.0 * coordinate_factor;

    let mut displacement = vec![[0.0; COMPONENTS]; source_mesh.vertices().len()];
    let mut velocity = vec![[0.0; COMPONENTS]; source_mesh.vertices().len()];
    for vertex in source_partition.solid_vertices() {
        let point = &source_mesh.vertices()[vertex.index()];
        let normalized = [point[0] / length_scale, point[1] / length_scale];
        displacement[vertex.index()] = [
            length_scale * 0.01 * normalized[1],
            length_scale * 0.005 * (normalized[0] - 0.5),
        ];
    }
    if violate_exterior {
        velocity[0][0] = velocity_scale * 1.0e-4;
    }
    let bubbles: std::collections::BTreeMap<_, _> = source_partition
        .fluid_cells()
        .iter()
        .copied()
        .map(|cell| (cell, [0.0; COMPONENTS]))
        .collect();
    let provisional = AleFsiState::<2>::new(
        0.75,
        &source_mesh,
        &source_partition,
        &source_motion,
        velocity.clone(),
        bubbles.clone(),
        vec![0.0; source_partition.fluid_vertices().len()],
        displacement.clone(),
    )?;
    let current = provisional.geometry().reconstruct_mesh(&source_mesh)?;
    let pressure = source_partition
        .fluid_vertices()
        .iter()
        .map(|vertex| {
            let point = &current.vertices()[vertex.index()];
            pressure_scale * (2.0 + point[0] / length_scale - 3.0 * point[1] / length_scale)
        })
        .collect();
    let source_state = AleFsiState::<2>::new(
        0.75,
        &source_mesh,
        &source_partition,
        &source_motion,
        velocity,
        bubbles,
        pressure,
        displacement,
    )?;
    project_simplicial_ale_fsi_remesh_2d(
        &source_mesh,
        &source_partition,
        &source_motion,
        &source_state,
        &target_mesh,
        &target_partition,
        &target_motion,
        FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 2.0, 2.0, 1.0)?,
        FixedReferenceFsiScale::<2>::new(length_scale, velocity_scale, pressure_scale)?,
        &triangle_duffy_gauss_legendre(5)?,
        projection_solver(),
    )
}

fn dimensionless_obligations(
    evidence: &crate::simplicial_ale_remesh::AleFsiRemeshProjectionEvidence2d,
) -> [f64; 6] {
    [
        evidence.dimensionless_displacement_trace_defect(),
        evidence.dimensionless_shared_velocity_trace_defect(),
        evidence.dimensionless_exterior_velocity_trace_defect(),
        evidence.dimensionless_weak_incompressibility_defect(),
        evidence.dimensionless_momentum_defect(),
        evidence.dimensionless_pressure_zeroth_moment_defect(),
    ]
}

fn projection_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn harmonic_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn two_domain_mesh(flip_diagonal: bool) -> SimplicialMesh {
    scaled_two_domain_mesh(flip_diagonal, 1.0)
}

fn scaled_two_domain_mesh(flip_diagonal: bool, coordinate_factor: f64) -> SimplicialMesh {
    let x_coordinates = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in x_coordinates {
            vertices.push(vec![x * coordinate_factor, y * coordinate_factor]);
        }
    }
    let width = x_coordinates.len();
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..width - 1 {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            if flip_diagonal {
                cells.push(vec![lower_left, lower_right, upper_left]);
                cells.push(vec![lower_right, upper_right, upper_left]);
            } else {
                cells.push(vec![lower_left, lower_right, upper_right]);
                cells.push(vec![lower_left, upper_right, upper_left]);
            }
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn partition(mesh: &SimplicialMesh) -> FixedReferenceFsiPartition<2> {
    let interface_x = mesh
        .vertices()
        .iter()
        .map(|vertex| vertex[0])
        .fold(f64::NEG_INFINITY, f64::max)
        / 2.0;
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 3.0;
        if centroid_x < interface_x {
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
                .all(|vertex| mesh.vertices()[vertex.index()][0] == interface_x)
        })
        .map(FacetId::new)
        .collect();
    FixedReferenceFsiPartition::<2>::new(mesh, fluid, solid, interface).unwrap()
}
