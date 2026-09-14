use std::num::NonZeroUsize;

use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{
    CellId, MeshQualityGate, OverlapCoordinateChart2d, SimplicialMesh, SimplicialRevisionOverlap2d,
    triangle_duffy_gauss_legendre,
};
use eqiora_realization::{
    AleGeometryQualityGate, ConformingTraceQuotient, P1HarmonicMeshMotionPolicy, TraceFieldEndpoint,
};
use eqiora_solver::{LinearSolver, SolverPlan};

use crate::simplicial_fsi::FixedReferenceFsiPartition;

use super::integration::{cell_basis, dense_zeroed, integrate_physical_triangle};
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
    let ids = PartitionIds::new();
    let source_partition = partition(&source, ids);
    let target_partition = partition(&target, ids);
    let overlap = material_overlap(
        &source,
        source_partition.domain_cells(ids.solid).unwrap(),
        &target,
        target_partition.domain_cells(ids.solid).unwrap(),
    )
    .unwrap();
    let source_velocity = vec![[0.0; COMPONENTS]; source.vertices().len()];
    let mut target_velocity = vec![[0.0; COMPONENTS]; target.vertices().len()];
    let interior_interface = target_partition
        .domain_vertices(ids.fluid)
        .unwrap()
        .iter()
        .filter(|vertex| {
            target_partition
                .domain_vertices(ids.solid)
                .unwrap()
                .contains(vertex)
        })
        .find(|vertex| target.vertices()[vertex.index()][1] == 0.5)
        .expect("fixture owns one non-exterior interface vertex");
    target_velocity[interior_interface.index()] = [1.0, 0.0];

    let shared = retained_interface_p1_trace_defect(
        ids.policy(),
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

fn two_domain_mesh(flip_diagonal: bool) -> SimplicialMesh {
    let x_coordinates = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in x_coordinates {
            vertices.push(vec![x, y]);
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

#[derive(Clone, Copy)]
struct PartitionIds {
    fluid: Id<kinds::Domain>,
    solid: Id<kinds::Domain>,
    fluid_velocity: Id<kinds::Field>,
    solid_velocity: Id<kinds::Field>,
    displacement: Id<kinds::Field>,
    interface: Id<kinds::Connection>,
}

impl PartitionIds {
    fn new() -> Self {
        Self {
            fluid: Id::new(),
            solid: Id::new(),
            fluid_velocity: Id::new(),
            solid_velocity: Id::new(),
            displacement: Id::new(),
            interface: Id::new(),
        }
    }

    fn policy(self) -> P1HarmonicMeshMotionPolicy {
        P1HarmonicMeshMotionPolicy::new(
            self.fluid,
            self.solid,
            self.displacement,
            self.interface,
            AleGeometryQualityGate::new(0.3).unwrap(),
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1.0e-12,
                1.0e-14,
                NonZeroUsize::new(500).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }
}

fn partition(mesh: &SimplicialMesh, ids: PartitionIds) -> FixedReferenceFsiPartition<2> {
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
    let quotient = ConformingTraceQuotient::new(
        ids.interface,
        TraceFieldEndpoint::new(ids.fluid, ids.fluid_velocity),
        TraceFieldEndpoint::new(ids.solid, ids.solid_velocity),
    )
    .unwrap();
    FixedReferenceFsiPartition::<2>::new(
        mesh,
        [(ids.fluid, fluid), (ids.solid, solid)],
        &[quotient],
    )
    .unwrap()
}
