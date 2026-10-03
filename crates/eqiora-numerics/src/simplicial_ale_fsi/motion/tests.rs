use std::num::NonZeroUsize;

use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{CellId, MeshEntity, MeshQualityGate};
use eqiora_realization::{AleGeometryQualityGate, ConformingTraceQuotient, TraceFieldEndpoint};
use eqiora_solver::{PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan};

use super::*;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

// Independent barycentric residual assembly, invalid-input rejection, and
// fail-closed solver cases form one private validation responsibility.
// Core motion identities stay beside the implementation they exercise.
mod validation;

const DIMENSION: usize = 2;
const COMPONENTS: usize = DIMENSION;

fn exact_id<K: eqiora_core::entity::Entity>(value: &str) -> Id<K> {
    Id::from_ulid(value.parse().expect("valid fixture ULID"))
}

fn fluid_domain() -> Id<kinds::Domain> {
    exact_id("01J00000000000000000000001")
}

fn solid_domain() -> Id<kinds::Domain> {
    exact_id("01J00000000000000000000002")
}

fn fluid_velocity() -> Id<kinds::Field> {
    exact_id("01J00000000000000000000003")
}

fn solid_velocity() -> Id<kinds::Field> {
    exact_id("01J00000000000000000000004")
}

fn solid_displacement() -> Id<kinds::Field> {
    exact_id("01J00000000000000000000005")
}

fn interface() -> Id<kinds::Connection> {
    exact_id("01J00000000000000000000006")
}

fn quotient() -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        interface(),
        TraceFieldEndpoint::new(fluid_domain(), fluid_velocity()),
        TraceFieldEndpoint::new(solid_domain(), solid_velocity()),
    )
    .expect("valid fixture quotient")
}

fn motion_policy(plan: SolverPlan) -> P1HarmonicMeshMotionPolicy {
    P1HarmonicMeshMotionPolicy::new(
        fluid_domain(),
        solid_domain(),
        solid_displacement(),
        interface(),
        AleGeometryQualityGate::new(0.01).expect("valid fixture quality gate"),
        plan,
    )
    .expect("valid fixture motion policy")
}

fn exact_partition<const D: usize>(
    mesh: &SimplicialMesh,
    fluid: Vec<CellId>,
    solid: Vec<CellId>,
) -> FixedReferenceFsiPartition<D> {
    FixedReferenceFsiPartition::new(
        mesh,
        [(fluid_domain(), fluid), (solid_domain(), solid)],
        &[quotient()],
    )
    .expect("valid exact test partition")
}

fn reference_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-13,
        1.0e-15,
        NonZeroUsize::new(10_000).expect("positive iteration limit"),
    )
    .expect("valid harmonic solver plan")
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Reproducible);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn seal_motion<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
) -> P1HarmonicMeshMotionAction<D> {
    let solver = reference_solver();
    P1HarmonicMeshMotionAction::new(mesh, partition, motion_policy(solver.plan()), solver)
        .expect("motion seals")
}

fn interface_vertices<const D: usize>(partition: &FixedReferenceFsiPartition<D>) -> Vec<VertexId> {
    partition
        .domain_vertices(fluid_domain())
        .expect("fluid Domain")
        .iter()
        .copied()
        .filter(|vertex| {
            partition
                .domain_vertices(solid_domain())
                .expect("solid Domain")
                .contains(vertex)
        })
        .collect()
}

fn refined_partition() -> (SimplicialMesh, FixedReferenceFsiPartition<2>) {
    partitioned_strip(&[0.0, 0.5, 1.0, 1.5, 2.0], &[0.0, 0.5, 1.0])
}

fn coarse_partition() -> (SimplicialMesh, FixedReferenceFsiPartition<2>) {
    partitioned_strip(&[0.0, 1.0, 2.0], &[0.0, 1.0])
}

fn refined_partition_3d() -> (SimplicialMesh, FixedReferenceFsiPartition<3>) {
    partitioned_block_3d(&[0.0, 0.5, 1.0, 2.0], &[0.0, 0.5, 1.0], &[0.0, 0.5, 1.0])
}

fn coarse_partition_3d() -> (SimplicialMesh, FixedReferenceFsiPartition<3>) {
    partitioned_block_3d(&[0.0, 1.0, 2.0], &[0.0, 1.0], &[0.0, 1.0])
}

fn partitioned_strip(
    x_coordinates: &[f64],
    y_coordinates: &[f64],
) -> (SimplicialMesh, FixedReferenceFsiPartition<2>) {
    let nx = x_coordinates.len();
    let vertices = y_coordinates
        .iter()
        .flat_map(|&y| x_coordinates.iter().map(move |&x| vec![x, y]))
        .collect::<Vec<_>>();
    let mut cells = Vec::new();
    let mut fluid_cells = Vec::new();
    let mut solid_cells = Vec::new();
    for x in 0..nx - 1 {
        for y in 0..y_coordinates.len() - 1 {
            let lower_left = y * nx + x;
            let lower_right = lower_left + 1;
            let upper_left = (y + 1) * nx + x;
            let upper_right = upper_left + 1;
            for triangle in [
                vec![lower_left, lower_right, upper_right],
                vec![lower_left, upper_right, upper_left],
            ] {
                let id = CellId::new(cells.len());
                if x_coordinates[x + 1] <= 1.0 {
                    fluid_cells.push(id);
                } else {
                    solid_cells.push(id);
                }
                cells.push(triangle);
            }
        }
    }
    let mesh = SimplicialMesh::new(
        DIMENSION,
        vertices,
        cells,
        MeshQualityGate::new(0.1).expect("valid test quality gate"),
    )
    .expect("valid conforming test mesh");
    let partition = exact_partition(&mesh, fluid_cells, solid_cells);
    (mesh, partition)
}

fn partitioned_block_3d(
    x_coordinates: &[f64],
    y_coordinates: &[f64],
    z_coordinates: &[f64],
) -> (SimplicialMesh, FixedReferenceFsiPartition<3>) {
    let nx = x_coordinates.len();
    let ny = y_coordinates.len();
    let vertex = |x: usize, y: usize, z: usize| z * ny * nx + y * nx + x;
    let vertices = z_coordinates
        .iter()
        .flat_map(|&z| {
            y_coordinates
                .iter()
                .flat_map(move |&y| x_coordinates.iter().map(move |&x| vec![x, y, z]))
        })
        .collect::<Vec<_>>();
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut cells = Vec::new();
    let mut fluid_cells = Vec::new();
    let mut solid_cells = Vec::new();
    for x in 0..nx - 1 {
        for y in 0..ny - 1 {
            for z in 0..z_coordinates.len() - 1 {
                for permutation in permutations {
                    let mut offset = [0, 0, 0];
                    let mut tetrahedron = vec![vertex(x, y, z)];
                    for axis in permutation {
                        offset[axis] = 1;
                        tetrahedron.push(vertex(x + offset[0], y + offset[1], z + offset[2]));
                    }
                    if signed_tetrahedron_measure(&vertices, &tetrahedron) < 0.0 {
                        tetrahedron.swap(1, 2);
                    }
                    let id = CellId::new(cells.len());
                    if x_coordinates[x + 1] <= 1.0 {
                        fluid_cells.push(id);
                    } else {
                        solid_cells.push(id);
                    }
                    cells.push(tetrahedron);
                }
            }
        }
    }
    let mesh = SimplicialMesh::new(
        3,
        vertices,
        cells,
        MeshQualityGate::new(0.02).expect("valid test quality gate"),
    )
    .expect("valid conforming tetrahedral test mesh");
    let partition = exact_partition(&mesh, fluid_cells, solid_cells);
    (mesh, partition)
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}

fn solid_field_3d(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
) -> BTreeMap<VertexId, [f64; 3]> {
    let mut field = BTreeMap::new();
    for vertex in partition
        .domain_vertices(solid_domain())
        .expect("solid Domain")
    {
        let point = &mesh.vertices()[vertex.index()];
        field.insert(
            *vertex,
            [
                0.01 + 0.02 * point[0] - 0.01 * point[1] + 0.03 * point[2],
                -0.02 + 0.01 * point[0] + 0.04 * point[1] - 0.02 * point[2],
                0.03 - 0.02 * point[0] + 0.01 * point[1] + 0.02 * point[2],
            ],
        );
    }
    field
}

fn solid_field(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    first: [f64; 3],
    second: [f64; 3],
) -> BTreeMap<VertexId, [f64; COMPONENTS]> {
    let mut field = BTreeMap::new();
    for vertex in partition
        .domain_vertices(solid_domain())
        .expect("solid Domain")
    {
        let point = &mesh.vertices()[vertex.index()];
        field.insert(
            *vertex,
            [
                first[0] + first[1] * point[0] + first[2] * point[1],
                second[0] + second[1] * point[0] + second[2] * point[1],
            ],
        );
    }
    field
}

#[test]
fn harmonic_motion_preserves_trace_exterior_and_independent_residual() {
    let (mesh, partition) = refined_partition();
    let solver = reference_solver();
    let motion = P1HarmonicMeshMotionAction::<2>::new(
        &mesh,
        &partition,
        motion_policy(solver.plan()),
        solver,
    )
    .expect("motion seals");
    assert_eq!(motion.reference_mesh(), &mesh);
    assert_eq!(motion.partition(), &partition);
    assert_eq!(motion.fluid_interior_vertices().len(), 1);
    assert_eq!(
        motion.influence_solve_reports().len(),
        motion.driver_vertices().len()
    );
    for report in motion.influence_solve_reports() {
        assert_eq!(report.backend(), REFERENCE_LINEAR_SOLVER.id());
        assert_eq!(report.solver_plan(), solver.plan());
        assert_eq!(report.algorithm(), LinearSolver::ConjugateGradient);
    }
    let solid = solid_field(&mesh, &partition, [0.01, 0.02, -0.03], [-0.02, 0.01, 0.04]);
    let displacement = motion
        .apply(solid_displacement(), &solid)
        .expect("harmonic action applies");

    for vertex in partition
        .domain_vertices(solid_domain())
        .expect("solid Domain")
    {
        assert_eq!(displacement[vertex.index()], solid[vertex]);
    }
    for vertex in motion.fixed_exterior_vertices() {
        assert_eq!(displacement[vertex.index()], [0.0; COMPONENTS]);
    }
    for vertex in interface_vertices(&partition) {
        assert_eq!(displacement[vertex.index()], solid[&vertex]);
    }
    assert!(interface_vertices(&partition).iter().any(|vertex| {
        mesh.is_boundary_entity(MeshEntity::new(0, vertex.index())) == Some(true)
            && !motion.fixed_exterior_vertices().contains(vertex)
    }));

    let interior = motion.fluid_interior_vertices()[0].index();
    let residual = independently_assemble_fluid_residual(&mesh, &partition, &displacement);
    for (component, value) in residual[interior].iter().enumerate() {
        assert!(
            value.abs() < 2.0e-14,
            "independent component {component} residual is {}",
            value
        );
    }
}

#[test]
fn influence_is_linear_and_jvp_is_the_same_exact_action() {
    let (mesh, partition) = refined_partition();
    let motion = seal_motion(&mesh, &partition);
    let left = solid_field(&mesh, &partition, [0.02, -0.01, 0.03], [0.01, 0.04, -0.02]);
    let right = solid_field(&mesh, &partition, [-0.03, 0.05, 0.01], [0.02, -0.02, 0.06]);
    let alpha = 1.75;
    let beta = -0.375;
    let combined = left
        .iter()
        .zip(&right)
        .map(|((vertex, left), (right_vertex, right))| {
            assert_eq!(vertex, right_vertex);
            (
                *vertex,
                [
                    alpha * left[0] + beta * right[0],
                    alpha * left[1] + beta * right[1],
                ],
            )
        })
        .collect::<BTreeMap<_, _>>();
    let applied_left = motion
        .apply(solid_displacement(), &left)
        .expect("left action applies");
    let applied_right = motion
        .apply(solid_displacement(), &right)
        .expect("right action applies");
    let applied_combined = motion
        .apply(solid_displacement(), &combined)
        .expect("combined action applies");
    let jvp = motion
        .apply_jvp(solid_displacement(), &combined)
        .expect("exact JVP applies");
    for vertex in 0..mesh.vertices().len() {
        for component in 0..COMPONENTS {
            let expected =
                alpha * applied_left[vertex][component] + beta * applied_right[vertex][component];
            assert!((applied_combined[vertex][component] - expected).abs() < 2.0e-14);
            assert_eq!(jvp[vertex][component], applied_combined[vertex][component]);
        }
    }
}

#[test]
fn tetrahedral_motion_has_one_shared_interface_driver_and_exact_jvp_action() {
    let (mesh, partition) = refined_partition_3d();
    let motion = seal_motion(&mesh, &partition);
    assert_eq!(motion.reference_mesh(), &mesh);
    assert_eq!(motion.partition(), &partition);
    assert_eq!(motion.fluid_interior_vertices().len(), 1);
    assert_eq!(
        motion.influence_solve_reports().len(),
        motion.driver_vertices().len()
    );
    for witness in &partition.traces()[0].facets {
        let facet_vertices = mesh
            .entity_vertices(witness.facet)
            .expect("shared triangular facet owns a closure");
        assert_eq!(facet_vertices.len(), 3);
        assert!(
            facet_vertices
                .iter()
                .map(|vertex| VertexId::new(vertex.index()))
                .all(|vertex| {
                    partition
                        .domain_vertices(fluid_domain())
                        .expect("fluid Domain")
                        .contains(&vertex)
                        && partition
                            .domain_vertices(solid_domain())
                            .expect("solid Domain")
                            .contains(&vertex)
                })
        );
    }

    let solid = solid_field_3d(&mesh, &partition);
    let displacement = motion
        .apply(solid_displacement(), &solid)
        .expect("3D harmonic action applies");
    let repeated = motion
        .apply(solid_displacement(), &solid)
        .expect("the same Dirichlet data has one sealed extension");
    let jvp = motion
        .apply_jvp(solid_displacement(), &solid)
        .expect("3D exact linear JVP applies");
    assert_eq!(displacement, repeated);
    assert_eq!(jvp, displacement);
    for vertex in partition
        .domain_vertices(solid_domain())
        .expect("solid Domain")
    {
        assert_eq!(displacement[vertex.index()], solid[vertex]);
    }
    for vertex in motion.fixed_exterior_vertices() {
        assert_eq!(displacement[vertex.index()], [0.0; 3]);
    }
    let interior = motion.fluid_interior_vertices()[0].index();
    assert!(
        displacement[interior]
            .iter()
            .all(|component| component.is_finite())
    );
    assert_ne!(displacement[interior], [0.0; 3]);
    assert_eq!(motion.relation.interior_stiffness().len(), 1);
    assert!(motion.relation.interior_stiffness()[0] > 0.0);
    for component in 0..3 {
        let prescribed_action = motion
            .relation
            .driver_vertices()
            .iter()
            .enumerate()
            .map(|(column, vertex)| {
                motion.relation.driver_stiffness()[column] * solid[vertex][component]
            })
            .sum::<f64>();
        let unique_extension = -prescribed_action / motion.relation.interior_stiffness()[0];
        assert!((displacement[interior][component] - unique_extension).abs() < 2.0e-14);
    }
}

#[test]
fn first_slice_rejects_a_partition_without_a_solved_fluid_interior() {
    let (mesh, partition) = coarse_partition();
    assert!(
        P1HarmonicMeshMotionAction::<2>::new(
            &mesh,
            &partition,
            motion_policy(reference_solver().plan()),
            reference_solver(),
        )
        .is_err()
    );
}

#[test]
fn tetrahedral_motion_rejects_incomplete_or_unsolved_dirichlet_closure() {
    let (mesh, partition) = coarse_partition_3d();
    assert!(
        P1HarmonicMeshMotionAction::<3>::new(
            &mesh,
            &partition,
            motion_policy(reference_solver().plan()),
            reference_solver(),
        )
        .is_err()
    );
}

fn independently_assemble_fluid_residual(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    displacement: &[[f64; COMPONENTS]],
) -> Vec<[f64; COMPONENTS]> {
    let mut residual = vec![[0.0; COMPONENTS]; mesh.vertices().len()];
    for cell in partition
        .domain_cells(fluid_domain())
        .expect("fluid Domain")
    {
        let vertices = &mesh.cells()[cell.index()];
        let point = |local: usize| &mesh.vertices()[vertices[local]];
        let twice_area = (point(1)[0] - point(0)[0]) * (point(2)[1] - point(0)[1])
            - (point(2)[0] - point(0)[0]) * (point(1)[1] - point(0)[1]);
        let b = [
            point(1)[1] - point(2)[1],
            point(2)[1] - point(0)[1],
            point(0)[1] - point(1)[1],
        ];
        let c = [
            point(2)[0] - point(1)[0],
            point(0)[0] - point(2)[0],
            point(1)[0] - point(0)[0],
        ];
        for row in 0..3 {
            for column in 0..3 {
                let stiffness = (b[row] * b[column] + c[row] * c[column]) / (2.0 * twice_area);
                for component in 0..COMPONENTS {
                    residual[vertices[row]][component] +=
                        stiffness * displacement[vertices[column]][component];
                }
            }
        }
    }
    residual
}
