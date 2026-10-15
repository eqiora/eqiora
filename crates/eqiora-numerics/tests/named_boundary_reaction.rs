use std::num::NonZeroUsize;

use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;

use eqiora_meshing::{
    MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh, simplex_centroid_rule,
    simplex_duffy_gauss_legendre,
};
use eqiora_numerics::fluid::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d, SimplicialMiniStokesSolution2d,
    solve_simplicial_mini_stokes_2d,
};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

const LEFT: &str = "left";
const RIGHT: &str = "right";

#[test]
fn named_surface_partition_sums_to_total_reaction() {
    let mesh = rectangle_triangles(2, 2);
    let solution = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT, RIGHT]));
    let total = solution.boundary_reaction().unwrap();
    let left = solution.named_boundary_reaction(LEFT).unwrap().unwrap();
    let right = solution.named_boundary_reaction(RIGHT).unwrap().unwrap();

    for component in 0..2 {
        let named_sum = left[component] + right[component];
        let scale = total[component]
            .abs()
            .max(left[component].abs() + right[component].abs())
            .max(1.0);
        let tolerance = 64.0 * f64::EPSILON * scale;
        assert!(
            (named_sum - total[component]).abs() <= tolerance,
            "component {component}: named sum {named_sum:e}, total {:e}, tolerance {tolerance:e}",
            total[component]
        );
    }
}

#[test]
fn overlapping_named_surfaces_are_rejected_before_solve() {
    let mesh = rectangle_triangles(2, 2);
    let left = side_facets(&mesh, 0, 0.0);
    let overlapping_vertex = mesh.entity_vertices(left[0]).unwrap()[0].index();
    let boundary = base_boundary(&mesh)
        .with_named_reaction_surface(&mesh, "cylinder", left.clone())
        .unwrap();
    let error = boundary
        .with_named_reaction_surface(&mesh, "duplicate-cylinder", [left[0]])
        .unwrap_err();

    assert!(
        error
            .message()
            .contains(&format!("vertex {overlapping_vertex}"))
    );
    assert!(error.message().contains("cylinder"));
    assert!(error.message().contains("duplicate-cylinder"));
}

#[test]
fn partial_named_surface_is_unaffected_by_naming_another_surface() {
    let mesh = rectangle_triangles(2, 2);
    let partial = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT]));
    let partitioned = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT, RIGHT]));
    let partial_left = partial.named_boundary_reaction(LEFT).unwrap().unwrap();
    let partitioned_left = partitioned.named_boundary_reaction(LEFT).unwrap().unwrap();

    assert_eq!(reaction_bits(partial_left), reaction_bits(partitioned_left));
    assert!(partial.named_boundary_reaction(RIGHT).unwrap().is_none());
    assert!(partial_left.iter().any(|value| value.abs() > 1.0e-12));
}

#[test]
fn naming_surfaces_does_not_change_total_reaction() {
    let mesh = rectangle_triangles(2, 2);
    let unnamed = solve(&mesh, base_boundary(&mesh));
    let named = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT, RIGHT]));

    assert_eq!(
        reaction_bits(unnamed.boundary_reaction().unwrap()),
        reaction_bits(named.boundary_reaction().unwrap())
    );
}

#[test]
fn named_surface_rejects_an_unconstrained_vertex() {
    let mesh = rectangle_triangles(2, 2);
    let top = side_facets(&mesh, 1, 2.0);
    let unconstrained_vertex = mesh
        .vertices()
        .iter()
        .position(|coordinates| coordinates == &[2.0, 2.0])
        .unwrap();
    let error = base_boundary(&mesh)
        .with_named_reaction_surface(&mesh, "free-top", top)
        .unwrap_err();

    assert!(error.message().contains("free-top"));
    assert!(
        error
            .message()
            .contains(&format!("unconstrained vertex {unconstrained_vertex}"))
    );
}

#[test]
fn named_surface_reactions_are_bit_deterministic() {
    let mesh = rectangle_triangles(2, 2);
    let first = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT, RIGHT]));
    let second = solve(&mesh, boundary_with_surfaces(&mesh, &[LEFT, RIGHT]));
    let names = [LEFT, RIGHT];
    let first_bits = names
        .iter()
        .flat_map(|name| reaction_bits(first.named_boundary_reaction(name).unwrap().unwrap()))
        .collect::<Vec<_>>();
    let second_bits = names
        .iter()
        .flat_map(|name| reaction_bits(second.named_boundary_reaction(name).unwrap().unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(first_bits, second_bits);
}

fn boundary_with_surfaces(mesh: &SimplicialMesh, names: &[&str]) -> SimplicialMiniStokesBoundary2d {
    let mut boundary = base_boundary(mesh);
    for name in names {
        let facets = match *name {
            LEFT => side_facets(mesh, 0, 0.0),
            RIGHT => side_facets(mesh, 0, 4.0),
            unexpected => panic!("unknown test surface {unexpected}"),
        };
        boundary = boundary
            .with_named_reaction_surface(mesh, *name, facets)
            .unwrap();
    }
    boundary
}

fn base_boundary(mesh: &SimplicialMesh) -> SimplicialMiniStokesBoundary2d {
    let facet_count = mesh.entity_count(1).unwrap();
    SimplicialMiniStokesBoundary2d::new(
        mesh,
        (0..facet_count).filter_map(|index| {
            let facet = MeshEntity::new(1, index);
            mesh.is_boundary_entity(facet).unwrap().then(|| {
                let vertices = mesh.entity_vertices(facet).unwrap();
                let essential = vertices.iter().all(|vertex| {
                    let x = mesh.vertices()[vertex.index()][0];
                    x == 0.0 || x == 4.0
                });
                let condition = if essential {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                } else {
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction { value: [0.0, 0.0] }
                };
                SimplicialMiniStokesBoundaryFacet2d::new(facet, condition)
            })
        }),
    )
    .unwrap()
}

fn side_facets(mesh: &SimplicialMesh, axis: usize, coordinate: f64) -> Vec<MeshEntity> {
    let facet_count = mesh.entity_count(1).unwrap();
    (0..facet_count)
        .filter_map(|index| {
            let facet = MeshEntity::new(1, index);
            mesh.is_boundary_entity(facet)
                .unwrap()
                .then(|| mesh.entity_vertices(facet).unwrap())
                .filter(|vertices| {
                    vertices
                        .iter()
                        .all(|vertex| mesh.vertices()[vertex.index()][axis] == coordinate)
                })
                .map(|_| facet)
        })
        .collect()
}

fn solve(
    mesh: &SimplicialMesh,
    boundary: SimplicialMiniStokesBoundary2d,
) -> SimplicialMiniStokesSolution2d {
    solve_simplicial_mini_stokes_2d(
        mesh,
        1.0,
        &|_| Ok([1.0, 0.25]),
        &boundary,
        &|_| Ok([0.0, 0.0]),
        &simplex_duffy_gauss_legendre(2, 3).unwrap(),
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        reference_solver(),
    )
    .unwrap()
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

fn reaction_bits(reaction: [f64; 2]) -> [u64; 2] {
    reaction.map(f64::to_bits)
}
