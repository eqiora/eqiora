use std::f64::consts::PI;
use std::num::NonZeroUsize;

use eqiora::compiler::compile;
use eqiora::differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_sensitivity,
};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::kernel::BoundarySide;
use eqiora::meshing::{MeshQualityGate, QuadratureRule, SimplicialMesh, simplex_centroid_rule};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use eqiora::{Id, entity::kinds};
use eqiora_numerics::{
    ale::SimplicialMeshVelocity, common::SpatialDesignCoordinate,
    scalar::ScalarEllipticSimplicialFemSolution,
    scalar::linearize_scalar_elliptic_simplicial_compliance,
    scalar::linearize_scalar_elliptic_simplicial_fem, scalar::lower_scalar_elliptic_cartesian,
    scalar::solve_scalar_elliptic_simplicial_fem,
};

const SOURCE: &str = include_str!(
    "../../../verify/differentiation/unstructured-shape-compliance/models/poisson.eqi"
);
const DOMAIN_DECLARATION: &str = "domain rectangle = box(0, 1, 0, 1);";
const CELLS_PER_AXIS: usize = 8;

#[test]
fn unstructured_simplex_shape_action_matches_rebuild_and_continuous_compliance() {
    let bounds = [1.15, 0.85];
    let (program, domain) = compile_program(bounds);
    let model = lower_scalar_elliptic_cartesian(&program).unwrap();
    let mesh = distorted_triangular_mesh(bounds, CELLS_PER_AXIS).unwrap();
    let quadrature = simplex_centroid_rule(2).unwrap();
    let solution = solve(&model, &mesh, &quadrature);
    assert!((solution.boundary_reaction_sum() + solution.integrated_source()).abs() < 2.0e-11);

    let selected = shape_coordinates(domain);
    let velocities = selected
        .iter()
        .map(|&coordinate| {
            SimplicialMeshVelocity::normalized_box_bound(&mesh, coordinate, model.bounds()).unwrap()
        })
        .collect::<Vec<_>>();
    let relation = linearize_scalar_elliptic_simplicial_fem(
        &model,
        &mesh,
        &solution,
        &quadrature,
        &selected,
        &velocities,
    )
    .unwrap();
    let objective = linearize_scalar_elliptic_simplicial_compliance(
        &model,
        &mesh,
        &solution,
        &quadrature,
        &selected,
        &velocities,
    )
    .unwrap();
    assert_eq!(relation.design_coordinates(), &selected);
    assert_eq!(relation.design_values(), &bounds);
    assert_relative_close(
        objective.value(),
        independent_compliance(&mesh, &solution),
        2.0e-14,
        2.0e-14,
    );

    let accepted = AcceptedOutputLinearization::new(&relation, &objective, 2.0e-10).unwrap();
    let direction = [0.3, -0.2];
    let forward = forward_sensitivity(
        accepted.relation(),
        &direction,
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver_request(),
    )
    .unwrap();
    let step = 1.0e-5;
    let plus_bounds = [
        bounds[0] + step * direction[0],
        bounds[1] + step * direction[1],
    ];
    let minus_bounds = [
        bounds[0] - step * direction[0],
        bounds[1] - step * direction[1],
    ];
    let (_, plus_values, _) = solved_revision(plus_bounds, CELLS_PER_AXIS);
    let (_, minus_values, _) = solved_revision(minus_bounds, CELLS_PER_AXIS);
    for ((computed, plus), minus) in forward.values().iter().zip(&plus_values).zip(&minus_values) {
        assert_relative_close(*computed, (plus - minus) / (2.0 * step), 1.2e-5, 2.0e-9);
    }

    let adjoint = adjoint_output_gradient(
        &accepted,
        &[1.0],
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver_request(),
    )
    .unwrap();
    for coordinate in 0..2 {
        let mut plus = bounds;
        let mut minus = bounds;
        plus[coordinate] += step;
        minus[coordinate] -= step;
        let expected = (solved_revision(plus, CELLS_PER_AXIS).2
            - solved_revision(minus, CELLS_PER_AXIS).2)
            / (2.0 * step);
        assert_relative_close(adjoint.gradient()[coordinate], expected, 1.2e-5, 2.0e-9);
    }

    let continuous = continuous_rectangle_compliance(bounds, 121);
    let errors = [4, 8, 16].map(|cells| (solved_revision(bounds, cells).2 - continuous).abs());
    assert!(
        errors[1] < 0.35 * errors[0],
        "compliance errors: {errors:?}"
    );
    assert!(
        errors[2] < 0.35 * errors[1],
        "compliance errors: {errors:?}"
    );

    let collapsed = distorted_triangular_mesh([1.0e-4, 1.0], CELLS_PER_AXIS).unwrap_err();
    assert_eq!(collapsed.code(), eqiora::diagnostic::codes::INVALID_MESH);
}

fn solved_revision(bounds: [f64; 2], cells: usize) -> (SimplicialMesh, Vec<f64>, f64) {
    let (program, _) = compile_program(bounds);
    let model = lower_scalar_elliptic_cartesian(&program).unwrap();
    let mesh = distorted_triangular_mesh(bounds, cells).unwrap();
    let quadrature = simplex_centroid_rule(2).unwrap();
    let solution = solve(&model, &mesh, &quadrature);
    let objective = independent_compliance(&mesh, &solution);
    (mesh, solution.algebraic_values().to_vec(), objective)
}

fn solve(
    model: &eqiora_numerics::scalar::ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
) -> ScalarEllipticSimplicialFemSolution {
    solve_scalar_elliptic_simplicial_fem(model, mesh, quadrature, solver_request()).unwrap()
}

fn compile_program(bounds: [f64; 2]) -> (KernelProgram, Id<kinds::Domain>) {
    let declaration = format!(
        "domain rectangle = box(0, {:.17}, 0, {:.17});",
        bounds[0], bounds[1]
    );
    let source = SOURCE.replacen(DOMAIN_DECLARATION, &declaration, 1);
    assert_ne!(source, SOURCE, "shape Domain declaration was not replaced");
    let mut compiled = compile("unstructured-shape-compliance.eqi", &source).unwrap();
    let domain = compiled[0]
        .symbols()
        .get("rectangle")
        .unwrap()
        .downcast::<kinds::Domain>()
        .unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        domain,
    )
}

fn shape_coordinates(domain: Id<kinds::Domain>) -> [SpatialDesignCoordinate; 2] {
    [
        SpatialDesignCoordinate::CartesianBound {
            domain,
            axis: 0,
            side: BoundarySide::Upper,
        },
        SpatialDesignCoordinate::CartesianBound {
            domain,
            axis: 1,
            side: BoundarySide::Upper,
        },
    ]
}

fn distorted_triangular_mesh(
    bounds: [f64; 2],
    cells: usize,
) -> Result<SimplicialMesh, eqiora::Diagnostic> {
    let mut vertices = Vec::with_capacity((cells + 1).pow(2));
    for row in 0..=cells {
        for column in 0..=cells {
            let mut x = column as f64 / cells as f64;
            let mut y = row as f64 / cells as f64;
            if column > 0 && column < cells && row > 0 && row < cells {
                let amplitude = 0.13 / cells as f64;
                x += amplitude * (PI * x).sin() * (2.0 * PI * y).sin();
                y += amplitude * (2.0 * PI * x).sin() * (PI * y).sin();
            }
            vertices.push(vec![bounds[0] * x, bounds[1] * y]);
        }
    }
    let vertex = |row: usize, column: usize| row * (cells + 1) + column;
    let mut triangles = Vec::with_capacity(2 * cells * cells);
    for row in 0..cells {
        for column in 0..cells {
            let lower_left = vertex(row, column);
            let lower_right = vertex(row, column + 1);
            let upper_left = vertex(row + 1, column);
            let upper_right = vertex(row + 1, column + 1);
            let pair = if (row + column) % 2 == 0 {
                [
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                ]
            } else {
                [
                    [lower_left, lower_right, upper_left],
                    [lower_right, upper_right, upper_left],
                ]
            };
            for (local, triangle) in pair.into_iter().enumerate() {
                let [a, b, c] = triangle;
                triangles.push(if (row + column + local) % 3 == 0 {
                    vec![b, c, a]
                } else {
                    vec![a, b, c]
                });
            }
        }
    }
    SimplicialMesh::new(2, vertices, triangles, MeshQualityGate::new(0.15).unwrap())
}

fn independent_compliance(
    mesh: &SimplicialMesh,
    solution: &ScalarEllipticSimplicialFemSolution,
) -> f64 {
    mesh.cells()
        .iter()
        .map(|cell| {
            let a = &mesh.vertices()[cell[0]];
            let b = &mesh.vertices()[cell[1]];
            let c = &mesh.vertices()[cell[2]];
            let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
            let average = cell
                .iter()
                .map(|&vertex| solution.field().vertex_values()[vertex])
                .sum::<f64>()
                / 3.0;
            area * average
        })
        .sum()
}

fn continuous_rectangle_compliance(bounds: [f64; 2], maximum_odd: usize) -> f64 {
    let [length_x, length_y] = bounds;
    let mut sum = 0.0;
    for m in (1..=maximum_odd).step_by(2) {
        for n in (1..=maximum_odd).step_by(2) {
            let m = m as f64;
            let n = n as f64;
            sum += 64.0 * length_x * length_y
                / (m.powi(2)
                    * n.powi(2)
                    * PI.powi(6)
                    * (m.powi(2) / length_x.powi(2) + n.powi(2) / length_y.powi(2)));
        }
    }
    sum
}

fn solver_request() -> LinearSolveRequest<'static> {
    LinearSolveRequest::new(
        &REFERENCE_LINEAR_SOLVER,
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(8192).unwrap(),
        )
        .unwrap(),
    )
}

fn assert_relative_close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    let tolerance = absolute + relative * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}",
    );
}
