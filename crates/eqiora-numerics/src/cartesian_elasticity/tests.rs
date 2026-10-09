use std::num::NonZeroUsize;

use eqiora_assembly::{AssemblyMap, CooAssembler, DofId, LocalContribution, LocalUnknown};
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

use super::*;
use crate::canonical::lower_scalar_elliptic_cartesian;

fn unit_cell_action(mu: f64, lambda: f64) -> (CartesianMesh, LocalLinearActionIr) {
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[1, 1]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(DIMENSION, 2).unwrap();
    let action =
        lower_cartesian_q1_linear_elasticity_local_action_2d(&mesh, mu, lambda, &quadrature)
            .unwrap();
    (mesh, action)
}

fn nodal_vector(mesh: &CartesianMesh, value: impl Fn(&[f64]) -> [f64; COMPONENTS]) -> Vec<f64> {
    (0..mesh.entity_count(0).unwrap())
        .flat_map(|vertex| value(&mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap()))
        .collect()
}

fn cell_nodal_vector(
    mesh: &CartesianMesh,
    value: impl Fn(&[f64]) -> [f64; COMPONENTS],
) -> Vec<f64> {
    mesh.entity_vertices(MeshEntity::new(DIMENSION, 0))
        .unwrap()
        .into_iter()
        .flat_map(|vertex| value(&mesh.vertex_coordinates(vertex).unwrap()))
        .collect()
}

fn energy(action: &LocalLinearActionIr, values: &[f64]) -> f64 {
    let mut applied = vec![0.0; action.output_len()];
    action.apply_reference(values, &mut applied).unwrap();
    0.5 * values
        .iter()
        .zip(applied)
        .map(|(left, right)| left * right)
        .sum::<f64>()
}

#[test]
fn every_essential_side_uses_exact_cartesian_vertex_topology() {
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 3]).unwrap();
    for axis in 0..DIMENSION {
        for side in 0..2 {
            let mut sides = [[false; 2]; DIMENSION];
            sides[axis][side] = true;
            let essential = CartesianEssentialSides2d::new(sides);
            let boundary_index = if side == 0 {
                0
            } else {
                mesh.axis_cell_count(axis).unwrap()
            };
            for vertex_index in 0..mesh.entity_count(0).unwrap() {
                let vertex = MeshEntity::new(0, vertex_index);
                let index = mesh.vertex_multi_index(vertex).unwrap();
                assert_eq!(
                    essential.constrains_vertex(&mesh, vertex),
                    index[axis] == boundary_index,
                    "axis {axis}, side {side}, vertex {index:?}",
                );
            }
        }
    }
}

#[test]
fn local_stiffness_is_symmetric_and_contains_cross_component_coupling() {
    let (_, action) = unit_cell_action(2.0, 3.0);
    let width = action.rows();
    let matrix = action.coefficients();
    for row in 0..width {
        for column in 0..width {
            assert!((matrix[row * width + column] - matrix[column * width + row]).abs() < 2.0e-15);
        }
    }
    assert!(
        (0..width).any(|row| {
            (0..width).any(|column| {
                row % COMPONENTS != column % COMPONENTS
                    && matrix[row * width + column].abs() > 1.0e-12
            })
        }),
        "vector elasticity must not degenerate into uncoupled scalar diffusion",
    );
}

#[test]
fn rigid_translation_and_infinitesimal_rotation_have_zero_energy() {
    let (mesh, action) = unit_cell_action(2.0, 3.0);
    let translation = cell_nodal_vector(&mesh, |_| [2.5, -4.0]);
    let rotation = cell_nodal_vector(&mesh, |point| [-point[1], point[0]]);
    assert!(energy(&action, &translation).abs() < 2.0e-14);
    assert!(energy(&action, &rotation).abs() < 2.0e-14);
}

#[test]
fn pure_shear_and_dilatation_match_analytical_energy() {
    let mu = 2.0;
    let lambda = 3.0;
    let (mesh, action) = unit_cell_action(mu, lambda);
    let pure_shear = cell_nodal_vector(&mesh, |point| [point[1], point[0]]);
    let dilatation = cell_nodal_vector(&mesh, |point| [point[0], point[1]]);
    assert!((energy(&action, &pure_shear) - 2.0 * mu).abs() < 2.0e-14);
    assert!((energy(&action, &dilatation) - 2.0 * (mu + lambda)).abs() < 4.0e-14);
}

#[test]
fn two_by_two_affine_patch_has_exact_center_and_boundary_equilibrium() {
    let lambda = 2.0;
    let mu = 3.0;
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 2]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(DIMENSION, 2).unwrap();
    let action =
        lower_cartesian_q1_linear_elasticity_local_action_2d(&mesh, mu, lambda, &quadrature)
            .unwrap();
    let vertex_count = mesh.entity_count(0).unwrap();
    let mut assembler = CooAssembler::new(vertex_count * COMPONENTS).unwrap();
    let local_width = action.rows();
    for cell_index in 0..mesh.entity_count(DIMENSION).unwrap() {
        let offset = cell_index * local_width * local_width;
        let local = LocalContribution::new(
            local_width,
            local_width,
            action.coefficients()[offset..offset + local_width * local_width].to_vec(),
            vec![0.0; local_width],
        )
        .unwrap();
        let vertices = mesh
            .entity_vertices(MeshEntity::new(DIMENSION, cell_index))
            .unwrap();
        let global = local_global_dofs(&vertices).unwrap();
        let map = AssemblyMap::new(
            global
                .iter()
                .map(|index| Some(DofId::new(*index)))
                .collect(),
            global
                .iter()
                .map(|index| LocalUnknown::Free(DofId::new(*index)))
                .collect(),
        )
        .unwrap();
        assembler.scatter(&map, &local).unwrap();
    }
    let system = assembler.finish().unwrap();
    let displacement = nodal_vector(&mesh, |point| {
        [
            2.0 * point[0] + 3.0 * point[1] + 1.0,
            5.0 * point[0] + 7.0 * point[1] - 2.0,
        ]
    });
    let reactions = system.matrix().multiply(&displacement).unwrap();
    let mut resultant = [0.0; COMPONENTS];
    let mut moment = 0.0;

    for vertex in 0..vertex_count {
        let entity = MeshEntity::new(0, vertex);
        let point = mesh.vertex_coordinates(entity).unwrap();
        let index = mesh.vertex_multi_index(entity).unwrap();
        let values = &displacement[vertex * COMPONENTS..(vertex + 1) * COMPONENTS];
        let reaction = &reactions[vertex * COMPONENTS..(vertex + 1) * COMPONENTS];
        if index == [1, 1] {
            assert!((values[0] - 3.5).abs() < 2.0e-15);
            assert!((values[1] - 4.0).abs() < 2.0e-15);
            assert!(reaction.iter().all(|value| value.abs() < 2.0e-14));
            continue;
        }

        let expected = match (index[0], index[1]) {
            (0, 0) => [-13.5, -21.0],
            (1, 0) => [-12.0, -30.0],
            (2, 0) => [1.5, -9.0],
            (0, 1) => [-15.0, -12.0],
            (2, 1) => [15.0, 12.0],
            (0, 2) => [-1.5, 9.0],
            (1, 2) => [12.0, 30.0],
            (2, 2) => [13.5, 21.0],
            _ => panic!("unexpected two-by-two patch vertex {index:?}"),
        };
        for component in 0..COMPONENTS {
            assert!((reaction[component] - expected[component]).abs() < 3.0e-14);
            resultant[component] += reaction[component];
        }
        moment += point[0] * reaction[1] - point[1] * reaction[0];
    }
    assert!(resultant.iter().all(|value| value.abs() < 6.0e-14));
    assert!(moment.abs() < 6.0e-14);
}

#[test]
fn vector_field_error_uses_value_and_frobenius_gradient_contracts() {
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 3]).unwrap();
    let values = nodal_vector(&mesh, |point| [-point[1], point[0]]);
    let field = CartesianQ1VectorField2d::new(mesh, values).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(DIMENSION, 2).unwrap();
    let error = field
        .error_norms(
            &|point| ([-point[1], point[0]], [[0.0, -1.0], [1.0, 0.0]]),
            &quadrature,
        )
        .unwrap();
    assert!(error.l2() < 2.0e-15);
    assert!(error.h1_seminorm() < 3.0e-15);
    assert!(error.h1() < 4.0e-15);
}

#[test]
fn canonical_potential_jvp_drives_load_and_full_reaction_balance() {
    let source = r#"
model potential_probe() {
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);

  variable probe: m ^ 3 on body in h1;
  relation balance on body {
-div(grad(probe)) - (coordinate(0) + 2 * coordinate(1)) = 0;
  }
  relation x_lower_value on x_lower { trace(probe) = 0; }
  relation x_upper_value on x_upper { trace(probe) = 0; }
  relation y_lower_value on y_lower { trace(probe) = 0; }
  relation y_upper_value on y_upper { trace(probe) = 0; }
}
"#;
    let mut compiled = compile("potential-probe.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let lowered = lower_scalar_elliptic_cartesian(&program).unwrap();

    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 2]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(DIMENSION, 2).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-13,
        1.0e-14,
        NonZeroUsize::new(64).unwrap(),
    )
    .unwrap();
    let solution = solve_cartesian_q1_linear_elasticity_2d(
        &mesh,
        2.0,
        3.0,
        lowered.source(),
        &quadrature,
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
    )
    .unwrap();
    for (actual, exact) in solution.integrated_body_force().into_iter().zip([1.0, 2.0]) {
        assert!((actual - exact).abs() < 2.0e-14);
    }
    for component in 0..COMPONENTS {
        assert!(
            (solution.boundary_reaction()[component] + solution.integrated_body_force()[component])
                .abs()
                < 2.0e-12
        );
    }
    assert_eq!(solution.assembly_report().target_count(), 2);
}
