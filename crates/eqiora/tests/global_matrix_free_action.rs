use std::num::NonZeroUsize;

use eqiora::assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyTarget, CsrMatrix, DofId,
    IndexedAssemblyWork, LocalContribution, LocalUnknown, PacketLinearSystem,
    REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora::meshing::{MeshEntity, MeshTopology, QuadratureRule};
use eqiora::solver::{
    DiagonalAvailability, LinearOperator, LinearOperatorProperties, LinearProblem,
    LinearSolveRequest, LinearSolver, OrientedLinearOperator, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use eqiora_meshing::CartesianMesh;
use eqiora_numerics::scalar::lower_cartesian_q1_diffusion_local_action;

fn maximum_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn apply_transpose(matrix: &CsrMatrix<f64>, input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; matrix.columns()];
    matrix
        .apply_oriented(
            eqiora_solver::LinearOperatorOrientation::Transposed,
            input,
            &mut output,
        )
        .unwrap();
    output
}

#[test]
fn nonsymmetric_packet_projection_matches_a_hand_calculated_oracle() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(3).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let packets = [
        AssemblyPacket::new(
            LocalContribution::new(
                3,
                4,
                vec![
                    2.0, 3.0, 5.0, 7.0, 13.0, 17.0, 19.0, 23.0, 31.0, 37.0, 41.0, 43.0,
                ],
                vec![11.0, 29.0, 47.0],
            )
            .unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(2)), None, Some(DofId::new(2))],
                    vec![
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Fixed(2.0),
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Free(DofId::new(0)),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap(),
        AssemblyPacket::new(
            LocalContribution::new(
                2,
                3,
                vec![53.0, 59.0, 61.0, 67.0, 71.0, 73.0],
                vec![79.0, 83.0],
            )
            .unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0)), Some(DofId::new(1))],
                    vec![
                        LocalUnknown::Free(DofId::new(2)),
                        LocalUnknown::Free(DofId::new(0)),
                        LocalUnknown::Fixed(-1.5),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap(),
        AssemblyPacket::new(
            LocalContribution::new(1, 2, vec![2.0, -3.0], vec![5.0]).unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(2))],
                    vec![
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Free(DofId::new(0)),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap(),
    ];
    let work = IndexedAssemblyWork::new(packets.len(), |index: usize| Ok(packets[index].clone()));
    let matrix_free = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let assembled = assembled.system(target).unwrap();

    let input = [2.0, -3.0, 5.0];
    let mut output = [f64::INFINITY; 3];
    matrix_free.operator().apply(&input, &mut output).unwrap();
    assert_eq!(output, [383.0, 477.0, -149.0]);
    assert_eq!(
        assembled.matrix().multiply(&input).unwrap(),
        [383.0, 477.0, -149.0]
    );

    let cotangent = [7.0, 11.0, -13.0];
    let mut transpose = [f64::INFINITY; 3];
    matrix_free
        .operator()
        .apply_oriented(
            eqiora_solver::LinearOperatorOrientation::Transposed,
            &cotangent,
            &mut transpose,
        )
        .unwrap();
    assert_eq!(transpose, [583.0, -1053.0, 1108.0]);
    assert_eq!(
        apply_transpose(assembled.matrix(), &cotangent),
        [583.0, -1053.0, 1108.0]
    );
    assert_eq!(matrix_free.right_hand_side(), &[170.5, 192.5, -17.0]);
    assert_eq!(assembled.rhs(), matrix_free.right_hand_side());
    assert_eq!(
        output
            .iter()
            .zip(cotangent)
            .map(|(left, right)| left * right)
            .sum::<f64>(),
        input
            .iter()
            .zip(transpose)
            .map(|(left, right)| left * right)
            .sum::<f64>()
    );
}

#[test]
fn packet_system_construction_rejects_incomplete_or_invalid_projection() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let empty = IndexedAssemblyWork::new(
        0,
        |_: usize| -> Result<AssemblyPacket<f64>, eqiora::Diagnostic> { unreachable!() },
    );
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &empty)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let local = LocalContribution::new(2, 1, vec![1.0, 2.0], vec![0.0, 0.0]).unwrap();
    let wrong_shape = AssemblyPacket::new(
        local,
        vec![TargetAssemblyMap::new(
            target,
            AssemblyMap::new(
                vec![Some(DofId::new(0))],
                vec![LocalUnknown::Free(DofId::new(0))],
            )
            .unwrap(),
        )],
    )
    .unwrap_err();
    assert_eq!(
        wrong_shape.code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let out_of_range = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(1))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &out_of_range)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let out_of_range_column = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(1))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &out_of_range_column)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let incomplete_plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let incomplete_target = incomplete_plan.target_id(0).unwrap();
    let missing_row = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                incomplete_target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&incomplete_plan, incomplete_target, &missing_row)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let cancellation = IndexedAssemblyWork::new(2, |index| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![if index == 0 { 1.0 } else { -1.0 }], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &cancellation)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let rhs_overflow = IndexedAssemblyWork::new(2, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![f64::MAX])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &rhs_overflow)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );

    let operator_overflow = IndexedAssemblyWork::new(2, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![f64::MAX], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    assert_eq!(
        PacketLinearSystem::from_work(&plan, target, &operator_overflow)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );
}

#[test]
fn packet_action_rejects_invalid_or_nonfinite_buffers() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let work = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![2.0], vec![3.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    let system = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
    let operator = system.operator();

    for result in [
        operator.apply(&[], &mut [0.0]),
        operator.apply(&[1.0], &mut []),
        operator.apply(&[f64::NAN], &mut [0.0]),
        operator.apply_oriented(
            eqiora_solver::LinearOperatorOrientation::Transposed,
            &[],
            &mut [0.0],
        ),
        operator.apply_oriented(
            eqiora_solver::LinearOperatorOrientation::Transposed,
            &[f64::INFINITY],
            &mut [0.0],
        ),
        operator.diagonal(&mut []).map(|_| ()),
    ] {
        assert_eq!(
            result.unwrap_err().code(),
            eqiora::diagnostic::codes::NUMERICAL_SOLVE_FAILED
        );
    }
    let row_action = operator.row_action().unwrap();
    let reversed_start = 1;
    let reversed_end = 0;
    assert_eq!(
        row_action
            .apply_rows(reversed_start..reversed_end, &[1.0], &mut [])
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::NUMERICAL_SOLVE_FAILED
    );
}

#[test]
fn cartesian_q1_packet_action_matches_csr_and_solves_in_one_through_three_dimensions() {
    for dimension in 1..=3 {
        let bounds = vec![[-0.75, 1.25]; dimension];
        let cells_per_axis = vec![3; dimension];
        let mesh = CartesianMesh::uniform(&bounds, &cells_per_axis).unwrap();
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2).unwrap();
        let local_action =
            lower_cartesian_q1_diffusion_local_action(&mesh, &|_: &[f64]| 1.25, &quadrature)
                .unwrap();
        let vertex_count = mesh.entity_count(0).unwrap();
        let cell_count = mesh.entity_count(dimension).unwrap();
        let width = local_action.columns();

        let boundary_value = |coordinates: &[f64]| {
            1.0 + coordinates
                .iter()
                .enumerate()
                .map(|(axis, coordinate)| (axis as f64 + 1.0) * 0.125 * coordinate)
                .sum::<f64>()
        };
        let mut fixed_values = vec![None; vertex_count];
        let mut free_indices = vec![None; vertex_count];
        let mut free_count = 0;
        for vertex_index in 0..vertex_count {
            let vertex = MeshEntity::new(0, vertex_index);
            if mesh.is_boundary_entity(vertex).unwrap() {
                fixed_values[vertex_index] =
                    Some(boundary_value(&mesh.vertex_coordinates(vertex).unwrap()));
            } else {
                free_indices[vertex_index] = Some(DofId::new(free_count));
                free_count += 1;
            }
        }

        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(free_count).unwrap()]).unwrap();
        let target = plan.target_id(0).unwrap();
        let work = IndexedAssemblyWork::new(cell_count, |cell_index| {
            let coefficient_offset = cell_index * width * width;
            let local = LocalContribution::new(
                width,
                width,
                local_action.coefficients()[coefficient_offset..coefficient_offset + width * width]
                    .to_vec(),
                vec![0.0; width],
            )?;
            let vertices = mesh
                .entity_vertices(MeshEntity::new(dimension, cell_index))
                .expect("Cartesian cell has a vertex closure");
            let map = AssemblyMap::new(
                vertices
                    .iter()
                    .map(|vertex| free_indices[vertex.index()])
                    .collect(),
                vertices
                    .iter()
                    .map(|vertex| {
                        fixed_values[vertex.index()].map_or_else(
                            || {
                                LocalUnknown::Free(
                                    free_indices[vertex.index()]
                                        .expect("unfixed vertex owns an equation"),
                                )
                            },
                            LocalUnknown::Fixed,
                        )
                    })
                    .collect(),
            )?;
            AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, map)])
        });

        let matrix_free = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
        let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        let assembled = assembled.system(target).unwrap();
        assert_eq!(matrix_free.packet_count(), cell_count);
        assert_eq!(matrix_free.right_hand_side(), assembled.rhs());

        let input = (0..free_count)
            .map(|index| (index as f64 + 0.25) / 7.0)
            .collect::<Vec<_>>();
        let mut matrix_free_output = vec![f64::INFINITY; free_count];
        matrix_free
            .operator()
            .apply(&input, &mut matrix_free_output)
            .unwrap();
        let assembled_output = assembled.matrix().multiply(&input).unwrap();
        assert!(
            maximum_difference(&matrix_free_output, &assembled_output) <= 3.0e-13,
            "dimension {dimension} forward action differs from assembled CSR"
        );

        let mut packed_input = Vec::with_capacity(local_action.input_len());
        for cell_index in 0..cell_count {
            let vertices = mesh
                .entity_vertices(MeshEntity::new(dimension, cell_index))
                .unwrap();
            packed_input.extend(
                vertices.iter().map(|vertex| {
                    free_indices[vertex.index()].map_or(0.0, |dof| input[dof.index()])
                }),
            );
        }
        let mut packed_output = vec![f64::INFINITY; local_action.output_len()];
        local_action
            .apply_reference(&packed_input, &mut packed_output)
            .unwrap();
        let mut local_scatter = vec![0.0; free_count];
        for cell_index in 0..cell_count {
            let vertices = mesh
                .entity_vertices(MeshEntity::new(dimension, cell_index))
                .unwrap();
            for (local_row, vertex) in vertices.iter().enumerate() {
                if let Some(dof) = free_indices[vertex.index()] {
                    local_scatter[dof.index()] += packed_output[cell_index * width + local_row];
                }
            }
        }
        assert!(
            maximum_difference(&local_scatter, &matrix_free_output) <= 3.0e-13,
            "dimension {dimension} anonymous local evaluator differs from global packet action"
        );

        let split = free_count / 2;
        let row_action = matrix_free
            .operator()
            .row_action()
            .expect("packet operator exposes independent normal row ranges");
        let mut row_output = vec![f64::INFINITY; free_count];
        row_action
            .apply_rows(0..split, &input, &mut row_output[..split])
            .unwrap();
        row_action
            .apply_rows(split..free_count, &input, &mut row_output[split..])
            .unwrap();
        assert_eq!(row_output, matrix_free_output);

        let cotangent = (0..free_count)
            .map(|index| (index as f64 - 0.5) / 9.0)
            .collect::<Vec<_>>();
        let mut matrix_free_transpose = vec![f64::INFINITY; free_count];
        matrix_free
            .operator()
            .apply_oriented(
                eqiora_solver::LinearOperatorOrientation::Transposed,
                &cotangent,
                &mut matrix_free_transpose,
            )
            .unwrap();
        let assembled_transpose = apply_transpose(assembled.matrix(), &cotangent);
        assert!(
            maximum_difference(&matrix_free_transpose, &assembled_transpose) <= 3.0e-13,
            "dimension {dimension} transpose action differs from assembled CSR"
        );

        let mut diagonal = vec![f64::INFINITY; free_count];
        assert_eq!(
            matrix_free.operator().diagonal(&mut diagonal).unwrap(),
            DiagonalAvailability::Available
        );
        for (row, actual) in diagonal.iter().enumerate() {
            let expected = assembled.matrix().entry(row, row).unwrap();
            assert!((actual - expected).abs() <= 3.0e-13);
        }

        let solver_plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-13,
            1.0e-13,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity);
        let matrix_free_problem = LinearProblem::new(
            matrix_free.operator(),
            matrix_free.right_hand_side(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let assembled_problem = LinearProblem::new(
            assembled.matrix(),
            assembled.rhs(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver_plan);
        let matrix_free_solution = solver.solve(&matrix_free_problem).unwrap();
        let assembled_solution = solver.solve(&assembled_problem).unwrap();
        assert!(
            maximum_difference(matrix_free_solution.values(), assembled_solution.values())
                <= 5.0e-12,
            "dimension {dimension} matrix-free and assembled solves differ"
        );

        let mut oracle_image = vec![0.0; free_count];
        assembled
            .matrix()
            .apply(matrix_free_solution.values(), &mut oracle_image)
            .unwrap();
        let oracle_residual = oracle_image
            .iter()
            .zip(assembled.rhs())
            .map(|(applied, rhs)| (rhs - applied).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(oracle_residual <= 2.0e-12);

        for (vertex_index, equation) in free_indices.iter().enumerate() {
            let Some(equation) = equation else {
                continue;
            };
            let expected = boundary_value(
                &mesh
                    .vertex_coordinates(MeshEntity::new(0, vertex_index))
                    .unwrap(),
            );
            assert!((matrix_free_solution.values()[equation.index()] - expected).abs() <= 2.0e-12);
        }
    }
}
