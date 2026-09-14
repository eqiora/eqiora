use std::num::NonZeroUsize;

use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_compiler::{ModelSymbols, compile};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    LinearOperatorProperties, LinearProblem, LinearSolver, LinearSolverBackend,
    REFERENCE_LINEAR_SOLVER, SolverPlan,
};

use super::*;

const AXIS: [f64; 4] = [0.0, 0.25, 0.75, 1.0];

fn boundaries(symbols: &ModelSymbols) -> BTreeMap<(usize, BoundarySide), RawId> {
    BTreeMap::from([
        ((0, BoundarySide::Lower), symbols.get("left").unwrap()),
        ((0, BoundarySide::Upper), symbols.get("right").unwrap()),
    ])
}

fn authored(reaction: &[Vec<f64>], reverse: bool) -> (String, Vec<String>) {
    let names = (0..reaction.len())
        .map(|index| {
            if reverse {
                format!("renamed_{index}")
            } else {
                format!("field_{index}")
            }
        })
        .collect::<Vec<_>>();
    let mut order = (0..reaction.len()).collect::<Vec<_>>();
    if reverse {
        order.reverse();
    }
    let mut source = String::from(
        "model Coupled() { domain body = box(0, 1); domain left = boundary(body, axis = 0, side = lower); domain right = boundary(body, axis = 0, side = upper); parameter inverse_area: 1 / m ^ 2 = 1;\n",
    );
    for &row in &order {
        source += &format!("variable {}: 1 on body;\n", names[row]);
    }
    for &row in &order {
        source += &format!(
            "relation balance_{row} on body {{ -div({} * grad({}))",
            row + 2,
            names[row]
        );
        for (column, entry) in reaction[row].iter().enumerate() {
            if *entry != 0.0 {
                source += &format!(" + ({entry}) * inverse_area * {}", names[column]);
            }
        }
        source += &format!(" - {} * inverse_area = 0; }}\n", row + 1);
        for side in ["left", "right"] {
            source += &format!(
                "relation {side}_{row} on {side} {{ trace({}) = 0; }}\n",
                names[row]
            );
        }
    }
    source += "}";
    (source, names)
}

fn compiled(source: &str) -> (CompiledLinearBlockForm, ModelSymbols) {
    let (transaction, model, symbols) = compile("assembly.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledLinearBlockForm::derive(
        &program,
        symbols.get("body").unwrap(),
        1,
        &std::collections::BTreeSet::new(),
    )
    .unwrap();
    (form, symbols)
}

/// Assemble the analytic interval integrals in authored Field order. This does
/// not consume production basis tabulation, element matrices or mesh incidence.
fn expected(reaction: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let count = reaction.len() * AXIS.len();
    let mut matrix = vec![0.0; count * count];
    let mut rhs = vec![0.0; count];
    for cell in 0..AXIS.len() - 1 {
        let h = AXIS[cell + 1] - AXIS[cell];
        for (row, entries) in reaction.iter().enumerate() {
            for test in 0..2 {
                let global_test = row * AXIS.len() + cell + test;
                rhs[global_test] += (row + 1) as f64 * h / 2.0;
                for (column, entry) in entries.iter().enumerate() {
                    for trial in 0..2 {
                        let stiffness = if row == column {
                            (row + 2) as f64 / h * if test == trial { 1.0 } else { -1.0 }
                        } else {
                            0.0
                        };
                        let mass = entry * h / 6.0 * if test == trial { 2.0 } else { 1.0 };
                        matrix[global_test * count + column * AXIS.len() + cell + trial] +=
                            stiffness + mass;
                    }
                }
            }
        }
    }
    (matrix, rhs)
}

fn dense(system: &LinearSystem) -> Vec<f64> {
    let matrix = system.matrix();
    let count = matrix.rows();
    let mut result = vec![0.0; count * count];
    for row in 0..count {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            result[row * count + matrix.column_indices()[entry]] = matrix.values()[entry];
        }
    }
    result
}

fn action(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    matrix
        .chunks_exact(vector.len())
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "entry {index}: {actual} != {expected}"
        );
    }
}

fn exercise(reaction: &[Vec<f64>], reverse: bool) {
    let (source, names) = authored(reaction, reverse);
    exercise_source(reaction, &source, &names);
}

fn exercise_source(reaction: &[Vec<f64>], source: &str, names: &[String]) {
    let (form, symbols) = compiled(source);
    let mesh = CartesianMesh::from_axes(vec![AXIS.to_vec()]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let assembled = CartesianLinearAssembly::assemble(
        &form,
        &mesh,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &boundaries(&symbols),
    )
    .unwrap();
    assert_eq!(assembled.fields, form.fields());
    let ids = names
        .iter()
        .map(|name| symbols.get(name).unwrap())
        .collect::<Vec<_>>();
    let field_order = assembled
        .fields
        .iter()
        .map(|(field, _)| ids.iter().position(|id| id == field).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(field_order.len(), reaction.len());
    let globals = field_order
        .iter()
        .flat_map(|field| (0..AXIS.len()).map(move |vertex| field * AXIS.len() + vertex))
        .collect::<Vec<_>>();
    let count = globals.len();
    let (authored_matrix, authored_rhs) = expected(reaction);
    let matrix = globals
        .iter()
        .flat_map(|&row| globals.iter().map(move |&column| (row, column)))
        .map(|(row, column)| authored_matrix[row * count + column])
        .collect::<Vec<_>>();
    let rhs = globals
        .iter()
        .map(|&global| authored_rhs[global])
        .collect::<Vec<_>>();
    close(&dense(&assembled.full_system), &matrix, 2e-14);
    close(assembled.full_system.rhs(), &rhs, 2e-14);

    let free = (0..count)
        .filter(|global| matches!(global % AXIS.len(), 1 | 2))
        .collect::<Vec<_>>();
    assert_eq!(assembled.constraints.free_globals(), free);
    for global in 0..count {
        assert_eq!(
            assembled.constraints.is_free(global).unwrap(),
            free.contains(&global)
        );
    }
    let reduced = free
        .iter()
        .flat_map(|&row| free.iter().map(move |&column| (row, column)))
        .map(|(row, column)| matrix[row * count + column])
        .collect::<Vec<_>>();
    let reduced_rhs = free.iter().map(|&global| rhs[global]).collect::<Vec<_>>();
    close(&dense(&assembled.system), &reduced, 2e-14);
    close(assembled.system.rhs(), &reduced_rhs, 2e-14);

    let direction = (0..free.len())
        .map(|index| (index + 1) as f64 / 7.0)
        .collect::<Vec<_>>();
    let mut jv = vec![0.0; free.len()];
    assembled
        .system
        .matrix()
        .multiply_into(&direction, &mut jv)
        .unwrap();
    close(&jv, &action(&reduced, &direction), 3e-14);

    let problem = LinearProblem::new(
        assembled.system.matrix(),
        assembled.system.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
    close(&action(&reduced, solution.values()), &reduced_rhs, 2e-12);
    let lifted = assembled.constraints.lift(solution.values()).unwrap();
    for (global, value) in lifted.iter().enumerate() {
        if !free.contains(&global) {
            assert_eq!(*value, 0.0);
        }
    }
    let expected_residual = action(&matrix, &lifted)
        .iter()
        .zip(&rhs)
        .map(|(left, right)| left - right)
        .collect::<Vec<_>>();
    close(
        &assembled
            .constraints
            .full_residual(&assembled.full_system, &lifted)
            .unwrap(),
        &expected_residual,
        3e-14,
    );
    for global in free {
        assert!(expected_residual[global].abs() < 2e-12);
    }
}

#[test]
fn two_fields_preserve_nonsymmetric_coupling_and_exact_identity_permutation() {
    let reaction = [vec![4.0, -1.0], vec![2.0, 3.0]];
    exercise(&reaction, false);
    exercise(&reaction, true);
}

#[test]
fn fieldwise_nonzero_trace_and_natural_load_recover_linear_fields() {
    let (source, names) = authored(&[vec![0.0; 2], vec![0.0; 2]], false);
    let source = source
        .replace("- 1 * inverse_area", "- 0 * inverse_area")
        .replace("- 2 * inverse_area", "- 0 * inverse_area")
        .replace(
            "parameter inverse_area:",
            "parameter q0: 1 / m = 2; parameter q1: 1 / m = 9; parameter inverse_area:",
        )
        .replace(
            "relation left_0 on left { trace(field_0) = 0; }",
            "relation left_0 on left { trace(field_0) = 2; }",
        )
        .replace(
            "relation left_1 on left { trace(field_1) = 0; }",
            "relation left_1 on left { trace(field_1) = 4; }",
        )
        .replace(
            "relation right_0 on right { trace(field_0) = 0; }",
            "relation right_0 on right { normal(2 * grad(field_0)) = q0; }",
        )
        .replace(
            "relation right_1 on right { trace(field_1) = 0; }",
            "relation right_1 on right { normal(3 * grad(field_1)) = q1; }",
        );
    let (form, symbols) = compiled(&source);
    let first = symbols.get(&names[0]).unwrap();
    let mesh = CartesianMesh::from_axes(vec![AXIS.to_vec()]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let assembled = CartesianLinearAssembly::assemble(
        &form,
        &mesh,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &boundaries(&symbols),
    )
    .unwrap();
    let problem = LinearProblem::new(
        assembled.system.matrix(),
        assembled.system.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
    let values = assembled.constraints.lift(solution.values()).unwrap();
    let expected = form
        .fields()
        .iter()
        .flat_map(|(field, _)| {
            AXIS.map(|x| {
                if *field == first {
                    2.0 + x
                } else {
                    4.0 + 3.0 * x
                }
            })
        })
        .collect::<Vec<_>>();
    close(&values, &expected, 2e-11);
    assert_eq!(assembled.constraints.free_count(), 6);
}

#[test]
fn three_fields_preserve_two_way_and_one_way_couplings_after_constraints() {
    let reaction = [
        vec![4.0, -1.0, 0.0],
        vec![2.0, 5.0, 3.0],
        vec![0.0, 0.0, 6.0],
    ];
    exercise(&reaction, false);
    exercise(&reaction, true);
}

#[test]
fn heterogeneous_length_and_time_fields_preserve_dimensional_general_assembly() {
    let reaction = [vec![4.0, -1.0], vec![2.0, 3.0]];
    let (source, names) = authored(&reaction, false);
    // u has units m, v has units s. Their residuals have units 1/m and
    // s/m²; cross coefficients therefore have units 1/(m*s) and s/m³.
    // Numerical values are coherent SI, with unit inheritance only at these
    // explicitly typed Parameter declaration initializers.
    let source = source
        .replace("variable field_0: 1 on body", "variable field_0: m on body")
        .replace("variable field_1: 1 on body", "variable field_1: s on body")
        .replace("parameter inverse_area:", "parameter cross_01: 1 / m / s = -1; parameter cross_10: s / m ^ 3 = 2; parameter force_0: 1 / m = 1; parameter force_1: s / m ^ 2 = 2; parameter inverse_area:")
        .replace("(-1) * inverse_area * field_1", "cross_01 * field_1")
        .replace("(2) * inverse_area * field_0", "cross_10 * field_0")
        .replace("- 1 * inverse_area =", "- force_0 =")
        .replace("- 2 * inverse_area =", "- force_1 =");
    let (form, _) = compiled(&source);
    assert_ne!(
        form.fields()[0].1.dimension(),
        form.fields()[1].1.dimension()
    );
    exercise_source(&reaction, &source, &names);
}

#[test]
fn scalar_storage_uses_consistent_mass_and_nonzero_boundary_history() {
    let source = r#"
model Heat() {
 domain body = box(0, 1);
 domain left = boundary(body, axis = 0, side = lower);
 domain right = boundary(body, axis = 0, side = upper);
 state u: 1 on body;
 parameter c: s / m^2 = 3;
 parameter q: 1 / m^2 = 4;
 initial { u = 2; }
 law balance on body { storage c * u; flux -grad(u); source q; }
 relation left_value on left { trace(u) = 2; }
 relation right_value on right { trace(u) = 2; }
}
"#;
    let (form, symbols) = compiled(source);
    assert!(form.is_transient());
    assert_eq!(
        form.initial_values().unwrap()[&symbols.get("u").unwrap()],
        2.0
    );
    assert!(form.volume().is_err());
    let capacity = symbols.get("c").unwrap().downcast().unwrap();
    let heating = symbols.get("q").unwrap().downcast().unwrap();
    let parameters = [capacity, heating];
    assert!(form.bind_parameter_point(&parameters, &[0.0, 4.0]).is_err());
    assert!(
        form.bind_parameter_point(&parameters, &[-3.0, 4.0])
            .is_err()
    );
    let second = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let form = form
        .bind_backward_euler(eqiora_core::DynQuantity::new(0.25, second))
        .unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.5, 1.0]]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let doubled = form.bind_parameter_point(&parameters, &[6.0, 4.0]).unwrap();
    let slower = CartesianLinearAssembly::assemble_backward_euler(
        &doubled,
        &mesh,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &boundaries(&symbols),
        &[2.0; 3],
    )
    .unwrap();
    // Doubled capacity doubles M alone, reducing the first temperature increment to 1/6.
    close(
        &[slower.system.rhs()[0] / dense(&slower.system)[0]],
        &[2.0 + 1.0 / 6.0],
        1e-12,
    );
    assert!(slower.into_single_field_canonical().is_err());
    let mut previous = vec![2.0; 3];
    // On each interval, exact integrals give M=c*h/6*[[2,1],[1,2]],
    // K=1/h*[[1,-1],[-1,1]], F=q*h/2*[1,1]. Thus the reduced
    // operator is 8 and u_n=2+(1-2^-n)/2 with both boundary histories retained.
    for expected in [2.25, 2.375, 2.4375] {
        let assembled = CartesianLinearAssembly::assemble_backward_euler(
            &form,
            &mesh,
            &quadrature,
            &REFERENCE_ASSEMBLY_BACKEND,
            &boundaries(&symbols),
            &previous,
        )
        .unwrap();
        close(
            &dense(&assembled.full_system),
            &[4.0, -1.0, 0.0, -1.0, 8.0, -1.0, 0.0, -1.0, 4.0],
            1e-12,
        );
        let value = assembled.system.rhs()[0] / dense(&assembled.system)[0];
        close(&[value], &[expected], 1e-12);
        previous[1] = value;
    }
    assert!(
        CartesianLinearAssembly::assemble(
            &form,
            &mesh,
            &quadrature,
            &REFERENCE_ASSEMBLY_BACKEND,
            &boundaries(&symbols)
        )
        .is_err()
    );
}
