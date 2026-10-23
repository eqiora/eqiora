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
use crate::form_compiler::spatial::CompiledSpatialForm;
use crate::region_assembly::mapping::RegionDofMap;
use eqiora_assembly::{AssemblyBackend, AssemblyPacketSetIdentityV1, LinearSystem};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};

struct AssembledFixture {
    mapping: RegionDofMap<f64>,
    system: LinearSystem<f64>,
    full_system: LinearSystem<f64>,
}

fn assemble(
    form: &CompiledSpatialForm<f64>,
    mesh: &CartesianMesh,
    quadrature: &QuadratureRule,
    boundaries: &BTreeMap<(usize, BoundarySide), RawId>,
) -> Result<AssembledFixture, Diagnostic> {
    let equations = ExecutableSpatialEquations {
        regions: vec![SpatialRegion {
            form: form.clone(),
            support: SpatialRegionSupport::Cartesian(ScalarRegionSupport::new(
                form.domain(),
                (0..mesh.topological_dimension())
                    .map(|axis| {
                        let values = mesh.axis_coordinates(axis).expect("Cartesian axis");
                        [values[0], values[values.len() - 1]]
                    })
                    .collect(),
                boundaries.clone(),
            )),
        }],
        interfaces: vec![],
    };
    let selected =
        equations.discretizations(Space::continuous_lagrange(std::num::NonZeroU16::MIN), None)?;
    let (mapping, mut input) = equations.cartesian_assembly(
        mesh,
        &selected,
        &eqiora_realization::SystemScaling::Dimensional,
    )?;
    for (_, rule) in &mut input.forms {
        *rule = quadrature.clone();
    }
    let prepared = mapping.prepare_assembly(
        AssemblyPacketSetIdentityV1::Unbound,
        mesh,
        input.forms,
        input.natural,
        None,
        None,
    )?;
    let (systems, _) = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&prepared.plan, &prepared.work)?
        .into_parts();
    let mut systems = systems.into_iter();
    Ok(AssembledFixture {
        mapping,
        system: systems.next().unwrap(),
        full_system: systems.next().unwrap(),
    })
}

fn residual(system: &LinearSystem<f64>, values: &[f64]) -> Vec<f64> {
    let mut result = vec![0.; values.len()];
    system.matrix().multiply_into(values, &mut result).unwrap();
    for (value, rhs) in result.iter_mut().zip(system.rhs()) {
        *value -= rhs;
    }
    result
}

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
        source += &format!("variable {}: 1 on body in smooth;\n", names[row]);
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

fn compiled(source: &str) -> (CompiledSpatialForm<f64>, ModelSymbols, KernelProgram) {
    let (transaction, model, symbols) = compile("assembly.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledSpatialForm::<f64>::derive_at_time(
        &program,
        symbols.get("body").unwrap(),
        1,
        &std::collections::BTreeSet::new(),
        None,
    )
    .unwrap();
    (form, symbols, program)
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

fn dense(system: &LinearSystem<f64>) -> Vec<f64> {
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
    let (form, symbols, _) = compiled(source);
    let mesh = CartesianMesh::from_axes(vec![AXIS.to_vec()]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let AssembledFixture {
        mapping,
        system,
        full_system,
    } = assemble(&form, &mesh, &quadrature, &boundaries(&symbols)).unwrap();
    let ids = names
        .iter()
        .map(|name| symbols.get(name).unwrap())
        .collect::<Vec<_>>();
    let field_order = form
        .fields()
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
    close(&dense(&full_system), &matrix, 2e-14);
    close(full_system.rhs(), &rhs, 2e-14);

    let free = (0..count)
        .filter(|global| matches!(global % AXIS.len(), 1 | 2))
        .collect::<Vec<_>>();
    let actual_free = mapping
        .keys()
        .filter(|key| mapping.free_dof(*key).is_some())
        .map(|key| mapping.global_dof(key).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual_free, free);
    for global in 0..count {
        assert_eq!(
            mapping
                .keys()
                .any(|key| mapping.global_dof(key) == Some(global)
                    && mapping.free_dof(key).is_some()),
            free.contains(&global)
        );
    }
    let reduced = free
        .iter()
        .flat_map(|&row| free.iter().map(move |&column| (row, column)))
        .map(|(row, column)| matrix[row * count + column])
        .collect::<Vec<_>>();
    let reduced_rhs = free.iter().map(|&global| rhs[global]).collect::<Vec<_>>();
    close(&dense(&system), &reduced, 2e-14);
    close(system.rhs(), &reduced_rhs, 2e-14);

    let direction = (0..free.len())
        .map(|index| (index + 1) as f64 / 7.0)
        .collect::<Vec<_>>();
    let mut jv = vec![0.0; free.len()];
    system.matrix().multiply_into(&direction, &mut jv).unwrap();
    close(&jv, &action(&reduced, &direction), 3e-14);

    let problem = LinearProblem::new(
        system.matrix(),
        system.rhs(),
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
    let lifted = mapping.lift(solution.values(), false).unwrap();
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
    close(&residual(&full_system, &lifted), &expected_residual, 3e-14);
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
    let (form, symbols, _) = compiled(&source);
    let first = symbols.get(&names[0]).unwrap();
    let mesh = CartesianMesh::from_axes(vec![AXIS.to_vec()]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let AssembledFixture {
        mapping, system, ..
    } = assemble(&form, &mesh, &quadrature, &boundaries(&symbols)).unwrap();
    let problem = LinearProblem::new(
        system.matrix(),
        system.rhs(),
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
    let values = mapping.lift(solution.values(), false).unwrap();
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
    assert_eq!(mapping.free_count(), 6);
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
    let (form, _, _) = compiled(&source);
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
 state u: 1 on body in h1;
 parameter c: s / m^2 = 3;
 parameter q: 1 / m^2 = 4;
 initial { u = 2; }
 law balance on body { storage c * u; flux -grad(u); source q; }
 relation left_value on left { trace(u) = 2; }
 relation right_value on right { trace(u) = 2; }
}
"#;
    let (form, symbols, program) = compiled(source);
    assert!(form.is_transient());
    assert_eq!(
        form.initial_values_at(&[0.0]).unwrap()[&symbols.get("u").unwrap()][0],
        2.0
    );
    let continuous = form.volume().unwrap();
    assert!(continuous.time_binding().is_none());
    let capacity = symbols.get("c").unwrap().downcast().unwrap();
    let heating = symbols.get("q").unwrap().downcast().unwrap();
    let parameters = [capacity, heating];
    for capacity in [0., -3.] {
        let bound = form
            .bind_inputs(
                &crate::parameter_point::ParameterPoint::bind(
                    &program,
                    &parameters,
                    &[capacity, 4.],
                )
                .unwrap(),
            )
            .unwrap();
        // Rebinding changes the certificate, not the source rate inventory.
        // Actual State acceptance owns singularity and rate compatibility.
        assert!(bound.is_transient());
    }
    let second = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let form = form
        .bind_backward_euler(eqiora_core::DynQuantity::new(0.25, second))
        .unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.5, 1.0]]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let doubled = form
        .bind_inputs(
            &crate::parameter_point::ParameterPoint::bind(&program, &parameters, &[6.0, 4.0])
                .unwrap(),
        )
        .unwrap();
    let geometry = mesh.geometry_map(MeshEntity::new(1, 0)).unwrap();
    // Continuous binding is valid, but affine evaluation cannot discard its
    // State/rate point or silently invent a backward-Euler step.
    let error = continuous
        .prepare_cell(&geometry, &quadrature)
        .unwrap()
        .evaluate(&BTreeMap::new())
        .unwrap_err();
    assert!(
        error.message().contains("explicit State/rate point"),
        "{error}"
    );
    let previous = BTreeMap::from([(symbols.get("u").unwrap(), vec![2.0; 2])]);
    // Exact one-cell integrals: M=c*h/6*[[2,1],[1,2]],
    // K=1/h*[[1,-1],[-1,1]], and F=q*h/2*[1,1].
    // The common transient Run separately checks assembly and repeated history.
    for (form, matrix, rhs) in [
        (&form, [4.0, -1.0, -1.0, 4.0], [7.0, 7.0]),
        (&doubled, [6.0, 0.0, 0.0, 6.0], [13.0, 13.0]),
    ] {
        let local = form
            .volume()
            .unwrap()
            .prepare_cell(&geometry, &quadrature)
            .unwrap()
            .evaluate(&previous)
            .unwrap();
        close(local.matrix(), &matrix, 1e-12);
        close(local.rhs(), &rhs, 1e-12);
    }
    assert!(assemble(&form, &mesh, &quadrature, &boundaries(&symbols)).is_err());
}
