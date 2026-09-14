use std::f64::consts::PI;

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{AffineGeometryMap, QuadratureRule, ReferenceCell};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::*;

const SOURCE: &str =
    include_str!("../../../../../crates/eqiora-api/packages/org.example.poisson/src/main.eqi");
const SOURCE_1D: &str =
    include_str!("../../../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");
const SOURCE_3D: &str =
    include_str!("../../../../../verify/numerics/cartesian-poisson-3d-fem-fvm/models/poisson.eqi");
const MATRIX: [f64; 16] = [
    0.833_333_333_333_333_3,
    -0.583_333_333_333_333_3,
    0.166_666_666_666_666_66,
    -0.416_666_666_666_666_63,
    -0.583_333_333_333_333_3,
    0.833_333_333_333_333_3,
    -0.416_666_666_666_666_63,
    0.166_666_666_666_666_66,
    0.166_666_666_666_666_66,
    -0.416_666_666_666_666_63,
    0.833_333_333_333_333_3,
    -0.583_333_333_333_333_3,
    -0.416_666_666_666_666_63,
    0.166_666_666_666_666_66,
    -0.583_333_333_333_333_3,
    0.833_333_333_333_333_3,
];
const LOAD: [f64; 4] = [
    0.425_495_714_381_214_03,
    0.474_820_601_775_891_64,
    0.242_871_390_835_767_61,
    0.271_025_855_380_221_5,
];
const MATRIX_3D_SCALED: [i32; 64] = [
    28, 6, 12, 2, -18, -13, -10, -7, 6, 28, 2, 12, -13, -18, -7, -10, 12, 2, 28, 6, -10, -7, -18,
    -13, 2, 12, 6, 28, -7, -10, -13, -18, -18, -13, -10, -7, 28, 6, 12, 2, -13, -18, -7, -10, 6,
    28, 2, 12, -10, -7, -18, -13, 12, 2, 28, 6, -7, -10, -13, -18, 2, 12, 6, 28,
];
const MATRIX_RELATIVE: f64 = 1.0e-14;
const LOAD_RELATIVE: f64 = 2.5e-2;
const ACTION_RELATIVE: f64 = 1.0e-7;
const STEP: f64 = 1.0e-6;

#[test]
fn e1_element_fixture_matches_frozen_matrix_loads_and_certificate() {
    let form = compiled_form(SOURCE);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let admitted = form.admit_quadrature(&quadrature).unwrap();
    let geometry = fixture_geometry();
    let source = |point: &[f64]| 2.0 * PI.powi(2) * (PI * point[0]).sin() * (PI * point[1]).sin();
    let local = admitted
        .evaluate(&geometry, &quadrature, &|_: &[f64]| 1.0, &source)
        .unwrap();
    assert_relative_slice(local.matrix(), &MATRIX, MATRIX_RELATIVE);
    assert_relative_slice(local.rhs(), &LOAD, LOAD_RELATIVE);

    let constant = admitted
        .evaluate(&geometry, &quadrature, &|_: &[f64]| 1.0, &|_: &[f64]| 1.0)
        .unwrap();
    assert_relative_slice(constant.rhs(), &[0.031_25; 4], MATRIX_RELATIVE);
    assert_matrix_invariants(local.matrix(), 4);

    let rules = form
        .certificate
        .entries
        .iter()
        .map(|entry| entry.rule_id)
        .collect::<Vec<_>>();
    assert_eq!(
        rules,
        [
            TEST_PAIRING,
            DIVERGENCE_BY_PARTS,
            ZERO_TEST_TRACE_DISCHARGE,
            ZERO_TEST_TRACE_DISCHARGE,
            ZERO_TEST_TRACE_DISCHARGE,
            ZERO_TEST_TRACE_DISCHARGE,
            SOURCE_PAIRING,
        ]
    );
    assert_eq!(form.parameters.len(), 2);
    form.validate_certificate().unwrap();
}

#[test]
fn e1_rejects_certificate_sign_slot_quadrature_aspect_and_coefficient_mutants() {
    let form = compiled_form(SOURCE);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let geometry = fixture_geometry();
    let source = |point: &[f64]| 2.0 * PI.powi(2) * (PI * point[0]).sin() * (PI * point[1]).sin();
    let admitted = form.admit_quadrature(&quadrature).unwrap();
    let local = admitted
        .evaluate(&geometry, &quadrature, &|_: &[f64]| 1.0, &source)
        .unwrap();

    let mut sign_mutant = form.clone();
    sign_mutant.certificate.entries[1].sign = WeakSign::Negative;
    assert_gate(sign_mutant.validate_certificate(), "derivation certificate");

    let mut slot_mutant = form.clone();
    slot_mutant.certificate.entries[1].slot = WeakTermSlot::Bilinear {
        test: MatrixSlot::Trial,
        trial: MatrixSlot::Test,
    };
    assert_gate(slot_mutant.validate_certificate(), "derivation certificate");

    let under_integrated = QuadratureRule::tensor_product_gauss_legendre(2, 1).unwrap();
    assert_gate(
        form.admit_quadrature(&under_integrated).map(|_| ()),
        "realization compatibility",
    );

    let aspect_mutant = permute_matrix(local.matrix(), [0, 2, 1, 3]);
    assert!(maximum_relative(&aspect_mutant, &MATRIX) > MATRIX_RELATIVE);

    let scaled = admitted
        .evaluate(&geometry, &quadrature, &|_: &[f64]| 2.0, &source)
        .unwrap();
    assert!(maximum_relative(scaled.matrix(), &MATRIX) > MATRIX_RELATIVE);
}

#[test]
fn e1_independent_actions_reject_corrupted_jvp_and_primal_as_vjp() {
    let form = compiled_form(SOURCE);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let source = |point: &[f64]| 2.0 * PI.powi(2) * (PI * point[0]).sin() * (PI * point[1]).sin();
    let local = form
        .admit_quadrature(&quadrature)
        .unwrap()
        .evaluate(&fixture_geometry(), &quadrature, &|_: &[f64]| 1.0, &source)
        .unwrap();
    let direction = normalized_direction(4);
    let state = direction.clone();

    let finite_difference =
        centered_residual_action(local.matrix(), local.rhs(), &state, &direction);
    let exact_jvp = matrix_action(local.matrix(), &direction);
    assert_relative_slice(&exact_jvp, &finite_difference, ACTION_RELATIVE);

    let mut corrupted = local.matrix().to_vec();
    for row in 0..4 {
        corrupted[row * 4 + 1] = 0.0;
    }
    let mutant_jvp = matrix_action(&corrupted, &direction);
    assert!(maximum_relative(&mutant_jvp, &finite_difference) > ACTION_RELATIVE);

    let directional_vjp = centered_scalar_gradient(local.matrix(), local.rhs(), &state, &direction);
    let exact_vjp = transpose_action(local.matrix(), &direction);
    assert_relative_slice(&exact_vjp, &directional_vjp, ACTION_RELATIVE);
    let primal_as_vjp = residual(local.matrix(), local.rhs(), &state);
    assert!(maximum_relative(&primal_as_vjp, &directional_vjp) > ACTION_RELATIVE);
}

#[test]
fn runtime_dimensional_1d_and_3d_elements_match_the_frozen_oracles() {
    let quadrature_1d = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let form_1d = compiled_form(SOURCE_1D);
    let local_1d = form_1d
        .admit_quadrature(&quadrature_1d)
        .unwrap()
        .evaluate(
            &fixture_geometry_1d(),
            &quadrature_1d,
            &|_: &[f64]| 1.0,
            &|_: &[f64]| 1.0,
        )
        .unwrap();
    assert_eq!(form_1d.dimension, 1);
    assert_matrix_invariants(local_1d.matrix(), 2);
    assert_relative_slice(local_1d.rhs(), &[0.125; 2], MATRIX_RELATIVE);
    assert_relative_slice(&[local_1d.rhs().iter().sum()], &[0.25], MATRIX_RELATIVE);

    let quadrature_3d = QuadratureRule::tensor_product_gauss_legendre(3, 2).unwrap();
    let form_3d = compiled_form(SOURCE_3D);
    let local_3d = form_3d
        .admit_quadrature(&quadrature_3d)
        .unwrap()
        .evaluate(
            &fixture_geometry_3d(),
            &quadrature_3d,
            &|_: &[f64]| 1.0,
            &|_: &[f64]| 1.0,
        )
        .unwrap();
    let matrix_3d = MATRIX_3D_SCALED
        .iter()
        .map(|entry| f64::from(*entry) / 192.0)
        .collect::<Vec<_>>();
    assert_eq!(form_3d.dimension, 3);
    assert_relative_slice(local_3d.matrix(), &matrix_3d, MATRIX_RELATIVE);
    assert_relative_slice(local_3d.rhs(), &[1.0 / 512.0; 8], MATRIX_RELATIVE);
    assert_matrix_invariants(local_3d.matrix(), 8);

    let direction = normalized_direction(8);
    let finite_difference =
        centered_residual_action(local_3d.matrix(), local_3d.rhs(), &direction, &direction);
    let exact_jvp = matrix_action(local_3d.matrix(), &direction);
    assert_relative_slice(&exact_jvp, &finite_difference, ACTION_RELATIVE);
    let directional_vjp =
        centered_scalar_gradient(local_3d.matrix(), local_3d.rhs(), &direction, &direction);
    let exact_vjp = transpose_action(local_3d.matrix(), &direction);
    assert_relative_slice(&exact_vjp, &directional_vjp, ACTION_RELATIVE);

    for (mutation, axes) in [
        ("swap-y-z", [0, 2, 1]),
        ("swap-x-y", [1, 0, 2]),
        ("cycle-x-y-z", [1, 2, 0]),
        ("cycle-x-z-y", [2, 0, 1]),
        ("swap-x-z", [2, 1, 0]),
    ] {
        let axis_mutant = permute_axis_bits(local_3d.matrix(), axes);
        assert!(
            maximum_relative(&axis_mutant, &matrix_3d) > MATRIX_RELATIVE,
            "3D anisotropic frozen-matrix gate accepted {mutation}"
        );
    }
}

#[test]
fn ineligible_natural_boundary_has_no_semantic_derivation() {
    let source = SOURCE.replace(
        "relation x_lower_value on x_lower { trace(potential) = 0; }",
        "relation x_lower_value on x_lower { normal(grad(potential)) = 0; }",
    );
    let program = compile_program(&source);
    let domain = box_domain(&program);
    assert!(derive_candidate(&program, domain).unwrap().is_none());
    crate::canonical::lower_scalar_elliptic_cartesian(&program).unwrap();
}

#[test]
fn complete_essential_boundary_accepts_nonzero_parameter_data() {
    let source = SOURCE
        .replace(
            "  parameter source_scale: 1 / m ^ 2 = 19.739208802178716;",
            "  parameter boundary_value: 1 = 2;\n  parameter source_scale: 1 / m ^ 2 = 19.739208802178716;",
        )
        .replace("trace(potential) = 0", "trace(potential) = boundary_value");
    let form = compiled_form(&source);
    assert_eq!(form.parameters.len(), 3);
    assert_eq!(form.formulation_description().1, "complete-essential");
    assert_eq!(
        form.formulation_description().2[2],
        ZERO_TEST_TRACE_DISCHARGE
    );
    form.validate_certificate().unwrap();
}

fn compiled_form(source: &str) -> DerivedScalarGalerkinForm {
    let program = compile_program(source);
    let domain = box_domain(&program);
    derive_candidate(&program, domain)
        .unwrap()
        .expect("exact package is compiler-eligible")
}

fn box_domain(program: &KernelProgram) -> RawId {
    program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap()
}

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("compiled-poisson.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn fixture_geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::hypercube(2).unwrap(),
        2,
        vec![0.375, 0.75],
        vec![0.125, 0.0, 0.0, 0.25],
    )
    .unwrap()
}

fn fixture_geometry_1d() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::hypercube(1).unwrap(),
        1,
        vec![0.125],
        vec![0.125],
    )
    .unwrap()
}

fn fixture_geometry_3d() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::hypercube(3).unwrap(),
        3,
        vec![0.125, 0.25, 0.0625],
        vec![0.125, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0625],
    )
    .unwrap()
}

fn assert_matrix_invariants(matrix: &[f64], dof_count: usize) {
    assert_eq!(matrix.len(), dof_count * dof_count);
    for row in 0..dof_count {
        assert!(matrix[row * dof_count + row] > 0.0);
        assert!(
            matrix[row * dof_count..(row + 1) * dof_count]
                .iter()
                .sum::<f64>()
                .abs()
                <= MATRIX_RELATIVE * matrix[row * dof_count + row].abs()
        );
        for column in 0..dof_count {
            let scale = matrix[row * dof_count + column]
                .abs()
                .max(matrix[column * dof_count + row].abs());
            assert!(
                (matrix[row * dof_count + column] - matrix[column * dof_count + row]).abs()
                    <= MATRIX_RELATIVE * scale
            );
        }
    }
}

fn assert_gate(result: Result<(), Diagnostic>, gate: &str) {
    let error = result.expect_err("mutant must fail closed");
    assert!(error.message().contains(gate), "{error:?}");
}

fn assert_relative_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        maximum_relative(actual, expected) <= tolerance,
        "actual={actual:?}, expected={expected:?}"
    );
}

fn maximum_relative(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

fn permute_matrix(matrix: &[f64], permutation: [usize; 4]) -> Vec<f64> {
    let mut permuted = vec![0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            permuted[row * 4 + column] = matrix[permutation[row] * 4 + permutation[column]];
        }
    }
    permuted
}

fn permute_axis_bits(matrix: &[f64], axes: [usize; 3]) -> Vec<f64> {
    let permutation = (0..8)
        .map(|vertex| {
            axes.iter()
                .enumerate()
                .map(|(axis, source_axis)| ((vertex >> source_axis) & 1) << axis)
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    let mut permuted = vec![0.0; 64];
    for row in 0..8 {
        for column in 0..8 {
            permuted[row * 8 + column] = matrix[permutation[row] * 8 + permutation[column]];
        }
    }
    permuted
}

fn normalized_direction(dimension: usize) -> Vec<f64> {
    let mut direction = (0..dimension)
        .map(|index| ((index + 1) as f64).sin())
        .collect::<Vec<_>>();
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    direction.iter_mut().for_each(|value| *value /= norm);
    direction
}

fn residual(matrix: &[f64], rhs: &[f64], state: &[f64]) -> Vec<f64> {
    matrix_action(matrix, state)
        .into_iter()
        .zip(rhs)
        .map(|(value, rhs)| value - rhs)
        .collect()
}

fn matrix_action(matrix: &[f64], input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|row| {
            matrix[row * input.len()..(row + 1) * input.len()]
                .iter()
                .zip(input)
                .map(|(entry, input)| entry * input)
                .sum()
        })
        .collect()
}

fn transpose_action(matrix: &[f64], input: &[f64]) -> Vec<f64> {
    let dimension = input.len();
    (0..dimension)
        .map(|column| {
            input
                .iter()
                .enumerate()
                .map(|(row, value)| matrix[row * dimension + column] * value)
                .sum()
        })
        .collect()
}

fn centered_residual_action(
    matrix: &[f64],
    rhs: &[f64],
    state: &[f64],
    direction: &[f64],
) -> Vec<f64> {
    let plus = state
        .iter()
        .zip(direction)
        .map(|(state, direction)| state + STEP * direction)
        .collect::<Vec<_>>();
    let minus = state
        .iter()
        .zip(direction)
        .map(|(state, direction)| state - STEP * direction)
        .collect::<Vec<_>>();
    residual(matrix, rhs, &plus)
        .into_iter()
        .zip(residual(matrix, rhs, &minus))
        .map(|(plus, minus)| (plus - minus) / (2.0 * STEP))
        .collect()
}

fn centered_scalar_gradient(
    matrix: &[f64],
    rhs: &[f64],
    state: &[f64],
    cotangent: &[f64],
) -> Vec<f64> {
    (0..state.len())
        .map(|coordinate| {
            let mut plus = state.to_vec();
            let mut minus = state.to_vec();
            plus[coordinate] += STEP;
            minus[coordinate] -= STEP;
            let scalar = |point: &[f64]| {
                residual(matrix, rhs, point)
                    .iter()
                    .zip(cotangent)
                    .map(|(residual, cotangent)| residual * cotangent)
                    .sum::<f64>()
            };
            (scalar(&plus) - scalar(&minus)) / (2.0 * STEP)
        })
        .collect()
}

#[test]
fn projected_zero_storage_does_not_hide_unconsumed_operations() {
    use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;
    let owner = Id::<kinds::Relation>::new().erase();
    for (domain, extra_operation) in [
        (ScalarDomain::Real, false),
        (ScalarDomain::Complex, false),
        (ScalarDomain::Real, true),
        (ScalarDomain::Complex, true),
    ] {
        let mut builder = ExprDagBuilder::new();
        let root = builder
            .constant(
                ValueLiteral::from_real(
                    ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                        .expect("fixture uses a numeric scalar domain"),
                    1.0,
                )
                .unwrap(),
            )
            .unwrap();
        let zero = builder
            .constant(
                ValueLiteral::from_real(
                    ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                        .expect("fixture uses a numeric scalar domain"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        // The extra exact zero is inert storage, even though its ExprId remains.
        // An unused operation on that zero is still an unconsumed operation.
        if extra_operation {
            builder.neg(zero).unwrap();
        }
        let dag = builder.finish([root]).unwrap();
        if extra_operation {
            assert_gate(require_closed_dag(&dag, owner), "derivation certificate");
        } else {
            require_closed_dag(&dag, owner).unwrap();
        }
    }
}

#[test]
fn non_numeric_or_nonzero_unused_storage_remains_unconsumed() {
    use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;
    let owner = Id::<kinds::Relation>::new().erase();
    for extra in [
        ValueLiteral::boolean(false),
        ValueLiteral::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("numeric scalar type"),
            0,
        )
        .unwrap(),
        ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("numeric scalar type"),
            2.0,
        )
        .unwrap(),
    ] {
        let mut builder = ExprDagBuilder::new();
        let root = builder
            .constant(
                ValueLiteral::from_real(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("numeric scalar type"),
                    1.0,
                )
                .unwrap(),
            )
            .unwrap();
        builder.constant(extra).unwrap();
        assert_gate(
            require_closed_dag(&builder.finish([root]).unwrap(), owner),
            "derivation certificate",
        );
    }
}

#[test]
fn reachable_extra_strong_form_term_is_not_dropped() {
    let source = SOURCE.replace(
        "-div(grad(potential))",
        "-div(grad(potential)) + div(grad(potential))",
    );
    assert_ne!(source, SOURCE);
    let program = compile_program(&source);
    let domain = box_domain(&program);
    assert_gate(
        derive_candidate(&program, domain).map(|_| ()),
        "unmatched signed leaves",
    );
}
