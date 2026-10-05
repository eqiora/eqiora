use super::*;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;

#[test]
fn source_square_root_halves_dimensions_and_checks_value_and_derivative_domains() {
    let source = r#"
model Root() {
  domain interval = box(0, 1);

  parameter area: m ^ 2 = 4000000 [mm ^ 2];
  variable length: m on interval;
  relation law on interval { length - math.sqrt(area) = 0; }
}
"#;
    let mut compiled = compile("sqrt.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation),
            _ => None,
        })
        .unwrap();
    let expression = &program.numerical_residuals(relation.id().erase()).unwrap();
    let Some(ExprNode::Sub(_, root)) = expression.node(expression.roots()[0]) else {
        panic!("length minus square root");
    };
    let tape = lower::<f64>(&program, expression, *root, relation.id().erase(), 1).unwrap();
    assert_eq!(tape.evaluate(&[0.0]).unwrap(), 2.0);
    assert_eq!(
        tape.evaluate_parameter_jvp(&[0.0], &[1.0]).unwrap(),
        (2.0, 0.25)
    );
    assert_eq!(
        tape.evaluate_parameter_vjp(&[0.0], 1.0).unwrap(),
        (2.0, vec![0.25])
    );
    let zero = tape
        .bind_parameter_point(tape.parameter_fields(), &[0.0])
        .unwrap();
    assert_eq!(zero.evaluate(&[0.0]).unwrap(), 0.0);
    assert!(zero.evaluate_parameter_jvp(&[0.0], &[0.0]).is_err());
    assert!(zero.evaluate_parameter_vjp(&[0.0], 0.0).is_err());
    let negative = tape
        .bind_parameter_point(tape.parameter_fields(), &[-1.0])
        .unwrap();
    assert!(negative.evaluate(&[0.0]).is_err());
    assert!(negative.evaluate_parameter_jvp(&[0.0], &[1.0]).is_err());
    assert!(negative.evaluate_parameter_vjp(&[0.0], 1.0).is_err());
    assert!(
        compile(
            "wrong.eqi",
            &source.replace("length: m on", "length: m ^ 2 on")
        )
        .is_err()
    );
}

#[test]
fn evaluates_distinct_axes_and_requires_exact_coordinate_shape() {
    let expression: ScalarSpatialExpression<f64> = ScalarSpatialExpression {
        coordinate_dimension: 2,
        instructions: vec![
            Instruction::Coordinate(0),
            Instruction::Coordinate(1),
            Instruction::Mul(1, 1),
            Instruction::Add(0, 2),
        ],
        root: 3,
        coordinate_dependent: true,
        parameter_fields: Vec::new(),
        parameter_values: Vec::new(),
    };

    assert_eq!(expression.coordinate_dimension(), 2);
    assert_eq!(expression.evaluate(&[2.0, 3.0]).unwrap(), 11.0);
    assert_eq!(
        expression.evaluate(&[2.0]).unwrap_err().code(),
        codes::OPERATOR_INPUT_MISMATCH
    );
    assert_eq!(
        expression.evaluate(&[2.0, f64::NAN]).unwrap_err().code(),
        codes::NONFINITE_EVALUATION
    );
}

#[test]
fn constant_tape_retains_coordinate_dimension() {
    let expression = ScalarSpatialExpression::constant(3, 4.5);
    assert_eq!(expression.coordinate_dimension(), 3);
    assert_eq!(expression.constant_value(), Some(4.5));
    assert_eq!(expression.evaluate(&[1.0, 2.0, 3.0]).unwrap(), 4.5);
}

#[test]
fn lowers_both_axes_from_one_canonical_plane_relation() {
    let source = r#"
model plane_source() {
  domain plane = box(0, 2, 0, 3);

  variable u: m on plane;
  relation identity on plane {
u - (coordinate(0) + coordinate(1)) = 0;
  }
}
"#;
    let mut compiled = compile("plane-source.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation),
            _ => None,
        })
        .unwrap();
    let residuals = program.numerical_residuals(relation.id().erase()).unwrap();
    let root = residuals.roots()[0];
    let source_root = match residuals.node(root) {
        Some(ExprNode::Sub(_, source)) => *source,
        _ => panic!("fixture has one field-minus-source residual"),
    };
    let lowered =
        lower::<f64>(&program, &residuals, source_root, relation.id().erase(), 2).unwrap();

    assert_eq!(lowered.coordinate_dimension(), 2);
    assert_eq!(lowered.evaluate(&[2.0, 3.0]).unwrap(), 5.0);
}

#[test]
fn retains_parameter_identity_and_evaluates_analytic_jvp() {
    let source = r#"
model parameterized_source() {
  domain interval = box(0, 2);

  variable u: m ^ 2 on interval;
  parameter amplitude: m = 3;
  relation identity on interval {
u - amplitude ^ 2 * math.sin(coordinate(0) / amplitude) = 0;
  }
}
"#;
    let mut compiled = compile("parameterized-source.eqi", source).unwrap();
    let amplitude = compiled[0]
        .symbols()
        .get("amplitude")
        .unwrap()
        .downcast::<kinds::Parameter>()
        .unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation),
            _ => None,
        })
        .unwrap();
    let residuals = program.numerical_residuals(relation.id().erase()).unwrap();
    let root = residuals.roots()[0];
    let source_root = match residuals.node(root) {
        Some(ExprNode::Sub(_, source)) => *source,
        _ => panic!("fixture has one field-minus-source residual"),
    };
    let lowered =
        lower::<f64>(&program, &residuals, source_root, relation.id().erase(), 1).unwrap();

    let coordinate = 0.4_f64;
    let (value, tangent) = lowered
        .evaluate_parameter_jvp(&[coordinate], &[2.0])
        .unwrap();
    let (_, combined_tangent) = lowered
        .evaluate_jvp(&[coordinate], &[0.25], &[2.0])
        .unwrap();
    let output_cotangent = 1.75;
    let (vjp_value, coordinate_cotangent, parameter_cotangent) = lowered
        .evaluate_vjp(&[coordinate], output_cotangent)
        .unwrap();
    let expected_value = 9.0 * (coordinate / 3.0).sin();
    let expected_tangent =
        2.0 * (6.0 * (coordinate / 3.0).sin() - coordinate * (coordinate / 3.0).cos());
    let expected_coordinate_derivative = 3.0 * (coordinate / 3.0).cos();
    let expected_parameter_derivative =
        6.0 * (coordinate / 3.0).sin() - coordinate * (coordinate / 3.0).cos();
    assert_eq!(lowered.parameter_fields(), &[amplitude]);
    assert_eq!(lowered.parameter_values(), &[3.0]);
    assert!((value - expected_value).abs() < 1.0e-14);
    assert!((vjp_value - expected_value).abs() < 1.0e-14);
    assert!((tangent - expected_tangent).abs() < 1.0e-14);
    assert!(
        (combined_tangent - (expected_tangent + 0.25 * expected_coordinate_derivative)).abs()
            < 1.0e-14
    );
    assert!(
        (coordinate_cotangent[0] - output_cotangent * expected_coordinate_derivative).abs()
            < 1.0e-14
    );
    assert!(
        (parameter_cotangent[0] - output_cotangent * expected_parameter_derivative).abs() < 1.0e-14
    );
    let jvp_pairing = output_cotangent * combined_tangent;
    let vjp_pairing = 0.25 * coordinate_cotangent[0] + 2.0 * parameter_cotangent[0];
    assert!((jvp_pairing - vjp_pairing).abs() < 1.0e-14);
}

#[test]
fn product_deduplicates_parameter_coordinates() {
    let parameter = Id::<kinds::Parameter>::new();
    let left = ScalarSpatialExpression {
        coordinate_dimension: 1,
        instructions: vec![Instruction::Parameter(0)],
        root: 0,
        coordinate_dependent: false,
        parameter_fields: vec![parameter],
        parameter_values: vec![2.0],
    };
    let right = ScalarSpatialExpression {
        coordinate_dimension: 1,
        instructions: vec![
            Instruction::Parameter(0),
            Instruction::Coordinate(0),
            Instruction::Add(0, 1),
        ],
        root: 2,
        coordinate_dependent: true,
        parameter_fields: vec![parameter],
        parameter_values: vec![2.0],
    };
    let product = left.multiply(right);

    assert_eq!(product.parameter_fields(), &[parameter]);
    assert_eq!(
        product.evaluate_parameter_jvp(&[3.0], &[1.0]).unwrap(),
        (10.0, 7.0)
    );
}

#[test]
fn coefficient_equality_is_identity_aware_not_value_inferred() {
    let first = Id::<kinds::Parameter>::new();
    let second = Id::<kinds::Parameter>::new();
    let coefficient = |parameter| ScalarSpatialExpression {
        coordinate_dimension: 2,
        instructions: vec![Instruction::Parameter(0)],
        root: 0,
        coordinate_dependent: false,
        parameter_fields: vec![parameter],
        parameter_values: vec![3.0],
    };

    let volume = coefficient(first);
    assert!(volume.is_same_coefficient_as(&volume.clone()));
    assert!(
        !volume.is_same_coefficient_as(&coefficient(second)),
        "equal revision-local values do not merge independent Parameters"
    );
    assert!(
        ScalarSpatialExpression::constant(2, 3.0)
            .is_same_coefficient_as(&ScalarSpatialExpression::constant(2, 3.0))
    );
}

#[test]
fn complex_coefficients_share_the_spatial_tape_without_losing_imaginary_parts() {
    use num_complex::Complex64 as C;
    let tape = ScalarSpatialExpression {
        coordinate_dimension: 5,
        instructions: vec![
            Instruction::Constant(C::new(1.0, 2.0)),
            Instruction::Coordinate(4),
            Instruction::Mul(0, 1),
            Instruction::Conjugate(2),
        ],
        root: 3,
        coordinate_dependent: true,
        parameter_fields: vec![],
        parameter_values: vec![],
    };
    assert_eq!(
        tape.evaluate(&[0.0, 0.0, 0.0, 0.0, 3.0]).unwrap(),
        C::new(3.0, -6.0)
    );
    assert!(tape.evaluate(&[0.0; 4]).is_err());
    assert!(tape.evaluate(&[0.0, 0.0, 0.0, 0.0, f64::INFINITY]).is_err());
    let mut root = ScalarSpatialExpression::constant(1, C::new(-4.0, 0.0));
    root.instructions.push(Instruction::Sqrt(0));
    root.root = 1;
    assert_eq!(root.evaluate(&[0.0]).unwrap(), C::new(0.0, 2.0));
    let real = ScalarSpatialExpression {
        coordinate_dimension: 1,
        instructions: vec![Instruction::Constant(-4.0), Instruction::Sqrt(0)],
        root: 1,
        coordinate_dependent: false,
        parameter_fields: vec![],
        parameter_values: vec![],
    };
    assert!(real.evaluate(&[0.0]).is_err());
}

#[test]
fn complex_parameter_points_preserve_identity_and_both_components() {
    use num_complex::Complex64 as C;
    let parameter = Id::<kinds::Parameter>::new();
    let tape = ScalarSpatialExpression {
        coordinate_dimension: 1,
        instructions: vec![Instruction::Parameter(0), Instruction::Conjugate(0)],
        root: 1,
        coordinate_dependent: false,
        parameter_fields: vec![parameter],
        parameter_values: vec![C::new(1.0, 2.0)],
    };
    let bound = tape
        .bind_parameter_point(&[parameter], &[C::new(3.0, -4.0)])
        .unwrap();
    assert_eq!(bound.parameter_fields(), &[parameter]);
    assert_eq!(bound.parameter_values(), &[C::new(3.0, -4.0)]);
    assert_eq!(bound.evaluate(&[0.0]).unwrap(), C::new(3.0, 4.0));
    assert_eq!(tape.evaluate(&[0.0]).unwrap(), C::new(1.0, -2.0));
    assert!(
        tape.bind_parameter_point(&[parameter], &[C::new(1.0, f64::NAN)])
            .is_err()
    );
    assert!(
        tape.bind_parameter_point(&[Id::new()], &[C::new(3.0, -4.0)])
            .is_err()
    );
    let product = bound.clone().multiply(bound);
    assert_eq!(product.parameter_fields(), &[parameter]);
    assert_eq!(product.evaluate(&[0.0]).unwrap(), C::new(-7.0, 24.0));
}

#[test]
fn source_complex_parameters_construction_and_conjugation_use_common_lowering() {
    use num_complex::Complex64 as C;
    let source = r#"
model ComplexCoefficient() {
  domain interval = box(0, 4);
  parameter amplitude: complex<1> = math.complex(1, 2);
  parameter scale: m = 1[m];
  variable u: complex<1> on interval;
  relation law on interval {
    u - math.conj(amplitude) * math.complex(coordinate(0) / scale, 2) = 0;
  }
}
"#;
    let mut compiled = compile("complex-coefficient.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation),
            _ => None,
        })
        .unwrap();
    let residuals = program.numerical_residuals(relation.id().erase()).unwrap();
    let Some(ExprNode::Sub(_, root)) = residuals.node(residuals.roots()[0]) else {
        panic!("one field-minus-coefficient residual");
    };
    let tape = lower::<C>(&program, &residuals, *root, relation.id().erase(), 1).unwrap();
    // (1 - 2i)(3 + 2i) = 7 - 4i; the first factor is explicitly conjugated.
    assert_eq!(tape.evaluate(&[3.0]).unwrap(), C::new(7.0, -4.0));
    assert_eq!(tape.parameter_fields().len(), 2);
    assert!(lower::<f64>(&program, &residuals, *root, relation.id().erase(), 1).is_err());
}
