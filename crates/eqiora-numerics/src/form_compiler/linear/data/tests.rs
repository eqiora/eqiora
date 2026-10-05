use super::*;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;
use num_complex::Complex64 as C;

#[test]
fn source_data_keeps_conjugation_parameters_and_known_field_resolution() {
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
    let (transaction, model, _) = compile("data.eqi", source).unwrap().remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) => Some(value),
            _ => None,
        })
        .unwrap();
    let dag = program.numerical_residuals(relation.id().erase()).unwrap();
    let Some(ExprNode::Sub(field_expr, root)) = dag.node(dag.roots()[0]) else {
        panic!("field minus coefficient");
    };
    let coefficients = BTreeMap::new();
    let context = Context::<C> {
        program: &program,
        dag: &dag,
        owner: relation.id().erase(),
        dimension: 1,
        coefficients: &coefficients,
    };
    let data = context.data(*root, 0).unwrap();
    // (1-2i)(x+2i) at x=3 is 7-4i, with derivative 1-2i.
    assert_eq!(data.evaluate(&[3.0]).unwrap(), C::new(7.0, -4.0));
    assert_eq!(
        data.coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        C::new(1.0, -2.0)
    );
    // Shift to 3+4i: sqrt=2+i and (1-2i)/(4+2i)=-i/2.
    let root_data = Data(Arc::new(Node::Math(
        UnaryMathFunction::Sqrt,
        data.clone().add(Data::constant(1, C::new(-4.0, 8.0))),
    )));
    assert_eq!(root_data.evaluate(&[3.0]).unwrap(), C::new(2.0, 1.0));
    assert!(
        (root_data
            .coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[3.0])
            .unwrap()
            - C::new(0.0, -0.5))
        .norm()
            < 1e-14
    );
    // Normalize the base to 1. Its derivative is (3-2i)/13.
    let power = Data(Arc::new(Node::Pow(
        data.clone().divide(Data::constant(1, C::new(7.0, -4.0))),
        i32::MIN,
    )));
    let expected = C::new(3.0, -2.0) * f64::from(i32::MIN) / 13.0;
    assert!(
        (power
            .coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[3.0])
            .unwrap()
            - expected)
            .norm()
            < 1e-6
    );
    let tape =
        spatial_expression::lower::<C>(&program, &dag, *root, relation.id().erase(), 1).unwrap();
    let fields = tape.parameter_fields();
    let rebound = data
        .bind_parameter_point(fields, &vec![C::new(2.0, 0.0); fields.len()])
        .unwrap();
    // amplitude=2 and scale=2 gives 2(x/2+2i).
    assert_eq!(rebound.evaluate(&[3.0]).unwrap(), C::new(3.0, 4.0));
    assert_eq!(
        rebound
            .coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        C::new(1.0, 0.0)
    );
    assert!(!data.same_coefficient(&rebound)); // Different bound values define different coefficient functions.
    assert!(data.bind_parameter_point(&[], &[]).is_err());
    assert!(
        data.bind_parameter_point(fields, &vec![C::new(0.0, f64::NAN); fields.len()])
            .is_err()
    );
    assert!(context.data(*field_expr, 0).is_err());
    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = dag.node(*field_expr) else {
        panic!("field");
    };
    let known = BTreeMap::from([(field.erase(), data.clone())]);
    let resolved = Context {
        coefficients: &known,
        ..context
    }
    .data(*field_expr, 0)
    .unwrap();
    assert!(resolved.same_coefficient(&data));
    assert_eq!(resolved.evaluate(&[3.0]).unwrap(), C::new(7.0, -4.0));
    let real_coefficients = BTreeMap::new();
    assert!(
        Context::<f64> {
            program: &program,
            dag: &dag,
            owner: relation.id().erase(),
            dimension: 1,
            coefficients: &real_coefficients
        }
        .data(*root, 0)
        .is_err()
    );
}

#[test]
fn composed_data_sqrt_derivative_checks_its_domain() {
    for value in [C::new(-4.0, 0.0), C::new(0.0, 0.0)] {
        let root = Data(Arc::new(Node::Math(
            UnaryMathFunction::Sqrt,
            Data::constant(1, value),
        )));
        assert!(root.evaluate(&[0.0]).is_ok());
        assert!(
            root.coordinate_derivative(0, 1)
                .unwrap()
                .evaluate(&[0.0])
                .is_err()
        );
    }
    let root = Data(Arc::new(Node::Math(
        UnaryMathFunction::Sqrt,
        Data::constant(1, C::new(3.0, 4.0)),
    )));
    assert_eq!(root.evaluate(&[0.0]).unwrap(), C::new(2.0, 1.0));
    assert_eq!(
        root.coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[0.0])
            .unwrap(),
        C::new(0.0, 0.0)
    );
}

#[test]
fn composed_power_derivative_preserves_minimum_integer_exponent() {
    let power = Data(Arc::new(Node::Pow(
        Data::constant(1, C::new(1.0, 0.0)),
        i32::MIN,
    )));
    assert_eq!(power.evaluate(&[0.0]).unwrap(), C::new(1.0, 0.0));
    assert_eq!(
        power
            .coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[0.0])
            .unwrap(),
        C::new(0.0, 0.0)
    );
}
