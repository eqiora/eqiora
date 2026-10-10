use super::*;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;
use num_complex::Complex64 as C;

#[test]
fn prescribed_time_rate_uses_canonical_scalar_value_and_spatial_action() {
    let source = r#"model PrescribedRate() {
        domain body=box(0,4);
        coordinate xi:m on body from body[0];
        parameter rate:1/s=0.5[1/s];
        variable velocity:m/s on body;
        relation prescribed on body { velocity=derivative((1+rate*time())*xi); }
    }"#;
    let (transaction, model, _) = compile("prescribed-rate.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let relation = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) => Some(value.id().erase()),
            _ => None,
        })
        .unwrap();
    let parameter = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let typed = crate::form_compiler::scalar::typed_relation(&program, relation).unwrap();
    let dag = typed.expression();
    let Some(ExprNode::Sub(field, rate)) = dag.node(dag.roots()[0]) else {
        panic!("prescribed rate equation")
    };
    let known = BTreeMap::new();
    let context = Context::<f64> {
        program: &program,
        dag,
        owner: relation,
        dimension: 1,
        coefficients: &known,
        time_s: Some(2.0),
    };
    let rate = context.data(*rate, 0).unwrap();
    assert_eq!(rate.evaluate(&[3.0]).unwrap(), 1.5);
    let derivative = rate.coordinate_derivative(0, 1).unwrap();
    assert_eq!(derivative.evaluate(&[3.0]).unwrap(), 0.5);
    assert_eq!(
        rate.bind_parameter_point(&[parameter], &[2.0])
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        6.0
    );
    assert_eq!(
        derivative
            .bind_parameter_point(&[parameter], &[2.0])
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        2.0
    );
    assert!(rate.bind_parameter_point(&[], &[]).is_err());
    assert!(context.data(*field, 0).is_err());
}

#[test]
fn affine_map_data_binds_time_and_preserves_parameter_identity() {
    // The diagonal 2x2 maps below have condition number <= 3. Allow a
    // 16-epsilon relative roundoff budget for normalization, LU and rescaling;
    // the oracle is the independent product of the two authored scale factors.
    let close = |actual: f64, expected: f64| {
        assert!((actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs().max(1.0));
    };
    let source = r#"model MapData() {
        domain reference=box(0,1,0,1);
        domain physical=box(-10,10,-10,10);
        coordinate xi:m on reference from reference[0];
        coordinate eta:m on reference from reference[1];
        coordinate x:m on physical from physical[0];
        coordinate y:m on physical from physical[1];
        parameter rate:1/s=0.5[1/s];
        variable u:1 on reference;
        relation mapped on reference {
            u=volume_jacobian(from=(eta,xi),at=(
                y=(1+0.25[1/s]*time())*eta,
                x=(1+rate*time())*xi+2[m/s]*time()));
        }
    }"#;
    for nonlinear in [false, true] {
        let source = if nonlinear {
            source.replace("*xi+2[m/s]", "*xi*xi/1[m]+2[m/s]")
        } else {
            source.to_owned()
        };
        let (transaction, model, _) = compile("map-data.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let owner = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(relation) => Some(relation.id().erase()),
                _ => None,
            })
            .unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
        let dag = typed.expression();
        let find = |predicate: fn(&ExprNode) -> bool| {
            dag.nodes()
                .iter()
                .position(predicate)
                .and_then(|index| dag.node_id(index as u32))
                .unwrap()
        };
        let factor = find(|node| matches!(node, ExprNode::CoordinateMapFactor { .. }));
        let time = find(|node| matches!(node, ExprNode::Symbol(SymbolRef::Time)));
        let parameter = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(parameter) => Some(parameter.id()),
                _ => None,
            })
            .unwrap();
        let known = BTreeMap::new();
        let mut context = Context::<f64> {
            program: &program,
            dag,
            owner,
            dimension: 2,
            coefficients: &known,
            time_s: None,
        };
        assert!(context.data(factor, 0).is_err());
        assert!(context.data(time, 0).is_err());
        for t in [0.0, 1.0, 2.0] {
            context.time_s = Some(t);
            assert_eq!(
                context.data(time, 0).unwrap().evaluate(&[0., 0.]).unwrap(),
                t
            );
            let data = context.data(factor, 0);
            if nonlinear {
                assert!(data.is_err());
                continue;
            }
            let data = data.unwrap();
            assert!(!data.spatial());
            close(
                data.evaluate(&[7., -3.]).unwrap(),
                (1. + 0.5 * t) * (1. + 0.25 * t),
            );
            let rebound = data.bind_parameter_point(&[parameter], &[1.0]).unwrap();
            close(
                rebound.evaluate(&[0., 0.]).unwrap(),
                (1. + t) * (1. + 0.25 * t),
            );
            assert!(data.bind_parameter_point(&[], &[]).is_err());
            if t == 1.0 {
                assert!(data.bind_parameter_point(&[parameter], &[-1.0]).is_err());
                assert!(
                    data.coordinate_derivative(0, 2)
                        .unwrap()
                        .bind_parameter_point(&[parameter], &[-1.0])
                        .is_err()
                );
            }
        }
    }
}

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
        time_s: None,
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
            time_s: None,
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
