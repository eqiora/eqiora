use super::*;
use crate::parameter_point::ParameterPoint;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;
use num_complex::Complex64 as C;

mod activity;

fn authored_coefficient<S: Coefficient>(source: &str) -> Data<S> {
    authored_coefficient_at(source, None)
}

fn authored_coefficient_at<S: Coefficient>(source: &str, time_s: Option<f64>) -> Data<S> {
    let (transaction, model, _) = compile("authored-coefficient.eqi", source)
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
    let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
        panic!("prescribed coefficient equation")
    };
    Context {
        program: &program,
        dag,
        owner,
        dimension: 1,
        coefficients: &BTreeMap::new(),
        time_s,
    }
    .data(*rhs, 0)
    .unwrap()
}

fn complex_coefficient(expression: &str) -> Data<C> {
    authored_coefficient(&format!(
        r#"model ComplexCoefficient() {{
        domain body=box(0,4);
        variable u:complex<1> on body;
        relation prescribed on body {{ u={expression}; }}
    }}"#
    ))
}

#[test]
fn authored_coefficient_degree_preserves_parameters_and_divisor_domains() {
    for (expression, expected) in [
        ("p*(coordinate(0)/1[m])^8", Some(8)),
        ("p*(coordinate(0)/1[m])^8/(coordinate(0)/1[m])", None),
        ("(coordinate(0)/1[m])^(-1)", None),
        ("math.sin(coordinate(0)/1[m])", None),
        ("math.sin(p)", Some(0)),
    ] {
        // A zero Parameter binding must not erase degree or divisor domains.
        let data = authored_coefficient::<f64>(&format!(
            r#"model Degree() {{
            domain body=box(0,4);
            parameter p:1=0;
            variable u:1 on body;
            relation prescribed on body {{u={expression};}}
        }}"#
        ));
        assert_eq!(data.polynomial_degree(), expected, "{expression}");
        if expression.contains("^8/") {
            assert!(data.evaluate(&[0.]).is_err());
        }
    }
}

#[test]
fn constant_coefficient_values_retain_spatial_primal_domains() {
    for (expression, invalid) in [
        ("math.sqrt(coordinate(0)/1[m])^0", -1.),
        ("((coordinate(0)/1[m])^(-1))^0", 0.),
    ] {
        let data = authored_coefficient::<f64>(&format!(
            r#"model Domain() {{
            domain body=box(-1,2);
            variable u:1 on body;
            relation prescribed on body {{u={expression};}}
        }}"#
        ));
        assert!(data.spatial(), "{expression}");
        assert_eq!(data.polynomial_degree(), None, "{expression}");
        assert_eq!(data.evaluate(&[1.]).unwrap(), 1.);
        assert!(data.evaluate(&[invalid]).is_err(), "{expression}");
    }
}

#[test]
fn composed_coefficient_actions_keep_outer_scales_through_parameter_derivatives() {
    for exponent in [-700, -400, 400, 700] {
        let scale = 2f64.powi(exponent);
        let source = format!(
            r#"
            space S=orthonormal(x,y);
            model Coefficients() {{
                domain body=box(0,1);
                parameter a:map<1,S,S>=linear_map(S,S,[[{scale},0],[0,{scale}]]);
                variable u:1 on body;
                relation prescribed on body {{u={scale}*matrix_trace(inverse(a));}}
            }}"#
        );
        let (transaction, model, symbols) = compile("coefficient-range.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let owner = symbols.get("prescribed").unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
        let dag = typed.expression();
        let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
            panic!("prescribed equation")
        };
        let context = Context::<f64> {
            program: &program,
            dag,
            owner,
            dimension: 1,
            coefficients: &BTreeMap::new(),
            time_s: None,
        };
        let data = context.data(*rhs, 0).unwrap();
        let selected =
            |coordinate: &ScalarSymbolCoordinate| f64::from(coordinate.component_index() == [0, 0]);
        // f(A)=s tr(A^-1). At A=sI, f=2, Df[E00]=-1/s and
        // D²f[E00,sE00]=2/s. The inner inverse partial 1/s² need not fit.
        assert_eq!(data.evaluate(&[0.]).unwrap(), 2.);
        let derivative = data.derivative(&selected, 1).unwrap();
        assert_eq!(derivative.evaluate(&[0.]).unwrap() * scale, -1.);
        assert_eq!(
            derivative
                .derivative(&|c| scale * selected(c), 1)
                .unwrap()
                .evaluate(&[0.])
                .unwrap()
                * scale,
            2.
        );
        assert_eq!(data.polynomial_degree(), Some(0));
        for point in [vec![], vec![0., 0.], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(data.evaluate(&point).is_err());
            assert!(derivative.evaluate(&point).is_err());
        }
    }
}

#[test]
fn coefficient_identity_preserves_physical_factor_across_boundary_restriction() {
    let source = r#"model Coordinates() {
        domain body = box(0,1,0,1);
        domain wall = boundary(body,axis=1,side=upper);
        domain away = box(0,1,0,1);
        variable u:m on body in h1;
        variable v:m on away in h1;
        relation volume on body { u=coordinate(0); }
        relation boundary on wall { trace(u)=coordinate(0); }
        relation foreign on away { v=coordinate(0); }
    }"#;
    let (transaction, model, symbols) = compile("coordinate-identity.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let data = |name| {
        let relation = symbols.get(name).unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, relation).unwrap();
        let dag = typed.expression();
        let node = dag
            .nodes()
            .iter()
            .position(|node| matches!(node, ExprNode::Symbol(SymbolRef::Coordinate { .. })))
            .unwrap();
        Context::<f64> {
            program: &program,
            dag,
            owner: relation,
            dimension: 2,
            coefficients: &BTreeMap::new(),
            time_s: None,
        }
        .data(dag.node_id(u32::try_from(node).unwrap()).unwrap(), 0)
        .unwrap()
    };
    let volume = data("volume");
    let boundary = data("boundary");
    let foreign = data("foreign");
    assert!(volume.same_coefficient(&boundary));
    // Equal values and equal dimensions cannot identify different physical factors.
    assert_eq!(
        volume.evaluate(&[0.25, 1.]).unwrap(),
        foreign.evaluate(&[0.25, 1.]).unwrap()
    );
    assert!(!volume.same_coefficient(&foreign));
    let volume = volume.coordinate_derivative(0, 2).unwrap();
    assert!(volume.same_coefficient(&boundary.coordinate_derivative(0, 2).unwrap()));
    assert!(!volume.same_coefficient(&foreign.coordinate_derivative(0, 2).unwrap()));
}

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
    // For rate*x, simultaneous directions dr=5 and dx=2 give 5*x+2*rate.
    let direction = |coordinate: &ScalarSymbolCoordinate| match coordinate.symbol() {
        SymbolRef::Parameter(id) if id == parameter => 5.,
        SymbolRef::Coordinate { axis: 0, .. } => 2.,
        _ => 0.,
    };
    let mixed_inputs = rate.derivative(&direction, 1).unwrap();
    assert_eq!(mixed_inputs.evaluate(&[3.]).unwrap(), 16.);
    // d_x(rate*x)=rate, hence d_direction(d_x)=5. The reverse order
    // differentiates 5*x+2*rate with respect to x and has the same value.
    assert_eq!(
        derivative
            .derivative(&direction, 1)
            .unwrap()
            .evaluate(&[3.])
            .unwrap(),
        5.
    );
    assert_eq!(
        mixed_inputs
            .coordinate_derivative(0, 1)
            .unwrap()
            .evaluate(&[3.])
            .unwrap(),
        5.
    );
    assert_eq!(
        mixed_inputs
            .bind_parameter_point(&ParameterPoint::bind(&program, &[parameter], &[2.]).unwrap(),)
            .unwrap()
            .evaluate(&[3.])
            .unwrap(),
        19.,
    );
    // A constant gradient still belongs to the declared one-dimensional space.
    for point in [vec![], vec![0.0, 0.0], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(rate.evaluate(&point).is_err());
        assert!(derivative.evaluate(&point).is_err());
    }
    let parameter_node = dag
        .nodes()
        .iter()
        .position(
            |node| matches!(node, ExprNode::Symbol(SymbolRef::Parameter(id)) if *id == parameter),
        )
        .and_then(|index| dag.node_id(index as u32))
        .unwrap();
    let parameter_data = context.data(parameter_node, 0).unwrap();
    let parameter_gradient = parameter_data.coordinate_derivative(0, 1).unwrap();
    assert_eq!(parameter_gradient.evaluate(&[3.0]).unwrap(), 0.0);
    assert!(
        parameter_gradient
            .bind_parameter_point(&ParameterPoint::default())
            .is_err()
    );
    let coordinate_node = dag
        .nodes()
        .iter()
        .position(|node| matches!(node, ExprNode::Symbol(SymbolRef::Coordinate { .. })))
        .and_then(|index| dag.node_id(index as u32))
        .unwrap();
    let coordinate_data = context.data(coordinate_node, 0).unwrap();
    assert_eq!(
        coordinate_data
            .coordinate_derivative(0, 1)
            .unwrap()
            .polynomial_degree(),
        Some(0)
    );
    assert_eq!(
        rate.bind_parameter_point(&ParameterPoint::bind(&program, &[parameter], &[2.0]).unwrap())
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        6.0
    );
    assert_eq!(
        derivative
            .bind_parameter_point(&ParameterPoint::bind(&program, &[parameter], &[2.0]).unwrap())
            .unwrap()
            .evaluate(&[3.0])
            .unwrap(),
        2.0
    );
    assert!(
        rate.bind_parameter_point(&ParameterPoint::default())
            .is_err()
    );
    assert!(context.data(*field, 0).is_err());
}

#[test]
fn map_data_binds_time_and_preserves_coordinate_and_parameter_actions() {
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
            let data = context.data(factor, 0).unwrap();
            assert_eq!(data.spatial(), nonlinear);
            if !nonlinear {
                assert_eq!(data.polynomial_degree(), Some(0));
            }
            // dx/dxi gains 2*xi/1[m] for the quadratic map, at xi=7[m].
            let spatial_factor = if nonlinear { 14. } else { 1. };
            close(
                data.evaluate(&[7., -3.]).unwrap(),
                spatial_factor * (1. + 0.5 * t) * (1. + 0.25 * t),
            );
            let rebound = data
                .bind_parameter_point(
                    &ParameterPoint::bind(&program, &[parameter], &[1.0]).unwrap(),
                )
                .unwrap();
            close(
                rebound.evaluate(&[7., -3.]).unwrap(),
                spatial_factor * (1. + t) * (1. + 0.25 * t),
            );
            let parameter_direction = |coordinate: &ScalarSymbolCoordinate| {
                f64::from(coordinate.symbol() == SymbolRef::Parameter(parameter))
            };
            close(
                data.derivative(&parameter_direction, 2)
                    .unwrap()
                    .evaluate(&[7., -3.])
                    .unwrap(),
                spatial_factor * t * (1. + 0.25 * t),
            );
            close(
                data.coordinate_derivative(0, 2)
                    .unwrap()
                    .derivative(&parameter_direction, 2)
                    .unwrap()
                    .evaluate(&[7., -3.])
                    .unwrap(),
                if nonlinear {
                    2. * t * (1. + 0.25 * t)
                } else {
                    0.
                },
            );
            assert!(
                data.bind_parameter_point(&ParameterPoint::default())
                    .is_err()
            );
            if t == 1.0 {
                let parameter_data = context
                    .data(
                        find(|node| matches!(node, ExprNode::Symbol(SymbolRef::Parameter(_)))),
                        0,
                    )
                    .unwrap();
                let coordinate_data = context
                    .data(
                        find(|node| {
                            matches!(
                                node,
                                ExprNode::Symbol(SymbolRef::Coordinate { axis: 0, .. })
                            )
                        }),
                        0,
                    )
                    .unwrap();
                let chart = crate::form_compiler::spatial::motion::UniformChart {
                    scale: 2.,
                    offset: [4., -6.],
                };
                let coefficient = parameter_data
                    .multiply(coordinate_data)
                    .on_uniform_chart(&chart);
                // At fixed x=2*xi+4, rate*xi has d/dx=rate/2 and
                // d/drate=xi. The Parameter direction must not be divided by 2.
                assert_eq!(coefficient.evaluate(&[10., -2.]).unwrap(), 1.5);
                assert_eq!(
                    coefficient
                        .coordinate_derivative(0, 2)
                        .unwrap()
                        .evaluate(&[10., -2.])
                        .unwrap(),
                    0.25
                );
                let direction = |coordinate: &ScalarSymbolCoordinate| match coordinate.symbol() {
                    SymbolRef::Parameter(id) if id == parameter => 5.,
                    SymbolRef::Coordinate { axis: 0, .. } => 2.,
                    _ => 0.,
                };
                assert_eq!(
                    coefficient
                        .derivative(&direction, 2)
                        .unwrap()
                        .evaluate(&[10., -2.])
                        .unwrap(),
                    15.5
                );
                assert!(
                    data.bind_parameter_point(
                        &ParameterPoint::bind(&program, &[parameter], &[-1.0]).unwrap()
                    )
                    .unwrap()
                    .evaluate(&[7., -3.])
                    .is_err()
                );
                assert!(
                    data.coordinate_derivative(0, 2)
                        .unwrap()
                        .bind_parameter_point(
                            &ParameterPoint::bind(&program, &[parameter], &[-1.0]).unwrap()
                        )
                        .unwrap()
                        .evaluate(&[7., -3.])
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn coordinate_map_factors_compose_with_scalar_coefficient_algebra() {
    for (factor, expected) in [
        ("jacobian_determinant", -6.),
        ("volume_jacobian", 6.),
        ("map_orientation", -3.),
    ] {
        let data = authored_coefficient::<f64>(&format!(
            r#"model MapCoefficient() {{
            domain body=box(0,1);
            coordinate xi:m on body from body[0];
            coordinate x:m on body from body[0];
            variable u:1 on body;
            relation prescribed on body {{u=3*{factor}(from=(xi),at=(x=-2*xi));}}
        }}"#
        ));
        assert_eq!(data.evaluate(&[0.25]).unwrap(), expected);
        assert_eq!(
            data.coordinate_derivative(0, 1)
                .unwrap()
                .evaluate(&[0.25])
                .unwrap(),
            0.
        );
        for point in [vec![], vec![0., 0.], vec![f64::NAN]] {
            assert!(data.evaluate(&point).is_err());
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
    let coefficient = "math.conj(amplitude) * math.complex(coordinate(0) / scale, 2)";
    let root_data = authored_coefficient::<C>(&source.replace(
        coefficient,
        &format!("math.sqrt(({coefficient}) + math.complex(-4,8))"),
    ));
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
    let power = authored_coefficient::<C>(&source.replace(
        coefficient,
        &format!("(({coefficient}) / math.complex(7,-4))^(-2147483648)"),
    ));
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
        .bind_parameter_point(
            &ParameterPoint::bind(
                &program,
                fields,
                &fields
                    .iter()
                    .flat_map(|id| {
                        let ty = program.typed_value(id.erase()).unwrap().value_type();
                        if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                            vec![2., 0.]
                        } else {
                            vec![2.]
                        }
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
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
    assert!(
        data.bind_parameter_point(&ParameterPoint::default())
            .is_err()
    );
    assert!(
        ParameterPoint::bind(
            &program,
            fields,
            &vec![
                f64::NAN;
                crate::parameter_point::coordinates(&program, fields)
                    .unwrap()
                    .len()
            ]
        )
        .is_err()
    );
    assert!(context.data(*field_expr, 0).is_err());
    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = dag.node(*field_expr) else {
        panic!("field");
    };
    let Some(KernelNode::Field(definition)) = program.node(field.erase()) else {
        panic!("known coefficient Field");
    };
    let prescribed =
        Components::prescribed(&context, definition.value_type(), Some(*root)).unwrap();
    let known = BTreeMap::from([(field.erase(), prescribed)]);
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
    for value in [-4, 0] {
        let root = complex_coefficient(&format!("math.sqrt(math.complex({value},0))"));
        assert!(root.evaluate(&[0.0]).is_ok());
        // Both are fixed complex values, independent of the physical coordinate.
        // Their spatial derivative is zero even on the principal branch cut.
        assert_eq!(
            root.coordinate_derivative(0, 1)
                .unwrap()
                .evaluate(&[0.0])
                .unwrap(),
            C::new(0., 0.)
        );
    }
    let root = complex_coefficient("math.sqrt(math.complex(3,4))");
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
    let power = complex_coefficient("math.complex(1,0)^(-2147483648)");
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

#[test]
fn component_data_preserves_complex_parameter_coordinates_and_spatial_derivatives() {
    let source = r#"model Components() {
        domain body=box(0,6);
        coordinate xi:m on body from body[0];
        parameter p:array<complex<1>,2>=[math.complex(2,-3),math.complex(-1,4)];
        variable u:array<complex<1>,2> on body;
        relation prescribed on body {
            [p[0]*u[1], p[1]*u[0]]=[p[1]*xi/1[m], math.conj(p[0])*(xi/1[m])^2];
        }
    }"#;
    let (transaction, model, _) = compile("component-data.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let owner = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) => Some(value.id().erase()),
            _ => None,
        })
        .unwrap();
    let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
    let dag = typed.expression();
    let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
        panic!("prescribed components")
    };
    let known = BTreeMap::new();
    let context = Context::<C> {
        program: &program,
        dag,
        owner,
        dimension: 1,
        coefficients: &known,
        time_s: None,
    };
    let components = context.components(*rhs, 0).unwrap();
    assert_eq!(components.len(), 2);
    let parameter = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let coordinates = crate::parameter_point::coordinates(&program, &[parameter]).unwrap();
    let direction = |coordinate: &ScalarSymbolCoordinate| {
        coordinates
            .iter()
            .position(|candidate| candidate == coordinate)
            .map_or(0., |index| [3., -2., 5., 7.][index])
    };
    let affine = context
        .affine_components(dag.roots()[0], 0)
        .unwrap()
        .unwrap();
    assert_eq!(affine.len(), 2);
    for (row, action) in affine.iter().enumerate() {
        for (_, component, derivative, coefficient) in &action.terms {
            assert!(!derivative);
            let crate::form_compiler::bilinear::BasisComponent::Value(column) = component else {
                panic!("value coefficient");
            };
            let expected = match (row, *column) {
                (0, 1) => C::new(3., -2.),
                (1, 0) => C::new(5., 7.),
                _ => C::new(0., 0.),
            };
            assert_eq!(
                coefficient
                    .derivative(&direction, 1)
                    .unwrap()
                    .evaluate(&[2.])
                    .unwrap(),
                expected
            );
        }
        // The offset is minus the prescribed right-hand side.
        let expected = [C::new(-10., -14.), C::new(-12., -8.)][row];
        assert_eq!(
            action
                .offset
                .derivative(&direction, 1)
                .unwrap()
                .evaluate(&[2.])
                .unwrap(),
            expected
        );
    }
    for x in [0., 2., 5.] {
        // p has real coordinates [p0.re,p0.im,p1.re,p1.im].
        // Conjugation reverses only the imaginary direction of channel zero.
        let expected_parameter = [
            [C::new(0., 0.), C::new(x * x, 0.)],
            [C::new(0., 0.), C::new(0., -x * x)],
            [C::new(x, 0.), C::new(0., 0.)],
            [C::new(0., x), C::new(0., 0.)],
        ];
        for (coordinate, expected) in coordinates.iter().zip(expected_parameter) {
            for (component, expected) in components.iter().zip(expected) {
                assert_eq!(
                    component
                        .derivative(&|input| f64::from(input == coordinate), 1)
                        .unwrap()
                        .evaluate(&[x])
                        .unwrap(),
                    expected
                );
            }
        }
        // Reversed channel selection and conjugation are authored independently.
        let expected = [C::new(-x, 4. * x), C::new(2. * x * x, 3. * x * x)];
        let derivative = [C::new(-1., 4.), C::new(4. * x, 6. * x)];
        for ((component, expected), derivative) in components.iter().zip(expected).zip(derivative) {
            assert_eq!(component.evaluate(&[x]).unwrap(), expected);
            assert_eq!(
                component
                    .coordinate_derivative(0, 1)
                    .unwrap()
                    .evaluate(&[x])
                    .unwrap(),
                derivative
            );
        }
    }
    for (index, node) in dag.nodes().iter().enumerate() {
        if let ExprNode::Index {
            value,
            index: component,
        } = node
            && matches!(
                dag.node(*value),
                Some(ExprNode::Symbol(SymbolRef::Parameter(_)))
            )
        {
            let scalar = context.data(dag.node_id(index as u32).unwrap(), 0).unwrap();
            assert_eq!(
                scalar.evaluate(&[3.]).unwrap(),
                [C::new(2., -3.), C::new(-1., 4.)][*component as usize]
            );
        }
    }
}

#[test]
fn mixed_scalar_coefficient_partials_preserve_complex_directions_and_domains() {
    fn check<S: Coefficient>(complex: bool, expression: &str, first: C, second: C) {
        let domain = if complex { "complex<m^2>" } else { "m^2" };
        let parameter = if complex {
            "complex<1>=math.complex(2,3)"
        } else {
            "1=2"
        };
        let source = format!(
            "model Mixed(){{domain body=box(0,4);coordinate x:m on body from body[0];parameter p:{parameter};variable u:{domain} on body;relation r on body{{u=({expression})*x^2;}}}}"
        );
        let (transaction, model, _) = compile("mixed-coefficient.eqi", &source)
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
        let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
            panic!("equation")
        };
        let known = BTreeMap::new();
        let context = Context::<S> {
            program: &program,
            dag,
            owner: relation,
            dimension: 1,
            coefficients: &known,
            time_s: None,
        };
        let data = context.data(*rhs, 0).unwrap();
        let close = |actual: S, expected: C| {
            assert!(
                (C::new(actual.re(), actual.im()) - expected).norm()
                    <= 128. * f64::EPSILON * (1. + expected.norm()),
                "{actual:?} != {expected}"
            );
        };
        for imaginary in [false, true].into_iter().take(if complex { 2 } else { 1 }) {
            let direction = |coordinate: &ScalarSymbolCoordinate| {
                f64::from(
                    coordinate.symbol() == SymbolRef::Parameter(parameter)
                        && coordinate.is_imaginary() == imaginary,
                )
            };
            let phase = if imaginary {
                C::new(0., 1.)
            } else {
                C::new(1., 0.)
            };
            let parameter_partial = data.derivative(&direction, 1).unwrap();
            // For g(p)*x², both mixed orders are 2*x*g'(p)*dp;
            // two equal Parameter directions give x²*g''(p)*dp².
            for x in [0.5, 3.] {
                close(
                    data.coordinate_derivative(0, 1)
                        .unwrap()
                        .derivative(&direction, 1)
                        .unwrap()
                        .evaluate(&[x])
                        .unwrap(),
                    2. * x * first * phase,
                );
                close(
                    parameter_partial
                        .coordinate_derivative(0, 1)
                        .unwrap()
                        .evaluate(&[x])
                        .unwrap(),
                    2. * x * first * phase,
                );
                close(
                    parameter_partial
                        .derivative(&direction, 1)
                        .unwrap()
                        .evaluate(&[x])
                        .unwrap(),
                    x * x * second * phase * phase,
                );
            }
            // Nested derivatives still require their original Parameter owner.
            assert!(
                parameter_partial
                    .coordinate_derivative(0, 1)
                    .unwrap()
                    .bind_parameter_point(&ParameterPoint::default())
                    .is_err()
            );
        }
    }
    for complex in [false, true] {
        let p = C::new(2., if complex { 3. } else { 0. });
        for (expression, first, second) in [
            ("p^3", 3. * p * p, 6. * p),
            ("1/p", -1. / (p * p), 2. / (p * p * p)),
            ("math.sin(p)", p.cos(), -p.sin()),
            ("math.cos(p)", -p.sin(), -p.cos()),
            ("math.exp(p)", p.exp(), p.exp()),
            ("math.log(p)", 1. / p, -1. / (p * p)),
            ("math.sqrt(p)", 0.5 / p.sqrt(), -0.25 / (p * p.sqrt())),
        ] {
            if complex {
                check::<C>(true, expression, first, second);
            } else {
                check::<f64>(false, expression, first, second);
            }
        }
    }
}

#[test]
fn prescribed_time_data_retains_its_exact_derivative_coordinate() {
    let direction =
        |coordinate: &ScalarSymbolCoordinate| f64::from(coordinate.symbol() == SymbolRef::Time);
    for (expression, value, first, second) in [
        ("time()/1[s]", 3., 1., 0.),
        ("2*(time()/1[s])^2", 18., 12., 4.),
    ] {
        let source = format!(
            "model TimedDatum() {{
            domain body=box(0,1); variable u:1 on body;
            relation prescribed on body {{u={expression};}}
        }}"
        );
        let data = authored_coefficient_at::<f64>(&source, Some(3.));
        assert_eq!(data.evaluate(&[0.25]).unwrap(), value);
        let rate = data.derivative(&direction, 1).unwrap();
        assert_eq!(rate.evaluate(&[0.25]).unwrap(), first, "{expression}");
        assert_eq!(
            rate.derivative(&direction, 1)
                .unwrap()
                .evaluate(&[0.25])
                .unwrap(),
            second
        );
    }
}

#[test]
fn prescribed_time_partials_preserve_domain_and_structural_activity() {
    let time =
        |coordinate: &ScalarSymbolCoordinate| f64::from(coordinate.symbol() == SymbolRef::Time);
    let parameter = |coordinate: &ScalarSymbolCoordinate| {
        f64::from(matches!(coordinate.symbol(), SymbolRef::Parameter(_)))
    };
    for expression in ["math.sqrt(time()/1[s])+p", "math.sqrt((time()/1[s])^2)+p"] {
        let source = format!(
            "model TimedDatum() {{
            domain body=box(0,1); parameter p:1=2; variable u:1 on body;
            relation prescribed on body {{u={expression};}}
        }}"
        );
        let data = authored_coefficient_at::<f64>(&source, Some(0.));
        assert_eq!(data.evaluate(&[0.25]).unwrap(), 2.);
        // Time stays frozen for a Parameter query, but a zero inner time
        // tangent must not hide the singular derivative of sqrt(t²) at zero.
        assert_eq!(
            data.derivative(&parameter, 1)
                .unwrap()
                .evaluate(&[0.25])
                .unwrap(),
            1.
        );
        assert!(
            data.derivative(&time, 1)
                .unwrap()
                .evaluate(&[0.25])
                .is_err()
        );
    }
    let data = authored_coefficient_at::<f64>(
        "model TimedDatum() {
        domain body=box(0,1); variable u:1 on body;
        relation prescribed on body {u=math.sqrt(time()/1[s]);}
    }",
        Some(-1.),
    );
    assert!(data.evaluate(&[0.25]).is_err());
    assert!(
        data.derivative(&parameter, 1)
            .unwrap()
            .evaluate(&[0.25])
            .is_err()
    );
}

#[test]
fn complex_time_parameter_products_share_the_retained_coordinate_inputs() {
    let source = "model TimedDatum() {
        domain body=box(0,1); parameter p:complex<1> = math.complex(2,3);
        variable u:complex<1> on body;
        relation prescribed on body {u=p*(time()/1[s])^2;}
    }";
    let time =
        |coordinate: &ScalarSymbolCoordinate| f64::from(coordinate.symbol() == SymbolRef::Time);
    for t in [0., 3.] {
        let data = authored_coefficient_at::<C>(source, Some(t));
        assert_eq!(data.evaluate(&[0.25]).unwrap(), C::new(2., 3.) * (t * t));
        let rate = data.derivative(&time, 1).unwrap();
        assert_eq!(rate.evaluate(&[0.25]).unwrap(), C::new(2., 3.) * (2. * t));
        assert_eq!(
            rate.derivative(&time, 1)
                .unwrap()
                .evaluate(&[0.25])
                .unwrap(),
            C::new(4., 6.)
        );
        for imaginary in [false, true] {
            let parameter = |coordinate: &ScalarSymbolCoordinate| {
                f64::from(
                    matches!(coordinate.symbol(), SymbolRef::Parameter(_))
                        && coordinate.is_imaginary() == imaginary,
                )
            };
            let expected = if imaginary {
                C::new(0., 2. * t)
            } else {
                C::new(2. * t, 0.)
            };
            assert_eq!(
                rate.derivative(&parameter, 1)
                    .unwrap()
                    .evaluate(&[0.25])
                    .unwrap(),
                expected
            );
            assert_eq!(
                data.derivative(&parameter, 1)
                    .unwrap()
                    .derivative(&time, 1)
                    .unwrap()
                    .evaluate(&[0.25])
                    .unwrap(),
                expected
            );
        }
    }
    // Rebinding Parameters must leave a bound model Time intact.
    let data = authored_coefficient_at::<f64>(
        "model TimedDatum() {
        domain body=box(0,1); variable u:1 on body;
        relation prescribed on body {u=time()/1[s];}
    }",
        Some(3.),
    );
    let rebound = data
        .bind_parameter_point(&ParameterPoint::default())
        .unwrap();
    assert_eq!(rebound.evaluate(&[0.25]).unwrap(), 3.);
    assert_eq!(
        rebound
            .derivative(&time, 1)
            .unwrap()
            .evaluate(&[0.25])
            .unwrap(),
        1.
    );
}
