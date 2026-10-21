//! A fixed singular outer operand is not an active differentiation coordinate.
use super::*;

#[test]
fn coefficient_partials_distinguish_frozen_inputs_from_zero_tangents() {
    for expression in ["math.sqrt(q)+p*x/1[m]", "math.sqrt(p*p)+q*x/1[m]"] {
        let source = format!(
            "model Activity() {{
            domain body=box(0,1); coordinate x:m on body from body[0];
            parameter p:1=0; parameter q:1=0;
            variable u:1 on body;
            relation r on body {{ u={expression}; }}
        }}"
        );
        let (transaction, model, symbols) = compile("activity.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let owner = symbols.get("r").unwrap();
        let parameter = symbols.get("p").unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
        let dag = typed.expression();
        let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
            panic!("equation")
        };
        let known = BTreeMap::new();
        let context = Context::<f64> {
            program: &program,
            dag,
            owner,
            dimension: 1,
            coefficients: &known,
            time_s: None,
        };
        let pointwise = pointwise::PointwiseData::lower(&context, *rhs, 0)
            .unwrap()
            .coordinate(&[], false, 1)
            .unwrap();
        let direction = |coordinate: &ScalarSymbolCoordinate| {
            f64::from(
                matches!(coordinate.symbol(), SymbolRef::Parameter(id) if id.erase()==parameter),
            )
        };
        for data in [context.data(*rhs, 0).unwrap(), pointwise] {
            assert_eq!(data.evaluate(&[0.5]).unwrap(), 0.);
            let partial = data.derivative(&direction, 1).unwrap();
            if expression.starts_with("math.sqrt(q)") {
                // q is held fixed at zero. The selected map is exactly p -> p*x,
                // so its first and mixed spatial derivatives are x and 1.
                assert_eq!(partial.evaluate(&[0.5]).unwrap(), 0.5);
                assert_eq!(
                    partial
                        .coordinate_derivative(0, 1)
                        .unwrap()
                        .evaluate(&[0.5])
                        .unwrap(),
                    1.
                );
                let parameters = [
                    parameter.downcast().unwrap(),
                    symbols.get("q").unwrap().downcast().unwrap(),
                ];
                let rebound = partial
                    .bind_parameter_point(
                        &ParameterPoint::bind(&program, &parameters, &[2., 4.]).unwrap(),
                    )
                    .unwrap();
                assert_eq!(rebound.evaluate(&[0.5]).unwrap(), 0.5);
                // Frozen means no derivative, not permission to erase the
                // original real square-root domain or exact Parameter binding.
                assert!(
                    partial
                        .bind_parameter_point(
                            &ParameterPoint::bind(&program, &parameters, &[2., -1.]).unwrap(),
                        )
                        .unwrap()
                        .evaluate(&[0.5])
                        .is_err()
                );
                assert!(
                    partial
                        .bind_parameter_point(&ParameterPoint::default())
                        .is_err()
                );
                assert_eq!(
                    data.coordinate_derivative(0, 1)
                        .unwrap()
                        .derivative(&direction, 1)
                        .unwrap()
                        .evaluate(&[0.5])
                        .unwrap(),
                    1.
                );
            } else {
                // sqrt(p²)=|p| is not differentiable at zero. A zero inner
                // numerical tangent must never turn p into a frozen input.
                assert!(partial.evaluate(&[0.5]).is_err());
            }
        }
    }
}

#[test]
fn affine_coefficient_partials_preserve_frozen_domains_and_active_zero_tangents() {
    for expression in ["math.sqrt(q)+p*x/1[m]", "math.sqrt(p*p)+q*x/1[m]"] {
        let source = format!(
            "model Activity() {{
            domain body=box(0,1); coordinate x:m on body from body[0];
            parameter p:1=0; parameter q:1=0; variable u:1 on body;
            relation r on body {{ u=({expression})*u; }}
        }}"
        );
        let (transaction, model, symbols) = compile("affine-activity.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let owner = symbols.get("r").unwrap();
        let parameter = symbols.get("p").unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
        let dag = typed.expression();
        let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
            panic!("equation")
        };
        let known = BTreeMap::new();
        let context = Context::<f64> {
            program: &program,
            dag,
            owner,
            dimension: 1,
            coefficients: &known,
            time_s: None,
        };
        let rows = context.affine_components(*rhs, 0).unwrap().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].terms.len(), 1);
        let data = &rows[0].terms[0].3;
        assert_eq!(data.evaluate(&[0.5]).unwrap(), 0.);
        let direction = |coordinate: &ScalarSymbolCoordinate| {
            f64::from(
                matches!(coordinate.symbol(), SymbolRef::Parameter(id) if id.erase()==parameter),
            )
        };
        let partial = data.derivative(&direction, 1).unwrap();
        if expression.starts_with("math.sqrt(q)") {
            assert_eq!(partial.evaluate(&[0.5]).unwrap(), 0.5);
            let parameters = [
                parameter.downcast().unwrap(),
                symbols.get("q").unwrap().downcast().unwrap(),
            ];
            for q in [0., 4., -1.] {
                let rebound = partial
                    .bind_parameter_point(
                        &ParameterPoint::bind(&program, &parameters, &[2., q]).unwrap(),
                    )
                    .unwrap();
                if q < 0. {
                    assert!(rebound.evaluate(&[0.5]).is_err());
                } else {
                    assert_eq!(rebound.evaluate(&[0.5]).unwrap(), 0.5);
                }
            }
        } else {
            // At p=0 the inner derivative is numerically zero, but p is active.
            assert!(partial.evaluate(&[0.5]).is_err());
        }
    }
}
