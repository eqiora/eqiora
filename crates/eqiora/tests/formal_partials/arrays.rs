//! Constructed arrays retain complete value domains through formal actions.
use super::typed_inputs::try_solve;

#[test]
fn constructed_array_actions_preserve_channel_axes_and_chain_rules() {
    for (expression, ty, expected) in [
        ("partial([x*x,math.log(x)][0],wrt=x)", "1", vec![4.]),
        (
            "partial(partial([x*x,math.log(x)][0],wrt=x),wrt=x)",
            "1",
            vec![2.],
        ),
        (
            "jvp([x*x,math.log(x)],[x],[1])",
            "array<1,2>",
            vec![4., 0.5],
        ),
        ("vjp([x*x,math.log(x)],x,[2,3])", "1", vec![9.5]),
        (
            "jvp([[x*x,x],[x,2*x]],[x],[1])",
            "array<array<1,2>,2>",
            vec![4., 1., 1., 2.],
        ),
    ] {
        let source = format!(
            "model M() {{parameter x:1=2;variable y:{ty};relation r {{y={expression};}}observable output:{ty}=y;}}"
        );
        let (document, plan, result) = try_solve(&source).unwrap();
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(
            observed.value().components().unwrap().collect::<Vec<_>>(),
            expected
                .into_iter()
                .map(|value| (value, 0.))
                .collect::<Vec<_>>(),
            "{expression}"
        );
    }
}

#[test]
fn array_actions_do_not_drop_unselected_or_zero_seed_domains() {
    for expression in [
        "partial([x*x,math.log(x)][0],wrt=x)",
        "jvp([x*x,math.log(x)],[x],[1])[0]",
        "vjp([x*x,math.log(x)],x,[1,0])",
    ] {
        for x in [2, -1] {
            let source = format!(
                "model M() {{parameter x:1={x}; variable y:1;relation r {{y={expression};}}}}"
            );
            match try_solve(&source) {
                Ok(_) => assert!(x > 0, "invalid array domain accepted: {expression}"),
                Err(error) => {
                    assert!(x < 0, "valid array action rejected: {expression}: {error}");
                    assert!(
                        error.contains("EQ0507") || error.contains("EQ0807"),
                        "{error}"
                    );
                }
            }
        }
    }
}

#[test]
fn constructed_array_actions_preserve_complex_products_and_units() {
    for (declarations, expression, ty, expected) in [
        (
            "parameter z:complex<1>=math.complex(3,4);",
            "jvp([z,math.conj(z)],[z],[math.complex(2,3)])",
            "array<complex<1>,2>",
            vec![(2., 3.), (2., -3.)],
        ),
        (
            "parameter z:complex<1>=math.complex(3,4);",
            "vjp([z,math.conj(z)],z,[math.complex(1,2),math.complex(3,4)])",
            "complex<1>",
            vec![(4., -2.)],
        ),
        (
            "parameter x:1=2;",
            "jvp([x,math.complex(x,x)],[x],[2])",
            "array<complex<1>,2>",
            vec![(2., 0.), (2., 2.)],
        ),
        (
            "parameter x:m=2[m];",
            "partial([x*x,1[m^2]*math.log(x/1[m])],wrt=x)",
            "array<m,2>",
            vec![(4., 0.), (0.5, 0.)],
        ),
    ] {
        // D[z,conj(z)] maps d to [d,conj(d)]. Under real dual pairing,
        // its adjoint maps [a,b] to a+conj(b). Units of d(x²)/dx are m.
        let source = format!(
            "model M() {{{declarations}variable y:{ty};relation r {{y={expression};}}observable output:{ty}=y;}}"
        );
        let (document, plan, result) = try_solve(&source).unwrap();
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(
            observed.value().components().unwrap().collect::<Vec<_>>(),
            expected,
            "{expression}"
        );
    }
}

#[test]
fn local_operator_array_actions_use_the_same_typed_composition() {
    for (input_type, argument, expression, output_type, expected) in [
        (
            "1",
            "2",
            "jvp([square(x=x),x],[x],[1])",
            "array<1,2>",
            vec![(4., 0.), (1., 0.)],
        ),
        (
            "1",
            "2",
            "partial([x*x,math.log(x)][0],wrt=x)",
            "1",
            vec![(4., 0.)],
        ),
        (
            "1",
            "2",
            "jvp([x*x,math.log(x)],[x],[1])[0]",
            "1",
            vec![(4., 0.)],
        ),
        (
            "1",
            "2",
            "vjp([x*x,math.log(x)],x,[2,3])",
            "1",
            vec![(9.5, 0.)],
        ),
        (
            "1",
            "2",
            "jvp([[x*x,x],[x,2*x]],[x],[1])",
            "array<array<1,2>,2>",
            vec![(4., 0.), (1., 0.), (1., 0.), (2., 0.)],
        ),
        (
            "complex<1>",
            "math.complex(3,4)",
            "jvp([x,math.conj(x)],[x],[math.complex(2,3)])",
            "array<complex<1>,2>",
            vec![(2., 3.), (2., -3.)],
        ),
        (
            "1",
            "2",
            "jvp([x,math.complex(x,x)],[x],[2])",
            "array<complex<1>,2>",
            vec![(2., 0.), (2., 2.)],
        ),
        (
            "m",
            "2[m]",
            "partial([x*x,1[m^2]*math.log(x/1[m])],wrt=x)",
            "array<m,2>",
            vec![(4., 0.), (0.5, 0.)],
        ),
    ] {
        let source = format!(
            "operator square(input x:scalar):scalar=x*x; operator action(input x:{input_type}):{output_type}={expression}; model M() {{variable y:{output_type};relation r {{y=action(x={argument});}}observable output:{output_type}=y;}}"
        );
        let (document, plan, result) =
            try_solve(&source).unwrap_or_else(|error| panic!("{expression}: {error}"));
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(
            observed.value().components().unwrap().collect::<Vec<_>>(),
            expected,
            "{expression}"
        );
    }
    for expression in [
        "partial([x*x,math.log(x)][0],wrt=x)",
        "jvp([x*x,math.log(x)],[x],[1])[0]",
        "vjp([x*x,math.log(x)],x,[1,0])",
    ] {
        let source = format!(
            "operator action(input x:1):1={expression}; model M() {{variable y:1;relation r {{y=action(x=-1);}}}}"
        );
        let Err(error) = try_solve(&source) else {
            panic!("discarded logarithm remains demanded");
        };
        assert!(
            error.contains("EQ0507") || error.contains("EQ0807"),
            "{expression}: {error}"
        );
    }
}
