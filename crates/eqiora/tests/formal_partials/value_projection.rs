//! A scalar projection retains the original complete output's domain through calculus.
use super::typed_inputs::try_solve;

#[test]
fn indexed_values_and_formal_actions_preserve_complete_output_domains() {
    for expression in [
        "f(x=x,z=z)[0]",
        "partial(f(x=x,z=z)[0],wrt=x)",
        "partial(partial(f(x=x,z=z)[0],wrt=x),wrt=x)",
        "jvp(f(x=x,z=z)[0],[x],[1])",
        "vjp(f(x=x,z=z)[0],x,1)",
    ] {
        for local_operator in [false, true] {
            for other in [1, -1] {
                let declaration = if local_operator {
                    format!("operator action(input x:1,input z:array<1,2>):1={expression};")
                } else {
                    String::new()
                };
                let value = if local_operator {
                    "action(x=x,z=z)"
                } else {
                    expression
                };
                let source = format!(
                    r#"
operator f(input x:1,input z:array<1,2>):array<1,2>=x*math.log(component(z,0));
{declaration}
model M() {{
 parameter x:1=1; parameter z:array<1,2>=[1,{other}];
 variable y:1; relation r {{ y={value}; }} observable output:1=y;
}}"#
                );
                let result = try_solve(&source);
                if other > 0 {
                    let (document, plan, result) = result.unwrap();
                    let bytes = plan.model_artifact().canonical_json().unwrap();
                    let replay =
                        eqiora::artifact::ModelEnvelope::from_json(&bytes, Default::default())
                            .unwrap();
                    assert_eq!(replay.canonical_json().unwrap(), bytes);
                    let observed = result
                        .observe(
                            plan.model_artifact(),
                            document.aliases()["output"].downcast().unwrap(),
                            &Default::default(),
                            None,
                        )
                        .unwrap();
                    assert_eq!(observed.value().real_scalar_value().unwrap().value(), 0.);
                } else {
                    let Err(error) = result else {
                        panic!(
                            "invalid complete output was admitted: {expression}; local={local_operator}"
                        );
                    };
                    // Ordinary evaluation reports the primitive domain error;
                    // affine Plan admission can reject the retained domain guard.
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
fn indexed_formal_domains_do_not_require_finite_primal_polynomial_values() {
    for expression in [
        "partial(f(x=x,z=z)[0],wrt=x)",
        "partial(partial(f(x=x,z=z)[0],wrt=x),wrt=x)",
    ] {
        let source = format!(
            r#"
operator f(input x:1,input z:array<1,2>):array<1,2>=x*x*component(z,0);
model M() {{ parameter x:1=1e200; parameter z:array<1,2>=[0,1];
 variable y:1; relation r {{ y={expression}; }} observable output:1=y; }}"#
        );
        // The source polynomial overflows binary64, but its selected derivative
        // is exactly zero. Formal validity keeps mathematical domains, not an
        // unnecessary numerical evaluation of the original polynomial.
        let (document, plan, result) = try_solve(&source).unwrap();
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(observed.value().real_scalar_value().unwrap().value(), 0.);
    }
}

#[test]
fn indexed_output_domains_remain_inside_the_selected_branch() {
    for expression in [
        "partial(if k>0 then f(x=x,z=z)[0] else x*x,wrt=x)",
        "if k>0 then partial(f(x=x,z=z)[0],wrt=x) else partial(x*x,wrt=x)",
    ] {
        let source = format!(
            r#"
operator f(input x:1,input z:array<1,2>):array<1,2>=x*math.log(component(z,0));
operator action(input x:1,input z:array<1,2>,input k:1):1={expression};
model M() {{ parameter x:1=3; parameter z:array<1,2>=[1,-1]; parameter k:1=-1;
 variable y:1; relation r {{ y=action(x=x,z=z,k=k); }} observable output:1=y; }}"#
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
        assert_eq!(observed.value().real_scalar_value().unwrap().value(), 6.);
    }
}
