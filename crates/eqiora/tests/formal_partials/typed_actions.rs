//! Non-holomorphic formal actions use the real dual pairing.

#[test]
fn actions_do_not_freeze_observable_dependencies() {
    for action in ["jvp(f,[x],[1])", "vjp(f,x,1)"] {
        let source = format!(
            "model M() {{ parameter x:1=3; observable f:1=x*x; variable y:1; relation r {{ y={action}; }} }}"
        );
        let Err(errors) = eqiora::api::ModelDocument::compile("observable-action.eqi", &source)
        else {
            panic!("a retained Observable cannot become a frozen formal input");
        };
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("Observable")),
            "{errors:?}"
        );
    }
}

#[test]
fn action_seeds_can_use_undifferentiated_observables() {
    for action in ["jvp(x*x,[x],[d])", "vjp(x*x,x,d)"] {
        let source = format!(
            "model M() {{ parameter x:1=3; observable d:1=x*x; variable y:1; relation r {{ y={action}; }} }}"
        );
        // The seed is held fixed by this action. Its own dependency would need
        // expansion only if a subsequent derivative selected the whole result.
        eqiora::api::ModelDocument::compile("observable-seed.eqi", &source).unwrap();
    }
}

#[test]
fn ignored_complex_blocks_keep_source_domain_checks() {
    for (reverse, component_only) in [(false, false), (true, false), (true, true)] {
        for local_operator in [false, true] {
            let ty = if reverse && !component_only {
                "array<complex<1>,2>"
            } else {
                "complex<1>"
            };
            let body = if component_only {
                "vjp(value(z=z,h=h),z,seed)[1]"
            } else if reverse {
                "vjp(value(z=z,h=h),z,seed)"
            } else {
                "jvp(value(z=z,h=h),[z],[dz])"
            };
            let expression = if local_operator {
                "action(z=z,dz=dz,seed=seed,h=h)"
            } else {
                body
            };
            let source = format!(
                r#"
operator value(input z:array<complex<1>,2>,input h:1):complex<1> = math.complex(math.log(h),4);
operator action(input z:array<complex<1>,2>,input dz:array<complex<1>,2>,input seed:complex<1>,input h:1):{ty} = {body};
model M() {{
 parameter z:array<complex<1>,2>=[math.complex(2,5),math.complex(7,-4)];
 parameter dz:array<complex<1>,2>=[math.complex(3,-4),math.complex(-2,3)];
 parameter seed:complex<1>=math.complex(7,11);
 parameter h:1=1;
 variable y:{ty}; relation r {{ y={expression}; }}
 observable output:{ty}=y;
}}
"#
            );
            let (document, plan, result) = super::typed_inputs::solve(&source);
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
                vec![(0., 0.); if reverse && !component_only { 2 } else { 1 }]
            );
            let invalid = source.replace("h:1=1;", "h:1=-1;");
            eqiora::api::ModelDocument::compile("guarded-action.eqi", &invalid).unwrap();
            assert!(
                super::typed_inputs::try_solve(&invalid).is_err(),
                "an ignored input must not erase log's source domain"
            );
        }
    }
}

#[test]
fn aliased_complex_call_arguments_keep_independent_formal_actions() {
    let source = r#"
operator forward(input p:complex<1>,input q:complex<1>,input d:complex<1>):complex<1> = jvp(p*math.conj(q),[p],[d]);
operator reverse(input p:complex<1>,input q:complex<1>,input seed:complex<1>):complex<1> = vjp(p*math.conj(q),p,seed);
model M() {
 parameter z:complex<1>=math.complex(2,5);
 parameter d:complex<1>=math.complex(3,-4);
 parameter seed:complex<1>=math.complex(7,11);
 variable f:complex<1>; variable r:complex<1>;
 relation law { f=forward(p=z,q=z,d=d); r=reverse(p=z,q=z,seed=seed); }
 observable forward_value:complex<1>=f;
 observable reverse_value:complex<1>=r;
}
"#;
    let (document, plan, result) = super::typed_inputs::solve(source);
    // Differentiate p*conj(q) before substituting p=q=z:
    // J_p*d=d*conj(q), J_p^T*seed=seed*q. Total derivatives differ.
    for (name, expected) in [
        ("forward_value", (-14., -23.)),
        ("reverse_value", (-41., 57.)),
    ] {
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()[name].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert_eq!(
            observed.value().components().unwrap().collect::<Vec<_>>(),
            vec![expected]
        );
    }
}

#[test]
fn indexed_complex_value_control() {
    check_indexed("z[0]*math.conj(z[1])", &[(-6., 43.)]);
}

#[test]
fn whole_identity_jvp_retains_its_complete_output() {
    check_indexed("jvp(z,[z],[dz])", &[(3., -4.), (-2., 3.)]);
}

#[test]
fn indexed_action_results_retain_their_local_axes() {
    check_indexed("jvp(z,[z],[dz])[1]", &[(-2., 3.)]);
    check_indexed("vjp(z[0]*math.conj(z[1]),z,seed)[1]", &[(69., 13.)]);
    // The second adjoint block is conj(seed)*z[0]; its direction is
    // conj(7+11i)*(3-4i)=-23-61i, independent of the other input component.
    check_indexed(
        "jvp(vjp(z[0]*math.conj(z[1]),z,seed),[z],[dz])[1]",
        &[(-23., -61.)],
    );
}

#[test]
fn indexed_complex_jvp_preserves_distinct_input_coordinates() {
    check_indexed("jvp(z[0]*math.conj(z[1]),[z],[dz])", &[(48., -32.)]);
}

#[test]
fn indexed_complex_vjp_returns_the_complete_selected_block() {
    check_indexed(
        "vjp(z[0]*math.conj(z[1]),z,seed)",
        &[(93., 49.), (69., 13.)],
    );
}

fn check_indexed(body: &str, expected: &[(f64, f64)]) {
    for local_operator in [false, true] {
        check_indexed_route(body, expected, local_operator);
    }
}

fn check_indexed_route(body: &str, expected: &[(f64, f64)], local_operator: bool) {
    let ty = if expected.len() == 1 {
        "complex<1>"
    } else {
        "array<complex<1>,2>"
    };
    let declaration = if local_operator {
        format!(
            "operator indexed(input z:array<complex<1>,2>,input dz:array<complex<1>,2>,input seed:complex<1>):{ty} = {body};"
        )
    } else {
        String::new()
    };
    let body = if local_operator {
        "indexed(z=z,dz=dz,seed=seed)"
    } else {
        body
    };
    let source = format!(
        r#"
{declaration}
model M() {{
 parameter z:array<complex<1>,2>=[math.complex(2,5),math.complex(7,-4)];
 parameter dz:array<complex<1>,2>=[math.complex(3,-4),math.complex(-2,3)];
 parameter seed:complex<1>=math.complex(7,11);
 variable y:{ty};
 relation r {{ y={body}; }}
 observable output:{ty}=y;
}}
"#
    );
    let (document, plan, result) = super::typed_inputs::solve(&source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    // For f=z0*conj(z1), Df[d]=d0*conj(z1)+z0*conj(d1),
    // and the real adjoint is (seed*z1, conj(seed)*z0).
    assert_eq!(
        observed.value().components().unwrap().collect::<Vec<_>>(),
        expected
    );
}

fn check(body: &str, expected: (i32, i32), local_operator: bool) {
    let declaration = if local_operator {
        format!(
            "operator response(input z:complex<1>,input dz:complex<1>,input seed:complex<1>):complex<1> = {body};"
        )
    } else {
        String::new()
    };
    let expression = if local_operator {
        "response(z=z,dz=dz,seed=seed)"
    } else {
        body
    };
    let source = format!(
        r#"
{declaration}
model M() {{
 parameter z:complex<1>=math.complex(2,5);
 parameter dz:complex<1>=math.complex(3,-4);
 parameter seed:complex<1>=math.complex(7,11);
 variable y:complex<1>;
 relation r {{ y={expression}; }}
 observable output:complex<1>=y;
}}
"#
    );
    let (document, plan, result) = super::typed_inputs::solve(&source);
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
        vec![(f64::from(expected.0), f64::from(expected.1))]
    );
}

#[test]
fn nonholomorphic_complex_value_control() {
    // f(a+ib)=(a²+b²-b)+i*a; f(2+5i)=24+2i.
    for local_operator in [false, true] {
        check(
            "z*math.conj(z)+math.complex(0,1)*z",
            (24, 2),
            local_operator,
        );
    }
}

#[test]
fn complex_formal_jvp_retains_both_real_coordinate_directions() {
    // Df(2,5)=[[4,9],[1,0]], so Df*(3,-4)=(-24,3).
    check(
        "jvp(z*math.conj(z)+math.complex(0,1)*z,[z],[dz])",
        (-24, 3),
        true,
    );
}

#[test]
fn complex_formal_vjp_uses_the_real_adjoint() {
    // Df(2,5)^T*(7,11)=(39,63). A holomorphic or unconjugated
    // transpose cannot satisfy this real-dual pairing for both directions.
    check(
        "vjp(z*math.conj(z)+math.complex(0,1)*z,z,seed)",
        (39, 63),
        true,
    );
}

#[test]
fn complex_model_jvp_shares_formal_action_meaning() {
    check(
        "jvp(z*math.conj(z)+math.complex(0,1)*z,[z],[dz])",
        (-24, 3),
        false,
    );
}

#[test]
fn complex_model_vjp_shares_the_real_adjoint() {
    check(
        "vjp(z*math.conj(z)+math.complex(0,1)*z,z,seed)",
        (39, 63),
        false,
    );
}

#[test]
fn nested_complex_actions_retain_independent_bound_coordinates() {
    for local_operator in [true, false] {
        // Hessian of Re(f) is 2I; Im(f) is linear. D²f[d,(-2,3)]=-36.
        check(
            "jvp(jvp(z*math.conj(z)+math.complex(0,1)*z,[z],[dz]),[z],[math.complex(-2,3)])",
            (-36, 0),
            local_operator,
        );
        // D(Df^T seed)[dz] = 2*Re(seed)*dz = 42-56i.
        check(
            "jvp(vjp(z*math.conj(z)+math.complex(0,1)*z,z,seed),[z],[dz])",
            (42, -56),
            local_operator,
        );
    }
}

#[test]
fn shaped_complex_jvp_projects_the_whole_direction() {
    check_shaped_jvp(true);
}

#[test]
fn shaped_model_jvp_projects_the_whole_direction() {
    check_shaped_jvp(false);
}

fn check_shaped_jvp(local_operator: bool) {
    let source = r#"
operator value(input z:array<complex<1>,2>):array<complex<1>,2> =
 component(z,0)*math.conj(component(z,0))+math.complex(0,1)*component(z,0);
operator response(input z:array<complex<1>,2>, input dz:array<complex<1>,2>):array<complex<1>,2> =
 jvp(value(z=z),[z],[dz]);
model M() {
 parameter z:array<complex<1>,2>=[math.complex(2,5),math.complex(7,-4)];
 parameter dz:array<complex<1>,2>=[math.complex(3,-4),math.complex(-2,3)];
 variable y:array<complex<1>,2>;
 relation r { y=response(z=z,dz=dz); }
 observable output:array<complex<1>,2>=y;
}
"#;
    let source = if local_operator {
        source.to_owned()
    } else {
        source.replace("y=response(z=z,dz=dz);", "y=jvp(value(z=z),[z],[dz]);")
    };
    let (document, plan, result) = super::typed_inputs::solve(&source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    // Df(a,b)(u,v)=(2*a*u+(2*b-1)*v,u), independently for each component.
    assert_eq!(
        observed.value().components().unwrap().collect::<Vec<_>>(),
        vec![(-24., 3.), (-55., -2.)]
    );
}

#[test]
fn shaped_vjp_contracts_every_output_component() {
    check_shaped_vjp(true);
}

#[test]
fn shaped_model_vjp_contracts_every_output_component() {
    check_shaped_vjp(false);
}

fn check_shaped_vjp(local_operator: bool) {
    let source = r#"
operator response(input z:array<complex<1>,2>):array<array<complex<1>,3>,2> =
 component(z,0)+math.conj(component(z,0));
operator pull(input z:array<complex<1>,2>, input seed:array<array<complex<1>,3>,2>):array<complex<1>,2> =
 vjp(response(z=z),z,seed);
model M() {
 parameter z:array<complex<1>,2>=[math.complex(2,5),math.complex(7,-4)];
 parameter seed:array<array<complex<1>,3>,2>=[
  [math.complex(1,2),math.complex(3,5),math.complex(7,11)],
  [math.complex(13,17),math.complex(19,23),math.complex(29,31)]];
 variable y:array<complex<1>,2>;
 relation r { y=pull(z=z,seed=seed); }
 observable output:array<complex<1>,2>=y;
}
"#;
    let source = if local_operator {
        source.to_owned()
    } else {
        source.replace("y=pull(z=z,seed=seed);", "y=vjp(response(z=z),z,seed);")
    };
    let (document, plan, result) = super::typed_inputs::solve(&source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    // Each of three outputs is 2*Re(z_i); the real adjoint sums all three seeds.
    assert_eq!(
        observed.value().components().unwrap().collect::<Vec<_>>(),
        vec![(22., 0.), (122., 0.)]
    );
}

#[test]
fn shaped_vjp_rebinds_rectangular_output_axes() {
    let source = r#"
operator response(input z:array<array<complex<1>,3>,2>):array<array<complex<1>,2>,3> =
 math.conj(component(z,1,0));
operator pull(input z:array<array<complex<1>,3>,2>, input seed:array<array<complex<1>,2>,3>):array<array<complex<1>,3>,2> =
 vjp(response(z=z),z,seed);
model M() {
 parameter z:array<array<complex<1>,3>,2>=[
  [math.complex(2,5),math.complex(7,-4),math.complex(3,1)],
  [math.complex(11,-2),math.complex(13,6),math.complex(-3,-9)]];
 parameter seed:array<array<complex<1>,2>,3>=[
  [math.complex(1,2),math.complex(3,5)],
  [math.complex(7,11),math.complex(13,17)],
  [math.complex(19,23),math.complex(29,31)]];
 variable y:array<array<complex<1>,3>,2>;
 relation r { y=pull(z=z,seed=seed); }
 observable output:array<array<complex<1>,3>,2>=y;
}
"#;
    let (document, plan, result) = super::typed_inputs::solve(source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    // Re <S, conj(dZ)^T> = Re <conj(S)^T, dZ> for these rectangular arrays.
    assert_eq!(
        observed.value().components().unwrap().collect::<Vec<_>>(),
        vec![
            (1., -2.),
            (7., -11.),
            (19., -23.),
            (3., -5.),
            (13., -17.),
            (29., -31.)
        ]
    );
}
