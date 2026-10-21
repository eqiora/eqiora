//! A held selector changes demand, not the independent coordinate being varied.
use super::*;

#[test]
fn held_branches_share_model_and_closed_operator_partial_actions() {
    let source = r#"
operator f(input x:1,input k:1):1=if k>0 then math.log(x) else math.log(-x);
operator dx(input x:1,input k:1):1=partial(if k>0 then math.log(x) else math.log(-x),wrt=x);
model M() {
 parameter x:1=2; parameter k:1=1; parameter y:1=3;
 let direct=if k>0 then math.log(x) else math.log(-x);
 relation r {
  partial(direct,wrt=x)=0.5;
  partial(partial(direct,wrt=x),wrt=x)=-0.25;
  partial(direct,wrt=y)=0;
  partial(f(x=x,k=k),wrt=x)=0.5;
  dx(x=x,k=k)=0.5;
  jvp(direct,[x],[3])=1.5;
  vjp(direct,x,4)=2;
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 7]);
    let negative = source
        .replace("x:1=2", "x:1=-2")
        .replace("k:1=1", "k:1=-1")
        .replace("=0.5", "=-0.5")
        .replace("=1.5", "=-1.5")
        .replace("=2;", "=-2;");
    assert_eq!(residuals(&negative), vec![0.; 7]);
}

#[test]
fn held_branches_retain_source_validity_for_zero_actions() {
    for expression in [
        "partial(f(x=x,k=k),wrt=y)",
        "jvp(f(x=x,k=k),[x],[0])",
        "vjp(f(x=x,k=k),x,0)",
    ] {
        let source = format!(
            r#"
operator f(input x:1,input k:1):1=if k>0 then math.log(x) else math.log(-x);
model M() {{ parameter x:1=-2; parameter k:1=1; parameter y:1=0;
 relation r {{ {expression}=0; }} }}"#
        );
        assert!(evaluated_values(&source).is_err(), "{expression}");
    }
}

#[test]
fn selecting_a_branch_is_not_a_globally_smooth_formal_operation() {
    for expression in [
        "partial(f(x=x,k=k),wrt=k)",
        "partial(partial(f(x=x,k=k),wrt=x),wrt=k)",
        "partial(partial(f(x=x,k=k),wrt=y),wrt=k)",
    ] {
        let source = format!(
            r#"
operator f(input x:1,input k:1):1=if k>0 then math.log(x) else math.log(-x);
model M() {{ parameter x:1=2; parameter k:1=1; parameter y:1=0;
 relation r {{ {expression}=0; }} }}"#
        );
        assert!(compile("held-branch.eqi", &source).is_err(), "{expression}");
    }
}

#[test]
fn held_branches_have_typed_zero_derivatives_inside_dimensioned_products() {
    for (selector, first, second) in [(1, 12, 6), (-1, 0, 0)] {
        let source = format!(
            r#"
operator f(input x:m,input k:1,input t:s):m^2*s=t*(if k>0 then x*x else 0[m^2]);
model M() {{ parameter x:m=2[m]; parameter t:s=3[s]; parameter k:1={selector};
 relation r {{
  partial(f(x=x,k=k,t=t),wrt=x)={first}[m*s];
  partial(partial(f(x=x,k=k,t=t),wrt=x),wrt=x)={second}[s];
 }} }}"#
        );
        assert_eq!(residuals(&source), vec![0.; 2]);
    }
    // The unselected branch and even its derivative overflow. A typed zero
    // cannot be manufactured by evaluating that derivative and subtracting it.
    let source = r#"
operator f(input x:m,input k:1):m^2=if k>0 then x*x else 0[m^2];
model M() { parameter x:m=1e308[m]; parameter k:1=-1;
 relation r { partial(f(x=x,k=k),wrt=x)=0[m]; } }"#;
    assert_eq!(residuals(source), vec![0.]);
}

#[test]
fn held_branch_quotient_curvature_retains_coefficient_units() {
    for (selector, first, second) in [(1, -0.25, 0.25), (-1, 0., 0.)] {
        let source = format!(
            r#"
operator f(input x:m,input k:1):m=if k>0 then 1[m^2]/x else 0[m];
model M() {{ parameter x:m=2[m]; parameter k:1={selector};
 relation r {{
  partial(f(x=x,k=k),wrt=x)={first};
  partial(partial(f(x=x,k=k),wrt=x),wrt=x)={second}[1/m];
 }} }}"#
        );
        assert_eq!(residuals(&source), vec![0.; 2]);
    }
}

#[test]
fn held_branches_keep_nested_and_short_circuit_predicates_lazy() {
    let source = r#"
model M() {
 parameter x:1=2; parameter k:1=-1; parameter h:1=-1; parameter y:1=0;
 let nested=if h>0 then (if math.log(k)>0 then math.log(x) else math.log(-x)) else x*x;
 let logical=if not (k>0) or math.log(k)>0 then math.log(x) else math.log(-x);
 relation r {
  partial(nested,wrt=x)=4;
  partial(partial(nested,wrt=x),wrt=x)=2;
  partial(nested,wrt=y)=0;
  partial(logical,wrt=x)=0.5;
  partial(partial(logical,wrt=x),wrt=x)=-0.25;
  partial(logical,wrt=y)=0;
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 6]);
}

#[test]
fn held_complex_branches_preserve_real_jacobian_and_adjoint_actions() {
    for (selector, forward, reverse) in [(1, (-28., 0.), (28., 70.)), (-1, (3., -4.), (7., 11.))] {
        for (expression, expected) in [
            ("jvp(f(z=z,k=k),[z],[d])", forward),
            ("vjp(f(z=z,k=k),z,seed)", reverse),
        ] {
            let source = format!(
                r#"
operator f(input z:complex<1>,input k:1):complex<1>=if k>0 then z*math.conj(z) else z;
model M() {{
 parameter z:complex<1>=math.complex(2,5); parameter d:complex<1>=math.complex(3,-4);
 parameter seed:complex<1>=math.complex(7,11); parameter k:1={selector};
 variable y:complex<1>; relation r {{ y={expression}; }} observable output:complex<1>=y;
}}"#
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
                vec![expected]
            );
        }
    }
}
