use super::{compile_source, geometry};
use eqiora_compiler::AuthoredFormExpressionV1 as E;

// This fixture evaluates only the scalar, polynomial wire vocabulary used below.
// It is not a numerical Form admission or a boundary-law checker.
fn density(e: &E, c: f64, cx: f64, x: f64, bulk: &str, gradient: &str) -> f64 {
    let eval = |e: &E| density(e, c, cx, x, bulk, gradient);
    match e {
        E::Rational {
            numerator,
            denominator,
            ..
        } => *numerator as f64 / *denominator as f64,
        E::Field { .. } => c,
        E::Parameter { ulid } if ulid == bulk => 2.0,
        E::Parameter { ulid } if ulid == gradient => 3.0,
        E::Direction { name, .. } if name == "eta" => 1.0 + x,
        E::Neg { value } => -eval(value),
        E::Add { left, right } => eval(left) + eval(right),
        E::Mul { left, right } => eval(left) * eval(right),
        E::Component { value, indices } if indices == &[0] => match value.as_ref() {
            E::Gradient { value } => match value.as_ref() {
                E::Field { .. } => cx,
                E::Direction { name, .. } if name == "eta" => 1.0,
                _ => panic!("unexpected differentiated input"),
            },
            _ => panic!("unexpected component input"),
        },
        _ => panic!("unexpected density node: {e:?}"),
    }
}

#[test]
fn unconstrained_direction_keeps_the_boundary_contribution() {
    let geometry = geometry();
    let source = r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J/m, parameter gradient:J*m
) {
    variable c:1 on body;
    relation stationarity on body { bulk*c-div(gradient*grad(c))=0; }
    relation left_natural on left { normal(grad(c))=0; }
    relation right_natural on right { normal(grad(c))=0; }
    observable energy:J=integral((bulk*c*c+gradient*contract(grad(c),grad(c),axes=((0,0),)))/2,measure(body));
    form stationary for stationarity {
        test eta:1 for c;
        variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))=0;
    }
}
"#;
    let compiled = compile_source(source, &geometry).unwrap();
    let form = compiled.authored_formulations().next().unwrap();
    assert!(form.projection().test_restrictions()[0].2.is_empty());
    let E::Variation { value, .. } = &form.projection().equations()[0].1 else {
        panic!("variation");
    };
    let E::Integrate { integrand, .. } = value.as_ref() else {
        panic!("fixed measure");
    };
    let bulk = compiled.symbols().get("bulk").unwrap().ulid().to_string();
    let gradient = compiled
        .symbols()
        .get("gradient")
        .unwrap()
        .ulid()
        .to_string();
    // For c=x^2, eta=1+x, a=2, k=3, the weak density is
    // 2*x^2*(1+x)+6*x. Simpson is exact for this cubic on [0,1].
    // The strong bulk term integrates to -47/6; the endpoint term is
    // [3*c_x*eta] = 12. This field deliberately violates the Neumann law:
    // merely declaring that law must not erase its boundary contribution.
    let at = |x: f64| density(integrand, x * x, 2.0 * x, x, &bulk, &gradient);
    let integral = (at(0.0) + 4.0 * at(0.5) + at(1.0)) / 6.0;
    assert!((integral - 25.0 / 6.0).abs() < 1e-12);
    assert!((integral - (-47.0 / 6.0) - 12.0).abs() < 1e-12);
    // For constant c=2 the actual normal gradient vanishes, even though
    // eta is nonzero at both endpoints. The density is 4*(1+x).
    for x in [0.0, 0.5, 1.0] {
        assert!(
            (density(integrand, 2.0, 0.0, x, &bulk, &gradient) - 4.0 * (1.0 + x)).abs() < 1e-12
        );
    }
    assert_eq!(
        eqiora_compiler::AuthoredFormulationProjection::decode(form.projection().canonical_bytes())
            .unwrap(),
        *form.projection()
    );
}
