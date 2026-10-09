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
        E::Number { value } => *value,
        E::Field { .. } => c,
        E::Parameter { ulid } if ulid == bulk => 2.0,
        E::Parameter { ulid } if ulid == gradient => 3.0,
        E::Direction { name, .. } if name == "eta" => 1.0 + x,
        E::Direction { name, .. } if name == "zeta" => 3.0 - x,
        E::Test { .. } => 1.0 + x,
        E::Trace { value } => eval(value),
        E::Neg { value } => -eval(value),
        E::Add { left, right } => eval(left) + eval(right),
        E::Mul { left, right } => eval(left) * eval(right),
        E::Sub { left, right } => eval(left) - eval(right),
        E::Div { left, right } => eval(left) / eval(right),
        E::Component { value, indices } if indices == &[0] => match value.as_ref() {
            E::Gradient { value } => match value.as_ref() {
                E::Field { .. } => cx,
                E::Direction { name, .. } if name == "eta" => 1.0,
                E::Direction { name, .. } if name == "zeta" => -1.0,
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

#[test]
fn surface_energy_retains_measure_trace_and_ordered_variations() {
    let geometry = geometry();
    for second in [false, true] {
        let first = "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))";
        let expression = if second {
            format!("variation({first},wrt=c,direction=zeta,holding=(bulk,gradient))")
        } else {
            first.to_owned()
        };
        let extra = if second {
            "test zeta:1 for c zero_on left;"
        } else {
            ""
        };
        let source = format!(
            r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J, parameter gradient:J
) {{
    variable c:1 on body;
    relation stationarity on body {{ bulk*c=0; }}
    observable energy:J=integral(bulk*trace(c)*trace(c)/2-gradient*trace(c),measure(right));
    form surface for stationarity {{
        test eta:1 for c zero_on left;
        {extra}
        {expression}=0;
    }}
}}
"#
        );
        let compiled = compile_source(&source, &geometry).unwrap();
        let form = compiled.authored_formulations().next().unwrap();
        let E::Variation {
            value, directions, ..
        } = &form.projection().equations()[0].1
        else {
            panic!("variation");
        };
        check_surface_replay(&compiled, &form.projection().equations()[0].1);
        assert_eq!(directions.len(), if second { 2 } else { 1 });
        let E::Integrate {
            domain_ulid,
            integrand,
        } = value.as_ref()
        else {
            panic!("surface measure");
        };
        assert_eq!(
            domain_ulid,
            &compiled.symbols().get("right").unwrap().ulid().to_string()
        );
        let bulk = compiled.symbols().get("bulk").unwrap().ulid().to_string();
        let gradient = compiled
            .symbols()
            .get("gradient")
            .unwrap()
            .ulid()
            .to_string();
        // The endpoint has unit zero-dimensional measure. Independently,
        // F=c^2-3c gives delta F=(2c-3)eta and delta² F=2 eta zeta.
        // c=4, eta=1+x, zeta=3-x, x=1 gives 10 and 8 respectively.
        let actual = density(integrand, 4.0, 0.0, 1.0, &bulk, &gradient);
        assert_eq!(actual, if second { 8.0 } else { 10.0 });
        if !second {
            let explicit = "integrate(right,(bulk*trace(c)-gradient)*trace(eta))";
            let explicit_source = source.replace(&expression, explicit);
            let explicit_form = compile_source(&explicit_source, &geometry).unwrap();
            let explicit_projection = explicit_form
                .authored_formulations()
                .next()
                .unwrap()
                .projection();
            let E::Integrate { integrand, .. } = &explicit_projection.equations()[0].1 else {
                panic!("explicit boundary integral");
            };
            let explicit_bulk = explicit_form
                .symbols()
                .get("bulk")
                .unwrap()
                .ulid()
                .to_string();
            let explicit_gradient = explicit_form
                .symbols()
                .get("gradient")
                .unwrap()
                .ulid()
                .to_string();
            assert_eq!(
                density(integrand, 4.0, 0.0, 1.0, &explicit_bulk, &explicit_gradient),
                actual
            );
            for invalid in [
                explicit_source.replace("trace(eta)", "eta"),
                explicit_source.replace("integrate(right,", "integrate(body,"),
                explicit_source.replace("trace(eta)", "trace(trace(eta))"),
                explicit_source.replace("trace(eta)", "integrate(right,trace(eta))"),
            ] {
                assert!(compile_source(&invalid, &geometry).is_err());
            }
        }
        let wire = std::str::from_utf8(form.projection().canonical_bytes()).unwrap();
        assert!(
            wire.contains("trace"),
            "the boundary restriction must remain explicit"
        );
        assert_eq!(
            eqiora_compiler::AuthoredFormulationProjection::decode(
                form.projection().canonical_bytes()
            )
            .unwrap(),
            *form.projection()
        );
        assert!(compile_source(&source.replace("trace(c)", "c"), &geometry).is_err());
        assert!(
            compile_source(
                &source.replace("observable energy:J=", "observable energy:J/m="),
                &geometry
            )
            .is_err()
        );
    }
}

// Infer the live density from the transaction, independently of the retained
// derivative. A canonical decode alone cannot establish this correspondence.
fn check_surface_replay(compiled: &eqiora_compiler::CompiledModel, variation: &E) {
    use eqiora_graph::Op;
    use eqiora_schema::kernel::typing::{
        ExpressionType, RootContract, SpatialSupport, TypedResidual,
    };
    use eqiora_schema::kernel::{KernelNode, SymbolRef};
    let nodes = compiled
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node, .. } => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    let functional = nodes
        .iter()
        .find_map(|node| match node {
            KernelNode::Observable(value) => Some(value),
            _ => None,
        })
        .unwrap();
    let volume = SpatialSupport::Volume {
        domain: compiled.symbols().get("body").unwrap(),
        dimensions: 1,
    };
    let boundary = SpatialSupport::Boundary {
        domain: compiled.symbols().get("right").unwrap(),
        parent: compiled.symbols().get("body").unwrap(),
        dimensions: 1,
    };
    let typed = TypedResidual::infer(
        functional.expression().clone(),
        Some(boundary.clone()),
        |on| (*boundary.domain() == on.erase()).then(|| boundary.clone()),
        RootContract::Observable,
        |symbol| {
            nodes
                .iter()
                .find_map(|node| match (symbol, node) {
                    (SymbolRef::Field(id), KernelNode::Field(field)) if id == field.id() => Some(
                        ExpressionType::new(field.value_type().clone(), Some(volume.clone())),
                    ),
                    (SymbolRef::Parameter(id), KernelNode::Parameter(parameter))
                        if id == parameter.id() =>
                    {
                        Some(ExpressionType::new(parameter.value_type().clone(), None))
                    }
                    _ => None,
                })
                .ok_or(())
        },
    )
    .unwrap();
    variation
        .check_functional_variation(&mut |id| {
            if id != functional.id() {
                return Err(eqiora_core::Diagnostic::error(
                    eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                    "variation energy differs from the exact live Observable",
                ));
            }
            Ok((functional.clone(), typed.clone()))
        })
        .unwrap();
    let mut wrong_boundary = variation.clone();
    let E::Variation { value, .. } = &mut wrong_boundary else {
        unreachable!()
    };
    let E::Integrate { domain_ulid, .. } = value.as_mut() else {
        unreachable!()
    };
    *domain_ulid = compiled.symbols().get("left").unwrap().ulid().to_string();
    assert!(
        wrong_boundary
            .check_functional_variation(&mut |id| {
                if id != functional.id() {
                    return Err(eqiora_core::Diagnostic::error(
                        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                        "variation energy differs from the exact live Observable",
                    ));
                }
                Ok((functional.clone(), typed.clone()))
            })
            .is_err()
    );
    let mut missing_trace = serde_json::to_value(variation).unwrap();
    fn remove_trace(value: &mut serde_json::Value) -> bool {
        if value.get("kind").and_then(serde_json::Value::as_str) == Some("trace") {
            *value = value.get("value").unwrap().clone();
            return true;
        }
        match value {
            serde_json::Value::Object(fields) => fields.values_mut().any(remove_trace),
            serde_json::Value::Array(items) => items.iter_mut().any(remove_trace),
            _ => false,
        }
    }
    assert!(remove_trace(&mut missing_trace));
    let missing_trace: E = serde_json::from_value(missing_trace).unwrap();
    assert!(
        missing_trace
            .check_functional_variation(&mut |id| {
                if id != functional.id() {
                    return Err(eqiora_core::Diagnostic::error(
                        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                        "variation energy differs from the exact live Observable",
                    ));
                }
                Ok((functional.clone(), typed.clone()))
            })
            .is_err()
    );
}

#[test]
fn composite_energy_retains_volume_surface_and_ordered_second_variation() {
    let geometry = geometry();
    for second in [false, true] {
        let first = "variation(total,wrt=c,direction=eta,holding=(bulk,gradient))";
        let expression = if second {
            format!("variation({first},wrt=c,direction=zeta,holding=(bulk,gradient))")
        } else {
            first.to_owned()
        };
        let extra = if second { "test zeta:1 for c;" } else { "" };
        let source = format!(
            r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J/m, parameter gradient:J*m
) {{
    variable c:1 on body;
    relation stationarity on body {{ bulk*c-div(gradient*grad(c))=0; }}
    relation left_natural on left {{ normal(grad(c))=0; }}
    relation right_natural on right {{ normal(grad(c))=0; }}
    observable energy:J=integral((bulk*c*c+gradient*contract(grad(c),grad(c),axes=((0,0),)))/2,measure(body));
    observable surface:J=integral(bulk*1[m]*trace(c)*trace(c)/2-gradient/1[m]*trace(c),measure(right));
    observable total:J=energy+surface;
    form stationary for stationarity {{
        test eta:1 for c;
        {extra}
        {expression}=0;
    }}
}}
"#
        );
        for combination in [
            "energy+surface",
            "energy-(-surface)",
            "energy+surface+energy-energy",
        ] {
            let source = source.replace("energy+surface;", &format!("{combination};"));
            let compiled = compile_source(&source, &geometry).unwrap();
            let form = compiled.authored_formulations().next().unwrap();
            let E::Variation {
                functional_ulid,
                value,
                ..
            } = &form.projection().equations()[0].1
            else {
                panic!("retained variation");
            };
            assert_eq!(
                functional_ulid,
                &compiled
                    .symbols()
                    .get("definition.total")
                    .unwrap()
                    .ulid()
                    .to_string()
            );
            let volume = compiled.symbols().get("body").unwrap().ulid().to_string();
            let boundary = compiled.symbols().get("right").unwrap().ulid().to_string();
            let bulk = compiled.symbols().get("bulk").unwrap().ulid().to_string();
            let gradient = compiled
                .symbols()
                .get("gradient")
                .unwrap()
                .ulid()
                .to_string();
            fn integral(e: &E, volume: &str, boundary: &str, bulk: &str, gradient: &str) -> f64 {
                let eval = |e| integral(e, volume, boundary, bulk, gradient);
                match e {
                    E::Add { left, right } => eval(left) + eval(right),
                    E::Sub { left, right } => eval(left) - eval(right),
                    E::Neg { value } => -eval(value),
                    E::Integrate {
                        domain_ulid,
                        integrand,
                    } => {
                        let at = |x: f64| density(integrand, x * x, 2.0 * x, x, bulk, gradient);
                        if domain_ulid == volume {
                            (at(0.0) + 4.0 * at(0.5) + at(1.0)) / 6.0
                        } else {
                            assert_eq!(domain_ulid, boundary);
                            at(1.0)
                        }
                    }
                    _ => panic!("unexpected composite variation term {e:?}"),
                }
            }
            // c=x², eta=1+x, zeta=3-x; a=2, k=3 on [0,1].
            // First bulk variation 25/6 and endpoint (2c-3)eta=-2 give 13/6.
            // Second bulk integral 2*eta*zeta+3*eta_x*zeta_x=13/3,
            // plus endpoint 2*eta*zeta=8, gives 37/3.
            let expected = if second { 37.0 / 3.0 } else { 13.0 / 6.0 };
            assert!(
                (integral(value, &volume, &boundary, &bulk, &gradient) - expected).abs() < 1e-12
            );
        }
        for (changed, message) in [
            (
                source.replace("holding=(bulk,gradient)", "holding=(bulk,)"),
                "holding must name exactly",
            ),
            (
                source.replace("energy+surface;", "energy*surface/1[J];"),
                "sum or difference",
            ),
        ] {
            let errors = compile_source(&changed, &geometry).unwrap_err();
            assert!(
                errors.iter().any(|error| error.message().contains(message)),
                "{errors:?}"
            );
        }
    }
}
