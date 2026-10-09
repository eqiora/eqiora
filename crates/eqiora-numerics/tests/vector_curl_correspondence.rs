use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::Diagnostic;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_numerics::check_authored_spatial_formulation;
use eqiora_sem::KernelProgram;

fn source(complex: bool) -> String {
    let mut source = String::from(
        "model M(){ domain body=box(0,1,0,1,0,1); variable u:vector<1,3> on body in smooth;",
    );
    let mut names = Vec::new();
    for axis in 0..3 {
        for side in ["lower", "upper"] {
            let name = format!("b{axis}{side}");
            source += &format!(
                "domain {name}=boundary(body,axis={axis},side={side}); relation fixed{axis}{side} on {name} {{trace(u)=0;}}"
            );
            names.push(name);
        }
    }
    source += &format!(
        "relation law on body {{curl(curl(u))+u*1[1/m^2]=u*0[1/m^2];}} form weak for law {{test eta:1 for u zero_on {}; integrate(body,dot(curl(eta),curl(u))+dot(eta,u)*1[1/m^2])=0;}}}}",
        names.join(",")
    );
    if complex {
        source = source
            .replace("vector<1,3>", "vector<complex<1>,3>")
            .replace("dot(", "inner(");
    }
    source
}
fn compile(source: &str) -> (KernelProgram, AuthoredFormulationProjection) {
    let compiled = eqiora_compiler::CompiledModel::compile_selected("curl.eqi", source, "M", &[])
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        form,
    )
}
fn check(source: &str) -> Result<(), Diagnostic> {
    let (program, form) = compile(source);
    check_authored_spatial_formulation(&program, &form)
}

#[test]
fn complete_real_and_complex_vector_green_identity_and_independent_component_rows() {
    for complex in [false, true] {
        let source = source(complex);
        check(&source).unwrap();
        let (program, form) = compile(&source);
        let expanded = expanded_form(&form, complex, false);
        check_authored_spatial_formulation(&program, &expanded).unwrap();
        let wrong_row = expanded_form(&form, complex, true);
        assert!(
            check_authored_spatial_formulation(&program, &wrong_row)
                .unwrap_err()
                .message()
                .contains("curl Green identity")
        );
        let pairing = if complex { "inner" } else { "dot" };
        for (broken, gate) in [
            (
                source.replace("curl(curl(u))+", "-curl(curl(u))+"),
                "curl Green identity",
            ),
            (
                source.replace(
                    &format!("{pairing}(curl(eta),curl(u))"),
                    &format!("-{pairing}(curl(eta),curl(u))"),
                ),
                "curl Green identity",
            ),
            (source.replace("zero_on b0lower,", "zero_on "), "zero_on"),
            (
                source.replace("trace(u)=0;", "trace(u)=trace(u);"),
                "homogeneous full trace",
            ),
            (
                source.replace("curl(curl(u))+", "curl(-curl(u))+"),
                "direct shared 3D curl-curl",
            ),
        ] {
            let error = check(&broken).unwrap_err();
            assert!(error.message().contains(gate), "{gate}: {error:?}");
        }
        if complex {
            for bad in [
                source.replace("inner(", "dot("),
                source.replace("inner(curl(eta),curl(u))", "inner(curl(u),curl(eta))"),
            ] {
                match eqiora_compiler::CompiledModel::compile_selected("curl.eqi", &bad, "M", &[]) {
                    Ok(_) => assert!(check(&bad).is_err()),
                    Err(errors) => assert!(
                        errors
                            .iter()
                            .any(|error| error.message().contains("conjugate")
                                || error.message().contains("dependence"))
                    ),
                }
            }
        }
    }
}

#[test]
fn boundary_completeness_and_source_identity_are_required() {
    let source = source(false);
    let missing = source.replace("relation fixed2upper on b2upper {trace(u)=0;}", "");
    assert!(
        check(&missing)
            .unwrap_err()
            .message()
            .contains("homogeneous full trace")
    );
    let duplicate = source.replace("relation law on body", "domain duplicate=boundary(body,axis=0,side=lower); relation duplicate_trace on duplicate {trace(u)=0;} relation law on body");
    assert!(
        check(&duplicate)
            .unwrap_err()
            .message()
            .contains("homogeneous full trace")
    );
    let (program, form) = compile(&source);
    let (foreign, _) = compile(&source.replace("box(0,1,0,1,0,1)", "box(0,2,0,1,0,1)"));
    assert!(check_authored_spatial_formulation(&foreign, &form).is_err());
    check_authored_spatial_formulation(&program, &form).unwrap();
}

// The exact component reference is independently assembled at the public wire
// boundary, where Component exists for functional variations as well as fields.
fn expanded_form(
    form: &AuthoredFormulationProjection,
    complex: bool,
    wrong: bool,
) -> AuthoredFormulationProjection {
    let field = &form.trial_ulids()[0];
    let gradient = |test, i, j| E::Component {
        value: Box::new(E::Gradient {
            value: Box::new(if test {
                E::Test {
                    field_ulid: field.clone(),
                }
            } else {
                E::Field {
                    ulid: field.clone(),
                }
            }),
        }),
        indices: vec![i, j],
    };
    let row = |test, i, j| E::Sub {
        left: Box::new(gradient(test, i, j)),
        right: Box::new(gradient(test, j, i)),
    };
    let pair = |left, right| {
        if complex {
            E::Inner {
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            E::Mul {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    };
    let mut sum = E::Number { value: 0.0 };
    // Right-handed curl rows: u_z,y-u_y,z; u_x,z-u_z,x; u_y,x-u_x,y.
    for (i, j) in [(2, 1), (0, 2), (1, 0)] {
        let test_row = if wrong && i == 2 {
            E::Add {
                left: Box::new(gradient(true, i, j)),
                right: Box::new(gradient(true, j, i)),
            }
        } else {
            row(true, i, j)
        };
        sum = E::Add {
            left: Box::new(sum),
            right: Box::new(pair(test_row, row(false, i, j))),
        };
    }
    let E::Integrate {
        domain_ulid,
        integrand,
    } = &form.equations()[0].1
    else {
        panic!("integral")
    };
    let E::Add {
        right: reaction, ..
    } = integrand.as_ref()
    else {
        panic!("reaction")
    };
    let left = E::Integrate {
        domain_ulid: domain_ulid.clone(),
        integrand: Box::new(E::Add {
            left: Box::new(sum),
            right: reaction.clone(),
        }),
    };
    replace_left(form, left)
}

fn replace_left(form: &AuthoredFormulationProjection, left: E) -> AuthoredFormulationProjection {
    let original = std::str::from_utf8(form.canonical_bytes()).unwrap();
    let old = serde_json::to_string(&form.equations()[0].1).unwrap();
    let new = serde_json::to_string(&left).unwrap();
    let bytes = original.replacen(&old, &new, 1);
    AuthoredFormulationProjection::decode(bytes.as_bytes()).unwrap()
}

#[test]
fn units_are_checked_before_polynomial_cancellation() {
    let (program, form) = compile(&source(false));
    let value = E::Mul {
        left: Box::new(form.equations()[0].1.clone()),
        right: Box::new(E::Rational {
            numerator: 1,
            denominator: 1,
            dimension: [(0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
        }),
    };
    let wrong = replace_left(&form, value);
    assert!(
        check_authored_spatial_formulation(&program, &wrong)
            .unwrap_err()
            .message()
            .contains("dimensions")
    );
}

#[test]
fn vector_energy_variation_replays_its_live_functional() {
    let source = source(false).replace("form weak for law", "observable energy:m=integral(0.5*contract(curl(u),curl(u),axes=((0,0),))+0.5*contract(u,u,axes=((0,0),))*1[1/m^2],measure(body)); form weak for law")
        .replace("integrate(body,dot(curl(eta),curl(u))+dot(eta,u)*1[1/m^2])", "variation(energy,wrt=u,direction=eta,holding=())");
    let (program, form) = compile(&source);
    check_authored_spatial_formulation(&program, &form).unwrap();
    let E::Variation {
        functional_ulid,
        wrt_ulid,
        directions,
        holding,
        ..
    } = &form.equations()[0].1
    else {
        panic!("variation");
    };
    let forged = replace_left(
        &form,
        E::Variation {
            functional_ulid: functional_ulid.clone(),
            wrt_ulid: wrt_ulid.clone(),
            directions: directions.clone(),
            holding: holding.clone(),
            value: Box::new(E::Number { value: 0.0 }),
        },
    );
    assert!(check_authored_spatial_formulation(&program, &forged).is_err());
}

#[test]
fn homogeneous_natural_curl_laws_discharge_their_exact_faces() {
    for complex in [false, true] {
        let original = source(complex);
        let natural = original
            .replace("trace(u)=0;", "tangential_trace(curl(u))=0;")
            .replace(
                " zero_on b0lower,b0upper,b1lower,b1upper,b2lower,b2upper",
                "",
            );
        check(&natural).unwrap();
        let hcurl = natural.replace("test eta:1 for u;", "test eta:1 for u in hcurl;");
        let (program, form) = compile(&hcurl);
        assert_eq!(form.test_restrictions()[0].4.as_deref(), Some("hcurl"));
        let replay = AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap();
        check_authored_spatial_formulation(&program, &replay).unwrap();
        let mixed = original
            .replace(
                "relation fixed0lower on b0lower {trace(u)=0;}",
                "relation fixed0lower on b0lower {tangential_trace(curl(u))=0;}",
            )
            .replace("zero_on b0lower,", "zero_on ");
        check(&mixed).unwrap();
        for wrong in [
            natural.replace("tangential_trace(curl(u))", "normal(curl(u))"),
            natural.replace("tangential_trace(curl(u))", "tangential_trace(u)"),
            natural.replace("tangential_trace(curl(u))", "tangential_trace(curl(-u))"),
        ] {
            let error = check(&wrong).unwrap_err();
            assert!(error.message().contains("tangential-curl law"), "{error:?}");
        }
        let unnecessary = natural.replace("test eta:1 for u;", "test eta:1 for u zero_on b0lower;");
        assert!(
            check(&unnecessary)
                .unwrap_err()
                .message()
                .contains("zero_on")
        );
        let missing = mixed.replace("zero_on b0upper,", "zero_on ");
        assert!(check(&missing).unwrap_err().message().contains("zero_on"));
    }
}
