use super::super::*;
use eqiora_graph::{GraphStore, InMemoryGraphStore};

fn product(left: E, right: E) -> E {
    E::Mul {
        left: Box::new(left),
        right: Box::new(right),
    }
}
fn difference(left: E, right: E) -> E {
    E::Sub {
        left: Box::new(left),
        right: Box::new(right),
    }
}
fn component(value: E, indices: Vec<u32>) -> E {
    E::Component {
        value: Box::new(value),
        indices,
    }
}

#[test]
fn source_curl_energy_variation_matches_authored_action_and_independent_components() {
    for dimensions in [2, 3] {
        let bounds = vec!["0,1"; dimensions].join(",");
        let (square, pairing, energy_unit) = if dimensions == 3 {
            (
                "contract(curl(u),curl(u),axes=((0,0),))",
                "dot(curl(eta),curl(u))",
                "J",
            )
        } else {
            ("curl(u)*curl(u)", "curl(eta)*curl(u)", "J/m")
        };
        let source = format!(
            r#"model M() {{
            domain body=box({bounds});
            domain face=boundary(body,axis=0,side=lower);
            variable u:vector<m,{dimensions}> on body;
            relation law on body {{ curl(curl(u))*3[Pa]=u*0[Pa/m^2]; }}
            observable energy:{energy_unit}=integral(1.5[Pa]*{square},measure(body));
            form derived for law {{
                test eta:m for u zero_on face;
                variation(energy,wrt=u,direction=eta,holding=())=0;
            }}
        }}"#
        );
        let compiled =
            eqiora_compiler::CompiledModel::compile_selected("curl-energy.eqi", &source, "M", &[])
                .unwrap_or_else(|errors| panic!("{errors:?}"));
        let derived_projection = compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .clone();
        let authored_source = source.replace("form derived", "form authored").replace(
            "variation(energy,wrt=u,direction=eta,holding=())",
            &format!("integrate(body,3[Pa]*{pairing})"),
        );
        let authored_compiled = eqiora_compiler::CompiledModel::compile_selected(
            "curl-energy.eqi",
            &authored_source,
            "M",
            &[],
        )
        .unwrap_or_else(|errors| panic!("{errors:?}"));
        let authored_projection = authored_compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .clone();
        let (derived, authored) = (&derived_projection, &authored_projection);
        assert_eq!(derived.trial_ulids(), authored.trial_ulids());
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let (_, left, right) = &authored.equations()[0];
        assert!(matches_weak_residual(
            derived, &program, dimensions, left, right
        ));
        let field = derived.trial_ulids()[0].clone();
        let gradient = |test: bool, i, j| {
            component(
                E::Gradient {
                    value: Box::new(if test {
                        E::Test {
                            field_ulid: field.clone(),
                        }
                    } else {
                        E::Field {
                            ulid: field.clone(),
                        }
                    }),
                },
                vec![i, j],
            )
        };
        // Independently d(3/2 |curl u|²)[eta] is
        // 3 sum_{i<j}(u_j,i-u_i,j)(eta_j,i-eta_i,j).
        let mut sum = E::Number { value: 0.0 };
        for i in 0..dimensions as u32 {
            for j in i + 1..dimensions as u32 {
                let term = product(
                    E::Number { value: 3.0 },
                    product(
                        difference(gradient(false, j, i), gradient(false, i, j)),
                        difference(gradient(true, j, i), gradient(true, i, j)),
                    ),
                );
                sum = E::Add {
                    left: Box::new(sum),
                    right: Box::new(term),
                };
            }
        }
        let expanded = E::Integrate {
            domain_ulid: derived.domain_ulid().unwrap().into(),
            integrand: Box::new(sum.clone()),
        };
        assert!(matches_weak_residual(
            derived, &program, dimensions, &expanded, right
        ));
        assert!(matches_weak_residual(
            authored, &program, dimensions, &expanded, right
        ));
        let defect = product(gradient(false, 0, 1), gradient(true, 1, 0));
        let wrong = E::Integrate {
            domain_ulid: derived.domain_ulid().unwrap().into(),
            integrand: Box::new(E::Add {
                left: Box::new(sum),
                right: Box::new(defect),
            }),
        };
        assert!(!matches_weak_residual(
            derived, &program, dimensions, &wrong, right
        ));
        assert!(!matches_weak_residual(
            authored, &program, dimensions, &wrong, right
        ));
    }
}

#[test]
fn complex_cross_pairing_matches_independent_rows_without_implicit_conjugation() {
    let source = r#"model M() {
        domain body=box(0,1,0,1,0,1);
        variable u:vector<complex<1>,3> on body;
        variable b:vector<1,3> on body;
        relation law on body { u=u; }
        form action for law {
            test eta:1 for u;
            integrate(body,inner(cross(eta,b),u))=0;
        }
    }"#;
    let compiled =
        eqiora_compiler::CompiledModel::compile_selected("cross-action.eqi", source, "M", &[])
            .unwrap();
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let u = projection.trial_ulids()[0].clone();
    let b = symbols.get("b").unwrap().ulid().to_string();
    let eta = |i| {
        component(
            E::Test {
                field_ulid: u.clone(),
            },
            vec![i],
        )
    };
    let coefficient = |i| component(E::Field { ulid: b.clone() }, vec![i]);
    let trial = |i| component(E::Field { ulid: u.clone() }, vec![i]);
    let mut sum = E::Number { value: 0.0 };
    // The independent right-handed rows are (eta_y*b_z-eta_z*b_y,
    // eta_z*b_x-eta_x*b_z, eta_x*b_y-eta_y*b_x). Only inner conjugates.
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        let cross = difference(
            product(eta(j), coefficient(k)),
            product(eta(k), coefficient(j)),
        );
        let term = E::Inner {
            left: Box::new(cross),
            right: Box::new(trial(i)),
        };
        sum = E::Add {
            left: Box::new(sum),
            right: Box::new(term),
        };
    }
    let expanded = E::Integrate {
        domain_ulid: projection.domain_ulid().unwrap().into(),
        integrand: Box::new(sum.clone()),
    };
    let zero = E::Number { value: 0.0 };
    assert!(matches_weak_residual(
        &projection,
        &program,
        3,
        &expanded,
        &zero
    ));
    let conjugated = E::Conjugate {
        value: Box::new(sum),
    };
    let wrong = E::Integrate {
        domain_ulid: projection.domain_ulid().unwrap().into(),
        integrand: Box::new(conjugated),
    };
    assert!(!matches_weak_residual(
        &projection,
        &program,
        3,
        &wrong,
        &zero
    ));
}

#[test]
fn nominal_three_coordinates_cannot_cancel_inside_a_physical_cross_product() {
    let source =
        "space S=orthonormal(a,b,c); model M(){ variable u:coordinates<1,S>; relation r{u=u;} }";
    let compiled =
        eqiora_compiler::CompiledModel::compile_selected("nominal.eqi", source, "M", &[]).unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("u").unwrap().ulid().to_string();
    let u = E::Field {
        ulid: field.clone(),
    };
    let forged = E::Cross {
        left: Box::new(u.clone()),
        right: Box::new(u),
    };
    let mut context = Context {
        name: "eta",
        field: &field,
        dimensions: 3,
        remaining: 65536,
        supports: field_supports(&program),
        symbols: symbol_types(&program),
    };
    assert!(context.vector(&forged, 0, 0).is_none());
    let forged_curl = E::Curl {
        value: Box::new(E::Field {
            ulid: field.clone(),
        }),
    };
    assert!(context.vector(&forged_curl, 0, 0).is_none());
}

#[test]
fn foreign_volume_operands_cannot_disappear_through_cross_cancellation() {
    let source = r#"model M() {
        domain a=box(0,1,0,1,0,1);
        domain b=box(0,1,0,1,0,1);
        variable u:vector<1,3> on a;
        variable v:vector<1,3> on b;
        variable q:1 on b;
        relation r on a {u=u;}
        relation s on b {v=v;}
        relation t on b {q=q;}
    }"#;
    let compiled =
        eqiora_compiler::CompiledModel::compile_selected("foreign-cross.eqi", source, "M", &[])
            .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("u").unwrap().ulid().to_string();
    let symbol = |name| E::Field {
        ulid: symbols.get(name).unwrap().ulid().to_string(),
    };
    let cross_self = |value: E| E::Cross {
        left: Box::new(value.clone()),
        right: Box::new(value),
    };
    let mut context = Context {
        name: "eta",
        field: &field,
        dimensions: 3,
        remaining: 65536,
        symbols: symbol_types(&program),
        supports: field_supports(&program),
    };
    let zero = Polynomial::constant(ExactRational::integer(0));
    for axis in 0..3 {
        assert_eq!(
            context.vector(&cross_self(symbol("u")), axis, 0),
            Some(zero.clone())
        );
        assert!(context.vector(&cross_self(symbol("v")), axis, 0).is_none());
        // A foreign scalar multiplier is still foreign even when both complete
        // vector operands coincide and the cross product would be zero.
        assert!(
            context
                .vector(&cross_self(product(symbol("q"), symbol("u"))), axis, 0)
                .is_none()
        );
        assert!(
            context
                .vector(
                    &E::Curl {
                        value: Box::new(symbol("v"))
                    },
                    axis,
                    0
                )
                .is_none()
        );
    }
}
