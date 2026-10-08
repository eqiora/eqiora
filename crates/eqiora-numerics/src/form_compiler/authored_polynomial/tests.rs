use super::*;
#[test]
fn integral_sums_keep_exact_measures_and_traces() {
    let test = E::Test {
        field_ulid: "u".into(),
    };
    let trace = E::Trace {
        value: Box::new(test.clone()),
    };
    let integral = |domain: &str, value: E| E::Integrate {
        domain_ulid: domain.into(),
        integrand: Box::new(value),
    };
    let bulk = integral("body", test.clone());
    let surface = integral("right", trace.clone());
    let sum = E::Add {
        left: Box::new(bulk.clone()),
        right: Box::new(surface.clone()),
    };
    let mut context = Context {
        name: "eta",
        field: "u",
        dimensions: 2,
        remaining: 65536,
        symbols: BTreeMap::from([(
            "u".into(),
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
        )]),
    };
    // Integration domains are independent formal measures, even for equal
    // constant densities. A boundary restriction is a distinct input atom.
    let expected = Polynomial::atom(Atom::Measure("body".into()))
        .checked_mul(&Polynomial::atom(Atom::Test(vec![])))
        .unwrap()
        .checked_add(
            &Polynomial::atom(Atom::Measure("right".into()))
                .checked_mul(&Polynomial::atom(Atom::TraceTest(vec![])))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(context.integral(&sum), Some(expected));
    assert_ne!(context.integral(&bulk), context.integral(&sum));
    assert_ne!(
        context.integral(&surface),
        context.integral(&integral("left", trace))
    );
    assert_ne!(
        context.integral(&surface),
        context.integral(&integral("right", test))
    );
    let opposite_sides = E::Sub {
        left: Box::new(integral("left", E::Number { value: 1.0 })),
        right: Box::new(integral("right", E::Number { value: 1.0 })),
    };
    assert_ne!(
        context.integral(&opposite_sides),
        context.integral(&E::Number { value: 0.0 })
    );
    let product = E::Mul {
        left: Box::new(bulk),
        right: Box::new(surface),
    };
    assert!(context.integral(&product).is_none());
}

#[test]
fn binary_coefficients_are_exact_and_fail_closed_outside_portable_bounds() {
    for (value, numerator, denominator) in [
        (0.0, 0, 1),
        (-0.0, 0, 1),
        (0.5, 1, 2),
        (-3.25, -13, 4),
        (0.1, 3602879701896397, 36028797018963968),
    ] {
        assert_eq!(
            number(value),
            Some(ExactRational::new(numerator, denominator).unwrap())
        );
    }
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::MIN_POSITIVE,
        f64::MAX,
        -2.0_f64.powi(127),
    ] {
        assert!(number(value).is_none());
    }
}

fn scalar_context(domain: eqiora_core::ScalarDomain) -> Context<'static> {
    Context {
        name: "eta",
        field: "u",
        dimensions: 2,
        remaining: 65536,
        symbols: ["u", "q"]
            .into_iter()
            .map(|name| {
                (
                    name.into(),
                    ValueType::scalar(domain, eqiora_core::DimExponents::DIMENSIONLESS).unwrap(),
                )
            })
            .collect(),
    }
}
fn complex(real: f64, imag: f64) -> E {
    E::Complex {
        real: Box::new(E::Number { value: real }),
        imag: Box::new(E::Number { value: imag }),
    }
}
fn inner(left: E, right: E) -> E {
    E::Inner {
        left: Box::new(left),
        right: Box::new(right),
    }
}
fn mul(left: E, right: E) -> E {
    E::Mul {
        left: Box::new(left),
        right: Box::new(right),
    }
}
fn conj(value: E) -> E {
    E::Conjugate {
        value: Box::new(value),
    }
}

#[test]
fn exact_complex_channels_distinguish_inner_order_phase_and_test_role() {
    let mut context = scalar_context(eqiora_core::ScalarDomain::Complex);
    // (1-2i)(3+4i) = 11-2i; ordinary multiplication gives -5+10i.
    assert_eq!(
        context.scalar(&inner(complex(1., 2.), complex(3., 4.)), 0),
        context.scalar(&complex(11., -2.), 0)
    );
    assert_eq!(
        context.scalar(&mul(complex(1., 2.), complex(3., 4.)), 0),
        context.scalar(&complex(-5., 10.), 0)
    );
    let test = E::Test {
        field_ulid: "u".into(),
    };
    let trial = E::Field { ulid: "u".into() };
    let phase = E::Parameter { ulid: "q".into() };
    let value = inner(test.clone(), mul(phase.clone(), trial.clone()));
    let actual = context.scalar(&value, 0).unwrap();
    assert_eq!(
        Some(actual.clone()),
        context.scalar(
            &mul(conj(test.clone()), mul(phase.clone(), trial.clone())),
            0
        )
    );
    for wrong in [
        inner(mul(phase.clone(), trial.clone()), test.clone()),
        inner(test.clone(), mul(conj(phase), trial.clone())),
        mul(test.clone(), trial.clone()),
        inner(test.clone(), conj(trial.clone())),
    ] {
        assert_ne!(Some(actual.clone()), context.scalar(&wrong, 0));
    }
    let grad = |value| E::Gradient {
        value: Box::new(value),
    };
    let pairing = inner(grad(test.clone()), grad(trial.clone()));
    let expanded = E::Dot {
        left: Box::new(conj(grad(test))),
        right: Box::new(grad(trial)),
    };
    assert_eq!(
        context.scalar(&pairing, 0).unwrap(),
        context.scalar(&expanded, 0).unwrap()
    );
    let real_trial = E::Field { ulid: "u".into() };
    let mut real = scalar_context(eqiora_core::ScalarDomain::Real);
    assert_eq!(
        real.scalar(&conj(real_trial.clone()), 0),
        real.scalar(&real_trial, 0)
    );
    assert_ne!(
        context.scalar(&conj(real_trial.clone()), 0),
        context.scalar(&real_trial, 0)
    );
}

#[test]
fn finite_inner_uses_every_declared_component_not_the_spatial_dimension() {
    use eqiora_core::{DimExponents, FiniteBasis, Id, ScalarDomain, entity::kinds};
    let basis = FiniteBasis::new(Id::<kinds::FiniteSpace>::new(), 4).unwrap();
    let ty =
        ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
    let mut context = scalar_context(ScalarDomain::Complex);
    context.symbols.insert("u".into(), ty);
    let test = E::Test {
        field_ulid: "u".into(),
    };
    let trial = E::Field { ulid: "u".into() };
    let component = |value: E, index| E::Component {
        value: Box::new(value),
        indices: vec![index],
    };
    let mut sum = E::Number { value: 0. };
    let mut partial = None;
    for axis in 0..4 {
        if axis == 3 {
            partial = Some(sum.clone());
        }
        sum = E::Add {
            left: Box::new(sum),
            right: Box::new(mul(
                conj(component(test.clone(), axis)),
                component(trial.clone(), axis),
            )),
        };
    }
    let actual = context
        .scalar(&inner(test.clone(), trial.clone()), 0)
        .unwrap();
    assert_eq!(Some(actual.clone()), context.scalar(&sum, 0));
    assert_ne!(Some(actual), context.scalar(&partial.unwrap(), 0));
    assert!(
        context
            .scalar(&inner(test, E::Parameter { ulid: "q".into() }), 0)
            .is_none()
    );
    assert!(context.scalar(&component(trial, 4), 0).is_none());
}

#[test]
fn live_strong_expression_and_authored_projection_keep_symbolic_complex_phase() {
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    let compiled = eqiora_compiler::compile("phase.eqi", "model Phase() { domain body=box(0,1); parameter q:complex<1>=math.complex(3,4); variable u:complex<1> on body; relation balance on body { math.conj(q)*u=math.complex(11,-2); } }").unwrap().remove(0);
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let typed =
        crate::form_compiler::scalar::typed_relation(&program, symbols.get("balance").unwrap())
            .unwrap();
    let field = symbols.get("u").unwrap().ulid().to_string();
    let parameter = E::Parameter {
        ulid: symbols.get("q").unwrap().ulid().to_string(),
    };
    let trial = E::Field {
        ulid: field.clone(),
    };
    let mut context = Context {
        name: "eta",
        field: &field,
        dimensions: 1,
        remaining: 65536,
        symbols: symbol_types(&program),
    };
    let actual = context
        .source(&typed, typed.expression().roots()[0], &[], 0)
        .unwrap();
    let expected = E::Sub {
        left: Box::new(mul(conj(parameter.clone()), trial.clone())),
        right: Box::new(complex(11., -2.)),
    };
    assert_eq!(Some(actual.clone()), context.scalar(&expected, 0));
    let wrong = E::Sub {
        left: Box::new(mul(parameter, trial)),
        right: Box::new(complex(11., -2.)),
    };
    assert_ne!(Some(actual), context.scalar(&wrong, 0));
}
