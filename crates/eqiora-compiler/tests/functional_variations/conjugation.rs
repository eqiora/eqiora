use super::*;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};

fn complex_form(expression: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    complex_form_with_fields(expression, "")
}

fn complex_form_with_fields(
    expression: &str,
    fields: &str,
) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let source = format!(
        r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J/m,
    parameter gradient:J*m,
    parameter phase:complex<1>=math.complex(0,1)
) {{
    {fields}
    variable c:complex<1> on body;
    relation stationarity on body {{ bulk*phase*c-div(gradient*grad(c))=0; }}
    relation left_value on left {{ trace(c)=0; }}
    relation right_value on right {{ trace(c)=0; }}
    form stationary for stationarity {{
        test eta:1 for c zero_on left,right;
        {expression}=0;
    }}
}}
"#
    );
    compile_source_with_values(
        &source,
        &geometry(),
        "model Values(){parameter bulk:1=2;parameter gradient:1=3;parameter phase:complex<1>=math.complex(0,1);}",
    )
}

#[test]
fn complex_scalar_domains_survive_authored_operators() {
    let compiled =
        complex_form("integrate(body,bulk*inner(eta,phase*c)+gradient*inner(grad(eta),grad(c)))")
            .unwrap_or_else(|errors| panic!("{errors:?}"));
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    assert_eq!(
        AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
        *form
    );
    // math.complex requires real inputs. Equal operands isolate scalar-domain
    // rejection from dimension, shape and support mismatch.
    for expression in [
        "c*2",
        "2*c",
        "c/2",
        "2/c",
        "c+2",
        "2-c",
        "c^2",
        "math.sin(c)",
        "math.conj(c)",
        "inner(eta,c)",
        "phase",
        "inner(grad(eta),grad(c))",
        "math.complex(1,2)",
    ] {
        let source = format!("integrate(body,math.complex({expression},{expression}))");
        let errors = complex_form(&source).unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .message()
                .contains("math.complex requires two equally dimensioned real scalars")),
            "{expression}: {errors:?}"
        );
    }
    let integral = "integrate(body,inner(eta,c))";
    assert!(
        complex_form(&format!("math.complex({integral},{integral})"))
            .unwrap_err()
            .iter()
            .any(|error| error
                .message()
                .contains("math.complex requires two equally dimensioned real scalars"))
    );
}

#[test]
fn complex_weak_form_classifies_test_and_trial_dependence_structurally() {
    for expression in [
        "inner(eta,c)",
        "math.conj(eta)*c",
        "inner(eta,math.sin(phase)*c)",
        "inner(eta,phase)",
        "inner(eta,c)/phase",
        "inner(eta,math.conj(math.conj(c)))",
    ] {
        complex_form(&format!("integrate(body,{expression})"))
            .unwrap_or_else(|errors| panic!("{expression}: {errors:?}"));
    }
    for expression in [
        "inner(c,eta)",
        "inner(math.conj(eta),c)",
        "math.conj(inner(eta,c))",
        "eta*c",
        "inner(eta,math.conj(c))",
        "inner(eta,c*c)",
        "inner(eta,math.sin(c))",
        "inner(eta,c)/c",
        "inner(eta,c)*inner(eta,c)",
        "inner(eta,c)+c",
    ] {
        let errors = complex_form(&format!("integrate(body,{expression})")).unwrap_err();
        assert!(errors.iter().any(|error| error.message().contains(
            "complex weak forms require conjugate-linear test dependence and linear trial dependence"
        )), "{expression}: {errors:?}");
    }
}

#[test]
fn authored_pairing_retains_conjugation_and_argument_order_in_current_wire() {
    let geometry = geometry();
    let plain = compile(
        "integrate(body,bulk*eta*c+gradient*dot(grad(eta),grad(c)))",
        &geometry,
    );
    let inner = compile(
        "integrate(body,bulk*inner(eta,c)+gradient*inner(grad(eta),grad(c)))",
        &geometry,
    );
    let conjugate = compile(
        "integrate(body,bulk*math.conj(eta)*c+gradient*inner(grad(eta),grad(c)))",
        &geometry,
    );
    assert_eq!(plain.transaction().ops(), inner.transaction().ops());
    assert_eq!(plain.transaction().ops(), conjugate.transaction().ops());
    let projection = |model: &CompiledModel| {
        model
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .clone()
    };
    let (plain, inner, conjugate) = (
        projection(&plain),
        projection(&inner),
        projection(&conjugate),
    );
    assert_ne!(plain.canonical_bytes(), inner.canonical_bytes());
    assert_ne!(inner.canonical_bytes(), conjugate.canonical_bytes());
    let [(_, E::Integrate { integrand, .. }, _)] = inner.equations() else {
        panic!("one retained integral");
    };
    let E::Add { left, .. } = integrand.as_ref() else {
        panic!("mass and gradient terms");
    };
    let E::Mul { right, .. } = left.as_ref() else {
        panic!("coefficient times inner product");
    };
    let E::Inner { left, right } = right.as_ref() else {
        panic!("explicit inner product");
    };
    let (E::Test { field_ulid }, E::Field { ulid }) = (left.as_ref(), right.as_ref()) else {
        panic!("the conjugated first argument is the test, second is the trial");
    };
    assert_eq!(field_ulid, ulid);
    for form in [&inner, &conjugate] {
        assert_eq!(
            AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
            *form
        );
        let text = std::str::from_utf8(form.canonical_bytes()).unwrap();
        assert!(text.contains("\"inner\""));
        assert!(
            AuthoredFormulationProjection::decode(
                text.replace("eqiora.authored-form/v13", "eqiora.authored-form/v9")
                    .as_bytes()
            )
            .is_err()
        );
    }
    assert!(
        std::str::from_utf8(conjugate.canonical_bytes())
            .unwrap()
            .contains("\"conjugate\"")
    );
    assert!(
        try_compile_profile(
            "integrate(body,bulk*inner(eta,grad(c)))",
            &geometry,
            "1",
            "J/m",
            "J*m",
            "1",
            ""
        )
        .is_err()
    );
}

fn replay_types(
    model: &CompiledModel,
) -> std::collections::BTreeMap<eqiora_core::RawId, eqiora_core::ValueType> {
    model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            eqiora_graph::Op::DefineKernelNode {
                node: eqiora_schema::kernel::KernelNode::Field(field),
            } => Some((field.id().erase(), field.value_type().clone())),
            eqiora_graph::Op::DefineKernelNode {
                node: eqiora_schema::kernel::KernelNode::Parameter(parameter),
            } => Some((parameter.id().erase(), parameter.value_type().clone())),
            _ => None,
        })
        .collect()
}

fn mutated_form(
    form: &AuthoredFormulationProjection,
    integrand: E,
) -> AuthoredFormulationProjection {
    let E::Integrate { integrand: old, .. } = &form.equations()[0].1 else {
        panic!("integral")
    };
    let old = format!("\"integrand\":{}", serde_json::to_string(old).unwrap());
    let new = format!(
        "\"integrand\":{}",
        serde_json::to_string(&integrand).unwrap()
    );
    let bytes = std::str::from_utf8(form.canonical_bytes())
        .unwrap()
        .replacen(&old, &new, 1);
    AuthoredFormulationProjection::decode(bytes.as_bytes()).unwrap()
}

#[test]
fn decoded_complex_dependence_rechecks_argument_order_and_live_types() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};
    let model = complex_form("integrate(body,inner(eta,c))").unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let types = replay_types(&model);
    let trial = projection.trial_ulids()[0].clone();
    let test = E::Test {
        field_ulid: trial.clone(),
    };
    let field = E::Field { ulid: trial };
    let conj = |value| E::Conjugate {
        value: Box::new(value),
    };
    let inner = |left, right| E::Inner {
        left: Box::new(left),
        right: Box::new(right),
    };
    let mul = |left, right| E::Mul {
        left: Box::new(left),
        right: Box::new(right),
    };
    for (expression, valid) in [
        (inner(test.clone(), field.clone()), true),
        (mul(conj(test.clone()), field.clone()), true),
        (inner(test.clone(), conj(conj(field.clone()))), true),
        (inner(field.clone(), test.clone()), false),
        (mul(test.clone(), field.clone()), false),
        (inner(test.clone(), conj(field.clone())), false),
        (
            inner(test.clone(), mul(field.clone(), field.clone())),
            false,
        ),
        (
            E::Div {
                left: Box::new(inner(test.clone(), field.clone())),
                right: Box::new(field.clone()),
            },
            false,
        ),
    ] {
        let decoded = mutated_form(projection, expression);
        assert_eq!(
            decoded
                .check_complex_dependence(&mut |id| Ok(types[&id].clone()))
                .is_ok(),
            valid,
            "{:?}",
            decoded.equations()
        );
    }
    let decoded = mutated_form(projection, mul(test, field));
    // Identical serialized multiplication is real-linear but fails when its live
    // argument domain is complex. The wire cannot assert that its inputs are real.
    let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    decoded
        .check_complex_dependence(&mut |_| Ok(real.clone()))
        .unwrap();
    assert!(
        decoded
            .check_complex_dependence(&mut |id| Ok(types[&id].clone()))
            .is_err()
    );
    let absent = eqiora_core::Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        "missing live definition",
    );
    assert!(
        decoded
            .check_complex_dependence(&mut |_| Err(absent.clone()))
            .is_err()
    );
}

#[test]
fn known_complex_fields_are_coefficients_and_real_arguments_still_require_linearity() {
    let model = complex_form_with_fields(
        "integrate(body,inner(eta,math.conj(known)*c))",
        "variable known:complex<1> on body;",
    )
    .unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let types = replay_types(&model);
    let decoded = AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap();
    decoded
        .check_complex_dependence(&mut |id| Ok(types[&id].clone()))
        .unwrap();
    assert_eq!(projection.trial_ulids().len(), 1);

    let model = complex_form("integrate(body,inner(eta,phase*c))").unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let mut types = replay_types(&model);
    // Keep phase complex while making both test and trial real in the live resolver.
    for (id, ty) in &mut types {
        if id.downcast::<eqiora_core::entity::kinds::Field>().is_some() {
            *ty = eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, ty.dimension())
                .unwrap();
        }
    }
    projection
        .check_complex_dependence(&mut |id| Ok(types[&id].clone()))
        .unwrap();
    let E::Integrate { integrand, .. } = &projection.equations()[0].1 else {
        panic!("integral")
    };
    let nonlinear = mutated_form(
        projection,
        E::Mul {
            left: integrand.clone(),
            right: integrand.clone(),
        },
    );
    assert!(
        nonlinear
            .check_complex_dependence(&mut |id| Ok(types[&id].clone()))
            .is_err()
    );
}

#[test]
fn body_parameter_coefficients_retain_closed_types_and_live_alias_dependencies() {
    let model = complex_form_with_fields(
        "integrate(body,gradient*inner(grad(eta),grad(c))+gradient*inner(eta,q*c))",
        "parameter q:complex<1/m^2>=math.complex(3,-1);",
    )
    .unwrap();
    let form = model.authored_formulations().next().unwrap().projection();
    let bytes = std::str::from_utf8(form.canonical_bytes()).unwrap();
    assert!(bytes.contains("\"kind\":\"complex\""));
    let dimension = eqiora_core::DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap();
    for numerator in [3, -1] {
        let expected = E::Rational {
            numerator,
            denominator: 1,
            dimension: dimension.exponents(),
        };
        assert!(bytes.contains(&serde_json::to_string(&expected).unwrap()));
    }
    let model = complex_form_with_fields(
        "integrate(body,bulk*inner(eta,q*c))",
        "parameter q:complex<1>=phase*2;",
    )
    .unwrap();
    let form = model.authored_formulations().next().unwrap().projection();
    let phase = model
        .symbols()
        .iter()
        .find(|(name, _)| name.ends_with(".phase") || *name == "phase")
        .unwrap()
        .1;
    assert!(
        std::str::from_utf8(form.canonical_bytes())
            .unwrap()
            .contains(&format!(
                "\"kind\":\"parameter\",\"ulid\":\"{}\"",
                phase.ulid()
            ))
    );
    let inline = complex_form_with_fields(
        "integrate(body,bulk*inner(eta,(phase*2)*c))",
        "parameter q:complex<1>=phase*2;",
    )
    .unwrap();
    assert_eq!(
        form.equations()[0].1,
        inline
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .equations()[0]
            .1
    );
}
