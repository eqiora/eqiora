//! A variation changes mathematical Formulation, not the authored Model.
use std::collections::BTreeMap;

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, GeometryGraph};

#[path = "functional_variations/conjugation.rs"]
mod conjugation;

fn geometry() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 1.0]).unwrap();
    graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![interval.boundaries()[0].into()]),
                ("right".to_owned(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap()
}

fn compile(expression: &str, geometry: &CanonicalGeometryV1) -> CompiledModel {
    compile_profile(expression, geometry, "1", "J/m", "J*m")
}

fn compile_profile(
    expression: &str,
    geometry: &CanonicalGeometryV1,
    field_unit: &str,
    bulk_unit: &str,
    gradient_unit: &str,
) -> CompiledModel {
    try_compile_profile(
        expression,
        geometry,
        field_unit,
        bulk_unit,
        gradient_unit,
        field_unit,
        "",
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"))
}

fn try_compile_profile(
    expression: &str,
    geometry: &CanonicalGeometryV1,
    field_unit: &str,
    bulk_unit: &str,
    gradient_unit: &str,
    test_unit: &str,
    additional_tests: &str,
) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let source = format!(
        r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:{bulk_unit},
    parameter gradient:{gradient_unit}
) {{
    variable c:{field_unit} on body;
    relation stationarity on body {{ bulk*c-div(gradient*grad(c))=0; }}
    relation left_value on left {{ trace(c)=0; }}
    relation right_value on right {{ trace(c)=0; }}
    observable energy:J=integral((bulk*c*c+gradient*contract(grad(c),grad(c),axes=((0,0),)))/2,measure(body));
    form stationary for stationarity {{
        test eta:{test_unit} for c zero_on left,right;
        {additional_tests}
        {expression}=0;
    }}
}}
"#
    );
    compile_source(&source, geometry)
}

fn compile_source(
    source: &str,
    geometry: &CanonicalGeometryV1,
) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let body = geometry.entity_set("body").unwrap();
    let mut bindings = ["body", "left", "right"]
        .map(|name| {
            (
                name,
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: (name != "body").then_some(body),
                },
            )
        })
        .to_vec();
    let values = eqiora_lang::parse(
        "values.eqi",
        "model Values(){parameter bulk:1=2;parameter gradient:1=3;}",
    )
    .into_document()
    .unwrap();
    for item in values.models()[0].items() {
        if let eqiora_lang::Item::Parameter(parameter) = item {
            bindings.push((
                parameter.name(),
                StaticBindingValue::Expression(parameter.value()),
            ));
        }
    }
    CompiledModel::compile_selected("variation.eqi", source, "Energy", &bindings)
}

#[test]
fn ordinary_gradient_energy_and_explicit_weak_form_compile() {
    let geometry = geometry();
    // Independently: delta F = integral(a*c*eta + k*c_x*eta_x) dx.
    // Integrating by parts gives (a*c-k*c_xx)*eta and [k*c_x*eta].
    // Only the explicit endpoint restriction makes that boundary term vanish.
    let explicit = compile(
        "integrate(body,bulk*c*eta+gradient*dot(grad(c),grad(eta)))",
        &geometry,
    );
    let form = explicit.authored_formulations().next().unwrap();
    assert_eq!(form.trials().len(), 1);
}

#[test]
fn authored_first_variation_uses_the_retained_energy() {
    let geometry = geometry();
    let derived = compile(
        "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))",
        &geometry,
    );
    let form = derived.authored_formulations().next().unwrap();
    assert_eq!(form.trials().len(), 1);
    let eqiora_compiler::AuthoredFormExpressionV1::Variation {
        directions,
        holding,
        wrt_ulid,
        ..
    } = &form.projection().equations()[0].1
    else {
        panic!("functional lineage must remain in the projection");
    };
    assert_eq!(directions, &["eta"]);
    assert_eq!(holding.len(), 2);
    assert_eq!(wrt_ulid, &form.trials()[0].ulid().to_string());
    assert_eq!(
        eqiora_compiler::AuthoredFormulationProjection::decode(form.projection().canonical_bytes())
            .unwrap(),
        *form.projection(),
    );
}

#[test]
fn variation_rejects_wrong_holding_and_direction_bindings() {
    let geometry = geometry();
    for (expression, diagnostic) in [
        (
            "variation(energy,wrt=c,direction=eta,holding=(bulk,))",
            "holding must name exactly",
        ),
        (
            "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient,bulk))",
            "repeated or varied",
        ),
        (
            "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient,c))",
            "repeated or varied",
        ),
        (
            "variation(energy,wrt=c,direction=bulk,holding=(bulk,gradient))",
            "explicitly declared test",
        ),
        (
            "variation(energy,wrt=bulk,direction=eta,holding=(gradient,))",
            "wrt must be a Field",
        ),
    ] {
        let errors =
            try_compile_profile(expression, &geometry, "1", "J/m", "J*m", "1", "").unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(diagnostic)),
            "{errors:?}"
        );
    }
    let errors = try_compile_profile(
        "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))",
        &geometry,
        "m",
        "J/m^3",
        "J/m",
        "1",
        "",
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("Field identity and dimension")),
        "{errors:?}"
    );
}

#[test]
fn authored_variation_direction_retains_the_field_dimension() {
    let geometry = geometry();
    // With c and eta in m, bulk*c*eta and gradient*c_x*eta_x both have
    // units J/m. Their directional integral has units J, not J/m.
    let derived = compile_profile(
        "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))",
        &geometry,
        "m",
        "J/m^3",
        "J/m",
    );
    assert_eq!(
        derived
            .authored_formulations()
            .next()
            .unwrap()
            .trials()
            .len(),
        1
    );
}

#[test]
fn nested_variations_retain_independent_direction_order() {
    let geometry = geometry();
    let mut identities = Vec::new();
    for (first, second) in [("eta", "zeta"), ("zeta", "eta")] {
        let expression = format!(
            "variation(variation(energy,wrt=c,direction={first},holding=(bulk,gradient)),wrt=c,direction={second},holding=(bulk,gradient))"
        );
        let compiled = try_compile_profile(
            &expression,
            &geometry,
            "1",
            "J/m",
            "J*m",
            "1",
            "test zeta:1 for c zero_on left,right;",
        )
        .unwrap();
        let form = compiled.authored_formulations().next().unwrap();
        assert_eq!(form.trials().len(), 1);
        assert_eq!(form.projection().test_restrictions().len(), 2);
        let eqiora_compiler::AuthoredFormExpressionV1::Variation { directions, .. } =
            &form.projection().equations()[0].1
        else {
            panic!("retained second variation");
        };
        assert_eq!(directions, &[first, second]);
        assert_eq!(
            eqiora_compiler::AuthoredFormulationProjection::decode(
                form.projection().canonical_bytes()
            )
            .unwrap(),
            *form.projection()
        );
        identities.push(form.source_identity().to_owned());
    }
    assert_ne!(identities[0], identities[1]);
}

#[test]
fn nested_variations_reject_dependent_directions_and_changed_holding() {
    let geometry = geometry();
    let first = "variation(energy,wrt=c,direction=eta,holding=(bulk,gradient))";
    for (expression, diagnostic) in [
        (
            format!("variation({first},wrt=c,direction=eta,holding=(bulk,gradient))"),
            "independent named direction",
        ),
        (
            format!("variation({first},wrt=c,direction=zeta,holding=(bulk,))"),
            "holding must name exactly",
        ),
        (
            format!(
                "variation(variation({first},wrt=c,direction=zeta,holding=(bulk,gradient)),wrt=c,direction=theta,holding=(bulk,gradient))"
            ),
            "only first and second",
        ),
    ] {
        let extra = "test zeta:1 for c zero_on left,right; test theta:1 for c zero_on left,right;";
        let errors =
            try_compile_profile(&expression, &geometry, "1", "J/m", "J*m", "1", extra).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(diagnostic)),
            "{errors:?}"
        );
    }
}

#[test]
fn explicit_weak_direction_retains_its_physical_dimension() {
    let geometry = geometry();
    let compiled = compile_profile(
        "integrate(body,bulk*c*eta+gradient*dot(grad(c),grad(eta)))",
        &geometry,
        "m",
        "J/m^3",
        "J/m",
    );
    let form = compiled.authored_formulations().next().unwrap();
    let expected = [(0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)];
    assert_eq!(form.projection().test_restrictions()[0].3, expected);
}

#[test]
fn second_variation_rejects_different_direction_restrictions() {
    let geometry = geometry();
    let expression = "variation(variation(energy,wrt=c,direction=eta,holding=(bulk,gradient)),wrt=c,direction=zeta,holding=(bulk,gradient))";
    let errors = try_compile_profile(
        expression,
        &geometry,
        "1",
        "J/m",
        "J*m",
        "1",
        "test zeta:1 for c zero_on left;",
    )
    .unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("identical dimension and boundary restrictions")),
        "{errors:?}"
    );

    let compiled = try_compile_profile(
        expression,
        &geometry,
        "1",
        "J/m",
        "J*m",
        "1",
        "test zeta:1 for c zero_on left,right;",
    )
    .unwrap();
    let form = compiled.authored_formulations().next().unwrap();
    let text = std::str::from_utf8(form.projection().canonical_bytes()).unwrap();
    let second = &form.projection().test_restrictions()[1];
    for dimension in [false, true] {
        let mut mutated = second.clone();
        if dimension {
            mutated.3[1].0 = 1;
        } else {
            mutated.2.pop();
        }
        let bytes = text.replace(
            &serde_json::to_string(second).unwrap(),
            &serde_json::to_string(&mutated).unwrap(),
        );
        let error =
            eqiora_compiler::AuthoredFormulationProjection::decode(bytes.as_bytes()).unwrap_err();
        assert!(
            error
                .message()
                .contains("identical dimension and boundary restrictions"),
            "{error:?}"
        );
    }
}

#[path = "functional_variations/boundary.rs"]
mod boundary;
