//! A variation changes mathematical Formulation, not the authored Model.
use std::collections::BTreeMap;

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, GeometryGraph};

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
    let source = format!(
        r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J/m,
    parameter gradient:J*m
) {{
    variable c:1 on body;
    relation stationarity on body {{ bulk*c-div(gradient*grad(c))=0; }}
    relation left_value on left {{ trace(c)=0; }}
    relation right_value on right {{ trace(c)=0; }}
    observable energy:J=integral((bulk*c*c+gradient*contract(grad(c),grad(c),axes=((0,0),)))/2,measure(body));
    form stationary for stationarity {{
        test eta:1 for c zero_on left,right;
        {expression}=0;
    }}
}}
"#
    );
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
    CompiledModel::compile_selected("variation.eqi", &source, "Energy", &bindings)
        .unwrap_or_else(|errors| panic!("{errors:?}"))
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
