use std::collections::BTreeMap;

use eqiora_api::ModelDocument;
use eqiora_compiler::{AuthoredFormExpressionV1, StaticBindingValue};
use eqiora_geometry::GeometryGraph;

#[test]
fn two_directions_of_one_field_reach_the_public_model_document() {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 1.0]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![interval.boundaries()[0].into()]),
                ("right".into(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    let source = r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter a:J/m
) {
    variable c:1 on body;
    relation balance on body { a*c=0; }
    relation left_value on left { trace(c)=0; }
    relation right_value on right { trace(c)=0; }
    observable energy:J=integral(a*c*c/2,measure(body));
    form hessian for balance {
        test eta:1 for c zero_on left,right;
        test zeta:1 for c zero_on left,right;
        variation(variation(energy,wrt=c,direction=eta,holding=(a,)),wrt=c,direction=zeta,holding=(a,))=0;
    }
}
"#;
    let values = eqiora_lang::parse("values.eqi", "model Values(){parameter a:1=2;}")
        .into_document()
        .unwrap();
    let eqiora_lang::Item::Parameter(parameter) = &values.models()[0].items()[0] else {
        panic!("parameter");
    };
    let body = geometry.entity_set("body").unwrap();
    let mut bindings = ["body", "left", "right"]
        .map(|name| {
            (
                name,
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: (name != "body").then_some(body),
                },
            )
        })
        .to_vec();
    bindings.push(("a", StaticBindingValue::Expression(parameter.value())));
    let module = eqiora_lang::Module::parse("energy.eqi", source).unwrap();
    let model = ModelDocument::compile_module(&module, Some("Energy"), &bindings).unwrap();
    let form = model.authored_formulations().next().unwrap();
    assert_eq!(form.projection().trial_ulids().len(), 1);
    assert_eq!(form.projection().test_restrictions().len(), 2);
    let AuthoredFormExpressionV1::Variation { directions, .. } =
        &form.projection().equations()[0].1
    else {
        panic!("retained variation");
    };
    assert_eq!(directions, &["eta", "zeta"]);
    check_live_variation_lineage(&model);
}

fn check_live_variation_lineage(model: &ModelDocument) {
    use eqiora_schema::kernel::KernelNode;
    let functional = model
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) => Some(value),
            _ => None,
        })
        .unwrap();
    let density = model.program().typed_observable(functional.id()).unwrap();
    let form = model.authored_formulations().next().unwrap();
    let expression = &form.projection().equations()[0].1;
    expression
        .check_functional_variation(functional, &density)
        .unwrap();
    for probe in ["energy", "holding", "body", "direction"] {
        let mut changed = expression.clone();
        let AuthoredFormExpressionV1::Variation {
            functional_ulid,
            directions,
            holding,
            value,
            ..
        } = &mut changed
        else {
            panic!("variation");
        };
        let expected = match probe {
            "energy" => {
                *functional_ulid = "01ARZ3NDEKTSV4RRFFQ69G5FAX".into();
                "exact live Observable"
            }
            "holding" => {
                holding.clear();
                "fixed bindings"
            }
            "body" => {
                **value = AuthoredFormExpressionV1::Number { value: 0.0 };
                "body differs"
            }
            "direction" => {
                directions[1] = directions[0].clone();
                "independent named directions"
            }
            _ => unreachable!(),
        };
        let error = changed
            .check_functional_variation(functional, &density)
            .unwrap_err();
        assert!(error.message().contains(expected), "{probe}: {error:?}");
    }
}
