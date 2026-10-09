use eqiora_api::ModelDocument;
use eqiora_compiler::StaticBindingValue;
use eqiora_geometry::GeometryGraph;
use std::collections::BTreeMap;

#[test]
fn single_vector_authoring_does_not_authenticate_an_incomplete_mixed_system() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0., 1.], [0., 1.]).unwrap();
    let names = ["left", "right", "bottom", "top"];
    let mut topology = BTreeMap::from([("body".into(), vec![rectangle.region().into()])]);
    for (name, boundary) in names.iter().zip(rectangle.boundaries()) {
        topology.insert((*name).into(), vec![boundary.into()]);
    }
    let geometry = graph.build(&rectangle, &topology).unwrap();
    let body = geometry.entity_set("body").unwrap();
    let members = names.map(|name| geometry.entity_set(name).unwrap());
    let mut bindings = vec![
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: body,
                parent: None,
            },
        ),
        (
            "surface",
            StaticBindingValue::CompleteExterior {
                geometry: &geometry,
                members: &members,
                parent: body,
            },
        ),
    ];
    let values = eqiora_lang::parse(
        "values.eqi",
        "model Values(){parameter mu:1=2.5;parameter load:1=3;parameter length:1=2;}",
    )
    .into_document()
    .unwrap();
    for (name, item) in ["mu", "load", "length"]
        .into_iter()
        .zip(values.models()[0].items())
    {
        let eqiora_lang::Item::Parameter(parameter) = item else {
            panic!("parameter");
        };
        bindings.push((name, StaticBindingValue::Expression(parameter.value())));
    }
    let source = r#"public component Flow(
        support body:volume(ambient_dimension=2),
        support surface:complete_exterior(parent=body),
        parameter mu:kg/m/s, parameter load:kg/m/s^2, parameter length:m
    ) {
        variable u:vector<m/s,2> on body in h1;
        variable p:kg/m/s^2 on body;
        variable F:kg/m/s^2 on body;
        relation force on body { F-load*coordinate(0)/length=0; }
        relation momentum on body { -div(2*mu*symmetric_part(grad(u))-isotropic_lift(p))-grad(F)=0; }
        relation continuity on body { div(u)=0; }
        relation fixed[face in surface] on face { trace(u)=0; }
        form weak for momentum,continuity {
            test v:1 for u zero_on surface;
            test q:1 for p;
            integrate(body,frobenius(grad(v),2*mu*symmetric_part(grad(u)))-p*div(v))=integrate(body,dot(v,grad(F)));
            integrate(body,q*div(u))=0;
        }
    }"#;
    let compile = |source: &str| {
        let module = eqiora_lang::Module::parse("mixed.eqi", source).unwrap();
        ModelDocument::compile_module(&module, Some("Flow"), &bindings)
            .unwrap_or_else(|errors| panic!("{errors:?}"))
    };
    let complete = compile(source);
    let full = complete
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    eqiora_numerics::check_authored_spatial_formulation(complete.program(), full).unwrap();
    let subset = source
        .replace("momentum,continuity", "momentum")
        .replace("test q:1 for p;", "")
        .replace("integrate(body,q*div(u))=0;", "");
    let partial = compile(&subset);
    let form = partial.authored_formulations().next().unwrap().projection();
    assert_eq!(form.trial_ulids().len(), 1);
    assert_eq!(form.equations().len(), 1);
    let error =
        eqiora_numerics::check_authored_spatial_formulation(partial.program(), form).unwrap_err();
    assert!(
        error
            .message()
            .contains("curl correspondence requires a spatial 3D Cartesian box"),
        "{error:?}"
    );
}

#[test]
fn oriented_form_rendering_retains_operators_and_exact_semantic_references() {
    for scalar in ["1", "complex<1>"] {
        for (integral, operator) in [
            (
                "integrate(body,dot(curl(v),curl(math.conj(math.conj(u)))))",
                "curl",
            ),
            (
                "integrate(face,dot(tangential_trace(v),trace(u)))",
                "tangential_trace",
            ),
            ("integrate(body,dot(cross(v,b),u))", "cross"),
        ] {
            let integral = if scalar == "complex<1>" {
                integral.replace("dot(", "inner(")
            } else {
                integral.to_owned()
            };
            let source = format!(
                r#"model M() {{
            domain body=box(0,1,0,1,0,1);
            domain face=boundary(body,axis=0,side=lower);
            variable u:vector<{scalar},3> on body;
            variable b:vector<1,3> on body;
            relation law on body {{ curl(curl(u))=u*0[1/m^2]; }}
            form weak for law {{ test v:1 for u zero_on face; {integral}=0; }}
        }}"#
            );
            let model = ModelDocument::compile("oriented-rendering.eqi", &source).unwrap();
            for profile in [
                eqiora_lang::NotationProfile::Plain,
                eqiora_lang::NotationProfile::Unicode,
                eqiora_lang::NotationProfile::Latex,
                eqiora_lang::NotationProfile::MathMl,
                eqiora_lang::NotationProfile::Speech,
            ] {
                let rendered = model.render_formulations(profile).unwrap().remove(0);
                assert!(rendered.plain().contains(operator), "{}", rendered.plain());
                for symbol in [
                    "u",
                    if operator == "tangential_trace" {
                        "face"
                    } else {
                        "body"
                    },
                ] {
                    assert!(
                        rendered
                            .references()
                            .iter()
                            .any(|r| r.graph_id() == Some(model.aliases()[symbol]))
                    );
                }
            }
        }
    }
}
