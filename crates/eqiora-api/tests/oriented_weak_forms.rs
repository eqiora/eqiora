use eqiora_api::ModelDocument;
use eqiora_compiler::StaticBindingValue;
use eqiora_geometry::GeometryGraph;
use std::collections::BTreeMap;

#[test]
fn weak_derivation_exposes_effective_equations_to_inspection_and_rendering() {
    let source = "model Pair() {
        domain body=box(0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        state u:1 on body in smooth; variable v:1 on body in smooth;
        initial {u=1;v=0;}
        relation first on body {derivative(u)-1[m^2/s]*div(grad(v))=0;}
        relation second on body {v+1[m^2]*div(grad(u))=0;}
        relation ul on left {normal(grad(u))=0;}
        relation ur on right {normal(grad(u))=0;}
        relation vl on left {normal(grad(v))=0;}
        relation vr on right {normal(grad(v))=0;}
        form paired for first,second {
            test wv:1 for v; test wu:1 for u;
            derive first using wu;
            derive second using wv;
        }
    }";
    let document = ModelDocument::compile("derived.eqi", source).unwrap();
    let form = document
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    assert!(form.requests_weak_derivation());
    assert_eq!(form.equations().len(), 2);
    assert_eq!(form.equation_test_fields().unwrap().len(), 2);
    let rendered = document
        .render_formulations(eqiora_lang::NotationProfile::Plain)
        .unwrap();
    assert_eq!(rendered.len(), 1);
    assert!(
        rendered[0].plain().contains("integral over"),
        "{}",
        rendered[0].plain()
    );
    let rechecked =
        eqiora_numerics::resolve_authored_spatial_formulation(document.program(), form).unwrap();
    assert_eq!(&rechecked, form);
}

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
            momentum: integrate(body,frobenius(grad(v),2*mu*symmetric_part(grad(u)))-p*div(v))=integrate(body,dot(v,grad(F)));
            continuity: integrate(body,q*div(u))=0;
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
    eqiora_numerics::resolve_authored_spatial_formulation(complete.program(), full).unwrap();
    let subset = source
        .replace("momentum,continuity", "momentum")
        .replace("test q:1 for p;", "")
        .replace("continuity: integrate(body,q*div(u))=0;", "")
        .replace("momentum: ", "");
    let partial = compile(&subset);
    let form = partial.authored_formulations().next().unwrap().projection();
    assert_eq!(form.trial_ulids().len(), 1);
    assert_eq!(form.equations().len(), 1);
    let error =
        eqiora_numerics::resolve_authored_spatial_formulation(partial.program(), form).unwrap_err();
    assert!(
        error
            .message()
            .contains("divergence constraint requires one unique gradient multiplier"),
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
            variable u:vector<{scalar},3> on body in h1;
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
