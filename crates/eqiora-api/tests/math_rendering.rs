use eqiora_api::ModelDocument;
use eqiora_lang::NotationProfile;
use eqiora_schema::kernel::KernelNode;

const PROFILES: [NotationProfile; 5] = [
    NotationProfile::Latex,
    NotationProfile::MathMl,
    NotationProfile::Unicode,
    NotationProfile::Plain,
    NotationProfile::Speech,
];

fn equation(document: &ModelDocument, profile: NotationProfile) -> eqiora_api::MathRendering {
    let id = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) => Some(value.id().into()),
            _ => None,
        })
        .unwrap();
    document.render_equations(id, profile).unwrap().remove(0)
}

#[test]
fn five_profiles_preserve_equation_order_grouping_and_exact_references() {
    let document = ModelDocument::compile("math.eqi", "model M(){parameter a @{a}:1=2;parameter b @{b}:1=3;variable x @{\\hat{x}_i}:1;relation law{x=(a-b)/(a+b);}}").unwrap();
    let plain = equation(&document, NotationProfile::Plain);
    assert_eq!(plain.plain(), "(hat(x)_{i}) = (((a) - (b)) / ((a) + (b)))");
    assert!(plain.speech().contains("minus"));
    assert!(plain.speech().contains("divided by"));
    for profile in PROFILES {
        let rendered = equation(&document, profile);
        assert_eq!(rendered.plain(), plain.plain());
        assert_eq!(rendered.speech(), plain.speech());
        assert_eq!(rendered.references(), plain.references());
        assert_eq!(rendered.references().len(), 3);
        assert!(!rendered.used_fallback());
        if profile == NotationProfile::MathMl {
            assert!(rendered.text().starts_with("<math xmlns="));
            assert!(rendered.text().contains("<mfrac>"));
            assert!(rendered.text().contains("<mover accent=\"true\">"));
        }
    }
}

#[test]
fn operator_fallback_preserves_definition_identity_without_name_inference() {
    let source = "operator magnitude(input x:V):V=math.abs(x);model M(){parameter x @{x}:V=-3;variable y:V;relation value{y=magnitude(x=x);}}";
    let document = ModelDocument::compile("math.eqi", source).unwrap();
    let renamed =
        ModelDocument::compile("math.eqi", &source.replace("magnitude", "gradient")).unwrap();
    let first = equation(&document, NotationProfile::Plain);
    let second = equation(&renamed, NotationProfile::Plain);
    let first_operator = first
        .references()
        .iter()
        .find_map(|reference| reference.operator())
        .unwrap();
    let second_operator = second
        .references()
        .iter()
        .find_map(|reference| reference.operator())
        .unwrap();
    assert_eq!(first_operator, second_operator);
    assert!(
        first
            .text()
            .contains(&format!("operator {first_operator}(x)"))
    );
    assert!(
        second
            .text()
            .contains(&format!("operator {first_operator}(x)"))
    );
    assert!(!second.text().contains("gradient"));
    assert!(
        !equation(&renamed, NotationProfile::Latex)
            .text()
            .contains("\\nabla")
    );
    let other = ModelDocument::compile("math.eqi", &source.replace("math.abs(x)", "-x")).unwrap();
    let other = equation(&other, NotationProfile::Plain);
    assert_ne!(
        first_operator,
        other
            .references()
            .iter()
            .find_map(|reference| reference.operator())
            .unwrap()
    );
}

#[test]
fn hostile_notation_is_rejected_before_any_renderer() {
    for notation in [
        r"\href{javascript:alert(1)}{x}",
        r"\input{secret}",
        r"<script>",
        r"x`onclick`",
        r"\htmlClass{unsafe}{x}",
    ] {
        assert!(
            ModelDocument::compile(
                "hostile.eqi",
                &format!("model M(){{variable x @{{{notation}}}:1;relation law{{x=0;}}}}")
            )
            .is_err(),
            "{notation}"
        );
    }
}

#[test]
fn borrowed_graph_targets_keep_every_declaration_without_selecting_an_alias() {
    let source = "component Reader(state value @{v}:1){relation law{value=0;}} model M(){state x @{x}:1;instance left:Reader(value=x);instance right:Reader(value=x);}";
    let document = ModelDocument::compile("borrow.eqi", source).unwrap();
    let target = document
        .notation()
        .iter()
        .find(|entry| entry.selector() == "x")
        .unwrap()
        .graph_id()
        .unwrap();
    let expected = document
        .notation()
        .iter()
        .map(|entry| entry.identity())
        .collect::<Vec<_>>();
    for profile in PROFILES {
        let rendered = equation(&document, profile);
        assert_eq!(rendered.references().len(), 1);
        assert_eq!(rendered.references()[0].graph_id(), Some(target));
        assert_eq!(rendered.references()[0].declarations(), expected);
        let fallback = eqiora_lang::NotationLabel::identifier(&format!("{target}_value")).unwrap();
        assert!(rendered.text().contains(&fallback.render(profile)));
    }
}

#[test]
fn discrete_state_operators_and_intrinsic_scripts_are_not_expression_powers() {
    let source = "model M(){clock tick=periodic(1[s]); state x @{x^{\\prime}}:1 at tick; initial{x=1;} relation update at tick{next(x)=pre(x)^2;}}";
    let document = ModelDocument::compile("state.eqi", source).unwrap();
    let relation = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(value) if !value.is_initial() => Some(value.id().into()),
            _ => None,
        })
        .unwrap();
    let latex = document
        .render_equations(relation, NotationProfile::Latex)
        .unwrap();
    assert!(latex[0].text().contains("\\prime"));
    assert!(latex[0].text().contains("^{2}"));
    let speech = document
        .render_equations(relation, NotationProfile::Speech)
        .unwrap();
    assert!(speech[0].text().contains("superscript prime end script"));
    assert!(speech[0].text().contains("raised to 2 end power"));
    assert!(speech[0].text().contains("next"));
    assert!(speech[0].text().contains("pre"));
}

#[test]
fn checked_type_presentation_keeps_rational_dimensions_and_axis_roles() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueShape, ValueType};
    let dimension =
        DimExponents::from_rationals([(0, 1), (1, 2), (-3, 2), (0, 1), (0, 1), (0, 1), (0, 1)])
            .unwrap();
    let scalar = ValueType::scalar(ScalarDomain::Complex, dimension).unwrap();
    let channel = scalar.clone().array(2).unwrap();
    let vector = ValueType::shaped(
        ScalarDomain::Complex,
        dimension,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    for profile in PROFILES {
        let channel = eqiora_api::MathRendering::value_type(&channel, profile).unwrap();
        let vector = eqiora_api::MathRendering::value_type(&vector, profile).unwrap();
        assert_ne!(channel.text(), vector.text());
        assert!(
            channel
                .plain()
                .contains("complex channel axis 2 invariant frame")
        );
        assert!(
            vector
                .plain()
                .contains("complex spatial axis 2 Cartesian frame")
        );
        assert!(
            vector
                .plain()
                .contains("m to power 1 over 2 times s to power -3 over 2")
        );
    }
}

#[test]
fn continuum_forms_keep_gradient_inner_test_and_support_nodes() {
    let source = "public component Diffusion(support body:volume(ambient_dimension=2),support surface:boundary(parent=body),parameter k @{k}:1,parameter f @{f}:1/m^2){variable u @{u}:1 on body;relation law on body{-div(k*grad(u))=f;}form weak for law { test w: 1 for u zero_on surface;integrate(body,dot(grad(w),k*grad(u)))=integrate(body,w*f);}}";
    let geometry = eqiora_geometry::CanonicalGeometryV1::from_circular_hole_named_roles(
        [[0.0, 2.0], [0.0, 1.0]],
        [0.5, 0.5],
        0.1,
        1e-12,
        "body",
        "left",
        "right",
        "walls",
        "walls",
        "hole",
    )
    .unwrap();
    let scalar = |dimension, value| {
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension).unwrap(),
            value,
        )
        .unwrap()
    };
    let k = scalar(eqiora_core::DimExponents::DIMENSIONLESS, 1.0);
    let f = scalar(
        eqiora_core::DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
        2.0,
    );
    let bindings = [
        (
            "body",
            eqiora_compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("body").unwrap(),
                parent: None,
            },
        ),
        (
            "surface",
            eqiora_compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("left").unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ),
        ("k", eqiora_compiler::StaticBindingValue::Value(&k)),
        ("f", eqiora_compiler::StaticBindingValue::Value(&f)),
    ];
    let document =
        ModelDocument::compile_selected("form.eqi", source, "Diffusion", &bindings).unwrap();
    for profile in PROFILES {
        let forms = document.render_formulations(profile).unwrap();
        assert_eq!(forms.len(), 1);
        let form = &forms[0];
        assert!(form.speech().contains("integral over"));
        assert!(form.speech().contains("real inner product of first"));
        assert!(form.speech().contains("gradient"));
        assert!(form.speech().contains("test of u"));
        assert!(
            form.references()
                .iter()
                .any(|reference| reference.graph_id().is_some() && reference.role().is_none())
        );
        if profile == NotationProfile::Latex {
            assert!(form.text().contains("\\nabla"));
            assert!(form.text().contains("\\int_"));
            assert!(form.text().contains("\\left\\langle"));
        }
    }
}

#[test]
fn composed_circuit_equations_keep_exact_port_roles_and_instance_labels() {
    let source = "connector Pin{across voltage:V;through current:A;} component Resistor(parameter resistance @{R}:V/A,port p @{v}:Pin,port n @{v}:Pin){relation law{p.voltage-n.voltage=resistance*p.current;p.current+n.current=0;}}model Circuit(){instance left:Resistor(resistance=2[V/A]);instance right:Resistor(resistance=3[V/A]);connect left.p,right.p;connect left.n,right.n;}";
    let document = ModelDocument::compile("circuit.eqi", source).unwrap();
    let mut roles = std::collections::BTreeSet::new();
    for node in document.program().nodes() {
        if let KernelNode::Relation(relation) = node {
            for equation in document
                .render_equations(relation.id().into(), NotationProfile::Plain)
                .unwrap()
            {
                for reference in equation.references() {
                    if let Some(role) = reference.role() {
                        roles.insert(role);
                    }
                    for id in reference.declarations() {
                        assert!(document.notation().get(*id).is_some());
                    }
                }
            }
        }
    }
    assert!(roles.contains(&eqiora_compiler::QuantityRole::Across));
    assert!(roles.contains(&eqiora_compiler::QuantityRole::Through));
}

#[test]
fn nominal_members_and_set_types_keep_exact_declarations() {
    let source = "enum Mode{Heating,Cooling} model M(){clock tick=periodic(1[s]);state mode:Mode at tick;initial{mode=Mode.Heating;}relation update at tick{next(mode)=Mode.Cooling;}}";
    let document = ModelDocument::compile("enum.eqi", source).unwrap();
    let rendered = equation(&document, NotationProfile::MathMl);
    assert!(rendered.references().iter().any(|reference| {
        reference
            .graph_id()
            .is_some_and(|id| id.kind() == eqiora_core::EntityKind::Enum)
    }));
    assert!(rendered.plain().contains("member "));
    let id = eqiora_core::Id::<eqiora_core::entity::kinds::IndexSet>::new();
    let value_type = eqiora_core::ValueType::index(id, 3).unwrap();
    let rendered =
        eqiora_api::MathRendering::value_type(&value_type, NotationProfile::Speech).unwrap();
    assert_eq!(rendered.references()[0].graph_id(), Some(id.into()));
    assert!(rendered.text().contains(&format!("index set {id}")));
}

#[test]
fn nominal_type_bounds_and_scalar_literal_dimensions_are_presented() {
    use eqiora_core::{Id, ValueType, entity::kinds};
    let index = Id::<kinds::IndexSet>::new();
    let enumeration = Id::<kinds::Enum>::new();
    for profile in PROFILES {
        for (first, second, marker) in [
            (
                ValueType::index(index, 3).unwrap(),
                ValueType::index(index, 4).unwrap(),
                "index extent 3",
            ),
            (
                ValueType::enumeration(enumeration, 2).unwrap(),
                ValueType::enumeration(enumeration, 3).unwrap(),
                "enum members 2",
            ),
        ] {
            let first = eqiora_api::MathRendering::value_type(&first, profile).unwrap();
            let second = eqiora_api::MathRendering::value_type(&second, profile).unwrap();
            assert_ne!(first.text(), second.text());
            assert!(first.speech().contains(marker));
            assert_eq!(first.references(), second.references());
        }
    }
    let meters = ModelDocument::compile(
        "units.eqi",
        "model M(){variable x @{x}:m;relation law{x=2[m];}}",
    )
    .unwrap();
    let seconds = ModelDocument::compile(
        "units.eqi",
        "model M(){variable x @{x}:s;relation law{x=2[s];}}",
    )
    .unwrap();
    for profile in PROFILES {
        let meters = equation(&meters, profile);
        let seconds = equation(&seconds, profile);
        assert_ne!(meters.text(), seconds.text());
        assert!(meters.plain().contains("dimensions m"));
        assert!(seconds.plain().contains("dimensions s"));
    }
    let scalar = ModelDocument::compile(
        "scalar.eqi",
        "model M(){variable x @{x}:1;relation law{x=2;}}",
    )
    .unwrap();
    assert_eq!(
        equation(&scalar, NotationProfile::Plain).text(),
        "(x) = (2)"
    );
}

#[test]
fn equation_rendering_never_relabels_constraints_as_equalities() {
    for condition in ["inequality(x >= 0);", "complementarity(x >= 0, y >= 0);"] {
        let document = ModelDocument::compile(
            "constraints.eqi",
            &format!("model M(){{variable x:1;variable y:1;relation law{{{condition}}}}}"),
        )
        .unwrap();
        let relation = document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(value) => Some(value.id().into()),
                _ => None,
            })
            .unwrap();
        for profile in PROFILES {
            let error = document.render_equations(relation, profile).unwrap_err();
            assert!(error.to_string().contains("inequality or complementarity"));
        }
    }
}
