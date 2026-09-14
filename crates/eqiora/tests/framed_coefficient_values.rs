//! Exact uniform coefficients in the model-global Cartesian frame; no solver claim.

use eqiora::api::ModelDocument;
use eqiora::artifact::{ModelDecoderLimits, ModelTransactionEnvelope};
use eqiora::graph::{EdgeKind, Op};
use eqiora::kernel::KernelNode;
use eqiora::language::{
    DecimalLiteral, DraftExpression, DraftParameter, DraftRelation, DraftSpatialDomain, Module,
};
use eqiora::{DimExponents, ScalarDomain, ValueFrame, ValueLiteral, ValueShape, ValueType};

const SOURCE: &str = r#"
model Coefficients() {
  domain body = box(0, 1, 0, 1);
  parameter direction: vector<V, 2> = tensor_value(frame=body, components=[2[V], 3[V]]);
  parameter stiffness: tensor<Pa, 2, 2> = tensor_value(frame=body, components=[[2[Pa], 3[Pa]], [5[Pa], 7[Pa]]]);
  parameter impedance: tensor<complex<Pa>, 2, 2> = tensor_value(frame=body, components=[[math.complex(2[Pa], 11[Pa]), math.complex(3[Pa], 13[Pa])], [math.complex(5[Pa], 17[Pa]), math.complex(7[Pa], 19[Pa])]]);
  relation witness { 0 = 0; }
}
"#;

// Last axis varies fastest. Unequal off-diagonal entries expose transposition;
// nonzero unequal imaginary components expose channel loss or reordering.
const VECTOR: [(f64, f64); 2] = [(2.0, 0.0), (3.0, 0.0)];
const MATRIX: [(f64, f64); 4] = [(2.0, 0.0), (3.0, 0.0), (5.0, 0.0), (7.0, 0.0)];
const COMPLEX_MATRIX: [(f64, f64); 4] = [(2.0, 11.0), (3.0, 13.0), (5.0, 17.0), (7.0, 19.0)];

fn coefficient_type(domain: ScalarDomain, dimension: [i32; 7], extents: &[u32]) -> ValueType {
    ValueType::shaped(
        domain,
        DimExponents::from_integers(dimension).unwrap(),
        ValueShape::new(extents.iter().copied()).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap()
}

fn expected_values() -> [(&'static str, ValueLiteral); 3] {
    [
        (
            "direction",
            ValueLiteral::new(
                coefficient_type(ScalarDomain::Real, [1, 2, -3, -1, 0, 0, 0], &[2]),
                VECTOR,
            )
            .unwrap(),
        ),
        (
            "stiffness",
            ValueLiteral::new(
                coefficient_type(ScalarDomain::Real, [1, -1, -2, 0, 0, 0, 0], &[2, 2]),
                MATRIX,
            )
            .unwrap(),
        ),
        (
            "impedance",
            ValueLiteral::new(
                coefficient_type(ScalarDomain::Complex, [1, -1, -2, 0, 0, 0, 0], &[2, 2]),
                COMPLEX_MATRIX,
            )
            .unwrap(),
        ),
    ]
}

fn native() -> ModelDocument {
    let body = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0), (0.0, 1.0)]);
    let mut declarations = vec![body.clone().into()];
    for (name, value) in expected_values() {
        declarations.push(DraftParameter::new(name, value).with_frame(&body).into());
    }
    let zero = || DraftExpression::constant(DecimalLiteral::parse("0").unwrap());
    declarations.push(DraftRelation::continuous("witness", [(zero(), zero())]).into());
    ModelDocument::compile_module(
        &Module::new("Coefficients", declarations).unwrap(),
        None,
        &[],
    )
    .unwrap()
}

fn assert_uniform_coefficients(document: &ModelDocument) {
    assert_eq!(
        document
            .program()
            .nodes()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count(),
        0
    );
    assert_eq!(
        document
            .program()
            .nodes()
            .filter(|node| matches!(node, KernelNode::Parameter(_)))
            .count(),
        3
    );
    for (name, expected) in expected_values() {
        let id = document.aliases()[name];
        let actual = document.program().typed_value(id).unwrap();
        assert_eq!(actual, &expected, "complete independently specified {name}");
        assert_eq!(actual.value_type().array_rank(), 0);
        assert!(
            !document.program().edges().iter().any(|edge| {
                edge.kind() == EdgeKind::DefinedOn && (edge.from() == id || edge.to() == id)
            }),
            "uniform coefficient must not acquire Field support ownership"
        );
    }
}

#[test]
fn nonzero_framed_coefficients_share_source_native_identity_and_exact_replay() {
    let source = ModelDocument::compile("framed-coefficients.eqi", SOURCE).unwrap();
    let native = native();
    assert_uniform_coefficients(&source);
    assert_uniform_coefficients(&native);
    assert!(source.structurally_equivalent(&native).unwrap());
    assert_eq!(
        source.structural_fingerprint().unwrap(),
        native.structural_fingerprint().unwrap()
    );
    for document in [&source, &native] {
        let bytes = document.canonical_json().unwrap();
        assert!(
            std::str::from_utf8(&bytes)
                .unwrap()
                .contains("eqiora.model-envelope/v27")
        );
        let replay = ModelDocument::replay(&bytes).unwrap();
        assert_eq!(replay.canonical_json().unwrap(), bytes);
        for (name, _) in expected_values() {
            let id = document.aliases()[name];
            assert_eq!(
                replay.program().typed_value(id),
                document.program().typed_value(id)
            );
        }
    }
}

#[test]
fn imaginary_coefficient_edit_preserves_parameter_id_type_and_other_components() {
    let document = ModelDocument::compile("framed-edit.eqi", SOURCE).unwrap();
    let target = document.aliases()["impedance"];
    let before = document.program().typed_value(target).unwrap();
    let replacement = ValueLiteral::new(
        before.value_type().clone(),
        [(2.0, 11.0), (3.0, 23.0), (5.0, 17.0), (7.0, 19.0)],
    )
    .unwrap();
    let plan = document
        .preview_value_edit(target, replacement.clone())
        .unwrap();
    let transaction = ModelTransactionEnvelope::from_json(
        &plan.transaction_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap()
    .to_transaction()
    .unwrap();
    assert_eq!(
        transaction.ops(),
        &[Op::SetValue {
            target,
            value: replacement.clone()
        }]
    );
    let changed = document
        .commit_value_edit(plan.clone())
        .unwrap()
        .into_document();
    let replay = ModelDocument::replay(&changed.canonical_json().unwrap()).unwrap();
    for child in [&changed, &replay] {
        assert_eq!(child.program().typed_value(target), Some(&replacement));
        assert_eq!(
            child
                .program()
                .nodes()
                .filter(|node| matches!(node, KernelNode::Field(_)))
                .count(),
            0
        );
        assert!(
            !child
                .program()
                .edges()
                .iter()
                .any(|edge| edge.kind() == EdgeKind::DefinedOn && edge.from() == target)
        );
        for name in ["direction", "stiffness"] {
            let id = document.aliases()[name];
            assert_eq!(
                child.program().typed_value(id),
                document.program().typed_value(id)
            );
        }
    }
    assert_eq!(
        document
            .program()
            .typed_value(target)
            .unwrap()
            .components()
            .unwrap()
            .collect::<Vec<_>>(),
        COMPLEX_MATRIX
    );
    assert!(changed.commit_value_edit(plan).is_err());
    let channels = ValueLiteral::new(
        ValueType::scalar(ScalarDomain::Complex, before.value_type().dimension())
            .expect("valid scalar type")
            .array(2)
            .unwrap()
            .array(2)
            .unwrap(),
        COMPLEX_MATRIX,
    )
    .unwrap();
    assert!(document.preview_value_edit(target, channels).is_err());
}

#[test]
fn explicit_frame_construction_rejects_channel_substitution_and_wrong_context() {
    ModelDocument::compile("frame-context.eqi", SOURCE).unwrap();
    for invalid in [
        SOURCE.replace(
            "tensor_value(frame=body, components=[2[V], 3[V]])",
            "[2[V], 3[V]]",
        ),
        SOURCE.replace("vector<V, 2>", "array<V, 2>"),
        SOURCE.replace("frame=body", "frame=missing"),
        SOURCE.replace("frame=body", "frame=\"body\""),
        SOURCE.replace("box(0, 1, 0, 1)", "box(0, 1, 0, 1, 0, 1)"),
    ] {
        let diagnostics = ModelDocument::compile("frame-context.eqi", &invalid).unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.source_span().is_some()),
            "{diagnostics:?}"
        );
    }
    // Two admitted supports in the same model-global Cartesian frame are not
    // automatically distinct nominal frames merely because Domain IDs differ.
    let shared_frame = SOURCE
        .replace(
            "domain body = box(0, 1, 0, 1);",
            "domain body = box(0, 1, 0, 1); domain other = box(0, 2, 0, 2);",
        )
        .replace("frame=body", "frame=other");
    assert_uniform_coefficients(
        &ModelDocument::compile("shared-frame.eqi", &shared_frame).unwrap(),
    );
}

#[test]
fn native_frame_reference_must_belong_to_the_actual_draft() {
    let included = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0), (0.0, 1.0)]);
    let foreign = DraftSpatialDomain::cartesian_box("body", [(0.0, 1.0), (0.0, 1.0)]);
    let (_, value) = expected_values().into_iter().next().unwrap();
    let witness = || {
        DraftRelation::continuous(
            "witness",
            [(
                DraftExpression::constant(DecimalLiteral::parse("0").unwrap()),
                DraftExpression::constant(DecimalLiteral::parse("0").unwrap()),
            )],
        )
    };
    let valid = Module::new(
        "ForeignFrame",
        [
            included.clone().into(),
            DraftParameter::new("direction", value.clone())
                .with_frame(&included)
                .into(),
            witness().into(),
        ],
    )
    .unwrap();
    ModelDocument::compile_module(&valid, None, &[]).unwrap();
    let parameter = DraftParameter::new("direction", value).with_frame(&foreign);
    assert!(
        Module::new(
            "ForeignFrame",
            [included.into(), parameter.into(), witness().into()]
        )
        .is_err()
    );
}

#[test]
fn selected_framed_parameter_uses_exact_caller_geometry_and_explicit_expression() {
    use eqiora::compiler::StaticBindingValue;
    use eqiora::geometry::CanonicalGeometryV1;
    use eqiora::language::{Item, parse};

    // Reuse the ordinary caller-Geometry constructor used by package binding tests;
    // no mesh, runtime state, package store, or numerical solver is required.
    let geometry = |volume| {
        CanonicalGeometryV1::from_circular_hole_named_roles(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            1.0e-12,
            volume,
            "inlet",
            "outlet",
            "walls",
            "walls",
            "cylinder",
        )
        .unwrap()
    };
    let body = geometry("body");
    let foreign = geometry("foreign");
    let expression = |frame: &str| {
        let source = format!(
            "model Binding() {{ parameter value: tensor<Pa, 2, 2> = tensor_value(frame={frame}, components=[[2[Pa], 3[Pa]], [5[Pa], 7[Pa]]]); }}"
        );
        let syntax = parse("binding.eqi", &source).into_document().unwrap();
        let Item::Parameter(parameter) = &syntax.models()[0].items()[0] else {
            panic!("Parameter expression fixture")
        };
        parameter.value().clone()
    };
    let value = expression("body");
    let invalid_frame = expression("missing");
    let expected = expected_values()
        .into_iter()
        .find(|(name, _)| *name == "stiffness")
        .unwrap()
        .1;

    for kind in ["component", "model"] {
        let source = format!(
            "public {kind} Selected(support body: volume(ambient_dimension=2), parameter stiffness: tensor<Pa, 2, 2>) {{ relation witness {{ 0 = 0; }} }}"
        );
        let support = || {
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &body,
                    selection: body.entity_set("body").unwrap(),
                    parent: None,
                },
            )
        };
        let document = ModelDocument::compile_selected(
            "selected-frame.eqi",
            &source,
            "Selected",
            &[
                support(),
                ("stiffness", StaticBindingValue::Expression(&value)),
            ],
        )
        .unwrap();
        let parameters = document
            .program()
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Parameter(parameter) => Some(parameter.id().erase()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(parameters.len(), 1);
        assert_eq!(
            document.program().typed_value(parameters[0]),
            Some(&expected)
        );
        assert_eq!(
            document
                .program()
                .nodes()
                .filter(|node| matches!(node, KernelNode::Field(_)))
                .count(),
            0
        );
        assert!(
            !document
                .program()
                .edges()
                .iter()
                .any(|edge| edge.kind() == EdgeKind::DefinedOn && edge.from() == parameters[0])
        );
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        assert_eq!(replay.program().typed_value(parameters[0]), Some(&expected));

        assert!(
            ModelDocument::compile_selected(
                "selected-frame.eqi",
                &source,
                "Selected",
                &[("stiffness", StaticBindingValue::Expression(&value)),]
            )
            .is_err(),
            "an explicit constructor cannot supply missing caller support"
        );
        assert!(
            ModelDocument::compile_selected(
                "selected-frame.eqi",
                &source,
                "Selected",
                &[
                    support(),
                    ("stiffness", StaticBindingValue::Expression(&invalid_frame)),
                ]
            )
            .is_err(),
            "external expression cannot name an unbound frame"
        );
        assert!(
            ModelDocument::compile_selected(
                "selected-frame.eqi",
                &source,
                "Selected",
                &[
                    (
                        "body",
                        StaticBindingValue::GeometrySupport {
                            geometry: &body,
                            selection: foreign.entity_set("foreign").unwrap(),
                            parent: None,
                        }
                    ),
                    ("stiffness", StaticBindingValue::Expression(&value)),
                ]
            )
            .is_err(),
            "foreign Geometry selection cannot borrow another authority's frame"
        );
    }
}
