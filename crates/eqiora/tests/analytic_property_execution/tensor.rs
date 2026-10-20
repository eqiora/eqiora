//! Tensor constitutive responses use the same value type and formal calculus.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::kernel::KernelNode;
use eqiora::sem::{Interpreter, ReferenceConfig};
use eqiora::{ScalarDomain, ValueFrame, ValueLiteral};

const SOURCE: &str = r#"
property contract Conductivity(input T:K,input p:1):tensor<complex<kg*m/s^3/K>,2,2> {
  derivatives first_partials;
  branch principal;
}
property release Measured:Conductivity {
  analytic {
    value = (math.complex(T/1[K],(T/1[K])^2)*delta(0,1)+p)*1[kg*m/s^3/K];
    source_unit:kg*m/s^3/K=1;
  }
  validity T>=0[K] and T<=4[K] and p>=0 and p<=5;
  outside reject;
  branch principal;
  citation org.example.tensor;
  license spdx.CC0_1_0;
}
component Consumer(parameter T:K,parameter p:1,property k:Conductivity) {
  variable re:kg*m/s^3/K;
  variable im:kg*m/s^3/K;
  relation law {
    re=math.real(matrix_trace(k(T=T,p=p)));
    im=math.imag(matrix_trace(k(T=T,p=p)));
  }
}
model Main() {
  instance first:Consumer(T=2[K],p=3,k=Measured);
  instance second:Consumer(T=1[K],p=2,k=Measured);
}
"#;

#[test]
fn tensor_property_contract_retains_values_units_partials_and_replay() {
    for source in [
        SOURCE.to_owned(),
        composed_source(),
        generic_composed_source(),
    ] {
        let document = ModelDocument::compile("tensor-property.eqi", &source).unwrap();
        let bytes = ModelEnvelope::from_program(document.program())
            .unwrap()
            .canonical_json()
            .unwrap();
        let reopened = ModelEnvelope::from_json(&bytes, Default::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(document.program(), &reopened);
        for program in [document.program(), &reopened] {
            let release = program
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Relation(value) => value.expression().properties().values().next(),
                    _ => None,
                })
                .unwrap();
            let value_type = release.meaning().value_type().unwrap();
            assert_eq!(
                value_type
                    .shape()
                    .extents()
                    .iter()
                    .map(|n| n.get())
                    .collect::<Vec<_>>(),
                [2, 2]
            );
            assert_eq!(value_type.frame(), ValueFrame::SpatialCartesian);
            assert_eq!(value_type.scalar_domain(), ScalarDomain::Complex);
            assert_eq!(release.inputs(), ["T", "p"]);
            assert_eq!(release.branch(), Some("principal"));
            assert!(release.guarded());
            let definition = release.meaning().definition().unwrap();
            let temperature = release.partial("T").unwrap();
            let pressure = release.partial("p").unwrap();
            for (partial, input) in [(&temperature, 0), (&pressure, 1)] {
                let result = partial.result_rule().exact_type().unwrap();
                assert_eq!(result.shape(), value_type.shape());
                assert_eq!(result.frame(), value_type.frame());
                assert_eq!(result.scalar_domain(), value_type.scalar_domain());
                assert_eq!(
                    result.dimension(),
                    value_type
                        .dimension()
                        .div(definition.formals()[input].dimension().unwrap())
                        .unwrap()
                );
            }
            // K_ij = (x + i*x²) δ_ij + p, x=T/(1 K), in conductivity units.
            // Holding p fixed: dK/dT = (1 + 2ix) δ_ij per kelvin.
            // Holding T fixed: dK/dp = 1 in every component.
            for (t, p) in [(2., 3.), (1., 2.)] {
                let arguments = [t, p]
                    .into_iter()
                    .zip(definition.formals())
                    .map(|(value, formal)| {
                        ValueLiteral::from_real(formal.exact_type().unwrap().clone(), value)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                for (law, expected) in [
                    (
                        definition,
                        [(t + p, t * t), (p, 0.), (p, 0.), (t + p, t * t)],
                    ),
                    (
                        &temperature,
                        [(1., 2. * t), (0., 0.), (0., 0.), (1., 2. * t)],
                    ),
                    (&pressure, [(1., 0.); 4]),
                ] {
                    let actual = eqiora::ir::ComponentScalarization::evaluate_pure_operator(
                        law,
                        &[&arguments[0], &arguments[1]],
                    )
                    .unwrap();
                    assert_eq!(actual.value_type(), law.result_rule().exact_type().unwrap());
                    let components = actual.components().unwrap().collect::<Vec<_>>();
                    assert_eq!(components.len(), expected.len());
                    for ((re, im), (expected_re, expected_im)) in
                        components.into_iter().zip(expected)
                    {
                        assert!((re - expected_re).abs() < 1e-13);
                        assert!((im - expected_im).abs() < 1e-13);
                    }
                }
            }
            for point in [[5., 3.], [2., 6.]] {
                let arguments = point
                    .into_iter()
                    .zip(definition.formals())
                    .map(|(value, formal)| {
                        ValueLiteral::from_real(formal.exact_type().unwrap().clone(), value)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                for law in [definition, &temperature, &pressure] {
                    assert!(
                        eqiora::ir::ComponentScalarization::evaluate_pure_operator(
                            law,
                            &[&arguments[0], &arguments[1]]
                        )
                        .is_err()
                    );
                }
            }
            let result = Interpreter::new()
                .run(program, ReferenceConfig::new(0., 1.).unwrap())
                .unwrap();
            for (name, expected) in [
                ("first.re", 10.),
                ("first.im", 8.),
                ("second.re", 6.),
                ("second.im", 2.),
            ] {
                let field = document.field_ref(name).unwrap().id().erase();
                assert!((result.last_value(field).unwrap().value() - expected).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn packaged_tensor_property_and_direct_law_agree_with_distinct_lineage() {
    let direct = SOURCE
        .replace(
            "matrix_trace(k(T=T,p=p))",
            "math.complex(2*T/1[K]+2*p,2*(T/1[K])^2)*1[kg*m/s^3/K]",
        )
        .replace(",property k:Conductivity", "")
        .replace(",k=Measured", "");
    let composed = composed_source();
    let generic = generic_composed_source();
    for (source, keeps_release) in [
        (SOURCE, true),
        (composed.as_str(), true),
        (generic.as_str(), true),
        (direct.as_str(), false),
    ] {
        let local = ModelDocument::compile("tensor-property.eqi", source).unwrap();
        let (declarations, consumer) = source.split_once("component Consumer").unwrap();
        let consumer = format!(
            "import org.example.Properties.main as properties;\npublic component Consumer{}",
            consumer
                .replace("k:Conductivity", "k:properties.Conductivity")
                .replace("k=Measured", "k=properties.Measured")
                .replace("model Main", "public model Main")
        );
        let packaged = super::packaged_sources(declarations, &consumer);
        for model in [&local, packaged.model()] {
            let bytes = ModelEnvelope::from_program(model.program())
                .unwrap()
                .canonical_json()
                .unwrap();
            let reopened = ModelEnvelope::from_json(&bytes, Default::default())
                .unwrap()
                .to_program()
                .unwrap();
            for program in [model.program(), &reopened] {
                let has_release = program.nodes().any(|node| {
                    matches!(node,
                    KernelNode::Relation(value) if !value.expression().properties().is_empty())
                });
                assert_eq!(has_release, keeps_release);
                let result = Interpreter::new()
                    .run(program, ReferenceConfig::new(0., 1.).unwrap())
                    .unwrap();
                for (name, expected) in [
                    ("first.re", 10.),
                    ("first.im", 8.),
                    ("second.re", 6.),
                    ("second.im", 2.),
                ] {
                    let field = model.field_ref(name).unwrap().id().erase();
                    assert!((result.last_value(field).unwrap().value() - expected).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn tensor_property_rejects_incompatible_contraction_input_and_branch() {
    for source in [
        SOURCE.replace(
            "tensor<complex<kg*m/s^3/K>,2,2>",
            "tensor<complex<kg*m/s^3/K>,3,2>",
        ),
        SOURCE.replace("k(T=T,p=p)", "k(T=1[s],p=p)"),
        SOURCE.replace("branch principal;\n  citation", "branch other;\n  citation"),
    ] {
        assert!(ModelDocument::compile("invalid-tensor-property.eqi", &source).is_err());
    }
}

fn composed_source() -> String {
    let body = "(math.complex(T/1[K],(T/1[K])^2)*delta(0,1)+p)*1[kg*m/s^3/K]";
    let law = format!(
        "operator conductivity(input T:K,input p:1):tensor<complex<kg*m/s^3/K>,2,2> = {body};\n"
    );
    law + &SOURCE.replace(body, "conductivity(T=T,p=p)")
}

fn generic_composed_source() -> String {
    composed_source().replacen("):tensor<complex<kg*m/s^3/K>,2,2>", "):spatial[2]", 1)
}
