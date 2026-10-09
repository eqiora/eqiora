//! Source assertions retain semantic identity and cannot be supplied by a caller's omission.
use eqiora_compiler::{compile, source_identity::LocalSourceIdentity};
use eqiora_graph::Op;
use eqiora_schema::kernel::{KernelNode, SpatialRegularity};

#[test]
fn regularity_survives_formatting_lowering_and_changes_source_identity() {
    let mut identities = Vec::new();
    for (syntax, expected) in [
        ("", SpatialRegularity::Unspecified),
        (" in l2", SpatialRegularity::L2),
        (" in h1", SpatialRegularity::H1),
        (" in hcurl", SpatialRegularity::HCurl),
        (" in hdiv", SpatialRegularity::HDiv),
        (" in smooth", SpatialRegularity::Smooth),
    ] {
        let source = format!(
            "model M() {{ domain body=box(0,1,0,1); variable u:vector<1,2> on body{syntax}; relation law on body {{ u=u; }} }}"
        );
        let document = eqiora_lang::parse("regularity.eqi", &source)
            .into_document()
            .unwrap();
        let formatted = eqiora_lang::format(&document);
        let reparsed = eqiora_lang::parse("formatted.eqi", &formatted)
            .into_document()
            .unwrap();
        let identity = LocalSourceIdentity::from_document(&document).unwrap();
        assert_eq!(
            identity,
            LocalSourceIdentity::from_document(&reparsed).unwrap()
        );
        assert!(!identities.contains(&identity));
        identities.push(identity);
        let compiled = compile("regularity.eqi", &formatted).unwrap();
        let fields = compiled[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Field(field),
                } => Some(field.spatial_regularity()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(fields, [expected]);
    }
}

#[test]
fn field_binding_checks_the_actual_assertion_through_forwarding() {
    let source = |assertion: &str| {
        format!(
            r#"
component Inner(support body: volume(ambient_dimension=2), variable u:1 on body in h1) {{
    relation law on body {{ u=u; }}
}}
component Forward(support body: volume(ambient_dimension=2), variable u:1 on body) {{
    instance inner: Inner(body=body,u=u);
}}
model M() {{
    domain body=box(0,1,0,1);
    variable u:1 on body{assertion};
    instance outer: Forward(body=body,u=u);
}}
"#
        )
    };
    // A generic forwarding declaration cannot promise H1 to its own child.
    // Check the direct requirement independently of that definition-time failure.
    let direct = |assertion| {
        source(assertion)
            .replace("instance outer: Forward", "instance outer: Inner")
            .replace(
                "instance inner: Inner(body=body,u=u);",
                "relation unused on body { u=u; }",
            )
    };
    for assertion in [" in h1", " in smooth"] {
        compile("regularity.eqi", &direct(assertion)).unwrap();
    }
    for assertion in ["", " in l2"] {
        let errors = compile("regularity.eqi", &direct(assertion)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("authored spatial regularity")),
            "{errors:?}"
        );
    }
    assert!(compile("forward.eqi", &source("")).is_err());
}

#[test]
fn unsupported_regularity_spelling_is_not_silently_ignored() {
    assert!(
        eqiora_lang::parse("bad.eqi", "model M() { variable u:1 in h2; }")
            .into_document()
            .is_err()
    );
}

#[test]
fn harmonic_amplitude_cannot_override_its_original_field_regularity() {
    let source = |assertion: &str| {
        format!(
            "model M() {{ domain body=box(0,1); state u:1 on body in h1; \
         relation evolution on body {{ derivative(u)=0[1/s]; }} \
         form response for evolution {{ \
         harmonic(angular_frequency=1[1/s],convention=negative_exponential,normalization=peak); \
         amplitude u_hat:complex<1> on body{assertion} for u; }} }}"
        )
    };
    let mut identities = Vec::new();
    let mut model_identity = None;
    for assertion in ["", " in h1"] {
        let authored = source(assertion);
        let document = eqiora_lang::parse("harmonic.eqi", &authored)
            .into_document()
            .unwrap();
        let formatted = eqiora_lang::format(&document);
        let reparsed = eqiora_lang::parse("formatted.eqi", &formatted)
            .into_document()
            .unwrap();
        let identity = LocalSourceIdentity::from_document(&document).unwrap();
        assert_eq!(
            identity,
            LocalSourceIdentity::from_document(&reparsed).unwrap()
        );
        match &model_identity {
            Some(previous) => assert_eq!(&identity, previous),
            None => model_identity = Some(identity),
        }
        // Harmonic declarations have a separate identity; the original Model
        // identity must remain unchanged when its sidecar assertion changes.
        let original =
            eqiora_compiler::CompiledModel::compile_selected("harmonic.eqi", &authored, "M", &[])
                .unwrap();
        let replayed =
            eqiora_compiler::CompiledModel::compile_selected("harmonic.eqi", &formatted, "M", &[])
                .unwrap();
        let form = original.authored_formulations().next().unwrap();
        let replayed_form = replayed.authored_formulations().next().unwrap();
        assert_eq!(form.projection(), replayed_form.projection());
        let form_identity = form.source_identity().to_owned();
        assert!(!identities.contains(&form_identity));
        identities.push(form_identity);
    }
    for assertion in [" in l2", " in smooth"] {
        let errors = compile("harmonic.eqi", &source(assertion)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("regularity assertion differs")),
            "{errors:?}"
        );
    }
}
