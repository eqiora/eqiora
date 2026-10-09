use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

fn source(operator: &str, target: &str, parent: &str) -> String {
    let shape = if operator == "trace" {
        "1"
    } else {
        "vector<1,2>"
    };
    format!(
        r#"model M() {{
        domain body=box(0,1,0,1);
        domain other=box(0,1,0,1);
        domain face=boundary(body,axis=0,side=lower);
        domain opposite=boundary(body,axis=0,side=upper);
        variable u:{shape} on body in h1;
        relation law on body {{ u=u; }}
        form weak for law {{
            test eta:1 for u;
            integrate(face,{operator}(eta,on={target},from={parent})*{operator}(u,on={target},from={parent}))=0;
        }}
    }}"#
    )
}

#[test]
fn explicit_form_targets_survive_replay_and_cannot_be_replaced_or_omitted() {
    for operator in ["trace", "normal", "tangential_trace"] {
        let model = CompiledModel::compile_selected(
            "form-target.eqi",
            &source(operator, "face", "body"),
            "M",
            &[],
        )
        .unwrap();
        let face = model.symbols().get("face").unwrap().ulid().to_string();
        let opposite = model.symbols().get("opposite").unwrap().ulid().to_string();
        let form = model.authored_formulations().next().unwrap().projection();
        assert_eq!(
            AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
            *form
        );
        let text = String::from_utf8(form.canonical_bytes().to_vec()).unwrap();
        let selection = format!("\"on_ulid\":\"{face}\"");
        assert_eq!(text.matches(&selection).count(), 2);
        let forged = text.replace(&selection, &format!("\"on_ulid\":\"{opposite}\""));
        assert!(
            AuthoredFormulationProjection::decode(forged.as_bytes())
                .unwrap_err()
                .message()
                .contains("exact canonical integration boundary")
        );
        let missing = text.replace(&format!("{selection},"), "");
        assert!(AuthoredFormulationProjection::decode(missing.as_bytes()).is_err());
        assert!(
            AuthoredFormulationProjection::decode(
                text.replace("eqiora.authored-form/v15", "eqiora.authored-form/v14")
                    .as_bytes()
            )
            .is_err()
        );
    }
}

#[test]
fn form_selectors_reject_a_different_face_or_parent_before_projection() {
    for operator in ["trace", "normal", "tangential_trace"] {
        for (target, parent, diagnostic) in [
            ("opposite", "body", "integrand support"),
            ("face", "other", "exact parent volume"),
            ("body", "body", "exact boundary"),
        ] {
            let errors = CompiledModel::compile_selected(
                "invalid-form-target.eqi",
                &source(operator, target, parent),
                "M",
                &[],
            )
            .unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains(diagnostic)),
                "{errors:?}"
            );
        }
    }
}

#[test]
fn hdiv_normal_trace_does_not_admit_a_full_trace_or_weaker_regularity() {
    for space in ["h1", "hdiv", "hcurl", "l2"] {
        let source = source("normal", "face", "body").replace(
            "test eta:1 for u;",
            &format!("test eta:1 for u in {space};"),
        );
        let result = CompiledModel::compile_selected("normal-space.eqi", &source, "M", &[]);
        if matches!(space, "h1" | "hdiv") {
            let model = result.unwrap();
            let form = model.authored_formulations().next().unwrap().projection();
            let text = String::from_utf8(form.canonical_bytes().to_vec()).unwrap();
            assert!(text.contains("normal-trace"));
            if space == "hdiv" {
                // A full vector trace is a stronger demand, even if a forged
                // projection keeps the same Field and boundary identities.
                let forged = text.replace("normal-trace", "trace");
                assert!(
                    AuthoredFormulationProjection::decode(forged.as_bytes())
                        .unwrap_err()
                        .message()
                        .contains("declared regularity")
                );
            }
        } else {
            assert!(
                result
                    .unwrap_err()
                    .iter()
                    .any(|error| error.message().contains("declared regularity"))
            );
        }
    }
}
