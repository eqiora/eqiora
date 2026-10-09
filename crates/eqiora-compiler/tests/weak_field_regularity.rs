//! A test-space assertion cannot supply the trial Field's boundary regularity.
use eqiora_compiler::CompiledModel;

fn source(operator: &str, regularity: &str, operand: &str) -> String {
    let shape = if operator == "trace" {
        "1"
    } else {
        "vector<1,2>"
    };
    format!(
        r#"model M() {{
        domain body=box(0,1,0,1);
        domain face=boundary(body,axis=0,side=lower);
        variable u:{shape} on body {regularity};
        relation law on body {{ u=u; }}
        form weak for law {{
            test eta:1 for u in h1;
            integrate(face,{operator}(eta,on=face)*{operator}({operand},on=face))=0;
        }}
    }}"#
    )
}

#[test]
fn weak_field_trace_positive_controls_use_the_fields_own_assertion() {
    for (operator, regularity) in [
        ("trace", "h1"),
        ("trace", "smooth"),
        ("normal", "hdiv"),
        ("normal", "h1"),
        ("tangential_trace", "hcurl"),
        ("tangential_trace", "h1"),
    ] {
        for operand in ["u", "2*u", "u+u"] {
            CompiledModel::compile_selected(
                "weak-field.eqi",
                &source(operator, &format!("in {regularity}"), operand),
                "M",
                &[],
            )
            .unwrap_or_else(|errors| panic!("{operator}/{regularity}/{operand}: {errors:?}"));
        }
    }
}

#[test]
fn weak_field_trace_rejects_missing_or_incompatible_field_assertions() {
    for (operator, regularity) in [
        ("trace", ""),
        ("trace", "in l2"),
        ("normal", ""),
        ("normal", "in l2"),
        ("normal", "in hcurl"),
        ("tangential_trace", ""),
        ("tangential_trace", "in l2"),
        ("tangential_trace", "in hdiv"),
    ] {
        let result = CompiledModel::compile_selected(
            "weak-field.eqi",
            &source(operator, regularity, "u"),
            "M",
            &[],
        );
        assert!(
            result.is_err(),
            "{operator} admitted a Field with `{regularity}` merely because its test is H1"
        );
        assert!(
            result
                .unwrap_err()
                .iter()
                .any(|error| error.message().contains("regularity")),
            "rejection must identify the regularity boundary"
        );
    }
}

#[test]
fn weak_field_derivative_trace_requires_the_explicit_smooth_profile() {
    let source = |regularity| {
        format!(
            r#"model M() {{
            domain body=box(0,1,0,1);
            domain face=boundary(body,axis=0,side=lower);
            variable u:1 on body in {regularity};
            relation law on body {{ u=u; }}
            form weak for law {{
                test eta:1 for u in h1;
                integrate(face,trace(eta,on=face)*normal(grad(u),on=face))=0;
            }}
        }}"#
        )
    };
    CompiledModel::compile_selected("smooth-field.eqi", &source("smooth"), "M", &[]).unwrap();
    let result = CompiledModel::compile_selected("h1-field.eqi", &source("h1"), "M", &[]);
    assert!(
        result.is_err(),
        "H1 alone cannot supply a gradient boundary trace"
    );
    assert!(
        result
            .unwrap_err()
            .iter()
            .any(|error| error.message().contains("regularity"))
    );
}
