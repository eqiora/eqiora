//! Support-sized declarations use the existing definition and occurrence checkers.

#[test]
fn support_sized_fields_and_expression_requirements_forward_through_nested_components() {
    for (dimension, bounds) in [(2, "0,1,0,1"), (3, "0,1,0,1,0,1")] {
        let source = format!(
            r#"
component Leaf(support body: volume(ambient_dimension={dimension}),
  variable velocity: vector<m/s, spatial> on body,
  expression prescribed: vector<m/s, spatial> on body) {{
  relation law on body {{ velocity=prescribed; }}
}}
component Wrapper(support body: volume(ambient_dimension={dimension}),
  variable velocity: vector<m/s, spatial> on body,
  expression prescribed: vector<m/s, spatial> on body) {{
  instance leaf: Leaf(body=body,velocity=velocity,prescribed=prescribed);
}}
model M() {{
  domain body=box({bounds});
  variable velocity:vector<m/s,{dimension}> on body;
  instance wrapper:Wrapper(body=body,velocity=velocity,prescribed=2*velocity);
}}
"#
        );
        crate::compile("support-sized.eqi", &source).unwrap();
    }
}

#[test]
fn support_sized_types_reject_foreign_parents_distinct_supports_and_channel_substitution() {
    let base = r#"
component Boundary(support body:volume(ambient_dimension=2),support face:boundary(parent=body),
  variable velocity:vector<m/s,spatial> on body) {
  relation law on face {trace(velocity)=0;}
}
model M(){
  domain body=box(0,1,0,1);domain other=box(0,1,0,1);
  domain face=boundary(other,axis=0,side=lower);
  variable velocity:vector<m/s,spatial> on body;
  instance boundary:Boundary(body=body,face=face,velocity=velocity);
}
"#;
    assert!(crate::compile("foreign.eqi", base).is_err());
    let valid = base.replace("boundary(other,", "boundary(body,");
    crate::compile("valid.eqi", &valid).unwrap();
    assert!(
        crate::compile(
            "distinct.eqi",
            &valid.replace(
                "velocity:vector<m/s,spatial> on body;",
                "velocity:vector<m/s,spatial> on other;"
            )
        )
        .is_err()
    );
    let unequal = valid
        .replace("other=box(0,1,0,1)", "other=box(0,1,0,1,0,1)")
        .replace(
            "velocity:vector<m/s,spatial> on body;",
            "velocity:vector<m/s,spatial> on other;",
        );
    assert!(crate::compile("unequal-supports.eqi", &unequal).is_err());
    assert!(
        crate::compile(
            "channels.eqi",
            &valid.replace(
                "velocity:vector<m/s,spatial> on body;",
                "velocity:array<m/s,2> on body;"
            )
        )
        .is_err()
    );
    assert!(
        crate::compile(
            "wrong-width.eqi",
            &valid.replace(
                "velocity:vector<m/s,spatial> on body;",
                "velocity:vector<m/s,3> on body;"
            )
        )
        .is_err()
    );
}

#[test]
fn support_sized_axes_do_not_weaken_unused_definition_validation() {
    let source = "component Bad(support body:volume(ambient_dimension=2),variable x:vector<m,spatial> on body){relation law on body{x=1[s];}} model M(){variable x:1;relation r{x=0;}}";
    assert!(crate::compile("unused.eqi", source).is_err());
    let missing = "component Bad(variable x:vector<m,spatial>){relation law{x=0;}} model M(){}";
    assert!(crate::compile("missing.eqi", missing).is_err());
}
