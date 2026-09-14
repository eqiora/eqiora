use eqiora_compiler::compile;

fn accepted(source: &str) {
    compile("support.eqi", source).unwrap_or_else(|errors| panic!("{source}\n{errors:?}"));
}

fn rejected(source: &str, message: &str) {
    eqiora_lang::parse("support.eqi", source)
        .into_document()
        .unwrap();
    let errors = compile("support.eqi", source).unwrap_err();
    assert!(
        errors.iter().any(|error| error.message().contains(message)),
        "{source}\n{errors:?}"
    );
    assert!(errors.iter().all(|error| error.source_span().is_some()));
}

#[test]
fn nominal_support_assertions_follow_forward_aliases_and_spatial_types() {
    accepted(
        "model M() { domain body=box(0,1,0,1); variable x:1 on body; let q:vector<1/m,2> on body=grad(value); let value on body=x; relation r on body { div(q)=0; } }",
    );
    rejected(
        "model M() { domain a=box(0,1); domain b=box(0,1); variable x:1 on a; let value on b=x; relation r on a { x=0; } }",
        "inferred exact support",
    );
    rejected(
        "model M() { domain body=box(0,1); variable x:1 on body; let value on absent=x; relation r on body { x=0; } }",
        "let alias spatial support",
    );
    rejected(
        "model M() { domain body=box(0,1); variable x:1 on body; let value on x=x; relation r on body { x=0; } }",
        "let alias spatial support",
    );
}

#[test]
fn component_assertions_are_checked_without_an_occurrence() {
    accepted(
        "component C(support body:volume(ambient_dimension=2), variable x:1 on body) { let q:vector<1/m,2> on body=grad(x); relation r on body { div(q)=0; } } model M() { variable seed:1; relation seed_equation { seed=0; } }",
    );
    rejected(
        "component C(support a:volume(ambient_dimension=1), support b:volume(ambient_dimension=1), variable x:1 on a) { let value on b=x; relation r on a { x=0; } } model M() { variable seed:1; relation seed_equation { seed=0; } }",
        "inferred exact support",
    );
}

#[test]
fn assertions_preserve_support_through_nested_occurrences() {
    accepted(
        "component Leaf(support body:volume(ambient_dimension=1), variable x:1 on body) { let q on body=grad(x); relation r on body { div(q)=0; } } component Wrapper(support body:volume(ambient_dimension=1), variable x:1 on body) { instance leaf:Leaf(body =body,x =x); } model M() { domain a=box(0,1); domain b=box(0,2); variable x:1 on a; variable y:1 on b; instance first:Wrapper(body =a,x =x); instance second:Wrapper(body =b,x =y); }",
    );
}

#[test]
fn static_and_support_free_aliases_cannot_acquire_support() {
    for expression in ["2", "p", "p+2", "math.sin(p)"] {
        rejected(
            &format!(
                "model M() {{ domain body=box(0,1); parameter p:1=2; let value on body={expression}; }}"
            ),
            "static let alias cannot assert spatial support",
        );
        rejected(
            &format!(
                "component C(support body:volume(ambient_dimension=1), parameter p:1) {{  let value on body={expression}; }} model M() {{ variable seed:1; relation seed_equation {{ seed=0; }} }}"
            ),
            "static let alias cannot assert spatial support",
        );
    }
    rejected(
        "model M() { domain body=box(0,1); state x:1; let value on body=x; relation r { x=0; } }",
        "inferred exact support",
    );
    rejected(
        "model M() { domain body=box(0,1); let value on body=time(); relation r { time()=0; } }",
        "inferred exact support",
    );
}

#[test]
fn on_assertions_do_not_supply_missing_operator_context() {
    for expression in ["coordinate(0)", "trace(x)", "normal(grad(x))"] {
        rejected(
            &format!(
                "model M() {{ domain body=box(0,1); variable x:1 on body; let value on body={expression}; relation r on body {{ x=0; }} }}"
            ),
            "no support context",
        );
        rejected(
            &format!(
                "component C(support body:volume(ambient_dimension=1), variable x:1 on body) {{ let value on body={expression}; }} model M() {{ variable seed:1; relation seed_equation {{ seed=0; }} }}"
            ),
            "no support context",
        );
    }
}
