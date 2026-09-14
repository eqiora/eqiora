use eqiora_compiler::compile;

const CLOCKS: &str =
    "clock a=periodic(1[s] / 1, phase = 0[s] / 1); clock b=periodic(1[s] / 1, phase = 0[s] / 1);";

fn accepted(body: &str) {
    let source = format!("model M() {{ {CLOCKS} {body} }}");
    compile("activation.eqi", &source).unwrap_or_else(|errors| panic!("{source}\n{errors:?}"));
}

fn rejected_in_both(body: &str, message: &str) {
    for source in [
        format!("model M() {{ {CLOCKS} {body} }}"),
        format!(
            "component Unused() {{ {CLOCKS} {body} }} model M() {{ variable seed:1; relation r {{ seed=0; }} }}"
        ),
    ] {
        eqiora_lang::parse("activation.eqi", &source)
            .into_document()
            .unwrap();
        let errors = compile("activation.eqi", &source).unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(message)),
            "{source}\n{errors:?}"
        );
        assert!(errors.iter().all(|error| error.source_span().is_some()));
    }
}

#[test]
fn clock_profiles_propagate_forward_with_static_neutral_dependencies() {
    accepted(
        "parameter p:1=2; state x:1 at a; let result at a=identity+p; let identity at a=x; relation r { result=0; }",
    );
    let source = format!(
        "component Unused() {{ {CLOCKS} state x:1 at a; let result at a=identity+2; let identity=x; }} model M() {{ variable seed:1; relation r {{ seed=0; }} }}"
    );
    compile("activation.eqi", &source).unwrap();
}

#[test]
fn assertions_reject_static_continuous_mixed_and_distinct_nominal_clocks() {
    for body in [
        "let value at a=2;",
        "parameter p:1=2; let value at a=p+1;",
        "parameter p:1=2; let value at a=math.sin(p);",
    ] {
        rejected_in_both(body, "static let alias cannot assert");
    }
    for body in [
        "state x:1; let value at a=x;",
        "let value at a=time();",
        "state x:1 at b; let value at a=x;",
        "state x:1 at a; state y:1 at b; let value at a=x+y;",
        "state x:1 at a; state y:1; let value at a=x+y;",
        "state x:1 at a; state y:1 at b; let mixed=x+y; let value at a=mixed-mixed;",
        "state x:1; let value at a=derivative(x);",
    ] {
        rejected_in_both(body, "exact declared dependency clock");
    }
    for target in ["missing", "x"] {
        rejected_in_both(
            &format!("state x:1 at a; let value at {target}=x;"),
            "let alias clock activation",
        );
    }
}

#[test]
fn assertions_do_not_create_read_fences_or_relax_evolution_obligations() {
    accepted("state x:1 at a; let read at a=x; relation r at b { read=0; }");
    accepted("state x:1 at a; state y:1 at b; let mixed=x+y; relation r { mixed=0; }");
    accepted(
        "state x:1 at a; let identity at a=x; let old at a=pre(identity); let future at a=next(identity); initial { old=0; } relation r at a { future=old+1; }",
    );
    for law in [
        "relation r at b { old=0; }",
        "relation r { old=0; }",
        "initial { future=0; }",
    ] {
        rejected_in_both(
            &format!(
                "state x:1 at a; let identity at a=x; let old at a=pre(identity); let future at a=next(identity); {law}"
            ),
            "clock",
        );
    }
}

#[test]
fn component_clock_assertions_follow_nested_binding_identity() {
    let source = format!(
        "component Delay(clock tick: periodic,state value:1 at tick) {{ let old at tick=pre(value); relation r at tick {{ next(value)=old; }} }} component Forward(clock tick: periodic,state value:1 at tick) {{ let read at tick=value; instance d:Delay(tick =tick,value =value); }} model M() {{ {CLOCKS} state x:1 at a; instance f:Forward(tick =a,value =x); }}"
    );
    compile("activation.eqi", &source).unwrap();
}

#[test]
fn current_ports_contribute_continuous_dependencies() {
    rejected_in_both(
        "port input:signal input 1; let read at a=input;",
        "exact declared dependency clock",
    );
    let source = format!(
        "component Child(output output:1) {{ relation r {{ output=2; }} }} model M() {{ {CLOCKS} instance child: Child(); let read at a=child.output; relation r {{ read=2; }} }}"
    );
    let errors = compile("activation.eqi", &source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("exact declared dependency clock")),
        "{errors:?}"
    );
}
