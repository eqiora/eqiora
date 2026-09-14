use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::KernelNode;

fn accepted(source: &str) {
    compile("runtime.eqi", source).unwrap_or_else(|errors| panic!("{source}\n{errors:?}"));
}
fn rejected(source: &str) {
    eqiora_lang::parse("runtime.eqi", source)
        .into_document()
        .unwrap();
    assert!(compile("runtime.eqi", source).is_err(), "{source}");
}

#[test]
fn runtime_scalar_aliases_and_identity_evolution_work_in_both_containers() {
    accepted(
        "model M() { state x: 1; let y = z * x; let z = x + 1; let identity = x; initial { identity = 2; } relation r { derivative(identity) = y / 1[s]; } }",
    );
    accepted(
        "component C(state x: 1) { let y = z * x; let z = x + 1; let identity = x; relation r { derivative(identity) = y / 1[s]; } } model M() { state x: 1; instance c: C(x = x); }",
    );
    accepted(
        "model M() { state x: 1; let composite = 2*x; relation r { derivative(composite) = 0; } }",
    );
    rejected(
        "model M() { variable x: 1; let composite = 2*x; relation r { derivative(composite) = 0; } }",
    );
}

#[test]
fn unused_aliases_validate_types_supports_and_full_dependency_cycles() {
    for body in [
        "state x: 1; let a = x + b; let b = a;",
        "state x: 1; let a: m = x;",
        "state x: 1; let a = x + 1[m];",
        "let a = missing;",
        "state x: 1; let a = grad(x);",
        "let a = coordinate(0);",
    ] {
        rejected(&format!("model M() {{ {body} }}"));
        rejected(&format!("component Unused() {{ {body} }} model M() {{}}"));
    }
    rejected(
        "model M() { domain a = box(0,1); domain b = box(0,1); variable x: 1 on a; variable y: 1 on b; let bad = x+y; }",
    );
    accepted(
        "model M() { domain body = box(0,1); variable x: 1 on body; let q = grad(x); let residual = div(q); relation r on body { residual = 0; } }",
    );
    rejected(
        "model M() { domain body = box(0,1); variable x: 1 on body; let trace_alias = trace(x); }",
    );
}

#[test]
fn runtime_effects_do_not_enter_static_bindings_transitively() {
    rejected(
        "component C(parameter p: 1) {  } model M() { state x: 1; let a = x; let b = a+1; instance c: C(p = b); }",
    );
    rejected(
        "component C(state x: 1) { let a=x; let b=a+1; instance d: D(p=b); } component D(parameter p: 1) {  } model M() {}",
    );
    rejected("model M() { variable x: 1; let a = x; domain body = box(0,a); }");
}

#[test]
fn deferred_clock_obligations_are_checked_at_each_use() {
    let prefix = "clock a = periodic(1[s] / 1, phase = 0[s] / 1); clock b = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at a; let read = x; let old = pre(read); let future = next(read);";
    accepted(&format!(
        "model M() {{ {prefix} initial {{ old = 0; }} relation r at a {{ future = old+1; }} }}"
    ));
    accepted(&format!(
        "model M() {{ {prefix} relation r at b {{ read=0; }} }}"
    ));
    // Unused pre/next aliases have intrinsic types without choosing a consumer clock.
    accepted(&format!(
        "model M() {{ {prefix} relation unused_alias_context {{ x=0; }} }}"
    ));
    for law in [
        "relation r at b { old=0; }",
        "relation r { old=0; }",
        "initial { future=0; }",
    ] {
        rejected(&format!("model M() {{ {prefix} {law} }}"));
    }
    rejected(
        "model M() { state x: 1; let dx=derivative(x); clock c = periodic(1[s] / 1, phase = 0[s] / 1); relation r at c { dx=0; } }",
    );
}

#[test]
fn runtime_diamond_stays_a_shared_dag_without_alias_entities() {
    let mut source = String::from("model M() { state x: 1; let a0 = x+x;");
    for index in 1..45 {
        source.push_str(&format!("let a{index}=a{}+a{};", index - 1, index - 1));
    }
    source.push_str("relation r { a44=0; } }");
    let compiled = compile("diamond.eqi", &source).unwrap();
    let model = &compiled[0];
    assert_eq!(model.symbols().iter().count(), 2);
    let nodes = model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nodes
            .iter()
            .filter(|n| matches!(n, KernelNode::Field(_)))
            .count(),
        1
    );
    let relations = nodes
        .iter()
        .filter_map(|n| match n {
            KernelNode::Relation(r) => Some(r),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(relations.len(), 1);
    assert!(relations[0].expression().nodes().len() < 100);
}

#[test]
fn parameter_only_scalar_math_aliases_remain_static_and_keep_domain_checks() {
    accepted("model M() { let a:1=math.sin(0); let b:1=math.sqrt(4); relation r { a+b=2; } }");
    rejected("model M() { let a:1=math.sin(1[m]); relation r { a=0; } }");
    accepted(
        "component C(parameter p: 1) {  relation r { p=0; } } model M() { parameter p: 1=4; let a=math.sqrt(p); let b=math.sin(a); instance c:C(p=b); }",
    );
    accepted(
        "component C(parameter p: 1) {  let a=math.sqrt(p); instance d:D(p=a); } component D(parameter p:1) {  relation r { p=0; } } model M() { parameter p:1=4; instance c:C(p=p); }",
    );
    rejected("model M() { parameter p:1=-1; let a=math.sqrt(p); relation r { a=0; } }");
    rejected(
        "component C(parameter p:1) {  let a=math.sqrt(p); relation r { a=0; } } model M() { instance c:C(p=-1); }",
    );
}

#[test]
fn long_identity_chain_preserves_eligibility_without_recursive_expansion() {
    let mut source = String::from("model M() { state x:1; let a0=x;");
    for index in 1..256 {
        source.push_str(&format!("let a{index}=a{};", index - 1));
    }
    source.push_str("relation r { derivative(a255)=0; } }");
    accepted(&source);
}

#[test]
fn runtime_aliases_can_read_public_child_ports_after_child_allocation() {
    accepted(
        "component Child(output output: 1) { relation r { output=2; } } model M() { instance child: Child(); let observed=child.output; relation r { observed=2; } }",
    );
    accepted(
        "component Child(output output: 1) { relation r { output=2; } } component Parent() { instance child: Child(); let observed=child.output; relation r { observed=2; } } model M() { instance parent: Parent(); }",
    );
}
