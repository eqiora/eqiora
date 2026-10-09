use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{FieldRole, KernelNode};

#[test]
fn flat_source_symbols_exclude_synthesized_owners_but_keep_kernel_nodes() {
    let source = "model M() { domain body = box(0,1); state x: 1 on body; initial { x = 1; } relation law on body { derivative(x) = 0; } }";
    let compiled = compile("symbols.eqi", source).unwrap().remove(0);
    assert_eq!(
        compiled
            .symbols()
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["body", "law", "x"]
    );
    let nodes = compiled
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
            .filter(|node| matches!(node, KernelNode::Representation(_)))
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Relation(relation) if relation.is_initial()))
            .count(),
        1
    );
    for (_, id) in compiled.symbols().iter() {
        assert!(nodes.iter().any(|node| node.id() == id));
    }
}

#[test]
fn native_symbols_exclude_unnamed_initial_relations() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};
    use eqiora_lang::{DraftDeclaration, DraftExpression, DraftField, FieldRoleSyntax, Module};
    let state = DraftField::new(
        "x",
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("admitted numeric scalar type"),
        FieldRoleSyntax::State,
    );
    let condition = (
        state.expression(),
        DraftExpression::constant(eqiora_lang::DecimalLiteral::parse("1").unwrap()),
    );
    let draft = Module::new(
        "M",
        [state.into(), DraftDeclaration::Initial(vec![condition])],
    )
    .unwrap();
    let compiled = eqiora_compiler::lower_module(&draft, None, &[]).unwrap();
    assert_eq!(
        compiled
            .symbols()
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["x"]
    );
    assert!(compiled.transaction().ops().iter().any(|op| matches!(op,
        Op::DefineKernelNode { node: KernelNode::Relation(relation) } if relation.is_initial()
    )));
}

fn nodes(source: &str) -> Vec<KernelNode> {
    compile("state.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}\n{source}"))[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn algebraic_state_support_and_initial_owners_are_independent() {
    let result = nodes(
        "model M() { domain body = box(0,1); variable a: 1; state x: 1; variable p: 1 on body; state t: 1 on body; initial { x = 1; t = 0; } relation law { derivative(x) = 0; a = x; } }",
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Field(f) if f.role() == FieldRole::State))
            .count(),
        2
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Field(f) if f.role() == FieldRole::Variable))
            .count(),
        2
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Representation(_)))
            .count(),
        1
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Relation(r) if r.is_initial()))
            .count(),
        1
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Activation(_)))
            .count(),
        1
    );
}

#[test]
fn only_declared_continuous_states_have_authored_time_derivatives() {
    nodes(
        "model M() { state x: 1; initial { derivative(x) = 0; } relation law { derivative(x) = 0; } }",
    );
    for source in [
        "model M() { variable x: 1; relation law { derivative(x) = 0; } }",
        "model M() { variable x: 1; relation law { derivative(derivative(x)) = 0; } }",
        "model M() { variable x: 1; initial { derivative(x) = 0; } }",
        "model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c; relation law { derivative(x) = 0; } }",
        "model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c; relation law { derivative(derivative(x)) = 0; } }",
    ] {
        assert!(compile("bad.eqi", source).is_err(), "{source}");
    }
}

#[test]
fn clocked_state_initializes_pre_and_requires_exact_clock() {
    nodes(
        "model M() { domain body = box(0,1); clock c = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 on body at c; initial { pre(x) = 0; } relation law on body at c { next(x) = pre(x); } }",
    );
    for source in [
        "model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); clock d = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c; relation law at d { next(x) = pre(x); } }",
        "model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); variable x: 1 at c; relation law at c { next(x) = pre(x); } }",
        "model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c; initial { next(x) = 0; } }",
    ] {
        assert!(compile("bad.eqi", source).is_err(), "{source}");
    }
}

#[test]
fn borrowed_roles_preserve_target_and_cannot_launder_state_eligibility() {
    let source = "component Reader(variable value: 1) { relation law { value = 0; } } model M() { state x: 1; instance r: Reader(value = x); }";
    assert_eq!(
        nodes(source)
            .iter()
            .filter(|n| matches!(n, KernelNode::Field(_)))
            .count(),
        1
    );
    for source in [
        "component Writer(state value: 1) { relation law { derivative(value) = 0; } } model M() { variable x: 1; instance w: Writer(value = x); }",
        "component Writer(state value: 1) { relation law { derivative(value) = 0; } } component Forward(variable value: 1) { instance w: Writer(value = value); } model M() { state x: 1; instance f: Forward(value = x); }",
        "component Reader(variable value: 1) { relation law { derivative(value) = 0; } } model M() { state x: 1; instance r: Reader(value = x); }",
    ] {
        assert!(compile("bad.eqi", source).is_err(), "{source}");
    }
}

#[test]
fn borrowed_clocks_and_states_forward_exact_targets() {
    let source = "component Delay(clock tick: periodic, state value: 1 at tick) { initial { pre(value) = 0; } relation law at tick { next(value) = pre(value); } } component Forward(clock tick: periodic, state value: 1 at tick) { instance d: Delay(tick = tick, value = value); } model M() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c; instance f: Forward(tick = c, value = x); }";
    let result = nodes(source);
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::Field(_)))
            .count(),
        1
    );
    assert_eq!(
        result
            .iter()
            .filter(|n| matches!(n, KernelNode::ClockDomain(_)))
            .count(),
        1
    );
    let wrong = source
        .replace("tick = c, value", "tick = other, value")
        .replace(
            "state x: 1 at c;",
            "clock other = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at c;",
        );
    assert!(compile("wrong.eqi", &wrong).is_err());
}

#[test]
fn initial_equation_identity_ignores_unrelated_declaration_order() {
    let first = nodes(
        "component Marker() {} model M() { state x: 1; variable a: 1; initial { x = 1; } initial { a = 2; } }",
    );
    let second = nodes(
        "component Marker() {} model M() { initial { a = 2; } variable a: 1; initial { x = 1; } state x: 1; }",
    );
    let initial_ids = |nodes: Vec<KernelNode>| {
        nodes
            .into_iter()
            .filter_map(|node| match node {
                KernelNode::Relation(relation) if relation.is_initial() => {
                    Some(relation.id().erase())
                }
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(initial_ids(first), initial_ids(second));
}

#[test]
fn initialization_has_no_implicit_values_or_scalar_broadcast() {
    assert!(
        !nodes("model M() { variable x: 1; state y: 1; relation r { x = y; } }")
            .iter()
            .any(|node| matches!(node, KernelNode::Relation(r) if r.is_initial()))
    );
    nodes(
        "model M() { domain body = box(0,1,0,1); state x: vector<m,2> on body; initial { x = 0; } }",
    );
    assert!(compile("broadcast.eqi", "model M() { domain body = box(0,1,0,1); state x: vector<m,2> on body; initial { x = 1[m]; } }").is_err());
}

#[test]
fn unused_component_clock_ownership_is_checked_at_its_definition() {
    for source in [
        "component C() { state x: 1 at missing; } model M() { variable y: 1; relation r { y = 0; } }",
        "component C(state x: 1 at hidden) { clock hidden = periodic(1[s] / 1, phase = 0[s] / 1); } model M() { variable y: 1; relation r { y = 0; } }",
    ] {
        assert!(compile("unused.eqi", source).is_err(), "{source}");
    }
}

#[test]
fn continuum_owner_is_shared_by_exact_support_across_components() {
    use eqiora_core::entity::EntityKind;
    use eqiora_graph::EdgeKind;
    let source = "component C(support body: volume(ambient_dimension = 1)) { variable load: 1 on body; relation r on body { load = 0; } } model M() { domain a = box(0,1); domain b = box(0,1); state x: 1 on a; variable y: 1 on a; variable z: 1 on b; instance c: C(body = a); relation r on a { x = y; } }";
    let compiled = compile("shared.eqi", source).unwrap();
    let model = &compiled[0];
    let representation = |field: &str| {
        let id = model.symbols().get(field).unwrap();
        model
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::DefinedOn,
                } if *from == id && to.kind() == EntityKind::Representation => Some(*to),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(representation("x"), representation("y"));
    assert_eq!(representation("x"), representation("c.load"));
    assert_ne!(representation("x"), representation("z"));
}
