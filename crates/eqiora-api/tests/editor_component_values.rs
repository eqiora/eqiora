use eqiora_api::editor::EditorWorkspaceSnapshot;
use eqiora_core::Span;

fn span(file: &str, source: &str, occurrence: &str, name: &str) -> Span {
    let start = source.find(occurrence).unwrap() as u32;
    Span {
        file: file.to_owned(),
        start,
        end: start + name.len() as u32,
    }
}

#[test]
fn owned_component_values_share_types_navigation_and_exact_references() {
    let source = "// 🧪\r\ncomponent Other(){variable value:s;} component C(parameter gain:1,input inlet:m,output outlet:m){parameter duration:s=1[s];parameter unused:1=0;variable value @{v}:m;port hidden:signal input m;relation r{value=inlet;outlet=value;hidden=value;duration=1[s];gain=1;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    let component = snapshot.symbols().iter().find(|s| s.name() == "C").unwrap();
    for (name, declaration, occurrence, detail) in [
        ("value", "value @{v}", "value=inlet", "dimension L"),
        ("gain", "gain:1", "gain=1", "parameter; Real"),
        ("duration", "duration:s", "duration=1[s]", "dimension T"),
        ("inlet", "inlet:m", "inlet;", "signal Input"),
        ("outlet", "outlet:m", "outlet=value", "signal Output"),
        ("hidden", "hidden:signal", "hidden=value", "signal Input"),
    ] {
        let offset = source.find(occurrence).unwrap() as u32;
        let hover = workspace.assistance(file, offset, name).unwrap();
        assert!(
            hover.detail().unwrap().contains(detail),
            "{name}: {:?}",
            hover.detail()
        );
        let symbol = component
            .children()
            .iter()
            .find(|s| s.name() == name)
            .unwrap();
        assert!(symbol.detail().unwrap().contains(detail));
        let position = snapshot.position(offset).unwrap();
        assert_eq!(
            workspace.value_definition_at_position(file, position),
            Some(span(file, source, declaration, name))
        );
        assert!(
            workspace
                .value_references_at_position(file, position, false)
                .unwrap()
                .contains(&span(file, source, occurrence, name))
        );
    }
    let declaration = span(file, source, "value @{v}", "value");
    let uses =
        ["value=inlet", "value;hidden", "value;duration"].map(|s| span(file, source, s, "value"));
    let position = snapshot.position(declaration.start).unwrap();
    assert_eq!(
        workspace.value_references_at_position(file, position, false),
        Some(uses.to_vec())
    );
    let mut including = vec![declaration];
    including.extend(uses);
    assert_eq!(
        workspace.value_references_at_position(file, position, true),
        Some(including)
    );
    let unused = snapshot
        .position(source.find("unused:1").unwrap() as u32)
        .unwrap();
    assert_eq!(
        workspace.value_references_at_position(file, unused, false),
        Some(vec![])
    );
}

#[test]
fn component_parameter_completion_uses_owned_unspecialized_types() {
    let source = "component C(parameter good:m,parameter bad:s){parameter selected:m=good;}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let (_, items) = workspace
        .completion(file, source.rfind("good;").unwrap() as u32)
        .unwrap();
    let good = items.iter().position(|s| s.name() == "good").unwrap();
    let bad = items.iter().position(|s| s.name() == "bad").unwrap();
    assert!(good < bad);
    assert!(items[good].detail().unwrap().contains("dimension L"));
}

#[test]
fn component_port_references_join_its_own_body_and_proven_model_child_uses() {
    let source = "component C(output value:1){relation r{value=1;}} model M(){instance child:C();relation r{child.value=2;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    let expected = vec![
        span(file, source, "value:1", "value"),
        span(file, source, "value=1", "value"),
        span(file, source, "value=2", "value"),
    ];
    for needle in ["value:1", "value=1", "value=2"] {
        let position = snapshot
            .position(source.find(needle).unwrap() as u32)
            .unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            Some(expected.clone())
        );
    }
}

#[test]
fn component_binders_never_lend_outer_value_identity() {
    for marked in [
        "component C(){parameter value:integer=1;indexset Rows=range(2);relation r[member in Rows]{|value=ordinal(member);}relation ordinary{value=1;}}",
        "component C(){parameter value:integer=1;indexset Rows=range(2);relation r{sum(|value,over=(member in Rows))=2;}relation ordinary{value=1;}}",
        "component Leaf(parameter n:integer){} component C(){parameter value:integer=1;indexset Rows=range(2);instance child[member in Rows]:Leaf(n=|value);relation ordinary{value=1;}}",
    ] {
        let source = marked.replace('|', "");
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{marked}: {:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let position = snapshot.position(marked.find('|').unwrap() as u32).unwrap();
        assert_eq!(workspace.value_definition_at_position(file, position), None);
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            None
        );
        let position = snapshot
            .position(source.find("value:integer").unwrap() as u32)
            .unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, position, false),
            Some(vec![span(file, &source, "value=1;", "value")])
        );
    }
}

#[test]
fn unsupported_component_contracts_and_names_do_not_gain_navigation() {
    for marked in [
        "component C(){let alias:integer=2;relation r{|alias=2;}}",
        "component C(clock tick:periodic){state value:1 at |tick;}",
        "component C(){variable value:m;} model M(){instance child:C();relation r{child.|value=1[m];}}",
        "component C(){port value:signal input 1;} model M(){instance child:C();relation r{child.|value=0;}}",
        "component C(parameter n:integer){variable value:array<1,n>;relation r{|value=value;}}",
        "component C(){parameter n:integer=2;let broken=1[m]+1[s];variable value:array<1,n>;relation r{|value=value;}}",
        "component C(){variable value:m;relation r{|value=1[s];}}",
        "component C(){variable value:m;relation r{|value",
        "component C(){variable value @{|v}:m;}",
    ] {
        let source = marked.replace('|', "");
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
        let file = workspace.files().next().unwrap();
        let position = workspace
            .document(file)
            .unwrap()
            .position(marked.find('|').unwrap() as u32)
            .unwrap();
        assert_eq!(
            workspace.value_definition_at_position(file, position),
            None,
            "{marked}"
        );
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            None,
            "{marked}"
        );
    }
}

#[test]
fn valid_borrowed_signature_field_does_not_become_an_owned_declaration() {
    let source = "component C(variable borrowed:m){relation r{borrowed=1[m];}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    let field = snapshot.symbols()[0]
        .children()
        .iter()
        .find(|s| s.name() == "borrowed")
        .unwrap();
    assert!(field.detail().is_none());
    for needle in ["borrowed:m", "borrowed=1"] {
        let offset = source.find(needle).unwrap() as u32;
        let position = snapshot.position(offset).unwrap();
        assert!(
            workspace
                .assistance(file, offset, "borrowed")
                .unwrap()
                .detail()
                .is_none_or(|detail| !detail.contains("dimension L"))
        );
        assert_eq!(workspace.value_definition_at_position(file, position), None);
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            None
        );
    }
}

#[test]
fn valid_specialization_and_record_members_do_not_supply_owned_source_targets() {
    for (source, needle) in [
        (
            "component C(parameter n:integer){variable value:array<1,n>;relation r{value=value;}} model M(){instance child:C(n=2);}",
            "value=value",
        ),
        (
            "record Config{gain:1} component C(parameter config:Config){relation r{config.gain=1;}}",
            "gain=1",
        ),
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{source}: {:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let position = snapshot
            .position(source.find(needle).unwrap() as u32)
            .unwrap();
        assert_eq!(workspace.value_definition_at_position(file, position), None);
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            None
        );
    }
}
