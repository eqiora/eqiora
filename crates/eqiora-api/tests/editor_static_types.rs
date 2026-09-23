use eqiora_api::editor::{EditorWorkspaceService, EditorWorkspaceSnapshot};

#[test]
fn model_static_extents_share_hover_outline_and_value_navigation() {
    for initializer in ["parameter n:integer=3;", "let n:integer=1+2;"] {
        let source = format!(
            "model Other(){{parameter n:integer=2;variable values:array<1,n>;}} model M(){{{initializer}variable values:array<1,n>;relation r{{values=[0,0,0];}}}}"
        );
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        for (model, shape) in [("Other", "shape [2]"), ("M", "shape [3]")] {
            let field = snapshot
                .symbols()
                .iter()
                .find(|s| s.name() == model)
                .unwrap()
                .children()
                .iter()
                .find(|s| s.name() == "values")
                .unwrap();
            let detail = field.detail().unwrap();
            assert!(detail.contains(shape), "{detail}");
            assert!(detail.contains("array rank 1"), "{detail}");
        }
        let offset = source.rfind("values=").unwrap() as u32;
        let hover = workspace.assistance(file, offset, "values").unwrap();
        assert!(hover.detail().unwrap().contains("shape [3]"));
        let position = snapshot.position(offset).unwrap();
        let target = workspace
            .value_definition_at_position(file, position)
            .unwrap();
        let start = source.rfind("variable values").unwrap() + "variable ".len();
        assert_eq!((target.start, target.end), (start as u32, start as u32 + 6));
        assert_eq!(target.file, file);
        // A static alias may supply an extent, but has no Field/Parameter declaration identity.
        if initializer.starts_with("let") {
            let alias_source = "model M(){let n:integer=1+2;relation r{n=3;}}";
            let alias = EditorWorkspaceSnapshot::analyze_standalone(1, alias_source);
            assert!(alias.diagnostics().is_empty());
            let file = alias.files().next().unwrap();
            let position = alias
                .document(file)
                .unwrap()
                .position(alias_source.rfind("n=3").unwrap() as u32)
                .unwrap();
            assert!(alias.value_definition_at_position(file, position).is_none());
        }
    }
}

#[test]
fn static_parameter_arrays_rank_completion_by_resolved_shape() {
    let source = "model M(){parameter n:integer=3;parameter pair:array<1,2> = [0,0];parameter triple:array<1,n> = [0,0,0];parameter selected:array<1,n> = triple;}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    // Query at the start of the complete value reference, with an empty prefix.
    let (_, items) = workspace
        .completion(file, source.rfind("triple;").unwrap() as u32)
        .unwrap();
    let triple = items.iter().position(|s| s.name() == "triple").unwrap();
    let pair = items.iter().position(|s| s.name() == "pair").unwrap();
    assert!(triple < pair);
    assert!(items[triple].detail().unwrap().contains("shape [3]"));
}

#[test]
fn static_type_projection_follows_preparation_versions_and_invalid_edits() {
    let source = "model M(){parameter n:integer=3;variable values:array<1,n>;}";
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(old.diagnostics().is_empty());
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let changed = source.replace("=3", "=4");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &changed))
        .unwrap();
    let file = current.files().next().unwrap();
    let values = current.document(file).unwrap().symbols()[0]
        .children()
        .iter()
        .find(|s| s.name() == "values")
        .unwrap();
    assert!(values.detail().unwrap().contains("shape [4]"));
    let invalid = source.replace("=3", "=0");
    let invalid = EditorWorkspaceSnapshot::analyze_standalone(3, invalid);
    assert!(!invalid.diagnostics().is_empty());
    let file = invalid.files().next().unwrap();
    let values = invalid.document(file).unwrap().symbols()[0]
        .children()
        .iter()
        .find(|s| s.name() == "values")
        .unwrap();
    assert!(values.detail().is_none());
    // A required value and a borrowed period remain symbolic, with their declared types.
    let free = EditorWorkspaceSnapshot::analyze_standalone(
        4,
        "model M(parameter n:integer,clock tick:periodic){parameter next:integer=n+1;parameter dt:s=period(tick);}",
    );
    assert!(free.diagnostics().is_empty(), "{:?}", free.diagnostics());
    let file = free.files().next().unwrap();
    for (name, expected) in [("next", "Integer"), ("dt", "dimension T")] {
        let symbol = free.document(file).unwrap().symbols()[0]
            .children()
            .iter()
            .find(|s| s.name() == name)
            .unwrap();
        assert!(symbol.detail().unwrap().contains(expected));
    }
}
