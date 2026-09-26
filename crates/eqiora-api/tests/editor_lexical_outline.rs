use eqiora_api::editor::{EditorService, EditorSymbolKind, EditorWorkspaceSnapshot};

#[test]
fn component_lets_and_model_relation_families_keep_authored_outline_and_prose() {
    let source = "// 🧪\r\ncomponent C(){\n/// Scaled value.\nlet gain:1=2;relation r{gain=2;}} model M(){clock tick=periodic(1[s]);indexset Rows=range(2);\n/// Each row.\nrelation balance[row in Rows]{period(tick)=1[s];}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (owner, name, kind, head, summary) in [
        (
            "C",
            "gain",
            EditorSymbolKind::Let,
            "let gain:1=2;",
            "Scaled value.",
        ),
        (
            "M",
            "balance",
            EditorSymbolKind::Relation,
            "relation balance[row in Rows]{period(tick)=1[s];}",
            "Each row.",
        ),
    ] {
        let symbol = snapshot
            .symbols()
            .iter()
            .find(|s| s.name() == owner)
            .unwrap()
            .children()
            .iter()
            .find(|s| s.name() == name)
            .unwrap();
        assert_eq!(symbol.kind(), kind);
        assert_eq!(
            &source[symbol.range().start() as usize..symbol.range().end() as usize],
            head
        );
        assert_eq!(symbol.doc_comment().unwrap().summary(), summary);
        assert!(symbol.detail().is_none());
        let offset = source
            .find(if name == "gain" {
                "gain:1"
            } else {
                "balance[row"
            })
            .unwrap() as u32;
        let hover = workspace.assistance(file, offset, name).unwrap();
        assert_eq!(hover.kind(), kind);
        assert_eq!(hover.doc_comment().unwrap().summary(), summary);
        assert!(hover.detail().unwrap().contains(name));
    }
    for needle in ["gain=2", "tick)=1"] {
        let position = snapshot
            .position(source.find(needle).unwrap() as u32)
            .unwrap();
        assert!(
            workspace
                .value_definition_at_position(file, position)
                .is_none()
        );
        assert!(
            workspace
                .value_references_at_position(file, position, true)
                .is_none()
        );
    }
    // A later incomplete declaration must not erase already recovered siblings.
    let incomplete = format!("{source} model Broken(){{variable missing:");
    let service = EditorService::new("recovery.eqi", 2, incomplete);
    assert!(!service.current().diagnostics().is_empty());
    for (owner, name) in [("C", "gain"), ("M", "balance")] {
        assert!(
            service
                .current()
                .symbols()
                .iter()
                .find(|s| s.name() == owner)
                .unwrap()
                .children()
                .iter()
                .any(|s| s.name() == name)
        );
    }
}

#[test]
fn index_sets_and_observables_keep_distinct_authored_symbols_without_value_navigation() {
    for owner in ["model", "component"] {
        let source = format!(
            "{owner} C(){{\n/// Bound indices.\nindexset Rows=range(2);\n/// Derived total.\nobservable total:1=2;}}"
        );
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        for (name, kind, declaration, summary) in [
            (
                "Rows",
                EditorSymbolKind::IndexSet,
                "indexset Rows=range(2);",
                "Bound indices.",
            ),
            (
                "total",
                EditorSymbolKind::Observable,
                "observable total:1=2;",
                "Derived total.",
            ),
        ] {
            let symbol = snapshot.symbols()[0]
                .children()
                .iter()
                .find(|s| s.name() == name)
                .unwrap();
            assert_eq!(symbol.kind(), kind);
            assert_eq!(
                &source[symbol.range().start() as usize..symbol.range().end() as usize],
                declaration
            );
            assert_eq!(symbol.doc_comment().unwrap().summary(), summary);
            assert!(symbol.detail().is_none());
            let offset = source
                .find(&format!("{name}="))
                .or_else(|| source.find(&format!("{name}:")))
                .unwrap() as u32;
            let hover = workspace.assistance(file, offset, name).unwrap();
            assert_eq!(hover.kind(), kind);
            assert_eq!(hover.doc_comment().unwrap().summary(), summary);
            assert!(
                hover
                    .detail()
                    .unwrap()
                    .contains(declaration.trim_end_matches(';'))
            );
            let position = snapshot.position(offset).unwrap();
            assert!(
                workspace
                    .value_definition_at_position(file, position)
                    .is_none()
            );
            assert!(
                workspace
                    .value_references_at_position(file, position, true)
                    .is_none()
            );
        }
    }
}
