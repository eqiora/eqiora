use eqiora_api::editor::{EditorSymbolKind, EditorWorkspaceSnapshot};
use eqiora_core::Span;

fn span(file: &str, source: &str, needle: &str, name: &str) -> Span {
    let start = source.find(needle).unwrap() as u32;
    Span {
        file: file.into(),
        start,
        end: start + name.len() as u32,
    }
}

#[test]
fn event_outline_prose_and_activation_navigation_preserve_source_identity() {
    for owner in ["model", "component"] {
        let source = format!(
            "// 🧪\r\n{owner} Other(){{state x:m;event hit=crossing(x,direction=any);relation r at hit{{next(x)=0[m];}}}} {owner} C(){{state x:m;\n/// Reset at the floor.\nevent hit=crossing(x,direction=falling);let before:m at hit=pre(x);relation reset at hit{{next(x)=1[m];}}event unused=crossing(x-2[m],direction=rising);}}"
        );
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, &source);
        assert!(
            workspace.diagnostics().is_empty(),
            "{:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let owner = snapshot.symbols().iter().find(|s| s.name() == "C").unwrap();
        let event = owner.children().iter().find(|s| s.name() == "hit").unwrap();
        assert_eq!(event.kind(), EditorSymbolKind::Event);
        assert!(
            event.detail().is_none(),
            "no inferred event time or guard type"
        );
        let target = span(file, &source, "hit=crossing(x,direction=falling)", "hit");
        let reference = span(file, &source, "hit{next(x)=1", "hit");
        assert_eq!(
            workspace
                .value_definition_at_position(file, snapshot.position(reference.start).unwrap()),
            Some(target.clone())
        );
        let alias = span(file, &source, "hit=pre", "hit");
        for cursor in [&target, &alias, &reference] {
            let position = snapshot.position(cursor.start).unwrap();
            assert_eq!(
                workspace.value_references_at_position(file, position, false),
                Some(vec![alias.clone(), reference.clone()])
            );
            assert_eq!(
                workspace.value_references_at_position(file, position, true),
                Some(vec![target.clone(), alias.clone(), reference.clone()])
            );
            let hover = workspace.assistance(file, cursor.start, "hit").unwrap();
            assert_eq!(hover.kind(), EditorSymbolKind::Event);
            assert!(
                hover
                    .detail()
                    .unwrap()
                    .contains("crossing(x,direction=falling)")
            );
            assert_eq!(
                hover.doc_comment().unwrap().summary(),
                "Reset at the floor."
            );
            // The existing safe Markdown renderer encodes dots to prevent autolinks.
            assert_eq!(
                hover.documentation().as_deref(),
                Some("Reset at the floor&#46;")
            );
            assert!(!hover.detail().unwrap().contains("periodic clock;"));
        }
        let unused = span(file, &source, "unused=crossing", "unused");
        let position = snapshot.position(unused.start).unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, position, false),
            Some(vec![])
        );
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            Some(vec![unused])
        );
    }
}

#[test]
fn event_direction_tokens_are_not_references_and_invalid_snapshots_clear_navigation() {
    let source = "model M(){state x:m;event falling=crossing(x,direction=falling);relation reset at falling{next(x)=1[m];}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    let target = span(file, source, "falling=crossing", "falling");
    let reference = span(file, source, "falling{", "falling");
    assert_eq!(
        workspace.value_references_at_position(
            file,
            snapshot.position(target.start).unwrap(),
            true
        ),
        Some(vec![target.clone(), reference])
    );
    let direction = snapshot
        .position(source.find("falling);").unwrap() as u32)
        .unwrap();
    assert!(
        workspace
            .value_definition_at_position(file, direction)
            .is_none()
    );
    assert!(
        workspace
            .value_references_at_position(file, direction, true)
            .is_none()
    );
    for invalid in [
        source.replace("direction=falling", "direction=unknown"),
        source.replace("1[m]", "1[s]"),
        source.replace("next(x)=1[m];}", "next(x)="),
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(2, &invalid);
        assert!(!workspace.diagnostics().is_empty());
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let position = snapshot
            .position(invalid.find("falling{").unwrap() as u32)
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
}
