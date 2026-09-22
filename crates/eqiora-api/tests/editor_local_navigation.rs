use eqiora_api::editor::{EditorPosition, EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_lang::TextRange;

fn target(marked: &str) -> Option<TextRange> {
    let offset = marked.find('|').unwrap() as u32;
    let source = marked.replacen('|', "", 1);
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    let file = workspace.files().next().unwrap();
    let position = workspace.document(file)?.position(offset)?;
    workspace.local_definition_at_position(file, position)
}

#[test]
fn local_navigation_returns_the_current_models_exact_declaration_name() {
    let source = "// 🧪\r\nmodel Other(){parameter rate:1=2;variable x:1;}\r\nmodel M(){parameter rate:1/s=1[1/s];state x:1;relation r{derivative(x)+rate*x=0;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (occurrence, declaration, prefix, name) in [
        ("x)+", "state x:1;relation", "state ", "x"),
        ("rate*x", "parameter rate:1/s", "parameter ", "rate"),
    ] {
        let expected = (source.find(declaration).unwrap() + prefix.len()) as u32;
        let position = snapshot
            .position(source.find(occurrence).unwrap() as u32)
            .unwrap();
        let range = workspace
            .local_definition_at_position(file, position)
            .unwrap();
        assert_eq!(
            range,
            TextRange::new(expected, expected + name.len() as u32)
        );
        assert_eq!(snapshot.position(range.start()).unwrap().line(), 2);
    }
    for position in [
        EditorPosition::new(0, 4),
        EditorPosition::new(99, 0),
        EditorPosition::new(2, 999),
    ] {
        assert!(
            workspace
                .local_definition_at_position(file, position)
                .is_none()
        );
    }
    assert!(
        workspace
            .local_definition_at_position("missing", EditorPosition::new(0, 0))
            .is_none()
    );
}

#[test]
fn local_navigation_never_invents_a_target_from_lexical_recovery() {
    for source in [
        "model A(){variable x:1;} model B(){relation r{|x=0;}}",
        "model M(){parameter value:m=1[m];indexset Rows=range(2);relation r{sum(ordinal(|value),over=(value in Rows))=1;}}",
        "component C(){variable x:1;relation r{|x=0;}} model M(){}",
        "component C(output x:1){} model M(){instance child:C();relation r{child.|x=0;}}",
        "model M(){port value:signal input 1;relation r{|value=0;}}",
        "model M(){parameter value:1=1;let alias=value;relation r{|alias=1;}}",
        "model M(){variable x:1;relation r{|x",
        "model M(){variable x:1;} // |x",
        "model M(){variable |x:1;}",
        "model M(){parameter x:1=1;variable y:|x;}",
        "model M(){variable x:1;relation r{x|=0;}}",
    ] {
        assert_eq!(target(source), None, "{source}");
    }
}

#[test]
fn valid_nested_binder_scopes_remain_unsupported() {
    for marked in [
        "model M(){parameter value:m=1[m];indexset Rows=range(2);relation r{sum(|value,over=(member in Rows))=2[m];}}",
        "model M(){parameter value:m=1[m];indexset Rows=range(2);relation r[member in Rows]{|value=1[m];}}",
        "component C(parameter length:integer){} model M(){parameter value:integer=1;indexset Rows=range(2);instance child[member in Rows]:C(length=|value);}",
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replace('|', ""));
        assert!(
            workspace.diagnostics().is_empty(),
            "{marked}: {:?}",
            workspace.diagnostics()
        );
        let file = workspace.files().next().unwrap();
        let snapshot = workspace.document(file).unwrap();
        let position = snapshot.position(marked.find('|').unwrap() as u32).unwrap();
        assert!(
            workspace
                .local_definition_at_position(file, position)
                .is_none(),
            "{marked}"
        );
    }
}

#[test]
fn local_navigation_tracks_unsaved_versions_and_rejects_stale_publication() {
    let old_source = "model M(){parameter rate:1=1;relation r{rate=0;}}";
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, old_source);
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.snapshot(1).is_err());
    assert!(service.replace(old).is_err());
    let source = old_source.replace("parameter rate", "\nparameter rate");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &source))
        .unwrap();
    let file = current.files().next().unwrap();
    let snapshot = current.document(file).unwrap();
    let position = snapshot
        .position(source.rfind("rate").unwrap() as u32)
        .unwrap();
    let target = current
        .local_definition_at_position(file, position)
        .unwrap();
    assert_eq!(
        snapshot.position(target.start()),
        Some(EditorPosition::new(1, 10))
    );
}
