use eqiora_api::editor::{EditorPosition, EditorWorkspaceService, EditorWorkspaceSnapshot};
use eqiora_core::Span;
use eqiora_lang::TextRange;

fn range(source: &str, occurrence: &str, name: &str) -> TextRange {
    let start = source.find(occurrence).unwrap() as u32;
    TextRange::new(start, start + name.len() as u32)
}

fn references(source: &str, occurrence: &str, include_declaration: bool) -> Option<Vec<TextRange>> {
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{source}: {:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let position = workspace
        .document(file)?
        .position(source.find(occurrence)? as u32)?;
    workspace
        .value_references_at_position(file, position, include_declaration)
        .map(|spans| {
            spans
                .into_iter()
                .map(|span| {
                    assert_eq!(span.file, file);
                    TextRange::new(span.start, span.end)
                })
                .collect()
        })
}

#[test]
fn declarations_and_values_return_only_the_current_models_exact_identifiers() {
    let source = "// 🧪\r\nmodel Other(){parameter rate:1=2;variable x:1;relation r{x=rate;}}\r\nmodel M(){parameter rate:1=1;parameter unused:1=0;variable x:1;relation r{x=(// 🧪\r\nrate)+rate*rate;}}";
    let last = source.rfind("rate;}").unwrap() as u32;
    let rate_uses = vec![
        range(source, "rate)+", "rate"),
        range(source, "rate*", "rate"),
        TextRange::new(last, last + 4),
    ];
    for origin in ["rate:1=1", "rate)+", "rate*", "rate*rate"] {
        assert_eq!(references(source, origin, false), Some(rate_uses.clone()));
        let mut expected = vec![range(source, "rate:1=1", "rate")];
        expected.extend(&rate_uses);
        assert_eq!(references(source, origin, true), Some(expected));
    }
    let x_use = range(source, "x=(", "x");
    assert_eq!(references(source, "x=(", false), Some(vec![x_use]));
    assert_eq!(references(source, "unused:", false), Some(vec![]));
    assert_eq!(
        references(source, "unused:", true),
        Some(vec![range(source, "unused:", "unused")])
    );
    assert_eq!(
        references(source, "rate:1=2", false),
        Some(vec![range(source, "rate;}", "rate")])
    );
}

#[test]
fn valid_binders_and_qualified_names_do_not_join_the_local_reference_set() {
    let source = "model M(){parameter value:m=1[m];indexset Rows=range(2);relation ordinary{value=1[m];}relation reduction{sum(value,over=(member in Rows))=2[m];}relation family[member in Rows]{value=1[m];}}";
    assert_eq!(
        references(source, "value:m", false),
        Some(vec![range(
            source,
            "value=1[m];}relation reduction",
            "value"
        )])
    );
    assert_eq!(references(source, "value,over", false), None);
    assert_eq!(references(source, "value=1[m];}}", false), None);
    let source = "record Config{gain:1,ready:bool} model M(){parameter config:Config=Config(ready=true,gain=4);parameter gain:1=2;variable y:1;relation law{y=config.gain+gain;}}";
    assert_eq!(
        references(source, "gain:1=2", false),
        Some(vec![range(source, "gain;}", "gain")])
    );
    assert_eq!(references(source, "gain+", false), None);
    assert_eq!(references(source, "config.gain", false), None);
    let source = "model M(){parameter m:1=1;variable y:m;}";
    assert_eq!(references(source, "m:1", false), Some(vec![]));
    assert_eq!(references(source, "m;", false), None);
}

#[test]
fn unsupported_positions_do_not_acquire_references_from_name_recovery() {
    for marked in [
        "component C(){parameter x:1=1;relation r{|x=1;}} model M(){}",
        "model M(){parameter x:1=1;let alias=x;relation r{|alias=1;}}",
        "model M(){parameter x:1=1;} // |x",
        "model M(){parameter x:1=1;relation r{x|=1;}}",
        "model M(){parameter x:1=1;variable y:|x;}",
        "model A(){parameter x:1=1;} model B(){relation r{|x=1;}}",
        "model M(){parameter x:1=1;indexset Rows=range(2);relation r[x in Rows]{ordinal(|x)=0;}}",
        "model M(){parameter x:1=1;relation r{|x",
    ] {
        let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, marked.replace('|', ""));
        let file = workspace.files().next().unwrap();
        let position = workspace
            .document(file)
            .unwrap()
            .position(marked.find('|').unwrap() as u32)
            .unwrap();
        assert!(
            workspace
                .value_references_at_position(file, position, true)
                .is_none(),
            "{marked}"
        );
    }
}

#[test]
fn references_use_current_unsaved_versions_and_validate_utf16_positions() {
    let source = "// 🧪\r\nmodel M(){parameter rate:1=1;relation r{rate=1;}}";
    let old = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    let mut service = EditorWorkspaceService::new(old.clone());
    service.begin(2).unwrap();
    assert!(service.current().is_none());
    assert!(service.replace(old).is_err());
    let source = source.replace("relation r{", "relation r{\r\n");
    let current = service
        .replace(EditorWorkspaceSnapshot::analyze_standalone(2, &source))
        .unwrap();
    let file = current.files().next().unwrap();
    let snapshot = current.document(file).unwrap();
    let position = snapshot
        .position(source.rfind("rate").unwrap() as u32)
        .unwrap();
    let actual = current
        .value_references_at_position(file, position, false)
        .unwrap();
    assert_eq!(actual, vec![at(file, range(&source, "rate=", "rate"))]);
    assert_eq!(
        snapshot.position(actual[0].start),
        Some(EditorPosition::new(2, 0))
    );
    for position in [
        EditorPosition::new(0, 4),
        EditorPosition::new(99, 0),
        EditorPosition::new(1, 999),
    ] {
        assert!(
            current
                .value_references_at_position(file, position, true)
                .is_none()
        );
    }
    assert!(
        current
            .value_references_at_position("missing", position, true)
            .is_none()
    );
}

#[test]
fn notation_between_name_and_type_preserves_exact_navigation() {
    let source = "// 🧪\r\nmodel Other(){parameter m:1=2;}model M(){parameter m @{m_0}:1=1;variable x @{x_i}:1;parameter unused @{u}:1=0;relation r{x=m;}}";
    let workspace = EditorWorkspaceSnapshot::analyze_standalone(1, source);
    assert!(
        workspace.diagnostics().is_empty(),
        "{:?}",
        workspace.diagnostics()
    );
    let file = workspace.files().next().unwrap();
    let snapshot = workspace.document(file).unwrap();
    for (name, declaration, reference) in [("m", "m @{", "m;"), ("x", "x @{", "x=m")] {
        let target = range(source, declaration, name);
        let occurrence = range(source, reference, name);
        let position = snapshot.position(occurrence.start()).unwrap();
        assert_eq!(
            workspace.value_definition_at_position(file, position),
            Some(eqiora_core::Span {
                file: file.to_owned(),
                start: target.start(),
                end: target.end()
            })
        );
        for cursor in [target.start(), occurrence.start()] {
            assert_eq!(
                workspace.value_references_at_position(
                    file,
                    snapshot.position(cursor).unwrap(),
                    true
                ),
                Some(vec![at(file, target), at(file, occurrence)])
            );
            assert_eq!(
                workspace.value_references_at_position(
                    file,
                    snapshot.position(cursor).unwrap(),
                    false
                ),
                Some(vec![at(file, occurrence)])
            );
        }
    }
    assert_eq!(references(source, "unused @{", false), Some(vec![]));
    assert_eq!(
        references(source, "unused @{", true),
        Some(vec![range(source, "unused @{", "unused")])
    );
    for notation in ["m_0", "x_i"] {
        let position = snapshot
            .position(source.find(notation).unwrap() as u32)
            .unwrap();
        assert_eq!(
            workspace.value_references_at_position(file, position, true),
            None
        );
        assert_eq!(workspace.value_definition_at_position(file, position), None);
    }
}

fn at(file: &str, range: TextRange) -> Span {
    Span {
        file: file.to_owned(),
        start: range.start(),
        end: range.end(),
    }
}
