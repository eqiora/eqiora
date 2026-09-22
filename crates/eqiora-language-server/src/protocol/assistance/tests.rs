use super::*;
use serde_json::json;

fn fixture(marked: &str) -> (ServerState, Uri, lsp_types::Position) {
    let offset = marked.find('|').expect("cursor");
    let source = marked.replacen('|', "", 1);
    let uri: Uri = "file:///workspace/main.eqi".parse().unwrap();
    let open = super::super::OpenDocument::new(uri.clone(), 1, source);
    let position = open
        .snapshot()
        .position(u32::try_from(offset).unwrap())
        .unwrap();
    let mut state = ServerState::new(vec![]);
    state.documents.insert(uri.as_str().into(), open);
    (
        state,
        uri,
        lsp_types::Position::new(position.line(), position.character()),
    )
}

fn complete(marked: &str) -> Vec<CompletionItem> {
    let (state, uri, position) = fixture(marked);
    let params =
        serde_json::from_value(json!({"textDocument":{"uri":uri},"position":position})).unwrap();
    let CompletionResponse::Array(items) = completion(params, &state).unwrap() else {
        panic!("array")
    };
    items
}

#[test]
fn protocol_preserves_compiler_ranking_in_client_sort_text() {
    let marked = "model M() { parameter value_bad:s=1[s]; parameter value_good:m=2[m]; parameter result:m=value_|; }";
    let (mut state, uri, position) = fixture(marked);
    let snapshot =
        eqiora::api::EditorWorkspaceSnapshot::analyze_standalone(1, marked.replace('|', ""));
    let file = snapshot.files().next().unwrap().to_owned();
    let group = state.group_for_uri(uri.as_str());
    state.workspaces.insert(
        group,
        super::super::WorkspaceAnalysis {
            snapshot,
            file_by_uri: [(uri.as_str().to_owned(), file.clone())].into(),
            uri_by_file: [(file, uri.clone())].into(),
        },
    );
    let params =
        serde_json::from_value(json!({"textDocument":{"uri":uri},"position":position})).unwrap();
    let CompletionResponse::Array(items) = completion(params, &state).unwrap() else {
        panic!("array")
    };
    assert_eq!(
        items
            .iter()
            .map(|item| item.label.as_str())
            .collect::<Vec<_>>(),
        ["value_good", "value_bad"]
    );
    assert!(items[0].sort_text < items[1].sort_text);
    assert!(items[0].detail.as_ref().unwrap().contains("dimension L"));
}

#[test]
fn navigation_hover_projects_current_types_and_exact_utf16_reference_range() {
    let marked = "// 🧪\nmodel M(){\n/// Measured length.\nvariable value:array<m,2>;relation r{va|lue=value;}}";
    let (mut state, uri, position) = fixture(marked);
    let snapshot =
        eqiora::api::EditorWorkspaceSnapshot::analyze_standalone(1, marked.replace('|', ""));
    let file = snapshot.files().next().unwrap().to_owned();
    let group = state.group_for_uri(uri.as_str());
    state.workspaces.insert(
        group,
        super::super::WorkspaceAnalysis {
            snapshot,
            file_by_uri: [(uri.as_str().to_owned(), file.clone())].into(),
            uri_by_file: [(file, uri.clone())].into(),
        },
    );
    let params =
        serde_json::from_value(json!({"textDocument":{"uri":uri},"position":position})).unwrap();
    let result = super::super::navigation::hover(params, &state)
        .unwrap()
        .unwrap();
    let range = result.range.unwrap();
    assert_eq!(
        range.start,
        lsp_types::Position::new(position.line, position.character - 2)
    );
    assert_eq!(
        range.end,
        lsp_types::Position::new(position.line, position.character + 3)
    );
    let HoverContents::Markup(contents) = result.contents else {
        panic!("Markdown")
    };
    for fact in ["dimension L", "shape [2]", "Variable", "Measured length"] {
        assert!(
            contents.value.contains(fact),
            "missing {fact:?} in {:?}",
            contents.value
        );
    }
}

fn signature(marked: &str) -> Option<SignatureHelp> {
    let (state, uri, position) = fixture(marked);
    let params =
        serde_json::from_value(json!({"textDocument":{"uri":uri},"position":position})).unwrap();
    signature_help(params, &state).unwrap()
}

#[test]
fn qualified_completion_replaces_the_whole_name_and_carries_documentation() {
    let items = complete("// 🧪\nmodel M() { let x = math.sq|rt(4); }");
    assert_eq!(items.len(), 1);
    let item = &items[0];
    assert_eq!(item.label, "math.sqrt");
    let Some(Documentation::MarkupContent(doc)) = &item.documentation else {
        panic!("Markdown")
    };
    assert!(doc.value.contains("x >= 0"));
    let Some(CompletionTextEdit::Edit(edit)) = &item.text_edit else {
        panic!("edit")
    };
    assert_eq!(edit.new_text, "math.sqrt");
    assert_eq!(edit.range.start, lsp_types::Position::new(1, 20));
    assert_eq!(edit.range.end, lsp_types::Position::new(1, 29));
    assert!(
        complete("model M() { let x = math.| }")
            .iter()
            .any(|i| i.label == "math.pi")
    );
}

#[test]
fn nested_and_incomplete_calls_track_active_parameter_without_counting_inner_commas() {
    let help = signature("model M() { let x = math.clamp(math.min(1, 2), 0, | }").unwrap();
    assert_eq!(help.signatures[0].label, "math.clamp(x, lower, upper)");
    assert_eq!(help.active_parameter, Some(2));
    assert!(
        help.signatures[0].parameters.as_ref().unwrap()[2]
            .documentation
            .is_some()
    );
    let help = signature("model M() { let x = math.clamp([1, 2][0], | }").unwrap();
    assert_eq!(help.active_parameter, Some(1));
    let help = signature("model M() { let x = math.clamp(math.min(1, | }").unwrap();
    assert_eq!(help.signatures[0].label, "math.min(x, y)");
    assert_eq!(help.active_parameter, Some(1));
}

#[test]
fn authored_signatures_keep_parameter_docs_and_named_argument_order() {
    let help = signature("/// Blend two values.\noperator blend(\n/// First value.\ninput x: 1,\n/// Second value.\ninput y: 1): 1 = x+y;\nmodel M() { relation r { blend(y = 2, x = |1) = 3; } }").unwrap();
    assert_eq!(help.active_parameter, Some(0));
    let info = &help.signatures[0];
    assert_eq!(info.label, "operator blend( input x: 1, input y: 1): 1");
    assert_eq!(info.parameters.as_ref().unwrap().len(), 2);
    let Some(Documentation::MarkupContent(doc)) =
        &info.parameters.as_ref().unwrap()[0].documentation
    else {
        panic!("parameter docs")
    };
    assert!(doc.value.contains("First value"));
    assert!(info.documentation.is_some());
}

#[test]
fn component_body_members_are_not_parameters_and_parameter_ranges_skip_callee_names() {
    let help = signature("component Part(\n/// Public gain.\nparameter gain: 1) { parameter internal: 1 = 2; }\nmodel M() { instance p: Part(gain = |1); }").unwrap();
    let parameters = help.signatures[0].parameters.as_ref().unwrap();
    assert_eq!(parameters.len(), 1);
    let Some(Documentation::MarkupContent(doc)) = &parameters[0].documentation else {
        panic!("docs")
    };
    assert!(doc.value.contains("Public gain"));
    let help = signature("model M() { let x = math.max(|1, 2); }").unwrap();
    assert_eq!(
        help.signatures[0].parameters.as_ref().unwrap()[0].label,
        ParameterLabel::LabelOffsets([9, 10])
    );
}

#[test]
fn completion_uses_lexical_scope_and_sanitizes_authored_markdown() {
    let items = complete(
        "component Other() { parameter secret: 1 = 2; }\nmodel M() {\n/// Local value. [run](command:delete)\nparameter local: 1 = 1;\nrelation r { loc| = 1; } }",
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].label, "local");
    let Some(Documentation::MarkupContent(doc)) = &items[0].documentation else {
        panic!("docs")
    };
    assert!(doc.value.contains("Local value"));
    assert!(!doc.value.contains("[run](command:"));
    assert!(
        complete(
            "component Other() { parameter secret: 1 = 2; }\nmodel M() { relation r { sec| = 1; } }"
        )
        .is_empty()
    );
}

#[test]
fn comments_notation_and_unknown_functions_have_no_assistance() {
    assert!(complete("// math.s|").is_empty());
    assert!(signature("model M() { // math.clamp(1, |\n}").is_none());
    assert!(signature("model M() { let x = math.cos(|0); }").is_none());
    assert!(signature("operator blend(input x: 1, |input y: 1): 1 = x+y;").is_none());
    assert!(
        signature("model M() { let x = time(|); }")
            .unwrap()
            .signatures[0]
            .parameters
            .as_ref()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn builtin_hover_and_local_hover_expose_docs_without_resolved_analysis() {
    for source in [
        "model M() { let x = math.sq|rt(4); }",
        "model M() {\n/// The gain.\nparameter gain: 1 = 2; relation r { ga|in = 0; } }",
    ] {
        let (state, uri, position) = fixture(source);
        let result = hover(&uri, position, &state).unwrap().unwrap();
        assert!(result.range.is_some());
        let HoverContents::Markup(contents) = result.contents else {
            panic!("Markdown")
        };
        assert!(contents.value.contains(if source.contains("math.") {
            "Real square root"
        } else {
            "The gain"
        }));
    }
}

#[test]
fn language_constructs_and_types_have_hover_and_documented_completion() {
    for (source, expected) in [
        ("mo|del M() {}", "executable model"),
        (
            "model M() { rel|ation law { 1 = 1; } }",
            "simultaneous mathematical equalities",
        ),
        (
            "model M() { par|ameter gain: 1 = 2; }",
            "static typed parameter",
        ),
        (
            "model M() { variable channels: arr|ay<V, 3>; }",
            "channel axis is not a spatial vector axis",
        ),
        (
            "model M() { parameter count: int|eger = 2; }",
            "checked arithmetic",
        ),
    ] {
        let (state, uri, position) = fixture(source);
        let result = hover(&uri, position, &state).unwrap().unwrap();
        let HoverContents::Markup(contents) = result.contents else {
            panic!("Markdown")
        };
        assert!(contents.value.contains(expected), "{}", contents.value);
        let items = complete(source);
        assert!(items.iter().any(|item| matches!(&item.documentation, Some(Documentation::MarkupContent(doc)) if doc.value.contains(expected))));
    }
    assert_eq!(complete("mod|")[0].kind, Some(CompletionItemKind::KEYWORD));
    assert_eq!(
        complete("model M() { variable x: arr|")[0].kind,
        Some(CompletionItemKind::CLASS)
    );
    assert!(signature("model M(|) {}").is_none());
    assert!(complete("// rel|").is_empty());
    assert!(
        complete("pur|").is_empty(),
        "obsolete pure operator syntax is not suggested"
    );
    assert!(
        complete("rea|").is_empty(),
        "there is no real<...> constructor"
    );
}

#[test]
fn unfinished_local_expressions_keep_current_scope_candidates_at_eof() {
    for source in [
        "model M(parameter gain: 1) { relation r { ga|",
        "model M() { parameter gain: 1 = 2; relation r { ga|",
        "model M() { state position: m; relation r { pos|",
        "component C() {} model M() { instance child: C(); relation r { chi|",
    ] {
        let expected = if source.contains("pos|") {
            "position"
        } else if source.contains("chi|") {
            "child"
        } else {
            "gain"
        };
        assert!(
            complete(source).iter().any(|item| item.label == expected),
            "missing {expected} in {source}"
        );
    }
    assert!(complete("model M() { parameter secret: 1 = 2; } sec|").is_empty());
}
