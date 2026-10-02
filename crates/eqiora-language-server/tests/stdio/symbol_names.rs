use super::*;

fn name_selection(source: &str, head: &str, prefix: &str, name: &str) -> Value {
    let before = &source[..source.find(head).unwrap() + prefix.len()];
    let line = before.bytes().filter(|byte| *byte == b'\n').count();
    let character = before.rsplit('\n').next().unwrap().encode_utf16().count();
    json!({"start":{"line":line,"character":character},"end":{"line":line,"character":character+name.encode_utf16().count()}})
}

#[test]
fn stdio_outline_selects_current_names_and_keeps_full_declaration_ranges() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\nmodel model(){variable variable @{v}:1;relation relation{variable=1;}}";
    let moved = format!("// moved\r\n{source}\r\nmodel Broken(){{");
    let imported = "import editor.workspace.library as editor; model M(){relation r{1=1;}}";
    let symbols = |id| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}});
    let change = |version, text: &str| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":text}]}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{"textDocument":{"documentSymbol":{"hierarchicalDocumentSymbolSupport":true,"symbolKind":{"valueSet":[3,5,8,25]}}}}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        symbols(2),
        change(2, &moved),
        symbols(3),
        change(1, source),
        symbols(4),
        change(3, imported),
        symbols(5),
        json!({"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ] {
        write_packet(&mut stdin, &message);
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    for (id, text) in [(2, source), (3, moved.as_str()), (4, moved.as_str())] {
        let root = &response(&messages, id)["result"][0];
        assert_eq!(root["name"], "model");
        assert_eq!(root["range"]["start"], source_position(text, "model model"));
        assert_eq!(
            root["selectionRange"],
            name_selection(text, "model model", "model ", "model")
        );
        for (name, head, prefix) in [
            ("variable", "variable variable @{v}", "variable "),
            ("relation", "relation relation", "relation "),
        ] {
            let child = root["children"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["name"] == name)
                .unwrap();
            assert_eq!(child["range"]["start"], source_position(text, head));
            assert_eq!(
                child["selectionRange"],
                name_selection(text, head, prefix, name)
            );
            assert_ne!(child["range"], child["selectionRange"]);
        }
    }
    let alias = response(&messages, 5)["result"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "editor")
        .unwrap();
    assert_eq!(alias["selectionRange"], alias["range"]);
    assert_eq!(alias["range"]["start"], json!({"line":0,"character":0}));
    assert_eq!(
        alias["range"]["end"],
        json!({"line":0,"character":"import editor.workspace.library as editor;".len()})
    );
}
