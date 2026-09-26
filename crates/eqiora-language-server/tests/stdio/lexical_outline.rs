use super::*;

#[test]
fn stdio_named_body_symbols_keep_distinct_kinds_and_authored_prose() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\ncomponent C(){indexset Columns=range(2);let gain:1=2;observable total:1=gain;} model M(){indexset Rows=range(2);\n/// Derived total.\nobservable total:1=2;relation balance[row in Rows]{1=1;}}";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(source,"total:1=2")}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,"total:1=2")}}),
        json!({"jsonrpc":"2.0","id":5,"method":"shutdown","params":null}),
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
    let diagnostics = messages
        .iter()
        .find(|m| m["method"] == "textDocument/publishDiagnostics")
        .unwrap();
    assert_eq!(diagnostics["params"]["diagnostics"], json!([]));
    for (owner, name, kind, label, head) in [
        (
            "C",
            "Columns",
            lsp_types::SymbolKind::ARRAY,
            "Index set",
            "indexset Columns",
        ),
        (
            "C",
            "gain",
            lsp_types::SymbolKind::CONSTANT,
            "Let",
            "let gain",
        ),
        (
            "C",
            "total",
            lsp_types::SymbolKind::VARIABLE,
            "Observable",
            "observable total:1=gain",
        ),
        (
            "M",
            "Rows",
            lsp_types::SymbolKind::ARRAY,
            "Index set",
            "indexset Rows",
        ),
        (
            "M",
            "total",
            lsp_types::SymbolKind::VARIABLE,
            "Observable",
            "observable total:1=2",
        ),
        (
            "M",
            "balance",
            lsp_types::SymbolKind::OPERATOR,
            "Relation",
            "relation balance",
        ),
    ] {
        let symbol = response(&messages, 2)["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == owner)
            .unwrap()["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == name)
            .unwrap();
        assert_eq!(symbol["kind"], json!(kind));
        assert_eq!(symbol["detail"], label);
        assert_eq!(symbol["range"]["start"], source_position(source, head));
    }
    let hover = response(&messages, 3)["result"]["contents"]["value"]
        .as_str()
        .unwrap();
    assert!(hover.contains("observable total:1=2"));
    assert!(hover.contains("Derived total&#46;"));
    assert!(response(&messages, 4)["result"].is_null());
}
