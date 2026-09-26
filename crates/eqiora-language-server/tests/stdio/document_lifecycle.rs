use super::*;

#[test]
fn stdio_duplicate_open_preserves_text_until_close_and_reopen() {
    let uri = "file:///workspace/main.eqi";
    let open = |version, name| json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":version,"text":format!("model {name}(){{relation r{{1=1;}}}}")}}});
    let symbols = |id| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}});
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
        open(5, "Current"),
        symbols(2),
        open(4, "OldDuplicate"),
        symbols(3),
        open(9, "NewDuplicate"),
        symbols(4),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":6},"contentChanges":[{"text":"model Updated(){relation r{1=1;}}"}]}}),
        symbols(5),
        json!({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":uri}}}),
        open(-3, "Reopened"),
        symbols(6),
        json!({"jsonrpc":"2.0","id":7,"method":"shutdown","params":null}),
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
    for (id, expected) in [
        (2, "Current"),
        (3, "Current"),
        (4, "Current"),
        (5, "Updated"),
        (6, "Reopened"),
    ] {
        assert_eq!(
            response(&messages, id)["result"][0]["name"],
            expected,
            "request {id}"
        );
    }
    let diagnostics: Vec<_> = messages
        .iter()
        .filter(|m| m["method"] == "textDocument/publishDiagnostics")
        .map(|m| &m["params"])
        .collect();
    for version in [5, 6, -3] {
        assert!(
            diagnostics
                .iter()
                .any(|d| d["version"] == version && d["diagnostics"] == json!([])),
            "accepted version {version}"
        );
    }
    assert!(
        diagnostics
            .iter()
            .all(|d| d["version"] != 4 && d["version"] != 9),
        "duplicate opens must not publish"
    );
}
