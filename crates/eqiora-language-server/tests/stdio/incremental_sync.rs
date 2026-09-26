use super::*;

#[test]
fn stdio_applies_ordered_utf16_edits_and_keeps_navigation_current() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\nmodel M() {\r\n parameter x:1=2;\r\n relation r {x=2;}\r\n}\r\n";
    let change = |version, changes| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":changes}});
    let query = |id, method, position| json!({"jsonrpc":"2.0","id":id,"method":method,"params":{"textDocument":{"uri":uri},"position":position}});
    let edit = |line, start, end, text| json!({"range":{"start":{"line":line,"character":start},"end":{"line":line,"character":end}},"rangeLength":999,"text":text});
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
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":-2,"text":source}}}),
        change(
            -1,
            json!([
                edit(0, 3, 5, "😀"),
                edit(2, 11, 12, "value"),
                edit(3, 13, 14, "value"),
                edit(0, 0, 0, "// inserted\n"),
            ]),
        ),
        query(
            2,
            "textDocument/definition",
            json!({"line":4,"character":13}),
        ),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/references","params":{"textDocument":{"uri":uri},"position":{"line":4,"character":13},"context":{"includeDeclaration":true}}}),
        change(
            0,
            json!([edit(3, 11, 16, "broken"), edit(1, 4, 5, "split")]),
        ),
        query(
            4,
            "textDocument/definition",
            json!({"line":4,"character":13}),
        ),
        change(-2, json!([{"text":"model Stale() {}"}])),
        query(
            5,
            "textDocument/definition",
            json!({"line":4,"character":13}),
        ),
        change(
            1,
            json!([{"text":"model Old() {}"}, edit(0, 6, 9, "Final")]),
        ),
        query(
            6,
            "textDocument/documentSymbol",
            json!({"line":0,"character":0}),
        ),
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
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["textDocumentSync"]["change"],
        2
    );
    let expected = json!({"uri":uri,"range":{"start":{"line":3,"character":11},"end":{"line":3,"character":16}}});
    for id in [2, 4, 5] {
        assert_eq!(response(&messages, id)["result"], expected);
    }
    assert_eq!(
        response(&messages, 3)["result"],
        json!([
            expected,
            {"uri":uri,"range":{"start":{"line":4,"character":13},"end":{"line":4,"character":18}}},
        ])
    );
    assert_eq!(response(&messages, 6)["result"][0]["name"], "Final");
    let diagnostics: Vec<_> = messages
        .iter()
        .filter(|m| m["method"] == "textDocument/publishDiagnostics")
        .map(|m| &m["params"])
        .collect();
    assert!(
        diagnostics
            .iter()
            .any(|d| d["version"] == -1 && d["diagnostics"] == json!([]))
    );
    assert!(
        diagnostics.iter().all(|d| d["version"] != 0),
        "rejected batch cannot publish its version"
    );
}
