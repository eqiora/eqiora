use super::*;

#[test]
fn stdio_documentation_is_shared_by_hover_completion_and_signature_help() {
    let source = "// 🧪\n/// Scale a value.\noperator scale(\n/// Input value.\ninput x: 1,\n/// Scale factor.\ninput factor: 1): 1 = x*factor;\nmodel M() { relation r { scale(factor = 2, x = math.sqrt(4)) = 4; } }\n";
    let uri = "file:///workspace/main.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(source,"sqrt(4)")}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/completion","params":{"textDocument":{"uri":uri},"position":source_position(source,"sqrt(4)")}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":uri},"position":source_position(source,"math.sqrt(4)")}}),
        json!({"jsonrpc":"2.0","id":5,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":uri},"position":source_position(source,"4))")}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":"model M() { let x = math.clamp(1, "}]}}),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":uri},"position":{"line":0,"character":34}}}),
        json!({"jsonrpc":"2.0","id":7,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().unwrap();
    for message in messages {
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
    let capabilities = &response(&messages, 1)["result"]["capabilities"];
    assert_eq!(
        capabilities["completionProvider"]["triggerCharacters"],
        json!(["."])
    );
    assert!(capabilities["signatureHelpProvider"].is_object());
    assert!(
        response(&messages, 2)["result"]["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("Real square root")
    );
    let items = response(&messages, 3)["result"].as_array().unwrap();
    assert!(items.iter().any(|item| {
        item["label"] == "math.sqrt"
            && item["documentation"]["value"]
                .as_str()
                .unwrap()
                .contains("Real square root")
    }));
    let authored = &response(&messages, 4)["result"];
    assert_eq!(authored["activeParameter"], 0);
    assert!(
        authored["signatures"][0]["documentation"]["value"]
            .as_str()
            .unwrap()
            .contains("Scale a value")
    );
    assert!(
        authored["signatures"][0]["parameters"][0]["documentation"]["value"]
            .as_str()
            .unwrap()
            .contains("Input value")
    );
    assert_eq!(
        response(&messages, 5)["result"]["signatures"][0]["label"],
        "math.sqrt(x)"
    );
    assert_eq!(response(&messages, 6)["result"]["activeParameter"], 1);
}
