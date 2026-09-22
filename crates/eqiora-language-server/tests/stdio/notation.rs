use super::*;

#[test]
fn stdio_declaration_notation_tracks_exact_imports_and_current_overlays() {
    let uri = "file:///workspace/main.eqi";
    let left_uri = "file:///workspace/left.eqi";
    let right_uri = "file:///workspace/right.eqi";
    let main = "// 🧪\nimport editor.workspace.left as left; import editor.workspace.right as right; model M(){instance a:left.Part();instance b:right.Part();}";
    let left = "/// Left declaration.\npublic component Part @{\\mathbf{x_i}}() {}";
    let right = "public component Part @{\\hat{q}^{2}}() {}";
    let changed = "public component Part @{\\alpha_2}() {}";
    let absent = "public component Part() {}";
    let rejected = "public component Part @{\\input{secret}}() {}";
    let request = |id, name| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/hover","params":{"textDocument":{"uri":uri},"position":source_position(main,name)}});
    let change = |version, text| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":left_uri,"version":version},"contentChanges":[{"text":text}]}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":left_uri,"languageId":"eqiora","version":1,"text":left}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":right_uri,"languageId":"eqiora","version":1,"text":right}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":main}}}),
        request(2, "left.Part"),
        request(3, "right.Part"),
        change(2, changed),
        change(1, left),
        request(4, "left.Part"),
        change(3, absent),
        request(5, "left.Part"),
        change(4, rejected),
        request(6, "left.Part"),
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
    for (id, expected, excluded) in [
        (2, "x_{i}", "hat(q)"),
        (3, "hat(q)^{2}", "x_{i}"),
        (4, "alpha_{2}", "x_{i}"),
    ] {
        let result = &response(&messages, id)["result"];
        assert_eq!(result["contents"]["kind"], "markdown");
        let rendered = result["contents"]["value"].as_str().unwrap();
        assert!(
            rendered.contains(&format!("Notation: `{expected}`")),
            "{rendered}"
        );
        assert!(!rendered.contains(excluded), "{rendered}");
    }
    assert!(
        response(&messages, 2)["result"]["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("Left declaration")
    );
    for id in [5, 6] {
        let result = &response(&messages, id)["result"];
        assert!(!result.to_string().contains("Notation:"), "{result}");
        assert!(!result.to_string().contains("alpha_{2}"), "{result}");
    }
}
