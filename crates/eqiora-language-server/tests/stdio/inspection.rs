use super::*;

#[test]
fn model_inspection_tracks_selected_unsaved_models_and_rejects_invalid_sources() {
    let source = "model A() { variable x: 1; relation balance { x = 1; } }\nmodel B() { variable y: 1; relation balance { y = 2; } }\n";
    let uri = "file:///workspace/decay.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"B","fingerprint":true}}),
        json!({"jsonrpc":"2.0","id":3,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"Missing"}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":source.replace("y = 2", "y = 3")}]}}),
        json!({"jsonrpc":"2.0","id":4,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"model":"B","fingerprint":true}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":3},"contentChanges":[{"text":"model Broken() { nonsense; }"}]}}),
        json!({"jsonrpc":"2.0","id":5,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri}}}),
        json!({"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}),
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
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["experimental"]["eqioraInspection"],
        1
    );
    let before = &response(&messages, 2)["result"];
    assert_eq!(before["model"], "B");
    assert_eq!(before["errors"], json!([]));
    assert_eq!(before["version"], 1);
    assert!(
        before["equations"][0]["plain"]
            .as_str()
            .unwrap()
            .contains("2")
    );
    assert!(before["fingerprint"].is_string());
    assert!(response(&messages, 3)["error"].is_object());
    let after = &response(&messages, 4)["result"];
    assert_eq!(after["version"], 2);
    assert!(
        after["equations"][0]["plain"]
            .as_str()
            .unwrap()
            .contains("3")
    );
    assert_ne!(before["fingerprint"], after["fingerprint"]);
    let invalid = response(&messages, 5);
    assert!(invalid["error"].is_object() || invalid["result"]["equations"] == json!([]));
}
