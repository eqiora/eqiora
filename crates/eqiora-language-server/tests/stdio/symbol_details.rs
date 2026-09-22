use super::*;

#[test]
fn stdio_outline_details_follow_current_authority_and_recover_after_invalid_edits() {
    let uri = "file:///workspace/main.eqi";
    let source = "model Other(){variable value:s;} model M(){variable value:m;parameter duration:s=1[s];port sensor:signal input K;}";
    let changed = source.replace("value:m", "value:K");
    let invalid = "model M(){variable value:K;relation r{value=1[m];}}";
    let incomplete = "model M(){variable value:K;relation r{value";
    let query = |id| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":uri}}});
    let change = |version, source: &str| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":version},"contentChanges":[{"text":source}]}});
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
        query(2),
        change(2, &changed),
        change(1, source),
        query(3),
        change(3, invalid),
        query(4),
        change(4, incomplete),
        query(5),
        change(5, &changed),
        query(6),
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
    let detail = |id, model: &str, name: &str| {
        response(&messages, id)["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|value| value["name"] == model)
            .unwrap()["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|value| value["name"] == name)
            .unwrap()["detail"]
            .as_str()
            .unwrap()
    };
    for (owner, name, expected) in [
        ("Other", "value", "dimension T"),
        ("M", "value", "dimension L"),
        ("M", "duration", "parameter; Real; dimension T"),
        ("M", "sensor", "signal Input; Real; dimension Θ"),
    ] {
        assert!(detail(2, owner, name).contains(expected));
    }
    for id in [3, 6] {
        assert!(detail(id, "M", "value").contains("dimension Θ"));
    }
    for id in [4, 5] {
        assert_eq!(detail(id, "M", "value"), "Field");
    }
}
