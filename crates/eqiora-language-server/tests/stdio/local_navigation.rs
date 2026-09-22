use super::*;

#[test]
fn stdio_local_definition_uses_current_model_and_document_version() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\nmodel Other(){parameter rate:1=2;variable x:1;}\r\nmodel M(){parameter rate:1=1;variable x:1;relation r{x=rate;}}";
    let moved = source.replace("model M(){", "model M(){\r\n");
    let invalid = "model M(){variable x:1;relation r{x";
    let shadowed = "model M(){parameter rate:1=1;indexset Rows=range(2);relation r[rate in Rows]{ordinal(rate)=0;}}";
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
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,"x=rate")}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,"rate;")}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":moved}]}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":1},"contentChanges":[{"text":source}]}}),
        json!({"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(&moved,"rate;")}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":3},"contentChanges":[{"text":invalid}]}}),
        json!({"jsonrpc":"2.0","id":5,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":{"line":0,"character":invalid.len()-1}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":4},"contentChanges":[{"text":shadowed}]}}),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(shadowed,"rate)=")}}),
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
    for (id, line, start, end) in [(2, 2, 38, 39), (3, 2, 20, 24), (4, 3, 10, 14)] {
        let result = &response(&messages, id)["result"];
        assert_eq!(result["uri"], uri, "{result}");
        assert_eq!(
            result["range"],
            json!({"start":{"line":line,"character":start},"end":{"line":line,"character":end}})
        );
    }
    for id in [5, 6] {
        assert!(response(&messages, id)["result"].is_null());
    }
}
