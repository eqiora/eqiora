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

#[test]
fn stdio_local_references_include_declarations_and_reject_stale_versions() {
    let uri = "file:///workspace/main.eqi";
    let source = "// 🧪\r\nmodel Other(){parameter rate:1=2;}\r\nmodel M(){parameter rate @{r_0}:1=1;parameter unused @{u}:1=0;variable x:1;relation r{x=(rate)+rate;}}";
    let moved = source.replace("relation r{", "relation r{\r\n");
    let invalid = "model M(){parameter rate @{r_0}:1=1;relation r{rate";
    let request = |id, source: &str, occurrence, include| {
        json!({
            "jsonrpc":"2.0","id":id,"method":"textDocument/references",
            "params":{"textDocument":{"uri":uri},"position":source_position(source,occurrence),"context":{"includeDeclaration":include}}
        })
    };
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
        request(2, source, "rate @{r_0}:1=1", false),
        request(3, source, "rate)+", true),
        request(4, source, "unused @{u}:", false),
        request(5, source, "unused @{u}:", true),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":moved}]}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":1},"contentChanges":[{"text":source}]}}),
        request(6, &moved, "rate @{r_0}:1=1", false),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":3},"contentChanges":[{"text":invalid}]}}),
        request(7, invalid, "rate @{r_0}:1=1", true),
        json!({"jsonrpc":"2.0","id":8,"method":"shutdown","params":null}),
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
    let location = |source: &str, occurrence: &str, length| {
        let start = source_position(source, occurrence);
        let end =
            json!({"line":start["line"],"character":start["character"].as_u64().unwrap()+length});
        json!({"uri":uri,"range":{"start":start,"end":end}})
    };
    let uses = vec![location(source, "rate)+", 4), location(source, "rate;}", 4)];
    assert_eq!(response(&messages, 2)["result"], json!(uses));
    assert_eq!(
        response(&messages, 3)["result"],
        json!([location(source, "rate @{r_0}:1=1", 4), uses[0], uses[1]])
    );
    assert_eq!(response(&messages, 4)["result"], json!([]));
    assert_eq!(
        response(&messages, 5)["result"],
        json!([location(source, "unused @{u}:", 6)])
    );
    assert_eq!(
        response(&messages, 6)["result"],
        json!([location(&moved, "rate)+", 4), location(&moved, "rate;}", 4)])
    );
    assert_eq!(response(&messages, 7)["result"], json!([]));
}

#[test]
fn stdio_public_port_definition_tracks_the_target_file_and_unsaved_version() {
    let uri = "file:///workspace/main.eqi";
    let target_uri = "file:///workspace/library.eqi";
    let source = "// 🧪\r\nimport editor.workspace.library as lib;model M(){instance child:lib.Part();relation r{child.value=0;}}";
    let library = "// 🧪\r\npublic component Part(output value @{v}:1){}";
    let moved = format!("// moved\r\n{library}");
    let broken = "public component Part(output value:1){";
    let query = |id| json!({"jsonrpc":"2.0","id":id,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,"value=0")}});
    let change = |version, text| json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":target_uri,"version":version},"contentChanges":[{"text":text}]}});
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{"workspace":{"workspaceFolders":true}},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":target_uri,"languageId":"eqiora","version":1,"text":library}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        query(2),
        change(2, moved.as_str()),
        change(1, library),
        query(3),
        change(3, broken),
        query(4),
        change(4, moved.as_str()),
        query(5),
        json!({"jsonrpc":"2.0","id":6,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,"child.value")}}),
        json!({"jsonrpc":"2.0","id":7,"method":"textDocument/definition","params":{"textDocument":{"uri":uri},"position":source_position(source,".value")}}),
        json!({"jsonrpc":"2.0","id":8,"method":"shutdown","params":null}),
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
    for (id, text) in [(2, library), (3, moved.as_str()), (5, moved.as_str())] {
        let result = &response(&messages, id)["result"];
        assert_eq!(result["uri"], target_uri);
        assert_eq!(result["range"]["start"], source_position(text, "value @{"));
        assert_eq!(
            result["range"]["end"]["line"],
            result["range"]["start"]["line"]
        );
        assert_eq!(
            result["range"]["end"]["character"].as_u64().unwrap(),
            result["range"]["start"]["character"].as_u64().unwrap() + 5
        );
    }
    for id in [4, 6, 7] {
        assert!(response(&messages, id)["result"].is_null());
    }
}
